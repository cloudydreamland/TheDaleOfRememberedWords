"""Retrieval-based chunking evaluation.

The idea: a chunking strategy is good if, for a question whose answer lives
at a known position, the chunk containing that position is retrievable.
We use a dependency-free BM25 over chunk texts and measure recall@k of the
gold chunk. Works on CPU, no API needed.
"""

from __future__ import annotations

import math
import re
from collections import Counter

try:
    import jieba

    jieba.setLogLevel(60)  # silence the "Building prefix dict" stderr noise in pipelines
    _HAS_JIEBA = True
except ImportError:  # pragma: no cover
    _HAS_JIEBA = False


def tokenize_zh(text: str) -> list[str]:
    """Jieba words when available; otherwise CJK bigrams + Latin words."""
    text = text.lower()
    if _HAS_JIEBA:
        return [t for t in jieba.cut(text) if not t.isspace()]
    tokens: list[str] = []
    for run in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text):
        if len(run) == 1 and "\u4e00" <= run <= "\u9fff":
            tokens.append(run)
        elif len(run) > 1:
            tokens.append(run)
    # add bigrams for CJK-heavy text when jieba is missing
    cjk_runs = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    for run in cjk_runs:
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


class BM25:
    """Okapi BM25, pure Python. k1=1.5, b=0.75."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs = [tokenize_zh(d) for d in docs]
        self.doc_lens = [len(d) for d in self.docs]
        self.avg_len = sum(self.doc_lens) / max(1, len(self.docs))
        self.tfs = [Counter(d) for d in self.docs]
        df: Counter = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        self.idf = {
            term: math.log((len(docs) - n + 0.5) / (n + 0.5) + 1) for term, n in df.items()
        }

    def score(self, query: str, index: int) -> float:
        q_tokens = tokenize_zh(query)
        tf = self.tfs[index]
        dl = self.doc_lens[index]
        s = 0.0
        for term in q_tokens:
            if term not in tf or term not in self.idf:
                continue
            f = tf[term]
            norm = f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / max(self.avg_len, 1)))
            s += self.idf[term] * norm
        return s

    def search(self, query: str, k: int = 3) -> list[tuple[int, float]]:
        scored = [(i, self.score(query, i)) for i in range(len(self.docs))]
        scored = [(i, s) for i, s in scored if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


class VectorRetriever:
    """Cosine-similarity retriever over chunk embeddings.

    Same ``search`` interface as :class:`BM25` so evaluation code can swap
    retrievers via ``recall_at_k(..., retriever_factory=...)``. Bring your
    own ``embed_fn`` (see :class:`worddael.embedders`) — vectors are computed
    once for the chunk texts; the query is embedded per call.
    """

    def __init__(self, docs: list[str], embed_fn) -> None:
        self._embed_fn = embed_fn
        self._vecs = embed_fn(docs) if docs else []

    @staticmethod
    def _cosine(a, b) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def search(self, query: str, k: int = 3) -> list[tuple[int, float]]:
        if not self._vecs:
            return []
        qv = self._embed_fn([query])[0]
        scored = [(i, self._cosine(qv, v)) for i, v in enumerate(self._vecs)]
        scored = [(i, s) for i, s in scored if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


def find_gold_chunk(chunks, gold: str, source: str) -> int | None:
    """Index of the first chunk fully containing an occurrence of ``gold``,
    or None if no chunk wholly contains it (a boundary-split gold is NOT a
    strict hit — see :func:`recall_at_k`'s union credit)."""
    seqs = find_gold_chunks(chunks, gold, source)
    return min(seqs) if seqs else None


def find_gold_chunks(chunks, gold: str, source: str) -> set[int]:
    """All chunk seqs *fully containing* any occurrence of ``gold``.

    A gold substring may legitimately appear more than once (repeated
    clauses, boilerplate); crediting only the first occurrence mis-scores
    retrievers that found a later, equally valid one. A gold split across a
    chunk boundary fully contains in NO chunk — that case is the union
    credit's job, not strict recall's."""
    seqs: set[int] = set()
    pos = source.find(gold)
    end = pos + len(gold)
    while pos >= 0:
        for c in chunks:
            if c.start <= pos and end <= c.end:
                seqs.add(c.seq)
        pos = source.find(gold, pos + 1)
        end = pos + len(gold)
    return seqs


def recall_at_k(
    chunks,
    source: str,
    questions: list[dict],
    k: int = 3,
    retriever_factory=BM25,
) -> dict:
    """recall@k of a chunking over one document's question set.

    Each question: ``{"q": str, "gold": str}`` where ``gold`` is a verbatim
    substring of ``source`` containing the answer. ``retriever_factory``
    maps a list of chunk texts to a retriever with ``search(query, k)``;
    defaults to BM25, pass ``lambda docs: VectorRetriever(docs, embed_fn)``
    for embedding retrieval.
    """
    docs = [c.text for c in chunks]
    scorer = retriever_factory(docs) if docs else None
    hits = strict_hits = union_hits = 0
    relevant_in_top = 0
    total = 0
    misses: list[str] = []
    for q in questions:
        gold_seqs = find_gold_chunks(chunks, q["gold"], source)
        gold_pos = source.find(q["gold"])
        if gold_pos < 0 or scorer is None:
            total += 1
            misses.append(q["q"])
            continue
        total += 1
        top = [i for i, _ in scorer.search(q["q"], k=k)]
        strict = bool(gold_seqs & set(top))
        # union credit: hygiene/budget boundaries may split a gold across two
        # adjacent chunks — gold_seqs is then EMPTY, but if the top-k chunks
        # together cover the gold (start in one, end in another), the answer
        # did arrive, just split. Reported separately, never merged into the
        # strict number.
        union = False
        if not strict:
            covered = [chunks[i] for i in top]
            g_end = gold_pos + len(q["gold"])
            start_covered = any(c.start <= gold_pos < c.end for c in covered)
            end_covered = any(c.start < g_end <= c.end for c in covered)
            union = start_covered and end_covered
        relevant_in_top += len(gold_seqs & set(top))
        if strict:
            strict_hits += 1
            hits += 1
        elif union:
            union_hits += 1
        else:
            misses.append(q["q"])
    result = {
        "recall": hits / max(1, total),
        "precision": relevant_in_top / max(1, total * k),
        "union_recall": (strict_hits + union_hits) / max(1, total),
        "strict_hits": strict_hits,
        "union_hits": union_hits,
        "hits": hits,
        "total": total,
        "misses": misses,
    }
    return result


def recall_at_k_parent_level(
    children,
    parents,
    source: str,
    questions: list[dict],
    k: int = 3,
) -> dict:
    """Small-to-big recall@k: retrieve over *children*, credit a hit if the
    gold position falls inside the *parent* of any top-k child.

    This mirrors real small-to-big RAG, where the index holds small chunks
    but the LLM sees the larger parent context.
    """
    scorer = BM25([c.text for c in children]) if children else None
    hits = 0
    total = 0
    misses: list[str] = []
    for q in questions:
        total += 1
        gold_pos = source.find(q["gold"])
        if gold_pos < 0 or scorer is None:
            misses.append(q["q"])
            continue
        top = scorer.search(q["q"], k=k)
        parent_seqs = set()
        for i, _ in top:
            ps = children[i].meta.get("parent_seq")
            if ps is not None:
                parent_seqs.add(ps)
        ok = False
        for p in parents:
            if p.start <= gold_pos < p.end and p.seq in parent_seqs:
                ok = True
                break
        if ok:
            hits += 1
        else:
            misses.append(q["q"])
    return {"recall": hits / max(1, total), "hits": hits, "total": total, "misses": misses}
