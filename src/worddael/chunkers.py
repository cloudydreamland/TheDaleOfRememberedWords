"""Chunker implementations.

Every chunker produces ``Chunk`` objects whose ``text`` is an exact slice of
the source string (``source[start:end]``), and whose spans tile the source
(the only gap allowed is none — tiling is complete). This is the invariant
that enables reliable citation in RAG systems, and it is fuzz-tested.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from .counters import CharCounter, TokenCounter, chars_for_tokens, get_counter
from .sentences import SENT_END, markdown_sections, split_sentences
from .types import Chunk

# Separator hierarchy for the recursive chunker, Chinese-first.
# Precompiled once: these run over every byte of every document.
_ZH_SENT_RE = re.compile(r"[。！？；…]")
_ZH_COMMA_RE = re.compile(r"[，、]")
_PARA_RE = re.compile(r"\n\s*\n")
_NEWLINE_RE = re.compile(r"\n")

_PUNCT_OPENERS = set("「『“《（〔［｛'")
_PUNCT_CLOSERS = set("」』”））》】〉，。！？；：、…!?;,.")
_SENT_END_SET = set(SENT_END)


def _split_keep(text: str, pattern) -> list[tuple[int, int]]:
    """Split by regex but keep separators; return spans tiling ``text``.
    ``pattern`` may be a compiled pattern or a string."""
    if isinstance(pattern, str):
        pattern = re.compile(pattern)
    spans: list[tuple[int, int]] = []
    pos = 0
    for m in pattern.finditer(text):
        if m.start() > pos:
            spans.append((pos, m.start()))
        spans.append((m.start(), m.end()))
        pos = m.end()
    if pos < len(text):
        spans.append((pos, len(text)))
    return spans


def _merge_spans(
    text: str, spans: list[tuple[int, int]], budget: int, measure
) -> list[tuple[int, int]]:
    """Greedily merge adjacent spans while the merged unit fits ``budget``.

    ``measure(text, s, e) -> int`` returns the size of a span. Merged spans
    stay contiguous, so offsets remain exact.
    """
    out: list[tuple[int, int]] = []
    cur_s, cur_e = spans[0]
    cur_size = measure(text, cur_s, cur_e)
    for s, e in spans[1:]:
        add = measure(text, s, e)
        if cur_size + add <= budget:
            cur_e = e
            cur_size += add
        else:
            out.append((cur_s, cur_e))
            cur_s, cur_e = s, e
            cur_size = add
    out.append((cur_s, cur_e))
    return out


def _hard_char_split(text: str, max_chars: int) -> list[tuple[int, int]]:
    step = max(1, max_chars)
    return [(s, min(s + step, len(text))) for s in range(0, len(text), step)]


def _hygiene_pass(
    text: str, spans: list[tuple[int, int]], max_shift: int = 8
) -> list[tuple[int, int]]:
    """Punctuation hygiene on interior tiling boundaries.

    Guarantees: a chunk never *ends* (after rstrip) on an opening
    punctuation (a quote or bracket it never closes) and never *starts*
    (after lstrip) on a closing punctuation (a dangling comma/period).
    Checks skip whitespace, because separators attach to chunk tails and
    would otherwise mask the offending character. Fixes move the boundary
    so the offending punct lands where it is legal — openers move to the
    right chunk's start, closers are absorbed by the left chunk's end.
    Tiling is preserved; chunks only shrink.
    """

    def last_nonws(b: int, lo: int) -> int:
        j = b - 1
        while j >= lo and text[j].isspace():
            j -= 1
        return j

    def first_nonws(b: int, hi: int) -> int:
        j = b
        while j < hi and text[j].isspace():
            j += 1
        return j

    out = list(spans)
    for i in range(len(out) - 1):
        s1, e1 = out[i]
        s2, e2 = out[i + 1]
        b = e1
        shift = 0
        while shift < max_shift and b - 1 > s1:
            j = last_nonws(b, s1)
            if j >= s1 and text[j] in _PUNCT_OPENERS:
                b = j  # opener moves to the right start — left chunk shrinks
                shift += 1
                continue
            k = first_nonws(b, e2)
            if (
                k < e2 - 1
                and text[k] in _PUNCT_CLOSERS
                and k + 1 - e1 <= 2  # absorbing a closer grows the left chunk; cap it
            ):
                b = k + 1
                shift += 1
                continue
            break
        if b != e1:
            out[i] = (s1, b)
            out[i + 1] = (b, e2)
    return out


def _snap_overlap_start(text: str, s: int, ns: int) -> int:
    """Align the overlap window start to the latest sentence start in
    ``[ns, s)``; fall back to skipping dangling closers, then to ``ns``."""
    for k in range(s - 1, ns, -1):
        j = k - 1
        while j >= 0 and text[j] in "」』”）)》】〉":
            j -= 1
        if j >= 0 and text[j] in _SENT_END_SET:
            return max(k, ns)
    while ns < s and text[ns] in _PUNCT_CLOSERS:
        ns += 1
    return ns


def _apply_overlap(
    text: str, spans: list[tuple[int, int]], overlap_chars: int, max_chars: int
) -> list[tuple[int, int, int]]:
    """Extend each span's start backwards into the previous chunk to create
    a sentence-aligned character overlap. Chunks remain exact source slices."""
    spans = _hygiene_pass(text, spans)
    if overlap_chars <= 0 or len(spans) <= 1:
        return [(s, e, 0) for s, e in spans]
    out: list[tuple[int, int, int]] = [(spans[0][0], spans[0][1], 0)]
    for i in range(1, len(spans)):
        s, e = spans[i]
        prev_s, _ = spans[i - 1]
        ns = s - min(overlap_chars, s)
        ns = _snap_overlap_start(text, s, ns)
        ns = max(ns, e - max_chars, prev_s + 1)
        # clamps may land back on a dangling closer — skip it (hygiene holds
        # even when the budget cap squeezes the overlap window)
        while ns < s and text[ns] in _PUNCT_CLOSERS:
            ns += 1
        if ns >= s:
            ns = s
        out.append((ns, e, s - ns))
    return out


class BaseChunker(ABC):
    strategy: str = ""
    explain: bool = False

    @abstractmethod
    def chunk(self, text: str) -> list[Chunk]: ...

    def _build(
        self,
        text: str,
        spans: list[tuple[int, int, int]],
        extra_meta=None,
        end_rule_overrides: dict[int, str] | None = None,
    ) -> list[Chunk]:
        metas: list[dict] = []
        for seq, (s, e, ov) in enumerate(spans):
            meta: dict = {"overlap_chars": ov}
            if extra_meta is not None:
                meta.update(extra_meta(seq, s, e) or {})
            metas.append(meta)
        if self.explain:
            from .explain import annotate  # noqa: PLC0415 — zero-cost unless requested

            annotate(text, spans, metas, end_rule_overrides)
        chunks: list[Chunk] = []
        for seq, ((s, e, ov), meta) in enumerate(zip(spans, metas)):
            chunks.append(
                Chunk(text=text[s:e], start=s, end=e, seq=seq, strategy=self.strategy, meta=meta)
            )
        return chunks


class TokenChunker(BaseChunker):
    """Pack sentence-sized units into a token budget.

    ``counter`` defaults to the dependency-free ``CharCounter`` heuristic;
    pass ``counter="jieba"`` / ``counter="tiktoken"`` (or an object with
    ``.count``) for exact counts.
    """

    strategy = "token"

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 0,
        counter: TokenCounter | str | None = None,
        explain: bool = False,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if not 0 <= overlap_tokens < max_tokens:
            raise ValueError("overlap_tokens must be in [0, max_tokens)")
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.explain = explain
        self.counter: TokenCounter = (
            get_counter(counter) if isinstance(counter, str) else (counter or CharCounter())
        )

    def chunk(self, text: str) -> list[Chunk]:
        if not text:
            return []
        cnt = self.counter.count
        atoms = split_sentences(text)
        sized = [(s, e, cnt(text[s:e])) for s, e in atoms]

        # Oversized single sentences get hard-split on a token-proportional
        # character estimate.
        expanded: list[tuple[int, int, int]] = []
        for s, e, t in sized:
            if t <= self.max_tokens:
                expanded.append((s, e, t))
                continue
            n_chars = e - s
            chars_per_tok = max(1, round(n_chars / max(t, 1)))
            win = chars_per_tok * self.max_tokens
            for ws in range(s, e, win):
                we = min(ws + win, e)
                expanded.append((ws, we, cnt(text[ws:we])))

        spans: list[tuple[int, int]] = []
        cur_s, cur_e, cur_t = expanded[0]
        for s, e, t in expanded[1:]:
            if cur_t + t <= self.max_tokens:
                cur_e = e
                cur_t += t
            else:
                spans.append((cur_s, cur_e))
                cur_s, cur_e, cur_t = s, e, t
        spans.append((cur_s, cur_e))

        ov_chars = 0
        if self.overlap_tokens:
            ov_chars = chars_for_tokens(self.counter, text, self.overlap_tokens)
        final = _apply_overlap(text, spans, ov_chars, max_chars=10**9)
        return self._build(text, final)


class SentenceChunker(BaseChunker):
    """Group whole Chinese-aware sentences up to ``max_chars`` each."""

    strategy = "sentence"

    def __init__(self, max_chars: int = 500, overlap_chars: int = 0, explain: bool = False) -> None:
        if max_chars < 1:
            raise ValueError("max_chars must be >= 1")
        if not 0 <= overlap_chars < max_chars:
            raise ValueError("overlap_chars must be in [0, max_chars)")
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.explain = explain

    def chunk(self, text: str) -> list[Chunk]:
        if not text:
            return []
        spans = _merge_spans(text, split_sentences(text), self.max_chars, lambda t, s, e: e - s)
        final = _apply_overlap(text, spans, self.overlap_chars, self.max_chars)
        return self._build(text, final)


class RecursiveChunker(BaseChunker):
    """Structure-aware recursive splitting with a Chinese-first separator
    hierarchy: paragraphs → lines → 句号/问号/感叹号 → 逗号/顿号 → characters.
    The workhorse for plain Chinese text."""

    strategy = "recursive"

    def __init__(self, max_chars: int = 500, overlap_chars: int = 50, explain: bool = False) -> None:
        if max_chars < 1:
            raise ValueError("max_chars must be >= 1")
        if not 0 <= overlap_chars < max_chars:
            raise ValueError("overlap_chars must be in [0, max_chars)")
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.explain = explain

    def _split(self, text: str, level: int) -> list[tuple[int, int]]:
        patterns = [_PARA_RE, _NEWLINE_RE, _ZH_SENT_RE, _ZH_COMMA_RE, ""]
        if level >= len(patterns):
            return _hard_char_split(text, self.max_chars)
        pattern = patterns[level]
        if not pattern:
            return _hard_char_split(text, self.max_chars)
        parts = _split_keep(text, pattern)
        if len(parts) <= 1:
            return self._split(text, level + 1)
        out: list[tuple[int, int]] = []
        for s, e in parts:
            if e - s <= self.max_chars:
                out.append((s, e))
            else:
                out.extend((s2 + s, e2 + s) for s2, e2 in self._split(text[s:e], level + 1))
        return out

    def chunk(self, text: str) -> list[Chunk]:
        if not text:
            return []
        raw = self._split(text, 0)
        # raw spans may be unordered across branches; restore source order
        raw.sort(key=lambda se: se[0])
        spans = _merge_spans(text, raw, self.max_chars, lambda t, s, e: e - s)
        final = _apply_overlap(text, spans, self.overlap_chars, self.max_chars)
        return self._build(text, final)


class MarkdownChunker(BaseChunker):
    """Split markdown by heading structure; oversized sections fall back to
    a recursive split inside the section. Code fences stay atomic. The
    heading path (e.g. ``("# 安装", "## 依赖")``) is stored in ``meta``."""

    strategy = "markdown"

    def __init__(self, max_chars: int = 500, overlap_chars: int = 50, explain: bool = False) -> None:
        if max_chars < 1:
            raise ValueError("max_chars must be >= 1")
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.explain = explain
        self._inner = RecursiveChunker(max_chars=max_chars, overlap_chars=overlap_chars)

    def chunk(self, text: str) -> list[Chunk]:
        if not text:
            return []
        sections = markdown_sections(text)
        spans: list[tuple[int, int, int]] = []
        headings_of: list[tuple[str, ...]] = []
        end_overrides: dict[int, str] = {}
        for si, sec in enumerate(sections):
            sec_text = text[sec.start : sec.end]
            if len(sec_text) <= self.max_chars:
                sec_spans = [(sec.start, sec.end, 0)]
            else:
                inner = self._inner.chunk(sec_text)
                sec_spans = [
                    (c.start + sec.start, c.end + sec.start, c.meta["overlap_chars"]) for c in inner
                ]
            for j, (s, e, ov) in enumerate(sec_spans):
                spans.append((s, e, ov))
                headings_of.append(sec.headings)
                # a section-final chunk ends because the next heading starts
                if j == len(sec_spans) - 1 and si < len(sections) - 1:
                    end_overrides[len(spans) - 1] = "heading"

        def extra_meta(seq: int, s: int, e: int):
            return {"headings": list(headings_of[seq])}

        return self._build(text, spans, extra_meta=extra_meta, end_rule_overrides=end_overrides)


class SemanticChunker(BaseChunker):
    """Split where embedding similarity between adjacent sentences drops.

    ``embed_fn`` maps a list of strings to a list of float vectors — pass any
    callable or a :class:`worddael.embedders.Embedder` instance (e.g.
    :class:`worddael.embedders.OpenAICompatibleEmbedder` for an embeddings API).
    The core library stays dependency-free: cosine similarity is computed in
    pure Python."""

    strategy = "semantic"

    def __init__(
        self,
        embed_fn,
        similarity_threshold: float = 0.55,
        max_chars: int = 500,
        min_chars: int = 50,
        overlap_chars: int = 0,
        explain: bool = False,
    ) -> None:
        if embed_fn is None:
            raise ValueError("SemanticChunker requires an embed_fn(list[str]) -> list[vector]")
        if not 0 <= overlap_chars < max_chars:
            raise ValueError("overlap_chars must be in [0, max_chars)")
        if min_chars > max_chars:
            raise ValueError("min_chars must be <= max_chars")
        self.embed_fn = embed_fn
        self.similarity_threshold = similarity_threshold
        self.max_chars = max_chars
        self.min_chars = min_chars
        self.overlap_chars = overlap_chars
        self.explain = explain

    @staticmethod
    def _cosine(a, b) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def chunk(self, text: str) -> list[Chunk]:
        if not text:
            return []
        atoms = split_sentences(text)
        texts = [text[s:e] for s, e in atoms]
        vecs = self.embed_fn(texts)
        if len(vecs) != len(texts):
            raise ValueError("embed_fn must return one vector per input string")

        spans: list[tuple[int, int]] = []
        cut_reasons: list[str] = []
        cur_s, cur_e = atoms[0]
        for i in range(1, len(atoms)):
            s, e = atoms[i]
            sim = self._cosine(vecs[i - 1], vecs[i])
            group_len = cur_e - cur_s
            # Cut on dissimilarity only once the current group is big enough;
            # this prevents fragmenting short adjacent sentences.
            if (sim < self.similarity_threshold and group_len >= self.min_chars) or (
                e - cur_s
            ) > self.max_chars:
                spans.append((cur_s, cur_e))
                cut_reasons.append("semantic_drop" if sim < self.similarity_threshold else "budget")
                cur_s, cur_e = s, e
            else:
                cur_e = e
        spans.append((cur_s, cur_e))
        cut_reasons.append("text_end")

        # Respect the size budget without merging across semantic cuts.
        final_spans: list[tuple[int, int]] = []
        final_reasons: list[str] = []
        for (s, e), reason in zip(spans, cut_reasons):
            if e - s <= self.max_chars:
                final_spans.append((s, e))
                final_reasons.append(reason)
            else:
                pieces = _hard_char_split(text[s:e], self.max_chars)
                for a, b in pieces:
                    final_spans.append((a + s, b + s))
                    final_reasons.append("budget" if b < e else reason)
        final = _apply_overlap(text, final_spans, self.overlap_chars, self.max_chars)
        overrides = dict(enumerate(final_reasons))
        return self._build(text, final, end_rule_overrides=overrides)


STRATEGIES = {
    "token": TokenChunker,
    "sentence": SentenceChunker,
    "recursive": RecursiveChunker,
    "markdown": MarkdownChunker,
    "semantic": SemanticChunker,
}
