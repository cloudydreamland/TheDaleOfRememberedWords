"""R12: precision@k and union (split-gold) credit in the retrieval protocol.

Grounding: gold-span evaluation à la LegalBench-RAG (span-level precision
and recall), plus an explicit split-gold credit — punctuation hygiene and
budget cuts may split a gold string across two adjacent chunks; the strict
metric then reports a miss even though the answer arrived in the top-k
context, just split. Union recall reports that case separately, never
merged into the strict number.
"""

from __future__ import annotations

from worddael.chunkers import RecursiveChunker
from worddael.eval.retrieval import recall_at_k
from worddael.eval.sample_data import SAMPLE_DOCS
from worddael.types import Chunk


def test_precision_at_k_counts_relevant_fraction():
    text = "报销政策说明开头。住宿费为每晚三百元。交通费每天八十元封顶。报销需要发票原件。结尾"
    # gold sits in one chunk; other retrieved chunks are noise
    chunks = RecursiveChunker(max_chars=30, overlap_chars=0).chunk(text)
    res = recall_at_k(chunks, text, [{"q": "住宿费每晚多少钱", "gold": "每晚三百元"}], k=3)
    assert 0.0 < res["precision"] <= 1 / 3  # 1 relevant chunk at best among k=3
    assert res["recall"] <= 1.0


def test_union_credit_for_boundary_split_gold():
    """Gold spans a chunk boundary: strict retrieval can never contain the
    whole gold in one chunk, but two top-k chunks cover it together →
    strict=miss, union=hit."""
    text = "AAAAAAAAAA前半部。BBBB后半部延续CCCCCCCCCC"
    gold = "前半部。BBBB"  # crosses the boundary at 13
    boundary = text.find(gold) + len("前半部。")
    chunks = [
        Chunk(text=text[0:boundary], start=0, end=boundary, seq=0, strategy="t", meta={}),
        Chunk(
            text=text[boundary:], start=boundary, end=len(text), seq=1, strategy="t", meta={}
        ),
    ]
    assert text[chunks[0].start : chunks[0].end].endswith("前半部。")
    assert gold not in chunks[0].text and gold not in chunks[1].text  # genuinely split
    res = recall_at_k(chunks, text, [{"q": "前半部和后半部分界", "gold": gold}], k=2)
    assert res["hits"] == 0, "strict must not merge the split-gold case"
    assert res["union_recall"] == 1.0, "union credit must recognise split-gold coverage"


def test_union_not_counted_when_gold_absent():
    text = "内容里根本没有那句话。"
    chunks = [Chunk(text=text, start=0, end=len(text), seq=0, strategy="t", meta={})]
    res = recall_at_k(chunks, text, [{"q": "?", "gold": "不存在的引用"}], k=1)
    assert res["hits"] == 0 and res["union_recall"] == 0.0


def test_corpus_recall_uses_new_fields():
    doc = SAMPLE_DOCS[0]
    chunks = RecursiveChunker(max_chars=150, overlap_chars=30).chunk(doc["text"])
    res = recall_at_k(chunks, doc["text"], doc["questions"], k=3)
    for key in ("recall", "precision", "union_recall", "strict_hits", "union_hits"):
        assert key in res
    assert res["union_recall"] >= res["recall"]  # union credit only widens
