"""Tests for iter7: VectorRetriever, retriever-swappable recall, calibration."""

from __future__ import annotations

import pytest

from worddael.chunkers import RecursiveChunker
from worddael.counters import CharCounter, JiebaCounter, calibration_report
from worddael.embedders import HashingEmbedder
from worddael.eval.report import retrieval_interaction_report
from worddael.eval.retrieval import BM25, VectorRetriever, recall_at_k
from worddael.eval.sample_data import SAMPLE_DOCS


def test_vector_retriever_ranks_similar_first():
    # trigram-sharing texts so the TOY embedder has signal (its near-orthogonality
    # on short unrelated texts is documented — real quality needs real embeddings)
    docs = [
        "报销发票制度：员工提交发票后按流程审核入库并打款",  # shares 报销发/销发票 trigrams with query
        "今天天气很好适合出门散步公园锻炼身体",
    ]
    retr = VectorRetriever(docs, HashingEmbedder(dim=256).embed)
    top = retr.search("报销发票怎么提交审核", k=2)
    assert top and top[0][0] == 0


def test_vector_retriever_filters_zero_similarity():
    def orthogonal_embed(texts):
        # apple-texts → [1,0], anything else → [0,1]: guaranteed cosine 0
        return [[1.0, 0.0] if "苹果" in t else [0.0, 1.0] for t in texts]

    retr = VectorRetriever(["苹果很甜很好吃"], orthogonal_embed)
    assert retr.search("今天天气怎么样", k=3) == []


def test_vector_retriever_empty_docs():
    retr = VectorRetriever([], HashingEmbedder().embed)
    assert retr.search("任何查询", k=3) == []


def test_recall_at_k_with_vector_retriever_factory():
    doc = SAMPLE_DOCS[0]
    chunks = RecursiveChunker(max_chars=150, overlap_chars=0).chunk(doc["text"])
    res = recall_at_k(
        chunks,
        doc["text"],
        doc["questions"],
        k=3,
        retriever_factory=lambda docs: VectorRetriever(docs, HashingEmbedder(dim=128).embed),
    )
    assert res["total"] == len(doc["questions"])
    assert 0.0 <= res["recall"] <= 1.0


def test_retrieval_interaction_report_shape():
    rows = retrieval_interaction_report(k=1, max_chars=150, overlap_chars=30)
    names = {r["strategy"] for r in rows}
    assert {"fixed-window", "recursive"} <= names
    for r in rows:
        assert 0.0 <= r["recall_bm25"] <= 1.0
        assert 0.0 <= r["recall_vector_toy"] <= 1.0


def test_calibration_report_perfect_for_same_counter():
    cal = calibration_report(CharCounter(), CharCounter(), ["abc", "更长的一段文本内容"])
    assert cal["mean_abs_rel_err"] == 0.0
    assert cal["n"] == 2


def test_calibration_report_jieba_reference():
    pytest.importorskip("jieba")
    texts = [d["text"] for d in SAMPLE_DOCS[:5]]
    cal = calibration_report(CharCounter(), JiebaCounter(), texts)
    assert cal["n"] == 5
    # heuristic vs word-level counting: expect drift, but bounded sanity
    assert 0.0 <= cal["mean_abs_rel_err"] < 1.0
    assert all(p["reference"] > 0 for p in cal["per_text"])


def test_calibration_report_empty():
    cal = calibration_report(CharCounter(), CharCounter(), [])
    assert cal["n"] == 0 and cal["mean_abs_rel_err"] == 0.0


def test_bm25_still_default_retriever():
    doc = SAMPLE_DOCS[1]
    chunks = RecursiveChunker(max_chars=150, overlap_chars=0).chunk(doc["text"])
    res_default = recall_at_k(chunks, doc["text"], doc["questions"], k=3)
    res_explicit = recall_at_k(chunks, doc["text"], doc["questions"], k=3, retriever_factory=BM25)
    assert res_default["recall"] == res_explicit["recall"]
