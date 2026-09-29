"""Tests for the eval subpackage (retrieval recall + judge dry-run + report)."""

from __future__ import annotations

from worddael.chunkers import RecursiveChunker
from worddael.eval.llm_judge import JudgeConfig, LLMJudge
from worddael.eval.report import FixedWindowChunker, run_report
from worddael.eval.retrieval import BM25, find_gold_chunk, recall_at_k, tokenize_zh
from worddael.eval.sample_data import SAMPLE_DOCS


def test_tokenize_zh_basic():
    toks = tokenize_zh("中文分词测试 chunking 123")
    assert any("中文" in t or "中文" == t for t in toks)
    assert "chunking" in toks or "chunk" in toks


def test_bm25_ranks_relevant_doc_first():
    docs = ["红烧肉的做法，焯水炒糖色炖煮", "今天天气很好，适合出门散步", "财务报销制度与发票要求"]
    bm = BM25(docs)
    top = bm.search("报销 发票 怎么处理", k=1)
    assert top and top[0][0] == 2


def test_gold_chunk_found_and_recall_sane():
    doc = SAMPLE_DOCS[0]
    chunks = RecursiveChunker(max_chars=120, overlap_chars=0).chunk(doc["text"])
    res = recall_at_k(chunks, doc["text"], doc["questions"], k=3)
    assert res["total"] == len(doc["questions"])
    assert 0.0 <= res["recall"] <= 1.0


def test_find_gold_chunk_none_when_absent():
    doc = SAMPLE_DOCS[0]
    chunks = RecursiveChunker(max_chars=100).chunk(doc["text"])
    assert find_gold_chunk(chunks, "这段话不存在于文档之中", doc["text"]) is None


def test_fixed_window_baseline_chunks():
    text = SAMPLE_DOCS[1]["text"]
    c = FixedWindowChunker(max_chars=100)
    chunks = c.chunk(text)
    assert all(ch.n_chars <= 100 for ch in chunks)
    assert all(ch.text == text[ch.start : ch.end] for ch in chunks)


def test_report_runs_all_strategies():
    rows = run_report(k=3, max_chars=200, overlap_chars=30)
    names = {r["strategy"] for r in rows}
    assert {"fixed-window", "sentence", "token", "recursive", "semantic-hash-toy"} <= names
    for r in rows:
        assert 0.0 <= r["recall@k"] <= 1.0


def test_corpus_integrity():
    """The benchmark must stay valid: 25+ docs, 3 unique questions each, and
    every gold string is a verbatim substring of its document."""
    assert len(SAMPLE_DOCS) >= 25
    for d in SAMPLE_DOCS:
        assert len(d["questions"]) == 3
        assert len({q["q"] for q in d["questions"]}) == 3
        assert len(d["text"]) >= 250, f"{d['id']} too short for a meaningful benchmark"
        for q in d["questions"]:
            assert q["gold"] in d["text"], f"gold not verbatim in {d['id']}: {q['gold']}"
    total_q = sum(len(d["questions"]) for d in SAMPLE_DOCS)
    assert total_q >= 75


def test_structure_aware_chunks_end_on_sentence_punctuation():
    """Design-guaranteed property: recursive chunks almost always end on a
    sentence boundary; blind fixed windows cut wherever the limit lands."""
    rows = {r["strategy"]: r for r in run_report(k=1, max_chars=150, overlap_chars=30)}
    assert rows["recursive"]["ends_on_punct_frac"] > 0.5
    assert (
        rows["recursive"]["ends_on_punct_frac"] >= rows["fixed-window"]["ends_on_punct_frac"]
    )


def test_judge_dry_run_costs_nothing():
    doc = SAMPLE_DOCS[2]
    chunks = RecursiveChunker(max_chars=200, overlap_chars=0).chunk(doc["text"])
    judge = LLMJudge(JudgeConfig())
    result = judge.score_pair(chunks[0].text, doc["questions"][0]["q"], dry_run=True)
    assert result["method"] == "dry-run-bm25"
    assert 1 <= result["score"] <= 5


def test_judge_cost_estimate():
    est = LLMJudge(JudgeConfig(price_per_1k_input=0.001)).estimate_cost(n_calls=100, avg_chars=300)
    assert est["n_calls"] == 100
    assert est["est_cost"] > 0


def test_report_is_byte_deterministic():
    """crc32 hashing + sorted everything → two runs must produce the exact
    same report. Guards the determinism guarantee end to end."""
    r1 = run_report(k=1, max_chars=150, overlap_chars=30)
    r2 = run_report(k=1, max_chars=150, overlap_chars=30)
    def strip(rows):
        return [{k: v for k, v in r.items() if k != "time_ms"} for r in rows]

    assert strip(r1) == strip(r2)
