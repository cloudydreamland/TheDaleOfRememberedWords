"""Tests for the QA generation scaffold — all LLM paths faked."""

from __future__ import annotations

import json

import pytest

from worddael.eval.qa_gen import QAGenerator, QAGenRunner, dry_run_pairs, load_docs_dir

DOC = (
    "第一章 总则。为规范流程特制定本制度，适用于全体正式员工。\n\n"
    "第二章 报销。住宿费报销标准为每晚三百元，需要提供发票。每月最多报销四次。\n\n"
    "第三章 时限。所有发票必须在费用发生后三十天内提交，逾期系统自动关闭入口。"
)


def test_dry_run_pairs_deterministic_and_valid():
    a = dry_run_pairs(DOC, n=3)
    b = dry_run_pairs(DOC, n=3)
    assert a == b  # deterministic
    assert 1 <= len(a) <= 3
    for p in a:
        assert p["method"] == "dry-run"
        assert p["gold"] in DOC  # the quality contract holds even in dry-run
        assert p["gold"].endswith(("。", "！", "？", "；"))


def test_generate_live_validates_gold_verbatim():
    gen = QAGenerator(api_key="k")

    def fake_call(doc_text):
        return [
            {"question": "住宿费标准？", "gold": "每晚三百元"},          # valid
            {"question": "编造的问题？", "gold": "这段话不在文档里"},  # bad gold → dropped
            {"question": "住宿费标准？", "gold": "每晚三百元"},          # dupe → dropped
        ]

    gen._call = fake_call  # type: ignore[method-assign]
    pairs, stats = gen.generate("doc1", DOC, dry_run=False)
    assert stats["raw"] == 3 and stats["kept"] == 1
    assert stats["dropped_bad_gold"] == 1 and stats["dropped_dupe"] == 1
    assert pairs == [{"question": "住宿费标准？", "gold": "每晚三百元", "method": "llm"}]


def test_generate_live_malformed_response_raises():
    gen = QAGenerator(api_key="k")
    gen._call = lambda doc_text: (_ for _ in ()).throw(ValueError("no JSON array"))
    with pytest.raises(ValueError):
        gen.generate("doc1", DOC, dry_run=False)


def test_runner_resume_and_aggregation(tmp_path):
    state = tmp_path / "qa.jsonl"
    docs = [{"id": "a", "title": "a", "text": DOC}, {"id": "b", "title": "b", "text": DOC}]
    runner = QAGenRunner(QAGenerator(), state_path=state, sleep=lambda s: None)

    agg1 = runner.run(docs, dry_run=True)
    assert agg1["docs"] == 2 and agg1["kept"] > 0 and agg1["skipped"] == 0
    agg2 = runner.run(docs, dry_run=True)
    assert agg2["skipped"] == 2 and agg2["docs"] == 0  # resume: nothing redone
    lines = state.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


def test_runner_retries_then_error_record(tmp_path):
    state = tmp_path / "qa.jsonl"
    gen = QAGenerator(api_key="k")

    def always_fail(doc_text):
        raise RuntimeError("api down")

    gen._call = always_fail  # type: ignore[method-assign]
    runner = QAGenRunner(gen, state_path=state, sleep=lambda s: None)
    agg = runner.run([{"id": "x", "title": "x", "text": DOC}], dry_run=False, max_retries=1)
    assert agg["errors"] == 1
    rec = json.loads(state.read_text(encoding="utf-8").strip())
    assert "api down" in rec["error"]


def test_load_docs_dir(tmp_path):
    (tmp_path / "b.md").write_text("markdown 文档内容。", encoding="utf-8")
    (tmp_path / "a.txt").write_text("纯文本文档内容。", encoding="utf-8")
    (tmp_path / "c.pdf").write_text("ignored", encoding="utf-8")
    docs = load_docs_dir(tmp_path)
    assert [d["id"] for d in docs] == ["a", "b"]  # sorted, pdf ignored
