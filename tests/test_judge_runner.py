"""Tests for JudgeRunner: resume, retries, rate limiting, cost cap, summarize.

All judge backends are fakes — no network, no sleeping (time is injected).
"""

from __future__ import annotations

import json

import pytest

from worddael.eval.judge_runner import JudgeRunner, build_pairs, pair_key, summarize
from worddael.eval.sample_data import SAMPLE_DOCS


class FakeJudge:
    def __init__(self, fail_times: int = 0, score: int = 4) -> None:
        self.calls: list[tuple] = []
        self.fail_times = fail_times
        self.score = score

    def estimate_cost(self, n_calls: int, avg_chars: int) -> dict:
        return {
            "n_calls": n_calls,
            "est_input_tokens": n_calls * avg_chars,
            "est_cost": 0.01 * n_calls,
            "currency_note": "fake",
        }

    def score_pair(self, chunk_text: str, question: str, dry_run: bool = True) -> dict:
        self.calls.append((chunk_text, question, dry_run))
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("transient failure")
        return {"score": self.score, "reason": "ok", "method": "dry-run-bm25" if dry_run else "llm"}


def _pairs(n: int = 3) -> list[dict]:
    return [
        {"strategy": "s", "doc_id": "d", "q_idx": i, "question": f"q{i}", "chunk_seq": 0, "chunk_text": f"chunk {i}"}
        for i in range(n)
    ]


def test_run_records_all_pairs(tmp_path):
    state = tmp_path / "judge.jsonl"
    runner = JudgeRunner(FakeJudge(), state_path=state, sleep=lambda s: None)
    stats = runner.run(_pairs(3), dry_run=True)
    assert stats.judged == 3 and stats.errors == 0 and stats.skipped == 0
    lines = state.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    keys = [json.loads(line)["key"] for line in lines]
    assert keys == [pair_key(p) for p in _pairs(3)]


def test_resume_skips_completed_pairs(tmp_path):
    state = tmp_path / "judge.jsonl"
    judge = FakeJudge()
    JudgeRunner(judge, state_path=state, sleep=lambda s: None).run(_pairs(3), dry_run=True)
    stats = JudgeRunner(judge, state_path=state, sleep=lambda s: None).run(_pairs(3), dry_run=True)
    assert stats.judged == 0
    assert stats.skipped == 3
    assert len(judge.calls) == 3  # no second-pass calls
    lines = state.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3  # no duplicate records


def test_rate_limiting_sleeps_between_calls(tmp_path):
    sleeps: list[float] = []
    ticks = iter([100.0, 100.0, 100.0, 100.0])  # fake monotonic clock
    runner = JudgeRunner(
        FakeJudge(),
        state_path=tmp_path / "s.jsonl",
        rpm=30,  # min_interval = 2.0s
        sleep=sleeps.append,
        monotonic=lambda: next(ticks, 100.0),
    )
    runner.run(_pairs(3), dry_run=True)
    assert sleeps == [2.0, 2.0]  # sleeps between call1->2 and call2->3


def test_retries_then_succeeds(tmp_path):
    sleeps: list[float] = []
    judge = FakeJudge(fail_times=2)
    runner = JudgeRunner(judge, state_path=tmp_path / "s.jsonl", max_retries=3, sleep=sleeps.append)
    stats = runner.run(_pairs(1), dry_run=True)
    assert stats.judged == 1 and stats.errors == 0
    assert sleeps == [2.0, 4.0]  # exponential backoff


def test_retries_exhausted_records_error(tmp_path):
    judge = FakeJudge(fail_times=10)
    runner = JudgeRunner(judge, state_path=tmp_path / "s.jsonl", max_retries=2, sleep=lambda s: None)
    stats = runner.run(_pairs(1), dry_run=True)
    assert stats.errors == 1 and stats.judged == 0
    rec = json.loads((tmp_path / "s.jsonl").read_text(encoding="utf-8").strip())
    assert rec["method"] == "error" and "transient" in rec["error"]


def test_cost_cap_stops_live_run(tmp_path):
    judge = FakeJudge()
    runner = JudgeRunner(judge, state_path=tmp_path / "s.jsonl", sleep=lambda s: None, cost_cap=0.015)
    stats = runner.run(_pairs(3), dry_run=False)  # est 0.01/call
    assert stats.cost_capped is True
    assert stats.judged == 1  # first call 0.01 fits; second would exceed cap
    assert len(judge.calls) == 1


def test_torn_last_line_tolerated(tmp_path):
    state = tmp_path / "s.jsonl"
    good = {"key": "s|d|0|0", "score": 4}
    state.write_text(json.dumps(good) + "\n{torn json", encoding="utf-8")
    runner = JudgeRunner(FakeJudge(), state_path=state, sleep=lambda s: None)
    assert set(runner.load_done()) == {"s|d|0|0"}


def test_build_pairs_crosses_chunks_and_questions():
    from worddael.chunkers import RecursiveChunker

    doc = SAMPLE_DOCS[0]
    chunker = RecursiveChunker(max_chars=150, overlap_chars=0)
    chunks = chunker.chunk(doc["text"])
    pairs = build_pairs({"recursive": chunker}, [doc])
    assert len(pairs) == len(chunks) * len(doc["questions"])
    assert all(p["strategy"] == "recursive" for p in pairs)
    assert {p["q_idx"] for p in pairs} == {0, 1, 2}


def test_summarize_aggregates_per_strategy(tmp_path):
    state = tmp_path / "s.jsonl"
    records = [
        {"strategy": "a", "score": 4, "method": "llm"},
        {"strategy": "a", "score": 2, "method": "llm"},
        {"strategy": "b", "score": -1, "method": "error"},
    ]
    state.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    summary = summarize(state)
    assert summary["a"]["n"] == 2 and summary["a"]["mean_score"] == 3.0
    assert summary["b"]["errors"] == 1 and summary["b"]["n"] == 0


def test_runner_validation():
    with pytest.raises(ValueError):
        JudgeRunner(FakeJudge(), state_path="x", rpm=0)
    with pytest.raises(ValueError):
        JudgeRunner(FakeJudge(), state_path="x", max_retries=-1)
