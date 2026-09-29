"""R10: data-engineer and UX round — schema versioning, error message quality."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from worddael import get_chunker
from worddael.chunkers import RecursiveChunker, SemanticChunker
from worddael.eval.judge_runner import JudgeRunner
from worddael.eval.qa_gen import QAGenerator, QAGenRunner
from worddael.parent_child import ParentChildChunker


class FakeJudge:
    def estimate_cost(self, n_calls: int, avg_chars: int) -> dict:
        return {"n_calls": n_calls, "est_cost": 0.0}

    def score_pair(self, chunk_text: str, question: str, dry_run: bool = True) -> dict:
        return {"score": 4, "reason": "ok", "method": "dry-run-bm25"}


def _pair() -> dict:
    return {"strategy": "s", "doc_id": "d", "q_idx": 0, "question": "q", "chunk_seq": 0, "chunk_text": "t"}


def test_judge_state_records_carry_schema_version(tmp_path):
    state = tmp_path / "j.jsonl"
    JudgeRunner(FakeJudge(), state_path=state, sleep=lambda s: None).run([_pair()], dry_run=True)
    rec = json.loads(state.read_text(encoding="utf-8").strip())
    assert rec["schema"] == 1


def test_legacy_state_without_schema_still_resumes(tmp_path):
    """A pre-schema state file (written before R10) must keep resuming —
    the reader tolerates missing fields."""
    state = tmp_path / "j.jsonl"
    state.write_text(json.dumps({"key": "s|d|0|0", "score": 4, "method": "dry-run-bm25"}), encoding="utf-8")
    stats = JudgeRunner(FakeJudge(), state_path=state, sleep=lambda s: None).run([_pair()], dry_run=True)
    assert stats.skipped == 1 and stats.judged == 0


def test_qa_state_records_carry_schema_version(tmp_path):
    state = tmp_path / "qa.jsonl"
    docs = [{"id": "a", "title": "a", "text": "报销标准为每晚三百元。" * 5}]
    QAGenRunner(QAGenerator(), state_path=state, sleep=lambda s: None).run(docs, dry_run=True)
    rec = json.loads(state.read_text(encoding="utf-8").strip().splitlines()[0])
    assert rec["schema"] == 1


def test_cli_missing_file_message_is_actionable(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "worddael.cli", "--compare", str(tmp_path / "nope.txt")],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1
    assert "file not found" in proc.stderr
    assert "worddael demo" in proc.stderr  # points the newcomer somewhere useful


def test_error_messages_state_the_valid_range():
    """UX contract: parameter validation errors say what IS valid."""
    with pytest.raises(ValueError, match=">= 1"):
        RecursiveChunker(max_chars=0)
    with pytest.raises(ValueError, match=r"\[0, max_chars\)"):
        SemanticChunker(embed_fn=lambda t: t, overlap_chars=99, max_chars=10)
    with pytest.raises(ValueError, match="< parent_max_chars"):
        ParentChildChunker(parent_max_chars=100, child_max_chars=100)
    with pytest.raises(ValueError, match="recursive"):
        get_chunker("nonsense")  # lists a valid option
