"""Tests for file IO and the CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from worddael import chunk_file, to_jsonl


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def test_chunk_file_and_jsonl_roundtrip(tmp_path):
    f = _write(tmp_path, "doc.txt", "第一句话，内容。第二句话，内容更长一些。第三句话。")
    chunks = chunk_file(f, max_chars=10)
    assert chunks
    out = to_jsonl(chunks, tmp_path / "out" / "chunks.jsonl")
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(chunks)
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["text"] == chunks[0].text


def test_chunk_file_md_defaults_to_markdown_strategy(tmp_path):
    f = _write(tmp_path, "doc.md", "# 标题\n\n" + "正文内容。" * 200)
    chunks = chunk_file(f, max_chars=50)
    assert chunks and chunks[0].strategy == "markdown"


def test_cli_jsonl_and_stats(tmp_path):
    f = _write(tmp_path, "doc.txt", "句子一。句子二。句子三。" * 30)
    out = tmp_path / "c.jsonl"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "worddael.cli",
            str(f),
            "--strategy",
            "recursive",
            "--max-chars",
            "40",
            "--jsonl",
            str(out),
            "--stats",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
    assert "chunks:" in proc.stderr
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert all(json.loads(line)["start"] >= 0 for line in lines)


def test_cli_version():
    proc = subprocess.run(
        [sys.executable, "-m", "worddael.cli", "--version"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "worddael" in proc.stdout


def test_cli_missing_file(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "worddael.cli", str(tmp_path / "nope.txt")],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1
    assert "error" in proc.stderr
