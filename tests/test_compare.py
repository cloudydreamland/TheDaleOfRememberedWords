"""Tests for the compare module and the --compare CLI flag."""

from __future__ import annotations

import subprocess
import sys

from worddael.compare import compare_text, format_table

TEXT = (
    "第一章 总则。为规范流程特制定本制度。适用全体员工。\n\n"
    "第二章 差旅。高铁二等座全额报销。住宿每晚三百元。\n\n"
    "第三章 时限。发票三十天内提交。逾期系统关闭入口。"
) * 3


def test_compare_text_returns_all_strategies():
    rows = compare_text(TEXT, max_chars=100, overlap_chars=20)
    names = {r["strategy"] for r in rows}
    assert {"fixed-window", "sentence", "token", "recursive", "semantic-hash-toy"} <= names
    for r in rows:
        assert r["chunks"] > 0
        assert r["min_chars"] > 0
        assert 0.99 <= r["coverage"] <= 1.6  # 1.0 tiling, more with overlap


def test_format_table_has_header_and_rows():
    rows = compare_text(TEXT, max_chars=120, overlap_chars=20)
    table = format_table(rows)
    assert "| strategy |" in table
    assert table.count("\n") >= len(rows)


def test_cli_compare(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text(TEXT, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "worddael.cli", str(f), "--compare", "--max-chars", "120"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "recursive" in proc.stdout and "fixed-window" in proc.stdout
