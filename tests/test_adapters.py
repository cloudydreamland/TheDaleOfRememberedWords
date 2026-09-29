"""Tests for R4: LangChain adapter and the `worddael demo` tour."""

from __future__ import annotations

import subprocess
import sys

import pytest

from worddael.adapters import LangChainChunker, has_langchain
from worddael.eval.sample_data import SAMPLE_DOCS

TEXT = SAMPLE_DOCS[0]["text"]


def test_langchain_available_in_dev_env():
    # the dev venv installs langchain-text-splitters; if absent, these tests skip
    if not has_langchain():
        pytest.skip("langchain-text-splitters not installed")


@pytest.mark.skipif(not has_langchain(), reason="langchain-text-splitters not installed")
def test_langchain_split_text_preserves_content():
    splitter = LangChainChunker(strategy="recursive", max_chars=80, overlap_chars=0)
    pieces = splitter.split_text(TEXT)
    assert pieces
    assert "".join(pieces) == TEXT  # LC API is strings-only; content must survive


@pytest.mark.skipif(not has_langchain(), reason="langchain-text-splitters not installed")
def test_langchain_is_a_real_text_splitter_subclass():
    from langchain_text_splitters import TextSplitter

    splitter = LangChainChunker(strategy="sentence", max_chars=60)
    assert isinstance(splitter, TextSplitter)
    # inherited machinery works end to end
    docs = splitter.create_documents([TEXT])
    assert docs
    assert all(d.page_content for d in docs)


@pytest.mark.skipif(not has_langchain(), reason="langchain-text-splitters not installed")
def test_langchain_markdown_strategy():
    md = "# 标题\n\n" + TEXT
    no_overlap = LangChainChunker(strategy="markdown", max_chars=100, overlap_chars=0)
    pieces = no_overlap.split_text(md)
    assert pieces and "".join(pieces) == md  # overlap=0 → lossless round-trip
    # with overlap the joined text is intentionally longer (context repetition)
    with_overlap = LangChainChunker(strategy="markdown", max_chars=100, overlap_chars=20)
    assert len("".join(with_overlap.split_text(md))) > len(md)


def test_cli_demo_runs():
    proc = subprocess.run(
        [sys.executable, "-m", "worddael.cli", "demo"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert proc.returncode == 0, proc.stderr
    assert "worddael demo" in proc.stdout
    assert "end_rule" in proc.stdout  # explainable chunking on display
    assert "策略对比" in proc.stdout
