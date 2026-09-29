"""Polish-round edge cases: degenerate inputs, jieba path, CLI overrides.

No new features — these lock in behaviours the release candidate relies on.
"""

from __future__ import annotations

import pytest

from worddael import chunk, chunk_file, get_chunker
from worddael.chunkers import RecursiveChunker, SentenceChunker, TokenChunker


def test_markdown_strategy_on_plain_text():
    """No headings at all → single implicit section, content preserved."""
    text = "这是没有标题的普通文本。" * 30
    chunks = get_chunker("markdown", max_chars=100, overlap_chars=0).chunk(text)
    assert chunks
    assert all(c.meta["headings"] == [] for c in chunks)
    assert all(c.text == text[c.start : c.end] for c in chunks)
    assert sum(c.end - c.start for c in chunks) == len(text)


def test_recursive_degenerate_budget_no_hang():
    """max_chars=1 must terminate and tile exactly (char-level chunks)."""
    text = "短句。"
    chunks = RecursiveChunker(max_chars=1, overlap_chars=0).chunk(text)
    assert [c.text for c in chunks] == list(text)
    cursor = 0
    for c in chunks:
        assert c.start == cursor
        cursor = c.end
    assert cursor == len(text)


def test_sentence_chunker_only_newlines():
    chunks = SentenceChunker(max_chars=50, overlap_chars=0).chunk("\n\n\n")
    joined = "".join(c.text for c in chunks)
    assert joined == "\n\n\n"


def test_token_chunker_jieba_counter():
    pytest.importorskip("jieba")
    from worddael.counters import JiebaCounter

    text = "中文分词计数测试。" * 50
    chunker = TokenChunker(max_tokens=30, overlap_tokens=5, counter=JiebaCounter())
    chunks = chunker.chunk(text)
    assert chunks
    for c in chunks:
        # content budget, plus the overlap allowance (same convention as test_offsets)
        assert chunker.counter.count(c.text) <= 35


def test_chunk_factory_dispatch_and_offsets():
    text = "工厂函数冒烟。第二句内容。" * 20
    for strategy in ["recursive", "sentence", "token", "markdown"]:
        chunks = chunk(text, strategy=strategy, max_chars=120, overlap_chars=0)
        assert chunks, strategy
        assert all(c.text == text[c.start : c.end] for c in chunks), strategy


def test_chunk_file_explicit_strategy_override(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("# 标题\n\n" + "正文内容。" * 100, encoding="utf-8")
    chunks = chunk_file(f, strategy="recursive", max_chars=80, overlap_chars=0)
    assert chunks and chunks[0].strategy == "recursive"


def test_all_offset_invariants_hold_under_fuzz_seeds_100_to_115():
    """Extended seed sweep beyond the parametrized 0–7 in test_offsets."""
    from conftest import random_zh_text

    for seed in range(100, 116):
        text = random_zh_text(seed, min_len=500, max_len=1500)
        for chunker in [
            RecursiveChunker(max_chars=73, overlap_chars=11),
            SentenceChunker(max_chars=131, overlap_chars=17),
            TokenChunker(max_tokens=97, overlap_tokens=13),
        ]:
            chunks = chunker.chunk(text)
            cursor = 0
            for c in chunks:
                assert c.text == text[c.start : c.end], (strategy_name(chunker), seed, c.seq)
                if c.meta.get("overlap_chars", 0) == 0:
                    assert c.start == cursor
                cursor = c.end
            assert cursor == len(text)


def strategy_name(chunker) -> str:
    return chunker.strategy
