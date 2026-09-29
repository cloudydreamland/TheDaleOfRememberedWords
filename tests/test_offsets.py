"""Fuzz tests for the core invariant: chunk.text == source[start:end],
and chunk spans tile the source exactly (no gaps, no duplicates)."""

from __future__ import annotations

import pytest
from conftest import random_zh_text

from worddael import (
    MarkdownChunker,
    RecursiveChunker,
    SentenceChunker,
    TokenChunker,
)
from worddael.eval.report import FixedWindowChunker

STRATEGIES = [
    lambda mc, oc: RecursiveChunker(max_chars=mc, overlap_chars=oc),
    lambda mc, oc: SentenceChunker(max_chars=mc, overlap_chars=oc),
    lambda mc, oc: TokenChunker(max_tokens=mc, overlap_tokens=oc),
    lambda mc, oc: FixedWindowChunker(max_chars=mc, overlap_chars=oc),
]


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("factory", STRATEGIES, ids=lambda f: getattr(f(10, 0), "strategy"))
def test_offsets_and_tiling_invariant(seed, factory):
    text = random_zh_text(seed)
    max_c, ov = (60, 15) if seed % 2 == 0 else (200, 50)
    chunker = factory(max_c, ov)
    chunks = chunker.chunk(text)

    assert chunks, "expected non-empty chunk list"
    cursor = 0
    for c in chunks:
        assert c.text == text[c.start : c.end], f"offset mismatch at seq {c.seq}"
        assert c.n_chars > 0
        if c.meta.get("overlap_chars", 0) == 0:
            assert c.start == cursor, f"gap/duplicate before seq {c.seq}"
        cursor = c.end
    assert cursor == len(text), "chunks do not cover the whole text"


def test_markdown_offsets_and_tiling():
    import random

    rng = random.Random(42)
    body = "".join(rng.choice(["结构介绍如下。", "参数说明详见附录！", "缓存默认关闭。\n"]) for _ in range(40))
    text = (
        "# 指南\n\n" + body + "\n## 安装\n\n```python\nprint('# not a heading')\n```\n\n内容继续。" * 3
    )
    chunker = MarkdownChunker(max_chars=120, overlap_chars=20)
    chunks = chunker.chunk(text)
    cursor = 0
    for c in chunks:
        assert c.text == text[c.start : c.end]
        if c.meta.get("overlap_chars", 0) == 0:
            assert c.start == cursor
        cursor = c.end
    assert cursor == len(text)


def test_no_chunk_exceeds_budget_without_overlap():
    text = random_zh_text(7, min_len=2000, max_len=3000)
    for chunker in [
        RecursiveChunker(max_chars=100, overlap_chars=0),
        SentenceChunker(max_chars=100, overlap_chars=0),
    ]:
        for c in chunker.chunk(text):
            # +2: punctuation hygiene may absorb up to 2 closing chars from
            # the next chunk's start (guarantees no dangling-closer starts)
            assert c.n_chars <= 102, f"{chunker.strategy} produced oversized chunk"


def test_overlap_extends_but_respects_budget_cap():
    text = random_zh_text(3, min_len=1500, max_len=2500)
    chunker = RecursiveChunker(max_chars=100, overlap_chars=30)
    chunks = chunker.chunk(text)
    for c in chunks[1:]:
        assert c.meta["overlap_chars"] <= 30
        assert c.n_chars <= 130  # content budget + overlap allowance


def test_empty_and_tiny_inputs():
    from worddael import chunk

    assert chunk("", strategy="recursive") == []
    assert chunk("", strategy="token") == []
    assert chunk("", strategy="markdown") == []
    tiny = chunk("短文本。", strategy="recursive")
    assert len(tiny) == 1 and tiny[0].text == "短文本。"


def test_mixed_chinese_english_offsets():
    text = (
        "CloudMsg SDK requires Python 3.10+. 安装方式是 pip install cloudmsg。"
        "版本号 3.14.1 不是句子结束. 但是这里以句号结尾。"
    )
    chunks = RecursiveChunker(max_chars=50, overlap_chars=0).chunk(text)
    joined = "".join(c.text for c in chunks)
    assert joined == text
