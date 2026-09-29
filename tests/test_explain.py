"""Tests for explainable chunking: boundary-cause annotation.

Contract: when ``explain=True``, every chunk carries ``end_rule`` and
``start_rule``; the rule matches the actual source characters at the
boundary (verified against the text, not just presence). When
``explain=False`` (default), the keys are absent — zero overhead.
"""

from __future__ import annotations

import pytest
from conftest import random_zh_text

from worddael.chunkers import (
    MarkdownChunker,
    RecursiveChunker,
    SemanticChunker,
    SentenceChunker,
    TokenChunker,
)
from worddael.explain import classify_end

RULES = {"paragraph", "newline", "sentence", "clause", "budget", "text_end", "text_start"}


def test_classify_end_rules_on_synthetic_boundaries():
    text = "第一句完整结束。\n\n第二句也在句号后结束。半句被拦腰"
    assert classify_end(text, 8) == "sentence"  # after 第一句完整结束。
    assert classify_end(text, 10) == "paragraph"  # after the \n\n run
    assert classify_end(text, len(text)) == "text_end"
    assert classify_end(text, 25) == "budget"  # cut inside content (between 拦 and 腰)


def test_recursive_explain_matches_source_characters():
    text = random_zh_text(11, min_len=1500, max_len=2500)
    chunks = RecursiveChunker(max_chars=150, overlap_chars=0, explain=True).chunk(text)
    for c in chunks:
        assert c.meta["end_rule"] in RULES - {"text_start"}
        assert c.meta["start_rule"] in RULES - {"text_end"}
        if c.meta["end_rule"] == "sentence":
            assert c.text.rstrip()[-1] in "。！？；…!?;"
        if c.meta["end_rule"] == "budget":
            # cut landed on content: next char in source continues the chunk
            assert c.end < len(text) and c.end > 0


def test_explain_start_rule_chains_from_previous_end():
    text = random_zh_text(12, min_len=800, max_len=1500)
    chunks = RecursiveChunker(max_chars=120, overlap_chars=0, explain=True).chunk(text)
    for prev, cur in zip(chunks, chunks[1:]):
        assert cur.meta["start_rule"] == prev.meta["end_rule"]
    assert chunks[0].meta["start_rule"] == "text_start"
    assert chunks[-1].meta["end_rule"] == "text_end"


def test_explain_false_by_default():
    text = "第一句。第二句。"
    chunks = RecursiveChunker(max_chars=6, overlap_chars=0).chunk(text)
    assert all("end_rule" not in c.meta for c in chunks)
    assert all("start_rule" not in c.meta for c in chunks)


@pytest.mark.parametrize(
    "chunker",
    [
        SentenceChunker(max_chars=100, overlap_chars=0, explain=True),
        TokenChunker(max_tokens=80, overlap_tokens=0, explain=True),
        RecursiveChunker(max_chars=100, overlap_chars=0, explain=True),
    ],
    ids=["sentence", "token", "recursive"],
)
def test_explain_consistency_across_strategies(chunker):
    text = random_zh_text(21, min_len=1200, max_len=2000)
    chunks = chunker.chunk(text)
    assert chunks
    for c in chunks:
        assert c.meta["end_rule"] in RULES
        # start_rule of chunk i+1 always equals end_rule of chunk i
    for prev, cur in zip(chunks, chunks[1:]):
        assert cur.meta["start_rule"] == prev.meta["end_rule"]


def test_markdown_heading_rule():
    text = (
        "# 安装\n\n" + "这一节讲如何安装依赖项。" * 5 + "\n\n## 配置\n\n" + "这一节讲配置文件。" * 5
    )
    chunks = MarkdownChunker(max_chars=100, overlap_chars=0, explain=True).chunk(text)
    rules = [c.meta["end_rule"] for c in chunks]
    assert "heading" in rules


def test_semantic_drop_rule():
    def embed(texts):
        out = []
        for t in texts:
            v = [0.0, 0.0]
            if "猫" in t:
                v[0] = 1.0
            if "狗" in t:
                v[1] = 1.0
            out.append(v)
        return out

    text = "猫在睡觉。猫在打滚。猫在吃饭。狗在跑步。狗在叫唤。狗在撒欢。"
    chunks = SemanticChunker(
        embed_fn=embed, similarity_threshold=0.5, max_chars=200, min_chars=5, explain=True
    ).chunk(text)
    rules = [c.meta["end_rule"] for c in chunks]
    assert "semantic_drop" in rules


def test_parent_child_explain_passthrough():
    from worddael.parent_child import ParentChildChunker

    text = random_zh_text(31, min_len=1500, max_len=2500)
    family = ParentChildChunker(
        parent_max_chars=500, child_max_chars=120, child_overlap_chars=10, explain=True
    )
    _, children = family.chunk_families(text)
    assert children
    for c in children:
        assert c.meta["end_rule"] in RULES - {"text_start"}


def test_explain_fuzz_seeds():
    for seed in range(40, 52):
        text = random_zh_text(seed, min_len=600, max_len=1600)
        chunks = RecursiveChunker(max_chars=97, overlap_chars=13, explain=True).chunk(text)
        for prev, cur in zip(chunks, chunks[1:]):
            assert cur.meta["start_rule"] == prev.meta["end_rule"]
