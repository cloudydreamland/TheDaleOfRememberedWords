"""Tests for punctuation hygiene, sentence-aligned overlap, gold disambiguation.

The guarantees under test (R3):
1. no chunk *ends* (rstripped) on an opening punct that it never closes;
2. no non-first chunk *starts* on a closing/dangling punct;
3. overlap windows start on a sentence start whenever one exists in the
   window (never mid-clause after a dangling closer);
4. a gold string appearing multiple times counts as retrieved if ANY of its
   occurrences' chunks are retrieved.
"""

from __future__ import annotations

import pytest
from conftest import random_zh_text

from worddael.chunkers import RecursiveChunker, SentenceChunker, TokenChunker
from worddael.eval.retrieval import find_gold_chunk, find_gold_chunks, recall_at_k
from worddael.types import Chunk

_OPENERS = set("「『“《（〔［｛")
_CLOSERS = set("」』”）)》】〉，。！？；：、")


@pytest.mark.parametrize("seed", range(60, 72))
@pytest.mark.parametrize(
    "chunker_factory",
    [
        lambda: RecursiveChunker(max_chars=89, overlap_chars=17),
        lambda: SentenceChunker(max_chars=110, overlap_chars=20),
        lambda: TokenChunker(max_tokens=97, overlap_tokens=15),
    ],
    ids=["recursive", "sentence", "token"],
)
def test_hygiene_guarantees_under_fuzz(seed, chunker_factory):
    text = random_zh_text(seed, min_len=800, max_len=1800)
    text = text.replace("「", "").replace("」", "")  # fuzz alphabet has no quotes; inject ours
    text = text[:40] + "「他说道：" + text[40:80] + "」" + text[80:]
    chunks = chunker_factory().chunk(text)
    for i, c in enumerate(chunks):
        body = c.text.rstrip()
        if body:
            assert body[-1] not in _OPENERS, f"chunk {c.seq} ends on opener: …{body[-6:]!r}"
        if i > 0 and c.meta.get("overlap_chars", 0) < (c.end - c.start) - 1:
            assert c.text[0] not in _CLOSERS, f"chunk {c.seq} starts on closer: {c.text[:6]!r}…"


def test_hygiene_moves_boundary_left_and_preserves_tiling():
    text = "前半段内容引用「一句关键的话。后半段继续讨论其他事情。结尾陈述句。"
    chunks = RecursiveChunker(max_chars=20, overlap_chars=0).chunk(text)
    cursor = 0
    for c in chunks:
        assert c.text == text[c.start : c.end]
        if c.meta["overlap_chars"] == 0:
            assert c.start == cursor
        cursor = c.end
    assert cursor == len(text)
    body = chunks[0].text.rstrip()
    assert not body or body[-1] not in _OPENERS


def test_overlap_start_aligns_to_sentence_when_available():
    text = (
        "第一句话用来填充位置。第二句话也在这里填充。第三句话内容比较长一些，"
        "包含了若干额外的词语和修饰成分，用来把这一段撑到一个可观的长度。第四句话收尾。"
    )
    chunks = RecursiveChunker(max_chars=40, overlap_chars=20).chunk(text)
    for c in chunks[1:]:
        if c.meta["overlap_chars"] > 0:
            # overlap starts at a sentence start (previous char is sentence end)
            assert text[c.start - 1] in "。！？；…」」”", (
                f"overlap starts mid-clause at {c.start}: …{text[c.start - 6 : c.start + 4]!r}"
            )


def test_gold_multiple_occurrences_disambiguated():
    text = "关键短语出现在开头。中间是一些无关内容。结尾又提到关键短语作为呼应。"
    chunk_a = Chunk(text=text[0:10], start=0, end=10, seq=0, strategy="t", meta={})
    chunk_b = Chunk(text=text[10:], start=10, end=len(text), seq=1, strategy="t", meta={})
    seqs = find_gold_chunks([chunk_a, chunk_b], "关键短语", text)
    assert seqs == {0, 1}  # both occurrences found
    assert find_gold_chunk([chunk_a, chunk_b], "关键短语", text) == 0  # back-compat: first

    # retrieving EITHER occurrence's chunk counts as a hit
    res = recall_at_k(
        [chunk_a, chunk_b],
        text,
        [{"q": "关键短语在哪里？", "gold": "关键短语"}],
        k=1,
    )
    assert res["hits"] == 1


def test_gold_absent_still_misses():
    text = "只有一次出现关键短语。"
    chunks = [Chunk(text=text, start=0, end=len(text), seq=0, strategy="t", meta={})]
    res = recall_at_k(chunks, text, [{"q": "另一个词？", "gold": "不存在的字符串"}], k=1)
    assert res["hits"] == 0 and res["misses"]
