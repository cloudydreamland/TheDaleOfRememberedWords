"""Traditional-Chinese spot checks — closing the honest limitation recorded
in docs/dogfood.md. Excerpts are public domain (清代小說, Gutenberg pg24264).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from worddael.chunkers import RecursiveChunker
from worddael.sentences import split_sentences

EXCERPT = (
    "方進來．鮑二家的笑說：“三人就在這里罷，茶也現成。”"
    "賈璉道：“我早听見了．如今且不用理他。”"
)

CORPUS = Path(__file__).resolve().parent.parent / "dogfood_corpus" / "hongloumeng_raw.txt"


def test_traditional_fullwidth_stop_splits():
    spans = split_sentences(EXCERPT)
    first = EXCERPT[spans[0][0] : spans[0][1]]
    assert first.startswith("方進來．")


def test_traditional_quotes_attach_to_sentence():
    spans = split_sentences(EXCERPT)
    second = EXCERPT[spans[1][0] : spans[1][1]]
    assert second.startswith("鮑二家的笑說：“")
    assert "。”" in second  # closer attached before the boundary


def test_traditional_chunking_offsets_hold():
    chunks = RecursiveChunker(max_chars=60, overlap_chars=10).chunk(EXCERPT)
    cursor = 0
    for c in chunks:
        assert c.text == EXCERPT[c.start : c.end]
        if c.meta["overlap_chars"] == 0:
            assert c.start == cursor
        cursor = c.end
    assert cursor == len(EXCERPT)


def test_real_corpus_still_chunks_if_downloaded():
    """The dogfood corpus (if present on this machine) must survive the full
    pipeline — guards the traditional-Chinese path against regressions."""
    if not CORPUS.exists():
        pytest.skip("dogfood corpus not downloaded (tools/dogfood.py fetches it)")
    text = CORPUS.read_text(encoding="utf-8", errors="replace")
    start = text.find("*** START")
    end = text.find("*** END")
    if start >= 0 and end >= 0:
        text = text[text.find("\n", start) + 1 : end]
    chunks = RecursiveChunker(max_chars=400, overlap_chars=50, explain=True).chunk(text.strip())
    assert len(chunks) > 1000
    rules = {c.meta["end_rule"] for c in chunks}
    assert "budget" in rules and ("newline" in rules or "paragraph" in rules)
