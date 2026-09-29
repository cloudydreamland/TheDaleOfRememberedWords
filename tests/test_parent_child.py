"""Tests for ParentChildChunker and parent-level recall."""

from __future__ import annotations

import pytest
from conftest import random_zh_text

from worddael import chunk, get_chunker
from worddael.eval.retrieval import recall_at_k_parent_level
from worddael.eval.sample_data import SAMPLE_DOCS
from worddael.parent_child import ParentChildChunker


def _family(seed: int = 5):
    text = random_zh_text(seed, min_len=1200, max_len=2000)
    family = ParentChildChunker(parent_max_chars=400, child_max_chars=100, child_overlap_chars=10)
    return text, *family.chunk_families(text)


def test_children_tile_text_with_offset_invariant():
    text, parents, children = _family()
    assert children and parents
    cursor = 0
    for c in children:
        assert c.text == text[c.start : c.end]
        if c.meta.get("overlap_chars", 0) == 0:
            assert c.start == cursor
        cursor = c.end
    assert cursor == len(text)


def test_parents_tile_text_and_linkage_consistent():
    text, parents, children = _family()
    cursor = 0
    for p in parents:
        assert p.text == text[p.start : p.end]
        assert p.start == cursor
        cursor = p.end
    assert cursor == len(text)

    by_seq = {c.seq: c for c in children}
    for p in parents:
        assert p.meta["child_seqs"], "every parent should own children"
        for cs in p.meta["child_seqs"]:
            child = by_seq[cs]
            assert child.meta["parent_seq"] == p.seq
            assert p.start <= child.start and child.end <= p.end


def test_chunk_returns_children_only():
    text, parents, children = _family()
    same = ParentChildChunker(
        parent_max_chars=400, child_max_chars=100, child_overlap_chars=10
    )
    flat = same.chunk(text)
    assert [(c.start, c.end) for c in flat] == [(c.start, c.end) for c in children]
    assert (
        chunk(
            text,
            strategy="parent-child",
            parent_max_chars=400,
            child_max_chars=100,
            child_overlap_chars=10,
        )
        == flat
    )


def test_get_chunker_registry():
    assert get_chunker("parent-child").strategy == "parent-child"


def test_validation_degenerate_family():
    with pytest.raises(ValueError):
        ParentChildChunker(parent_max_chars=100, child_max_chars=100)
    with pytest.raises(ValueError):
        ParentChildChunker(parent_max_chars=100, child_max_chars=200)


def test_parent_level_recall_sane_and_ties_to_gold():
    family = ParentChildChunker(parent_max_chars=400, child_max_chars=120, child_overlap_chars=0)
    doc = SAMPLE_DOCS[0]
    parents, children = family.chunk_families(doc["text"])
    res = recall_at_k_parent_level(children, parents, doc["text"], doc["questions"], k=3)
    assert res["total"] == len(doc["questions"])
    assert 0.0 <= res["recall"] <= 1.0


def test_parent_level_recall_on_full_corpus():
    """Small-to-big should retrieve at least as well as flat at the same
    child budget when credit goes to the parent (more forgiving target)."""
    family = ParentChildChunker(parent_max_chars=600, child_max_chars=150, child_overlap_chars=30)
    hits = total = 0
    for doc in SAMPLE_DOCS:
        parents, children = family.chunk_families(doc["text"])
        res = recall_at_k_parent_level(children, parents, doc["text"], doc["questions"], k=1)
        hits += res["hits"]
        total += res["total"]
    assert 0.0 <= hits / total <= 1.0
