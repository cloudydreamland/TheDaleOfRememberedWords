"""Worddael — Chinese-first text chunking for RAG.

把中文文本切成好用的块，同时保留精确到字符的原文偏移量，
让 RAG 的每一条检索结果都能被可靠引用。

Quickstart::

    from worddael import chunk

    chunks = chunk(long_text, strategy="recursive", max_chars=500, overlap_chars=50)
    for c in chunks:
        print(c.seq, c.start, c.end, c.text[:20])
"""

from __future__ import annotations

from .chunkers import (
    STRATEGIES,
    BaseChunker,
    MarkdownChunker,
    RecursiveChunker,
    SemanticChunker,
    SentenceChunker,
    TokenChunker,
)
from .counters import CharCounter, get_counter
from .embedders import (
    Embedder,
    EmbedderError,
    HashingEmbedder,
    OpenAICompatibleEmbedder,
)
from .io_utils import chunk_file, to_jsonl
from .parent_child import ParentChildChunker
from .types import Chunk

__version__ = "0.1.0rc3"

# parent-child families are produced by a dedicated module; register the
# strategy here so get_chunker("parent-child") works like any other.
STRATEGIES["parent-child"] = ParentChildChunker

__all__ = [
    "BaseChunker",
    "CharCounter",
    "Chunk",
    "Embedder",
    "EmbedderError",
    "HashingEmbedder",
    "MarkdownChunker",
    "OpenAICompatibleEmbedder",
    "ParentChildChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "SentenceChunker",
    "STRATEGIES",
    "TokenChunker",
    "__version__",
    "chunk",
    "chunk_file",
    "get_chunker",
    "get_counter",
    "to_jsonl",
]


def get_chunker(strategy: str = "recursive", **kwargs) -> BaseChunker:
    """Build a chunker by name. See ``STRATEGIES`` for available names."""
    try:
        cls = STRATEGIES[strategy]
    except KeyError:
        raise ValueError(
            f"unknown strategy {strategy!r}; expected one of {sorted(STRATEGIES)}"
        ) from None
    return cls(**kwargs)


def chunk(text: str, strategy: str = "recursive", **kwargs) -> list[Chunk]:
    """One-shot helper: ``chunk(text, strategy, **chunker_kwargs)``.

    Uniform budget names: ``max_chars``/``overlap_chars`` are accepted by
    every strategy. For ``strategy="token"`` they are interpreted as token
    budgets (mapped to ``max_tokens``/``overlap_tokens``) — tokens are what
    that strategy measures, and the char-named arguments keep call sites
    strategy-agnostic.
    """
    if strategy == "token":
        if "max_tokens" not in kwargs and "max_chars" in kwargs:
            kwargs["max_tokens"] = kwargs.pop("max_chars")
        if "overlap_tokens" not in kwargs and "overlap_chars" in kwargs:
            kwargs["overlap_tokens"] = kwargs.pop("overlap_chars")
    return get_chunker(strategy, **kwargs).chunk(text)
