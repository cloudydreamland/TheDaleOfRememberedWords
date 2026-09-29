"""Ecosystem adapters: plug worddael into frameworks users already run.

LangChain: :class:`LangChainChunker` subclasses
``langchain_text_splitters.TextSplitter`` (optional import), so it works
with ``split_documents`` / ``create_documents`` and drops into any existing
RAG pipeline as a drop-in replacement for RecursiveCharacterTextSplitter.
"""

from __future__ import annotations

from . import get_chunker
from .chunkers import BaseChunker

try:  # optional dependency — the adapter degrades to a plain object otherwise
    from langchain_text_splitters import TextSplitter as _LCTextSplitter

    _HAS_LANGCHAIN = True
except ImportError:  # pragma: no cover
    _LCTextSplitter = object  # type: ignore[assignment,misc]
    _HAS_LANGCHAIN = False

__all__ = ["LangChainChunker", "has_langchain"]


def has_langchain() -> bool:
    return _HAS_LANGCHAIN


class LangChainChunker(_LCTextSplitter):
    """worddael chunking behind the LangChain ``TextSplitter`` interface.

    Example::

        from worddael.adapters import LangChainChunker
        splitter = LangChainChunker(strategy="recursive", max_chars=500, overlap_chars=50)
        docs = splitter.create_documents([long_text])   # full LC Document flow
    """

    def __init__(self, strategy: str = "recursive", **chunker_kwargs) -> None:
        if _HAS_LANGCHAIN:
            super().__init__()
        self._chunker: BaseChunker = get_chunker(strategy, **chunker_kwargs)

    def split_text(self, text: str) -> list[str]:
        return [c.text for c in self._chunker.chunk(text)]
