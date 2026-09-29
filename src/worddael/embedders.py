"""Embedding functions for SemanticChunker.

Zero required dependencies. Two flavors:

- ``HashingEmbedder``: deterministic character-trigram hashing. A pure-Python
  stand-in so the semantic pipeline runs anywhere — explicitly NOT a quality
  claim; use a real embedding model for real semantic chunking. Hashing is
  crc32-based, so vectors are stable across processes (Python's built-in
  ``hash()`` is not — it is randomized per process).
- ``OpenAICompatibleEmbedder``: calls any OpenAI-compatible ``/embeddings``
  endpoint (vLLM, Zhipu, DeepSeek, OpenAI, ...). The API key is resolved
  lazily at call time: explicit ``api_key=`` wins, else the environment
  variable named by ``api_key_env``.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
import zlib
from typing import Protocol


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class EmbedderError(RuntimeError):
    """Raised when an embedding backend fails or returns malformed data."""


class HashingEmbedder:
    """Deterministic char-trigram hashing into a fixed-dim L1-normalized vector."""

    def __init__(self, dim: int = 96) -> None:
        if dim < 8:
            raise ValueError("dim must be >= 8")
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for t in texts:
            v = [0.0] * self.dim
            t = t.replace("\n", "")
            for i in range(len(t) - 2):
                gram = t[i : i + 3]
                v[zlib.crc32(gram.encode("utf-8")) % self.dim] += 1.0
            norm = sum(x * x for x in v) ** 0.5 or 1.0
            vectors.append([x / norm for x in v])
        return vectors

    def __call__(self, texts: list[str]) -> list[list[float]]:
        # delegate at runtime so instance-level overrides of .embed are honored
        return self.embed(texts)


class OpenAICompatibleEmbedder:
    """Client for OpenAI-compatible embedding endpoints.

    Example::

        embedder = OpenAICompatibleEmbedder(
            base_url="https://open.bigmodel.cn/api/paas/v4",
            model="embedding-3",
            api_key_env="ZHIPUAI_API_KEY",
        )
        chunks = SemanticChunker(embed_fn=embedder).chunk(long_text)
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        batch_size: int = 16,
        timeout: int = 60,
        extra: dict | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url is required")
        if not model:
            raise ValueError("model is required")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.api_key_env = api_key_env
        self.batch_size = batch_size
        self.timeout = timeout
        self.extra = extra or {}

    def _resolve_key(self) -> str:
        if self.api_key:
            return self.api_key
        key = os.environ.get(self.api_key_env)
        if not key:
            raise EmbedderError(
                f"missing API key: pass api_key=... or set the {self.api_key_env} environment variable"
            )
        return key

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._embed_batch(texts[i : i + self.batch_size]))
        return out

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": list(batch), **self.extra}
        headers = {"Content-Type": "application/json"}
        if self.api_key or self.api_key_env:
            headers["Authorization"] = f"Bearer {self._resolve_key()}"
        req = urllib.request.Request(
            self.base_url + "/embeddings",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")[:200]
            except Exception:  # noqa: BLE001 — best-effort diagnostics only
                pass
            raise EmbedderError(f"embeddings HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise EmbedderError(f"embeddings request failed: {exc.reason}") from exc

        items = data.get("data")
        if not isinstance(items, list) or len(items) != len(batch):
            raise EmbedderError(
                f"unexpected embeddings response: expected {len(batch)} items, got {items!r:.120}"
            )
        items.sort(key=lambda item: item.get("index", 0))
        vectors = [item["embedding"] for item in items]
        if any(not isinstance(v, list) for v in vectors):
            raise EmbedderError("embeddings response contained a non-list vector")
        return vectors

    def __call__(self, texts: list[str]) -> list[list[float]]:
        # delegate at runtime so instance-level overrides of .embed are honored
        return self.embed(texts)
