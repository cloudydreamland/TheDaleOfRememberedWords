"""Tests for embedders — all HTTP paths are mocked, no network, no models."""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from worddael.chunkers import SemanticChunker
from worddael.embedders import EmbedderError, HashingEmbedder, OpenAICompatibleEmbedder


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _fake_embeddings_response(texts: list[str], dim: int = 8) -> dict:
    vectors = [[float((len(t) + i) % dim + 1)] * dim for i, t in enumerate(texts)]
    return {"data": [{"index": i, "embedding": v} for i, v in enumerate(vectors)]}


def test_hashing_embedder_deterministic_across_instances():
    a = HashingEmbedder(dim=64).embed(["中文文本切分测试", "第二段内容"])
    b = HashingEmbedder(dim=64).embed(["中文文本切分测试", "第二段内容"])
    assert a == b  # crc32-based: stable across processes, unlike hash()


def test_hashing_embedder_normalized_and_dim():
    emb = HashingEmbedder(dim=32)
    vecs = emb.embed(["Hello, world", "你好，世界。"])
    assert all(len(v) == 32 for v in vecs)
    for v in vecs:
        norm = sum(x * x for x in v) ** 0.5
        assert norm == pytest.approx(1.0)


def test_hashing_embedder_similarity_orders_correctly():
    emb = HashingEmbedder(dim=256)
    a, = emb.embed(["今天天气很好适合出门散步"])
    b, = emb.embed(["今天天气很好适合出门跑步"])
    c, = emb.embed(["财务报销制度发票税号要求"])
    ab = sum(x * y for x, y in zip(a, b))
    ac = sum(x * y for x, y in zip(a, c))
    assert ab > ac


def test_hashing_embedder_dim_validation():
    with pytest.raises(ValueError):
        HashingEmbedder(dim=4)


def test_openai_embedder_batches_and_preserves_order(monkeypatch):
    calls: list[dict] = []

    def fake_urlopen(req, timeout=None):
        payload = json.loads(req.data.decode("utf-8"))
        calls.append(payload)
        return _FakeResponse(_fake_embeddings_response(payload["input"]))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setenv("TEST_EMBED_KEY", "sk-night-test")

    embedder = OpenAICompatibleEmbedder(
        base_url="https://api.example.com/v1",
        model="text-embedding-test",
        api_key_env="TEST_EMBED_KEY",
        batch_size=2,
    )
    texts = ["一", "二", "三", "四", "五"]
    vectors = embedder.embed(texts)

    assert len(calls) == 3  # ceil(5/2)
    assert [len(c["input"]) for c in calls] == [2, 2, 1]
    assert all(c["model"] == "text-embedding-test" for c in calls)
    assert len(vectors) == 5
    # order preserved: vector[0] corresponds to text "一"
    assert vectors[0] == _fake_embeddings_response(["一"])["data"][0]["embedding"]


def test_openai_embedder_auth_header_from_env(monkeypatch):
    seen: dict = {}

    def fake_urlopen(req, timeout=None):
        seen["auth"] = req.get_header("Authorization")
        return _FakeResponse(_fake_embeddings_response(["x"]))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setenv("QIEGAO_TEST_KEY", "secret-123")
    OpenAICompatibleEmbedder(
        base_url="https://api.example.com", model="m", api_key_env="QIEGAO_TEST_KEY"
    ).embed(["x"])
    assert seen["auth"] == "Bearer secret-123"


def test_openai_embedder_explicit_key_wins(monkeypatch):
    seen: dict = {}

    def fake_urlopen(req, timeout=None):
        seen["auth"] = req.get_header("Authorization")
        return _FakeResponse(_fake_embeddings_response(["x"]))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setenv("QIEGAO_TEST_KEY", "env-key")
    OpenAICompatibleEmbedder(
        base_url="https://api.example.com", model="m", api_key="explicit-key"
    ).embed(["x"])
    assert seen["auth"] == "Bearer explicit-key"


def test_openai_embedder_missing_key_raises(monkeypatch):
    monkeypatch.delenv("QIEGAO_MISSING_KEY", raising=False)
    embedder = OpenAICompatibleEmbedder(
        base_url="https://api.example.com", model="m", api_key_env="QIEGAO_MISSING_KEY"
    )
    with pytest.raises(EmbedderError, match="missing API key"):
        embedder.embed(["x"])


def test_openai_embedder_empty_input_no_http(monkeypatch):
    def fake_urlopen(req, timeout=None):  # pragma: no cover — must not be called
        raise AssertionError("network call attempted for empty input")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert OpenAICompatibleEmbedder(base_url="https://x", model="m").embed([]) == []


def test_openai_embedder_http_error_wrapped(monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            "https://api.example.com/embeddings", 500, "boom", {}, io.BytesIO(b"exploded")
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    embedder = OpenAICompatibleEmbedder(base_url="https://api.example.com", model="m", api_key="k")
    with pytest.raises(EmbedderError, match="HTTP 500"):
        embedder.embed(["x"])


def test_openai_embedder_malformed_response(monkeypatch):
    def fake_urlopen(req, timeout=None):
        return _FakeResponse({"data": [{"index": 0, "embedding": [1.0]}]})  # asked for 2

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    embedder = OpenAICompatibleEmbedder(base_url="https://api.example.com", model="m", api_key="k")
    with pytest.raises(EmbedderError, match="unexpected embeddings response"):
        embedder.embed(["x", "y"])


def test_semantic_chunker_accepts_embedder_instance():
    """Glue test: an Embedder object (callable) plugs straight into the chunker."""
    text = "第一句话讲安装。第二句话讲配置。第三句话讲部署。"
    embedder = OpenAICompatibleEmbedder(base_url="https://x", model="m", api_key="k")
    embedder.embed = HashingEmbedder(dim=16).embed  # type: ignore[method-assign] — mock backend
    chunks = SemanticChunker(
        embed_fn=embedder,
        similarity_threshold=0.99,  # force cuts at every sentence boundary
        max_chars=100,
        min_chars=1,
    ).chunk(text)
    assert len(chunks) >= 3
    assert all(c.text == text[c.start : c.end] for c in chunks)
