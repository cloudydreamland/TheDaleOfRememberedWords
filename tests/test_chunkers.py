"""Behavioural tests for sentence splitting, recursive merging, budgets."""

from __future__ import annotations

from worddael.chunkers import RecursiveChunker
from worddael.sentences import markdown_sections, split_sentences
from worddael.types import Chunk


def test_sentence_boundaries_chinese_punctuation():
    text = "第一句。第二句！第三句？最后一句没有结尾"
    spans = split_sentences(text)
    joined = "".join(text[s:e] for s, e in spans)
    assert joined == text
    assert text[spans[0][0] : spans[0][1]] == "第一句。"
    assert text[spans[1][0] : spans[1][1]] == "第二句！"


def test_closing_quotes_attach_to_sentence():
    text = "他说「你好」？然后离开了。”完"
    spans = split_sentences(text)
    first = text[spans[0][0] : spans[0][1]]
    assert first.startswith("他说")
    assert "？" in first


def test_ascii_period_not_a_boundary():
    text = "版本号是 3.14.1，注意小数点。这是下一句"
    spans = split_sentences(text)
    assert text[spans[0][0] : spans[0][1]] == "版本号是 3.14.1，注意小数点。"


def test_recursive_prefers_sentence_boundaries():
    text = "这是一句话，内容比较长一点。这是另一句话，也在同一个段落里面。最后还有第三句话呢。"
    chunks = RecursiveChunker(max_chars=20, overlap_chars=0).chunk(text)
    ends_on_punct = sum(1 for c in chunks if c.text[-1] in "。？！")
    assert ends_on_punct >= len(chunks) - 1


def test_recursive_merges_small_sentences():
    text = "短句一。短句二。短句三。短句四。短句五。"
    chunks = RecursiveChunker(max_chars=100, overlap_chars=0).chunk(text)
    assert len(chunks) == 1


def test_token_chunker_respects_token_budget():
    from worddael.chunkers import TokenChunker
    from worddael.counters import CharCounter

    text = "这是一段测试文本。" * 200
    counter = CharCounter()
    chunker = TokenChunker(max_tokens=64, overlap_tokens=0, counter=counter)
    for c in chunker.chunk(text):
        assert counter.count(c.text) <= 64


def test_token_chunker_validation():
    import pytest

    from worddael.chunkers import TokenChunker

    with pytest.raises(ValueError):
        TokenChunker(max_tokens=0)
    with pytest.raises(ValueError):
        TokenChunker(max_tokens=10, overlap_tokens=10)


def test_chunk_has_strategy_and_seq():
    text = "第一句。第二句。第三句。第四句。"
    chunks = RecursiveChunker(max_chars=8, overlap_chars=0).chunk(text)
    assert [c.seq for c in chunks] == list(range(len(chunks)))
    assert all(c.strategy == "recursive" for c in chunks)


def test_chunk_dataclass():
    c = Chunk(text="abc", start=0, end=3, seq=0, strategy="t", meta={})
    assert c.n_chars == 3
    assert c.to_dict()["end"] == 3


def test_markdown_section_parsing():
    text = (
        "# 安装\n\n先安装依赖。\n\n## 依赖列表\n\n需要 python 3.10。\n\n"
        "```python\n# ## 这不是标题\nprint(1)\n```\n\n## 配置\n\n修改 yaml。"
    )
    sections = markdown_sections(text)
    paths = [s.headings for s in sections]
    assert paths[0] == ("安装",) or paths[0] == ()
    assert ("安装", "依赖列表") in [p for p in paths if len(p) == 2]
    assert ("安装", "配置") in [p for p in paths if len(p) == 2]
    # fenced '# ## 这不是标题' must not become a heading
    assert all("这不是标题" not in s.headings for s in sections)


def test_markdown_chunker_heading_metadata():
    text = "# 指南\n\n" + "这一节内容很长。" * 50 + "\n\n## 小节\n\n很短的内容。"
    from worddael.chunkers import MarkdownChunker

    chunks = MarkdownChunker(max_chars=100, overlap_chars=0).chunk(text)
    with_headings = [c for c in chunks if c.meta.get("headings")]
    assert with_headings
    assert any(c.meta["headings"] == ["指南"] for c in with_headings)


def test_semantic_chunks_at_topic_shift():
    from worddael.chunkers import SemanticChunker

    # toy embedder: vector is bag-of-char indicator -> topic A vs B disjoint
    def embed(texts):
        out = []
        for t in texts:
            v = [0.0, 0.0, 0.0]
            if "猫" in t:
                v[0] = 1.0
            if "狗" in t:
                v[1] = 1.0
            if "云" in t:
                v[2] = 1.0
            out.append(v)
        return out

    text = "猫在睡觉。猫在吃饭。狗在跑步。狗在叫。"
    chunks = SemanticChunker(
        embed_fn=embed, similarity_threshold=0.5, max_chars=100, min_chars=5
    ).chunk(text)
    assert len(chunks) >= 2
    texts = [c.text for c in chunks]
    assert any("猫" in t for t in texts) and any("狗" in t for t in texts)


def test_semantic_requires_embed_fn():
    import pytest

    from worddael.chunkers import SemanticChunker

    with pytest.raises(ValueError):
        SemanticChunker(embed_fn=None)


def test_get_chunker_factory():
    from worddael import get_chunker

    assert get_chunker("recursive").strategy == "recursive"
    import pytest

    with pytest.raises(ValueError):
        get_chunker("nope")


def test_factory_one_shot():
    from worddael import chunk

    chunks = chunk("内容一二三。内容四五六。", strategy="sentence", max_chars=10)
    assert all(isinstance(c.n_chars, int) for c in chunks)
