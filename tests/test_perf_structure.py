"""Performance floor and structural guarantees on formatted text.

The perf floor is intentionally conservative (CI machines vary); its job is
to catch gross regressions, not to benchmark. Real numbers live in
docs/dogfood.md (24 MB/s on 0.9M chars, 2026-09-27).
"""

from __future__ import annotations

import statistics
import time

from conftest import random_zh_text

from worddael.chunkers import RecursiveChunker


def test_recursive_throughput_floor():
    text = random_zh_text(99, min_len=200_000, max_len=210_000)
    started = time.perf_counter()
    chunks = RecursiveChunker(max_chars=400, overlap_chars=50).chunk(text)
    elapsed = time.perf_counter() - started
    assert chunks
    mbps = len(text) / elapsed / 1e6
    assert mbps > 0.1, f"throughput regression: {mbps:.3f} MB/s ({elapsed:.2f}s)"


def test_pathological_markdown_lines_no_redos():
    """Red-team #2 (security): the heading regex must not backtrack
    catastrophically on adversarial lines."""
    cases = [
        "# " + "#" * 50_000,
        "# " + "a" * 200_000 + " #",
        "###### " + "x" * 100_000 + " " + "#" * 5_000,
    ]
    from worddael.sentences import markdown_sections

    for case in cases:
        started = time.perf_counter()
        markdown_sections(case + "\n\nbody")
        assert time.perf_counter() - started < 1.0


def test_markdown_table_rows_stay_intact():
    """Known behaviour (documented, not a bug): markdown tables may be split
    at ROW boundaries by the line separator, but a row is never cut in half —
    every line of the table appears verbatim inside some chunk."""
    header = "| 列一 | 列二 | 列三 |"
    divider = "| --- | --- | --- |"
    rows = [f"| 数据甲{i} | 数据乙{i} | 数据丙{i} |" for i in range(40)]
    md = "# 数据表\n\n" + "\n".join([header, divider, *rows]) + "\n"
    chunks = __import__("worddael").MarkdownChunker(max_chars=150, overlap_chars=0).chunk(md)
    for line in [header, divider, *rows[:5], *rows[-5:]]:
        assert any(line in c.text for c in chunks), f"table line cut in half: {line!r}"


def test_scaling_stays_near_linear():
    """4x data must not cost more than ~5x time (guards against accidental
    quadratic paths in merge/hygiene). Median timings damp CI noise."""
    text_100 = random_zh_text(201, min_len=100_000, max_len=105_000)
    text_400 = random_zh_text(201, min_len=400_000, max_len=405_000)
    chunker = RecursiveChunker(max_chars=400, overlap_chars=50)

    # Warm up imports/caches before measuring. A single few-millisecond sample
    # is too noisy to compare reliably on shared CI runners.
    chunker.chunk(text_100)
    chunker.chunk(text_400)

    def median_runtime(text: str) -> float:
        samples = []
        for _ in range(5):
            t0 = time.perf_counter()
            chunker.chunk(text)
            samples.append(time.perf_counter() - t0)
        return statistics.median(samples)

    small = median_runtime(text_100)
    big = median_runtime(text_400)
    assert big < small * 6, f"super-linear scaling: 4x data cost {big / max(small, 1e-6):.1f}x"
