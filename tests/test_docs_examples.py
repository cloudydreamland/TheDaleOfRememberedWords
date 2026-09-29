"""Docs truth audit: every code pattern shown in README must keep working.

These tests mirror the README quickstart snippets 1:1 (the snippets read
user files, so here they run against the built-in sample corpus — the API
surface exercised is identical). If a snippet's API changes and these fail,
update the docs in the same commit.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from worddael import SemanticChunker, chunk
from worddael.eval.sample_data import SAMPLE_DOCS


def test_readme_recursive_snippet():
    """README: chunk() + 偏移不变量断言。"""
    text = SAMPLE_DOCS[0]["text"]
    chunks = chunk(text, strategy="recursive", max_chars=500, overlap_chars=50)
    assert chunks
    for c in chunks:
        assert c.text == text[c.start : c.end]  # README promises this holds


def test_readme_semantic_snippet():
    """README: SemanticChunker + OpenAICompatibleEmbedder + HashingEmbedder."""
    from worddael import HashingEmbedder, OpenAICompatibleEmbedder

    text = SAMPLE_DOCS[3]["text"]
    embedder = OpenAICompatibleEmbedder(
        base_url="https://open.bigmodel.cn/api/paas/v4",
        model="embedding-3",
        api_key_env="ZHIPUAI_API_KEY",
    )
    # no network in tests: swap in the hashing backend via the documented
    # instance-override behaviour, then chunk
    embedder.embed = HashingEmbedder().embed  # type: ignore[method-assign]
    chunks = SemanticChunker(embed_fn=embedder, similarity_threshold=0.55, max_chars=500, min_chars=80).chunk(text)
    assert chunks

    toy = SemanticChunker(embed_fn=HashingEmbedder()).chunk(text)
    assert toy


def test_readme_cli_examples():
    """README: `worddael demo` and the manual.txt one-liner (with the real
    --overlap-chars flag, not an argparse abbreviation)."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "manual.txt"
        f.write_text(SAMPLE_DOCS[1]["text"], encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, "-m", "worddael.cli", str(f), "--strategy", "recursive",
             "--max-chars", "400", "--overlap-chars", "50", "--jsonl", str(Path(td) / "c.jsonl")],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert (Path(td) / "c.jsonl").exists()


def test_readme_strategy_table_matches_registry():
    """Every strategy documented in the README table exists in STRATEGIES."""
    from worddael import STRATEGIES

    readme = open("README.md", encoding="utf-8").read()
    documented = {"recursive", "markdown", "sentence", "token", "semantic", "parent-child"}
    missing = {s for s in documented if f"| `{s}`" not in readme}
    assert not missing, f"strategies missing from README table: {missing}"
    assert documented <= set(STRATEGIES)


def test_readme_internal_links_resolve():
    """Doc files referenced from README must exist."""
    from pathlib import Path

    readme = Path("README.md").read_text(encoding="utf-8")
    import re

    links = re.findall(r"\]\((docs/[^)#]+|GAP_PROOF\.md|benchmarks/[^)#]+)\)", readme)
    assert links, "expected doc links in README"
    for rel in links:
        assert Path(rel).exists(), f"README links to missing file: {rel}"


def test_benchmarks_results_file_is_current():
    """benchmarks/results.md must be regenerable byte-for-byte from the
    current code — stale benchmarks break the version-regression contract.
    (Deterministic by design; see test_report_is_byte_deterministic.)"""
    results = Path("benchmarks/results.md")
    if not results.exists():
        pytest.skip("benchmarks/results.md not generated yet")
    from worddael.eval.report import full_report

    assert full_report() == results.read_text(encoding="utf-8")
