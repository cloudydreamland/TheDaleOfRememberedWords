"""Strategy comparison report generator.

Runs every built-in strategy over the sample corpus, measures recall@k and
chunk-shape statistics, and writes a markdown report. CPU-only, zero API
cost by default; add ``--judge --live`` (with an API key in the environment)
to layer LLM answerability judging on top.

Usage::

    python -m worddael.eval.report --out benchmarks/results.md
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from ..chunkers import RecursiveChunker, SemanticChunker, SentenceChunker, TokenChunker
from ..embedders import HashingEmbedder
from .llm_judge import JudgeConfig, LLMJudge
from .retrieval import BM25, VectorRetriever, recall_at_k
from .sample_data import SAMPLE_DOCS


class FixedWindowChunker:
    """Naive fixed-length baseline: cut every N chars, ignore all structure.
    This is what you get with a careless LangChain-style default."""

    strategy = "fixed-window"

    def __init__(self, max_chars: int = 500, overlap_chars: int = 0) -> None:
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def chunk(self, text: str):
        from ..types import Chunk  # noqa: PLC0415

        step = max(1, self.max_chars - self.overlap_chars)
        chunks = []
        for seq, s in enumerate(range(0, len(text), step)):
            e = min(s + self.max_chars, len(text))
            chunks.append(
                Chunk(
                    text=text[s:e],
                    start=s,
                    end=e,
                    seq=seq,
                    strategy=self.strategy,
                    meta={"overlap_chars": self.overlap_chars if seq > 0 else 0},
                )
            )
            if e == len(text):
                break
        return chunks


def _hashing_embed(texts: list[str], dim: int = 96):
    """Deterministic trigram hashing — see worddael.embedders.HashingEmbedder.
    A TOY stand-in so the semantic pipeline is exercisable end-to-end with
    zero dependencies; not a quality claim."""
    return HashingEmbedder(dim=dim).embed(texts)


def build_strategies(max_chars: int, overlap_chars: int) -> dict:
    return {
        "fixed-window": FixedWindowChunker(max_chars=max_chars),
        "sentence": SentenceChunker(max_chars=max_chars, overlap_chars=overlap_chars),
        "token": TokenChunker(max_tokens=max_chars, overlap_tokens=overlap_chars),
        "recursive": RecursiveChunker(max_chars=max_chars, overlap_chars=overlap_chars),
        "semantic-hash-toy": SemanticChunker(
            embed_fn=_hashing_embed,
            max_chars=max_chars,
            min_chars=max(20, max_chars // 3),
            overlap_chars=overlap_chars,
        ),
    }


def run_report(k: int = 3, max_chars: int = 300, overlap_chars: int = 50) -> list[dict]:
    rows = []
    strategies = build_strategies(max_chars, overlap_chars)
    punct = "。！？；…!?;"
    for name, chunker in strategies.items():
        hits = 0
        total = 0
        n_chunks = 0
        lens = []
        end_ok = 0
        precision_hits = 0.0
        union_hits = 0.0
        misses: list[str] = []
        started = time.perf_counter()
        for doc in SAMPLE_DOCS:
            chunks = chunker.chunk(doc["text"])
            res = recall_at_k(chunks, doc["text"], doc["questions"], k=k)
            hits += res["hits"]
            total += res["total"]
            n_chunks += len(chunks)
            lens.extend(c.n_chars for c in chunks)
            end_ok += sum(
                1
                for c in chunks
                if c.text.rstrip() and c.text.rstrip()[-1] in punct
            )
            misses.extend(f"{doc['id']}|{m}" for m in res["misses"])
            precision_hits += res["precision"] * res["total"] * k
            union_hits += res["union_recall"] * res["total"]
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "strategy": name,
                "recall@k": round(hits / max(1, total), 3),
                "precision@k": round(precision_hits / max(1, total * k), 3),
                "union_recall": round(union_hits / max(1, total), 3),
                "k": k,
                "hits": hits,
                "total": total,
                "n_chunks": n_chunks,
                "avg_chars": round(sum(lens) / max(1, len(lens)), 1),
                "max_chars_seen": max(lens) if lens else 0,
                "ends_on_punct": f"{end_ok}/{len(lens)}",
                "ends_on_punct_frac": round(end_ok / max(1, len(lens)), 3),
                "misses": misses,
                "time_ms": round(elapsed * 1000, 1),
            }
        )
    return rows


def retrieval_interaction_report(k: int = 1, max_chars: int = 150, overlap_chars: int = 30) -> list[dict]:
    """Strategy rows under two retrievers: BM25 vs toy-vector cosine.

    The vector column validates the retriever-swapping infrastructure with
    HashingEmbedder stand-in vectors — explicitly NOT a quality claim; real
    embeddings will change these numbers."""

    def hash_vector_retriever(docs):
        from ..embedders import HashingEmbedder  # noqa: PLC0415

        return VectorRetriever(docs, HashingEmbedder(dim=96).embed)

    retrievers = {"bm25": BM25, "vector-hash-toy": hash_vector_retriever}
    strategies = build_strategies(max_chars, overlap_chars)
    rows = []
    for name, chunker in strategies.items():
        scores = {rname: [0, 0] for rname in retrievers}
        for doc in SAMPLE_DOCS:
            chunks = chunker.chunk(doc["text"])
            for rname, factory in retrievers.items():
                res = recall_at_k(chunks, doc["text"], doc["questions"], k=k, retriever_factory=factory)
                scores[rname][0] += res["hits"]
                scores[rname][1] += res["total"]
        rows.append(
            {
                "strategy": name,
                "recall_bm25": round(scores["bm25"][0] / max(1, scores["bm25"][1]), 3),
                "recall_vector_toy": round(scores["vector-hash-toy"][0] / max(1, scores["vector-hash-toy"][1]), 3),
            }
        )
    return rows


def format_markdown(rows: list[dict], cols: list[str] | None = None) -> str:
    if cols is None:
        cols = [
            "strategy",
            "recall@k",
            "precision@k",
            "union_recall",
            "hits",
            "total",
            "n_chunks",
            "avg_chars",
        ]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def parent_child_report(
    k_list=(1, 3), max_chars: int = 150, parent_max_chars: int = 600, overlap_chars: int = 30
) -> list[dict]:
    """Flat recursive vs small-to-big (retrieve children, credit parents).

    Both rows use the same child budget, so the only difference is what the
    retriever's hit is credited against: the small chunk itself vs its
    parent context."""
    from ..parent_child import ParentChildChunker
    from .retrieval import recall_at_k_parent_level

    flat = RecursiveChunker(max_chars=max_chars, overlap_chars=overlap_chars)
    family = ParentChildChunker(
        parent_max_chars=parent_max_chars,
        child_max_chars=max_chars,
        child_overlap_chars=overlap_chars,
    )
    rows = []
    for k in k_list:
        flat_hits = flat_total = pc_hits = pc_total = 0
        n_parents = n_children = 0
        for doc in SAMPLE_DOCS:
            flat_chunks = flat.chunk(doc["text"])
            flat_res = recall_at_k(flat_chunks, doc["text"], doc["questions"], k=k)
            flat_hits += flat_res["hits"]
            flat_total += flat_res["total"]
            parents, children = family.chunk_families(doc["text"])
            pc_res = recall_at_k_parent_level(children, parents, doc["text"], doc["questions"], k=k)
            pc_hits += pc_res["hits"]
            pc_total += pc_res["total"]
            n_parents += len(parents)
            n_children += len(children)
        rows.append(
            {
                "k": k,
                "flat_recursive": round(flat_hits / max(1, flat_total), 3),
                "parent_child": round(pc_hits / max(1, pc_total), 3),
                "parents": n_parents,
                "children": n_children,
            }
        )
    return rows


def format_parent_child_table(rows: list[dict]) -> str:
    cols = ["k", "flat_recursive", "parent_child", "parents", "children"]
    lines = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join("---" for _ in cols) + "|",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def full_report(k_list=(1, 3), max_chars: int = 150, overlap_chars: int = 30) -> str:
    """Dual-k markdown report with corpus metadata and failure lists."""
    from .sample_data import SAMPLE_DOCS as _docs

    n_docs = len(_docs)
    n_q = sum(len(d["questions"]) for d in _docs)
    n_chars = sum(len(d["text"]) for d in _docs)
    lines = [
        "# worddael 内置基准报告",
        "",
        f"- 语料：{n_docs} 篇原创中文文档 / {n_q} 个问题 / 共 {n_chars} 字",
        f"- 切分预算：max_chars={max_chars}, overlap_chars={overlap_chars}",
        "- 指标：recall@k = 答案所在块被 BM25 检索命中的比例",
        "- 诚实声明：这是包内置的小型基准，用于版本间回归比较，不是公开排行榜结论",
        "",
    ]
    all_rows: dict[int, list[dict]] = {}
    for k in k_list:
        rows = run_report(k=k, max_chars=max_chars, overlap_chars=overlap_chars)
        all_rows[k] = rows
        lines.append(f"## recall@{k}（max_chars={max_chars}）")
        lines.append("")
        lines.append(format_markdown(rows))
        lines.append("")

    misses = all_rows[k_list[0]]
    lines.append(f"## 失败清单（k={k_list[0]}，每策略最多列 8 条）")
    lines.append("")
    for r in misses:
        shown = r["misses"][:8]
        lines.append(f"### {r['strategy']}（{len(r['misses'])} 未命中）")
        for m in shown:
            lines.append(f"- {m}")
        lines.append("")

    pc_rows = parent_child_report(k_list=k_list, max_chars=max_chars, overlap_chars=overlap_chars)
    lines.append("## 小-大检索对照（同样 150 字子块：检索子块、按父块 600 字判分 vs 按子块判分）")
    lines.append("")
    lines.append(format_parent_child_table(pc_rows))
    lines.append("")

    lines.append("## 分块 × 检索方式交互（recall@1）")
    lines.append("")
    lines.append(
        "向量列使用 HashingEmbedder 玩具向量，**仅验证检索器基础设施，不是质量结论**；"
        "真实 embedding 下的数字待接入真实模型后重新生成。"
    )
    lines.append("")
    lines.append(
        format_markdown(
            retrieval_interaction_report(max_chars=max_chars, overlap_chars=overlap_chars),
            cols=["strategy", "recall_bm25", "recall_vector_toy"],
        )
    )
    lines.append("")

    lines.append("## 计数器校准")
    lines.append("")
    try:
        from ..counters import CharCounter, JiebaCounter, calibration_report

        cal = calibration_report(CharCounter(), JiebaCounter(), [d["text"] for d in _docs])
        lines.append(
            f"- CharCounter（每 CJK 字符 ≈1 token）相对 jieba 词计数：平均绝对相对误差 "
            f"**{cal['mean_abs_rel_err']:.1%}**（最大 {cal['max_abs_rel_err']:.1%}，{cal['n']} 篇）"
        )
        lines.append("- 注意：jieba 词数本身 ≠ 模型 BPE token 数，此对照只量化启发式与词级计数的偏差方向和量级。")
    except ImportError:
        lines.append("- （未安装 jieba，跳过 jieba 对照）")
    try:
        from ..counters import CharCounter, TiktokenCounter, calibration_report

        cal_bpe = calibration_report(CharCounter(), TiktokenCounter("cl100k_base"), [d["text"] for d in _docs])
        lines.append(
            f"- CharCounter 相对 tiktoken/cl100k_base（英文语料的 BPE 参照）：平均绝对相对误差 "
            f"**{cal_bpe['mean_abs_rel_err']:.1%}**（最大 {cal_bpe['max_abs_rel_err']:.1%}，{cal_bpe['n']} 篇）"
        )
    except ImportError:
        lines.append("- （未安装 tiktoken，跳过 BPE 对照：`pip install worddael[tiktoken]`）")
    lines.append("- 解读：启发式计数用于预算切分足够（预算本就是近似目标）；**不可用于计费估算**。")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="worddael-eval-report")
    parser.add_argument("--k", type=int, default=None, help="single-k quick view (default: full report)")
    parser.add_argument("--max-chars", type=int, default=150)
    parser.add_argument("--overlap-chars", type=int, default=30)
    parser.add_argument("--out", type=str, default=None, help="write markdown report to this path")
    parser.add_argument(
        "--judge-estimate",
        action="store_true",
        help="print what a live LLM-judge run over this corpus would cost",
    )
    args = parser.parse_args(argv)

    if args.k is not None:
        report = format_markdown(run_report(k=args.k, max_chars=args.max_chars, overlap_chars=args.overlap_chars))
    else:
        report = full_report(max_chars=args.max_chars, overlap_chars=args.overlap_chars)

    if args.judge_estimate:
        judge = LLMJudge(JudgeConfig())
        n_q = sum(len(d["questions"]) for d in SAMPLE_DOCS)
        n_chunks_per_doc = len(RecursiveChunker(max_chars=args.max_chars).chunk("x" * 5000))
        est = judge.estimate_cost(n_q * n_chunks_per_doc, avg_chars=350)
        report += (
            "\nLLM-judge live run estimate: "
            + json_dumps(est)
            + "\n(dry-run mode costs nothing and needs no API key)\n"
        )

    print(report)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"written -> {out}")
    return 0


def json_dumps(d: dict) -> str:
    import json  # noqa: PLC0415

    return json.dumps(d, ensure_ascii=False)


if __name__ == "__main__":
    raise SystemExit(main())
