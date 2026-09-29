"""LLM-based evaluation harness for chunking quality (token-hungry by design).

This is where "unlimited tokens" becomes a moat: answerability judging over
large question sets is exactly the kind of evaluation that is too expensive
for most teams to run properly.

Two modes:
- ``dry_run=True`` (default): a lexical heuristic stands in for the LLM so
  the whole pipeline can be exercised with zero API cost. Results are
  clearly marked ``method="dry-run-bm25"`` and are NOT quality claims.
- live mode: any OpenAI-compatible /chat/completions endpoint (vLLM, Zhipu,
  DeepSeek, OpenAI, ...). Uses urllib only — no required dependency.
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass, field

from ..counters import CharCounter
from .retrieval import BM25, tokenize_zh


@dataclass
class JudgeConfig:
    base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    model: str = "glm-4-flash"
    api_key: str | None = None
    price_per_1k_input: float = 0.0
    price_per_1k_output: float = 0.0
    timeout: int = 60
    extra: dict = field(default_factory=dict)


PROMPT_TEMPLATE = """你是一个严格的评测员。下面是一段从文档中切出的片段和一个问题。

片段：
{chunk}

问题：{question}

请判断这个片段是否包含回答该问题所需的信息。只输出 JSON：
{{"score": 1-5, "reason": "一句话理由"}}
1=完全无关，3=沾边但不足以回答，5=包含完整答案所需信息。"""


def _heuristic_score(chunk_text: str, question: str, scorer: BM25) -> tuple[int, str]:
    top = scorer.search(question, k=1)
    if not top:
        return 1, "empty chunk"
    _, score = top[0]
    q_tokens = set(tokenize_zh(question))
    c_tokens = set(tokenize_zh(chunk_text))
    overlap = len(q_tokens & c_tokens) / max(1, len(q_tokens))
    if score > 0 and overlap >= 0.5:
        return 4, "high lexical overlap with question"
    if score > 0:
        return 3, "moderate lexical overlap"
    return 1, "no lexical overlap"


class LLMJudge:
    """Answerability judge over (chunk, question) pairs."""

    def __init__(self, config: JudgeConfig | None = None) -> None:
        self.config = config or JudgeConfig()
        self._counter = CharCounter()

    def estimate_cost(self, n_calls: int, avg_chars: int) -> dict:
        """Rough pre-flight cost estimate using the heuristic counter."""
        tokens_per_call = self._counter.count("x" * avg_chars) + 100  # prompt overhead
        input_cost = n_calls * tokens_per_call / 1000 * self.config.price_per_1k_input
        output_cost = n_calls * 50 / 1000 * self.config.price_per_1k_output
        return {
            "n_calls": n_calls,
            "est_input_tokens": n_calls * tokens_per_call,
            "est_cost": round(input_cost + output_cost, 4),
            "currency_note": "price_per_1k fields are in your provider's billing unit",
        }

    def _call_llm(self, chunk_text: str, question: str) -> dict:
        cfg = self.config
        prompt = PROMPT_TEMPLATE.format(chunk=chunk_text[:2000], question=question)
        payload = {
            "model": cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            **cfg.extra,
        }
        headers = {"Content-Type": "application/json"}
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
        req = urllib.request.Request(
            cfg.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=cfg.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        m = re.search(r"\{[^{}]*\}", content, re.DOTALL)
        if not m:
            return {"score": -1, "reason": f"unparseable: {content[:80]}"}
        parsed = json.loads(m.group(0))
        return {"score": int(parsed.get("score", -1)), "reason": str(parsed.get("reason", ""))[:200]}

    def score_pair(
        self,
        chunk_text: str,
        question: str,
        scorer: BM25 | None = None,
        dry_run: bool = True,
    ) -> dict:
        if dry_run:
            lexical = scorer or BM25([chunk_text])
            score, reason = _heuristic_score(chunk_text, question, lexical)
            return {"score": score, "reason": reason, "method": "dry-run-bm25"}
        result = self._call_llm(chunk_text, question)
        result["method"] = "llm"
        return result
