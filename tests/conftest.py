"""Shared fixtures/helpers."""

from __future__ import annotations

import random

ZH_CHARS = "数据文本切分算法自然语言处理检索增强生成模型训练测试中文支持偏移量准确引用结构化解析器块大小限制重叠窗口策略递归句号问号感叹号逗号顿号分号冒号引号括号"

ZH_SENTENCES = [
    "本制度适用于全体正式员工。",
    "员工出差乘坐高铁可全额报销！",
    "报销时限为三十天内提交；逾期系统自动关闭。",
    "系统支持中文分块和英文混排处理。",
    "第二代产品续航提升至十四天。",
    "配置参数保存在 config.yaml 文件中。",
]


def random_zh_text(seed: int, min_len: int = 3000, max_len: int = 8000) -> str:
    rng = random.Random(seed)
    parts: list[str] = []
    total = 0
    target = rng.randint(min_len, max_len)
    while total < target:
        kind = rng.random()
        if kind < 0.15:
            parts.append("\n\n")
            total += 2
        elif kind < 0.25:
            parts.append("\n")
            total += 1
        elif kind < 0.32:
            # a latin token run (mixed-language reality)
            run = "".join(rng.choice("abcdefgh0123456789 ") for _ in range(rng.randint(3, 30)))
            parts.append(run)
            total += len(run)
        else:
            s = rng.choice(ZH_SENTENCES)
            parts.append(s)
            total += len(s)
    return "".join(parts)
