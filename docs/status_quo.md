# STATUS_QUO：现状切分的真实坏例子（可复现）

> 所有片段与数字来自 `tools/status_quo_experiment.py` 在 2026-09-26 的真实运行输出（复现命令在文末），不是构造的示意。
> 实验对象：同一段 202 字的中文制度文档，同一预算（chunk_size/max_chars = 80），对比三种切法。

## A. LangChain `RecursiveCharacterTextSplitter` 默认参数（RAG 教程最常见的用法）

分隔符为英文默认 `["\n\n", "\n", " ", ""]`，无任何中文标点：

```
chunk0 @0:  「第三章」                                    ← 标题被孤立成 3 字块
chunk1 @4:  差旅报销。员工出差乘坐高铁的，…报销标准为每…
chunk2 @83: 「限八十元。」                                ← 答案"每天上限八十元"被腰斩
chunk3 @90: 第四章 报销时限。…
```

结构指标：**50% 的块从句子中间开始**；问题答案「每天上限八十元」**被切成两半**，分别落在 chunk1 尾部和 chunk2。检索时任何一个半句都无法独立回答"市内交通每天报销上限"。

## B. LangChain 手工传入中文分隔符（`["\n\n", "。", "，", ""]`）

做对了第一步——但这是**每个中文使用者都要重新发明一次**的配置，而且仍有问题：

```
chunk1 @66: 「。市内交通费凭发票据实报销，每天上限八十元。」   ← 以句号开头的悬垂块
```

结构指标：33% 的块从句中开始（分隔符 `keep_separator=True` 默认把句号留在下一块开头）；`starts` 偏移量库本身不提供，引用回溯要自己算。

## C. worddael `RecursiveChunker(max_chars=80, overlap_chars=0)`

```
chunk0 @0:  第三章 差旅报销。…住宿费报销…
chunk1 @67: 市内交通费凭发票据实报销，每天上限八十元。       ← 答案完整，句子边界干净
chunk2 @90: 第四章 报销时限。…
```

结构指标：**100% 的块结束在句尾标点，0% 从句中开始**，4 个测试答案全部完整；每个块自带 `start/end` 字符偏移，`chunk.text == source[start:end]` 由模糊测试保证。

## 汇总对比

| 指标 | LC 默认参数 | LC 手工中文分隔符 | worddael recursive |
|---|---|---|---|
| 块结束在句尾标点 | 50% | 66.7% | **100%** |
| 块从句中开始 | 50% | 33.3% | **0%** |
| 4 个答案完整性 | 3/4（一处腰斩） | 4/4 | **4/4** |
| 字符偏移量 | 无（自己算） | 无（自己算） | **内置且模糊测试保证** |

## 与 chonkie 1.7.0 的功能对照（2026-09-27 安装实测）

`pip install chonkie` 后直接检查其公开 API 面（非 README 转述）：

| 能力 | chonkie 1.7.0 实测 | worddael |
|---|---|---|
| 切分器清单 | Cloud/Code/Late/Neural/Recursive/Semantic/Sentence/Slumber/Token | recursive/markdown/sentence/token/semantic/**parent-child** |
| 父子块（小-大检索） | **无**（chunkers 子模块无 parent/family/hierarchy 命名，Chunk 无父链接字段） | ✅ 一等公民 + 父块级评测 |
| 边界成因解释 | **无** | ✅ end_rule/start_rule |
| 标点卫生保证 | **无** | ✅ fuzz 测试锁定 |
| 偏移不变量 fuzz 声明 | 无此承诺 | ✅ 46+ 种子模糊测试 |
| 中文句界（引号闭合/省略号/ASCII 句点保护） | 无专项 | ✅ |
| 速度 | 官方宣称快 | 我们 24.6 MB/s（紅樓夢实测），不做跨库对比 |

复现：`pip install chonkie && python -c "from chonkie import chunker; print([n for n in dir(chunker) if not n.startswith('_')])"`

## 诚实说明

1. LangChain 并非不能配置好——B 方案已大幅改善，本文件论证的是**默认路径的坑 + 配置好的隐性成本**（每个团队手传一遍分隔符、手算一遍偏移、没有任何测试保证）。
2. 单文档示例不构成统计结论；规模化数字见 `benchmarks/results.md`（25 篇 / 75 问：recursive recall@1 0.96 vs fixed-window 0.907，见 benchmarks/results.md）。
3. 实验仅对比切分环节本身，不评价 LangChain 框架整体。

## 复现

```bash
pip install -e ".[jieba,dev]" langchain-text-splitters
python tools/status_quo_experiment.py
```
