# 评测指南：拿到 API key 之后如何跑大规模 LLM 判分

> 面向：想在真实 LLM 判分下评测分块策略的使用者（也是 worddael 评测方法论的说明书）。
> 原则：先干跑、再估价、后实跑；一切结果可断点续跑、可审计。

## 0. 概念

- **干跑（dry-run）**：LLMJudge 用 BM25 词法重叠做启发式打分，零成本、零网络。用于验证管线与数据流，**分数不是质量结论**。
- **实跑（live）**：真实调用 OpenAI 兼容 `/chat/completions` 端点逐对判分（1–5 分）。每条记录带时间戳与方法标记，落盘 JSONL。

## 1. 干跑（无需任何 key）

```bash
python -m worddael.eval.judge_runner --state results/judge.jsonl --rpm 60000
```

输出：总数/判定数/跳过数/错误数，以及按策略汇总的平均分。

## 2. 估价

```bash
python -m worddael.eval.report --judge-estimate
```

内置语料（25 篇 / 75 问）在全策略下的判分对数量约为 1000+ 对。实跑成本估算公式：

```
est_cost ≈ n_pairs × (input_tokens × price_in + 50 × price_out) / 1000
```

建议先只跑一个策略验证质量（`--strategy recursive`），确认 prompt 输出稳定后再全量。

## 3. 实跑（以 Zhipu 为例）

```bash
export ZHIPUAI_API_KEY="你的key"
python -m worddael.eval.judge_runner \
    --state results/judge_live.jsonl \
    --live --api-key-env ZHIPUAI_API_KEY \
    --base-url https://open.bigmodel.cn/api/paas/v4 \
    --model glm-4-flash \
    --rpm 30 --max-retries 4 --cost-cap 5.0
```

参数语义：

| 参数 | 语义 |
|---|---|
| `--state` | JSONL 落盘文件。**断点续跑靠它**：已有 key 的对直接跳过，中断后重跑同一命令即可继续 |
| `--rpm` | 每分钟请求数（进程内限速，按墙钟计算） |
| `--max-retries` | 异常重试次数，指数退避（2s、4s、8s…）；重试耗尽记录为 error 记录，不丢数据 |
| `--cost-cap` | 累计估算成本上限，超过即安全停止（估算用启发式计数器，偏保守） |

## 4. 汇总

```bash
python -c "from worddael.eval.judge_runner import summarize; import json; print(json.dumps(summarize('results/judge_live.jsonl'), ensure_ascii=False, indent=2))"
```

每个策略得到 `{n, errors, mean_score}`。**解读纪律**：

1. 干跑分数必须与实跑分开呈现，不得混用；
2. LLM 判分有系统性偏差（长文本偏好、词面重叠偏好），结论要配合 `benchmarks/results.md` 的 BM25 recall@k 交叉验证；
3. 报告任何数字时注明语料规模与切分预算。

## 5. 记录格式

state 文件每行一条：

```json
{"key": "recursive|policy|0|3", "strategy": "recursive", "doc_id": "policy", "q_idx": 0,
 "question": "出差的住宿费报销标准是多少钱一晚？", "chunk_seq": 3, 
 "ts": "2026-09-26T03:20:00", "score": 5, "reason": "...", "method": "llm"}
```

损坏的最后一行（进程被杀导致）在续跑时自动忽略。

## 6. 安全注意事项（红队 #2 / S2）

- **文档内容视为不可信输入**：qa_gen 与 LLMJudge 都会把语料文本拼进 prompt——恶意语料可操纵生成或判分结果（提示词注入）。判分输出只用于聚合统计，不要把单条高分记录当作可信事实。
- 生成的 QA 对入库前必须过 gold 逐字校验（`QAGenerator` 已内置），且建议人工抽检 5%。
- base_url/api_key 通过参数或环境变量显式配置，库不会隐式外呼；请勿把 key 写进 state 文件目录。

## 6. 扩展到自己的语料

`build_pairs(build_strategies(max_chars, overlap_chars), your_docs)` 接受任何
`[{"id":..., "text":..., "questions":[{"q":..., "gold":...}]}]` 形状的文档列表。
question 里的 `gold` 在 judge 路径不使用（只用于 BM25 recall），可以留空字符串但建议保留以便交叉验证。

## 7. 方法论定位（与业界实践对齐，2026-09 检索）

worddael 的评测协议采用业界共识的三层架构：

1. **确定性检索指标**（本库内置）：gold-span 的 recall@k / precision@k / union_recall——便宜、可复现、进 CI；
2. **LLM 判分**（LLMJudge + JudgeRunner）：语义层面的 answerability，需要校准；
3. **人工抽检**：校准 judge 与抽查生成质量。

业界参照：LegalBench-RAG 的 gold-snippet 协议（span 级 precision/recall）；故障归因（chunking vs retrieval vs generation 的 oracle 上界分析）列为本库 backlog；判分校准（judge 与人工一致率）待有 key 后进行。

数字纪律重申：本库 75 问小基准只用于版本回归，跨项目比较需要百篇级语料（qa_gen 拿到 key 后可扩）。

## 8. Worked example：怎么读 `benchmarks/results.md`（真实数字，2026-09-27）

以当前 25 篇 / 75 问 / max_chars=150 的内置基准为例：

**recall@1 与 recall@3 要连读。** recursive recall@1 0.96、recall@3 0.973——松一圈容差只换来 +1.3pp，说明它的块边界与问题相关性强；fixed-window 从 0.907 到 0.96（+5.3pp），说明它 k=1 的失误里有一部分是"答案块排到了第 2-3 位"——结构差的切分器更依赖大 k 容错。

**precision@k 在小语料上有天花板。** 每个问题通常只有 1 个相关块，k=3 时 precision@3 理论上限就是 1/3≈0.33——表中 0.32–0.42 不是"噪声多"，是 gold 唯一性使然。precision 的区分力要等 k 减小或 gold 多样化（多出现 gold 会让 precision 上探）。

**union_recall 盯住"被切开的答案"。** fixed-window union 0.973 > strict 0.96：有约 1.3% 的问题其 gold 被预算硬切成了两半、两半又都被检回——这批在 strict 里记失误，在 union 里记部分成功。两列的差值就是"切分把答案切碎"的直接度量，**差值大的切分策略在给 LLM 喂上下文时风险更高**（半句答案不足以作答）。

**失败清单是最值钱的部分。** 每条 `doc|问题` 都能人工复核：是切分切断答案（切分之过）、还是 BM25 词不匹配（检索之过）、还是问题本身歧义（语料之过）——故障归因从这里开始。
