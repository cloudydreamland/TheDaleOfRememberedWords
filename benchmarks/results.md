# worddael 内置基准报告

- 语料：25 篇原创中文文档 / 75 个问题 / 共 9044 字
- 切分预算：max_chars=150, overlap_chars=30
- 指标：recall@k = 答案所在块被 BM25 检索命中的比例
- 诚实声明：这是包内置的小型基准，用于版本间回归比较，不是公开排行榜结论

## recall@1（max_chars=150）

| strategy | recall@k | precision@k | union_recall | hits | total | n_chunks | avg_chars |
|---|---|---|---|---|---|---|---|
| fixed-window | 0.907 | 0.907 | 0.907 | 68 | 75 | 73 | 123.9 |
| sentence | 0.92 | 0.92 | 0.92 | 69 | 75 | 77 | 126.0 |
| token | 0.933 | 0.933 | 0.933 | 70 | 75 | 75 | 131.6 |
| recursive | 0.96 | 0.96 | 0.96 | 72 | 75 | 84 | 110.0 |
| semantic-hash-toy | 0.813 | 0.813 | 0.813 | 61 | 75 | 142 | 85.7 |


## recall@3（max_chars=150）

| strategy | recall@k | precision@k | union_recall | hits | total | n_chunks | avg_chars |
|---|---|---|---|---|---|---|---|
| fixed-window | 0.96 | 0.32 | 0.973 | 72 | 75 | 73 | 123.9 |
| sentence | 0.987 | 0.347 | 0.987 | 74 | 75 | 77 | 126.0 |
| token | 0.987 | 0.356 | 0.987 | 74 | 75 | 75 | 131.6 |
| recursive | 0.973 | 0.324 | 0.973 | 73 | 75 | 84 | 110.0 |
| semantic-hash-toy | 0.973 | 0.422 | 0.973 | 73 | 75 | 142 | 85.7 |


## 失败清单（k=1，每策略最多列 8 条）

### fixed-window（7 未命中）
- policy|发票超过多久就不能报销了？
- product|手表开不了机怎么办？
- recipe|炒糖色炒过了会怎么样？
- swim|初学者适合多深的泳池？
- library|逾期还书怎么收费？
- coffee|闷蒸要多久？
- wedding|婚礼应急包里有什么？

### sentence（6 未命中）
- policy|发票超过多久就不能报销了？
- library|逾期还书怎么收费？
- interview|被拒后多久可以再投同一家公司？
- coffee|闷蒸要多久？
- astro|入门望远镜看什么参数？
- astro|什么时候观测行星最好？

### token（5 未命中）
- policy|发票超过多久就不能报销了？
- library|逾期还书怎么收费？
- interview|被拒后多久可以再投同一家公司？
- coffee|闷蒸要多久？
- astro|什么时候观测行星最好？

### recursive（3 未命中）
- policy|发票超过多久就不能报销了？
- ecommerce|生鲜坏了怎么处理？
- coffee|闷蒸要多久？

### semantic-hash-toy（14 未命中）
- policy|发票超过多久就不能报销了？
- travel|古城墙从哪里上比较好？
- product|XWatch 5 开启全天血氧监测后续航多久？
- rental|墙面正常褪色房东能扣押金吗？
- succulent|多肉夏天怎么浇水？
- swim|初学者适合多深的泳池？
- ecommerce|什么商品不支持七天无理由退货？
- ecommerce|生鲜坏了怎么处理？

## 小-大检索对照（同样 150 字子块：检索子块、按父块 600 字判分 vs 按子块判分）

| k | flat_recursive | parent_child | parents | children |
|---|---|---|---|---|
| 1 | 0.96 | 0.987 | 25 | 84 |
| 3 | 0.973 | 0.987 | 25 | 84 |

## 分块 × 检索方式交互（recall@1）

向量列使用 HashingEmbedder 玩具向量，**仅验证检索器基础设施，不是质量结论**；真实 embedding 下的数字待接入真实模型后重新生成。

| strategy | recall_bm25 | recall_vector_toy |
|---|---|---|
| fixed-window | 0.907 | 0.453 |
| sentence | 0.92 | 0.48 |
| token | 0.933 | 0.507 |
| recursive | 0.96 | 0.52 |
| semantic-hash-toy | 0.813 | 0.32 |


## 计数器校准

- CharCounter（每 CJK 字符 ≈1 token）相对 jieba 词计数：平均绝对相对误差 **78.0%**（最大 90.8%，25 篇）
- 注意：jieba 词数本身 ≠ 模型 BPE token 数，此对照只量化启发式与词级计数的偏差方向和量级。
- （未安装 tiktoken，跳过 BPE 对照：`pip install worddael[tiktoken]`）
- 解读：启发式计数用于预算切分足够（预算本就是近似目标）；**不可用于计费估算**。
