# worddael 内置基准报告

- 语料：25 篇原创中文文档 / 75 个问题 / 共 9044 字
- 切分预算：max_chars=150, overlap_chars=30
- 指标：recall@k = 答案所在块被 BM25 检索命中的比例
- 诚实声明：这是包内置的小型基准，用于版本间回归比较，不是公开排行榜结论

## recall@1（max_chars=150）

| strategy | recall@k | precision@k | union_recall | hits | total | n_chunks | avg_chars |
|---|---|---|---|---|---|---|---|
| fixed-window | 0.947 | 0.947 | 0.947 | 71 | 75 | 73 | 123.9 |
| sentence | 0.973 | 0.973 | 0.973 | 73 | 75 | 77 | 126.0 |
| token | 0.947 | 0.947 | 0.947 | 71 | 75 | 75 | 131.6 |
| recursive | 0.973 | 0.973 | 0.973 | 73 | 75 | 84 | 110.0 |
| semantic-hash-toy | 0.92 | 0.92 | 0.92 | 69 | 75 | 142 | 85.7 |


## recall@3（max_chars=150）

| strategy | recall@k | precision@k | union_recall | hits | total | n_chunks | avg_chars |
|---|---|---|---|---|---|---|---|
| fixed-window | 0.973 | 0.324 | 0.987 | 73 | 75 | 73 | 123.9 |
| sentence | 1.0 | 0.351 | 1.0 | 75 | 75 | 77 | 126.0 |
| token | 1.0 | 0.36 | 1.0 | 75 | 75 | 75 | 131.6 |
| recursive | 1.0 | 0.333 | 1.0 | 75 | 75 | 84 | 110.0 |
| semantic-hash-toy | 1.0 | 0.444 | 1.0 | 75 | 75 | 142 | 85.7 |


## 失败清单（k=1，每策略最多列 8 条）

### fixed-window（4 未命中）
- product|手表开不了机怎么办？
- swim|初学者适合多深的泳池？
- cat|猫不能吃哪些东西？
- wedding|婚礼应急包里有什么？

### sentence（2 未命中）
- rental|提前退租违约金一般多少？
- cat|猫不能吃哪些东西？

### token（4 未命中）
- rental|提前退租违约金一般多少？
- attendance|迟到多少分钟算迟到？
- cat|猫不能吃哪些东西？
- astro|什么时候观测行星最好？

### recursive（2 未命中）
- ecommerce|生鲜坏了怎么处理？
- cat|猫不能吃哪些东西？

### semantic-hash-toy（6 未命中）
- attendance|迟到多少分钟算迟到？
- succulent|多肉夏天怎么浇水？
- swim|初学者适合多深的泳池？
- ecommerce|什么商品不支持七天无理由退货？
- cat|猫不能吃哪些东西？
- astro|什么时候观测行星最好？

## 小-大检索对照（同样 150 字子块：检索子块、按父块 600 字判分 vs 按子块判分）

| k | flat_recursive | parent_child | parents | children |
|---|---|---|---|---|
| 1 | 0.973 | 1.0 | 25 | 84 |
| 3 | 1.0 | 1.0 | 25 | 84 |

## 分块 × 检索方式交互（recall@1）

向量列使用 HashingEmbedder 玩具向量，**仅验证检索器基础设施，不是质量结论**；真实 embedding 下的数字待接入真实模型后重新生成。

| strategy | recall_bm25 | recall_vector_toy |
|---|---|---|
| fixed-window | 0.947 | 0.453 |
| sentence | 0.973 | 0.48 |
| token | 0.947 | 0.507 |
| recursive | 0.973 | 0.52 |
| semantic-hash-toy | 0.92 | 0.32 |


## 计数器校准

- （未安装 jieba，跳过 jieba 对照）
- （未安装 tiktoken，跳过 BPE 对照：`pip install worddael[tiktoken]`）
- 解读：启发式计数用于预算切分足够（预算本就是近似目标）；**不可用于计费估算**。
