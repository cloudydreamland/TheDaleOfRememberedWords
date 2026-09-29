# REPORT — worddael 两晚红队驱动迭代总结（终版，2026-09-27 06:45）

> 给刚醒来的你。一句话版本：**两晚 23 轮迭代，四个查证过的新颖特性落地，210 项测试全绿，v0.1.0rc3 可发布；三个真实阻塞项需要你。**

## 一、当前状态（可信核查）

- `pytest`：**210 passed**（偏移不变量模糊测试 46+ 种子、标点卫生 fuzz、缩放/性能地板、ReDoS 回归、文档示例镜像、基准时效性守卫、报告确定性）
- `ruff check src tests`：**0 error**
- 版本 **v0.1.0rc3**（pyproject / `__version__` / wheel METADATA 三处一致），工作区干净，45 个提交
- CI：GitHub Actions 就绪（Linux + Windows × Python 3.10–3.13 矩阵），等仓库创建后即跑

## 二、三阶段产出清单

### 第一晚（09-26 凌晨）：从零到库
骨架 + 五策略 + 偏移不变量 fuzz + 迷你基准 + 评测框架（iter0–iter8，详见 git 历史与下文"历史报告"存档）。

### 第二阶段（09-26 晚）："继续迭代"
- `ParentChildChunker`（父子块小-大检索——竞品扫描发现的手搓痛点库级化）
- `VectorRetriever` + 分块×检索交互评测 + 计数器校准 + Windows CI
- `qa_gen` 生成式 QA 扩充脚手架（gold 逐字硬校验、断点续跑）

### 第三阶段（09-26 深夜–09-27 晨）：红队驱动 23 轮
| 轮 | 质疑角色 | 产出 |
|---|---|---|
| R1 | 三角色开题 | 缺口新颖性验证：**可解释切分、标点卫生全网无先例**；chonkie 无父子块 |
| R2 | 开发者 | **可解释切分**：后验边界成因分类器（end_rule/start_rule），解释永不漂移，默认零开销 |
| R3 | 开发者 | **标点卫生保证**（块尾不开口/块首不闭口标点，预算只缩不涨）+ overlap 句界对齐 + gold 多出现消歧 |
| R4 | PM | **LangChain TextSplitter 适配器**（真实集成测试）+ `worddael demo` 一键演示 |
| R5 | PM/老板 | **《紅樓夢》90.6 万字真实语料实测**：24.6 MB/s、卫生违例 4→0、发现全角句点'．'漏洞 |
| R6 | 安全/运维 | ReDoS 排除+回归锁定、jieba 日志噪音修复、提示词注入风险披露、架构文档 + non-goals |
| R7/R20 | 技术写作/审计 | 文档示例全部有测试镜像；基准时效性守卫（results.md 必须可逐字节重生成）；零 TODO |
| R8 | 性能工程师 | 扩展性线性实证（4x 数据 3.6-3.9x 耗时）+ 回归测试；预编译热路径正则 |
| R9 | 老板 | **chonkie 1.7.0 安装级证据**：九类切分器无父子块、Chunk 无父链接字段（附复现命令） |
| R10 | 数据工程师/UX | state 文件 schema 版本化（向后兼容）+ 可操作错误信息契约测试 |
| R11 | 语言数据工程师 | 全角句点'．'切分、Unicode 空白（\u3000/NBSP）全面感知——真实语料驱动的修复 |
| R12 | 数据科学家 | **precision@k + union（切分 gold）宽严双口径**；严格判定改为"完整包含"语义（修复跨块 gold 误判） |
| R13–R19 | 研究/写作/运维 | 三层评测方法论定位、繁体专项测试、报告逐字节确定性（含回归测试）、英文 quickstart、FAQ、worked example |
| R21–R23 | 发布工程 | 版本收敛 0.1.0rc3、CHANGELOG 整理、README 全文审读刷新 |

意外事故如实记录：一次 `git add -A` 误扫入你放在仓库目录的 baidu/bing 解析文件——已 soft-reset 修复（文件保留在磁盘），并用 `.git/info/exclude` 防止再犯（WORKLOG 有专节）。

## 三、核心数字（全部来自真实运行，脚本在仓库内可复现）

| 声明 | 数字 | 出处 |
|---|---|---|
| 真实语料吞吐 | 24.6 MB/s（90.6 万字紅樓夢，recursive） | `tools/dogfood.py` → docs/dogfood.md |
| 标点卫生违例 | 0 / 2350 块（修复前 4） | 同上 |
| recall@1（75 问基准） | recursive **0.96** vs fixed-window 0.907 | `python -m worddael.eval.report` |
| 小-大检索增益 | parent-child 0.987 vs 平铺 0.96（k=1） | benchmarks/results.md |
| 对 LangChain 默认切分 | 答案腰斩、50% 块从句中开始（worddael：0） | `tools/status_quo_experiment.py` |
| 测试 | 210 passed，ruff 0 | CI 可复验 |

## 四、诚实的未完成项（阻塞原因）

1. **真实 LLM 判分 / 百篇级 QA 生成**——机器环境无 API key；管线全部就绪且有成本熔断（docs/eval_guide.md 第 3 节）
2. **真实 embedding 模型下的语义切分数字**——需要装模型或 API；toy 向量数字已如实标注
3. **GitHub 仓库 / PyPI / HF**——需要你的账号；README 徽章 URL 仍是占位符
4. 基准仍偏小（25 篇/75 问）：precision@k 区分力要等语料扩展；跨项目比较声明仍被纪律性禁止

## 五、你醒来后的行动清单（按序）

1. `git log --oneline` 过一眼（45 提交）+ `pytest` 复核
2. **建 GitHub 仓库 → push → 替换 README 徽章占位 URL**
3. **提供 LLM API key** → 先 `--strategy recursive --cost-cap 1.0` 小规模验证，再跑百篇级 QA 生成 + 真实判分
4. 按 docs/launch_checklist.md 走发布（素材已备齐：status_quo 坏例子、dogfood 真实数字、worked example 读法）
5. 可选：给 chonkie 提中文 chunker issue（话术思路在清单里）

## 六、面试叙事（已可支撑，一句版）

"我发现中文 RAG 的分块环节没有标准件：用可复现实验证明默认方案的切分缺陷；做了一个零依赖、偏移量可证（fuzz）、边界可解释（全网无先例）、标点卫生有保证的库；在 90 万字真实小说上实测，并用严格/宽松双口径的评测协议量化了每种切分策略的取舍。"

---

# 历史报告存档

## 第一晚（09-26 凌晨，iter0–iter5 + 打磨）
103→136 tests，rc1。完成 embedding 层、25 篇基准、LLM 判分引擎、竞品取证、compare 子命令。当时记录的诚实缺口（k=3 趋同、toy 语义数字、无 key 无 push）至今仍未关闭，处置见上表。

## 第二阶段（09-26 晚，iter6–iter8）
125→136 tests，rc2。父子块（k=1 0.987 vs 0.960）、VectorRetriever+校准+windows CI、qa_gen 脚手架（冒烟 5 篇→10 对全有效）。
