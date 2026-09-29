# 发布清单（Launch Checklist）

> 目标：v0.1.0 正式发布时按序执行。原则：一次发布一个渠道，观察 48 小时反馈再推下一个。

## 前置（代码侧，全部完成后才允许发布）

- [ ] GitHub 仓库已创建并 push（README 徽章 URL 从 `user/worddael` 改为真实 org/name）
- [ ] CI 在 GitHub Actions 上跑绿（Python 3.10–3.13 矩阵）
- [ ] `pip install worddael` 在全新 venv 验证通过（PyPI 发 token 后）
- [ ] benchmarks/results.md 数字与仓库当前版本一致（重新跑一遍 `python -m worddael.eval.report`）
- [ ] LICENSE / CONTRIBUTING / CHANGELOG 齐全；git tag v0.1.0

## PyPI

- [ ] 注册 PyPI 账号，配置 trusted publisher
- [ ] `python -m build` + `twine check`，先发 0.1.0（不要发 0.0.x 占位）

## 首发内容（一次写好，多渠道复用）

- [ ] 主题：《我们测了 5 种中文分块策略：RAG 命中率差 27%》——以 `benchmarks/results.md` + `docs/status_quo.md` 真实数据为主体，结尾自然引出 worddael
- [ ] 30 秒 GIF：终端里 `worddael doc.md --stats` 的偏移量+句尾统计演示
- [ ] 英文 README 段落（已有）+ 中文主 README（已有）

## 渠道顺序

1. [ ] 知乎（中文首发，RAG/LLM 话题）+ 掘金同步
2. [ ] V2EX / 即刻 / 朋友圈技术群
3. [ ] Reddit r/Rag + r/LocalLLaMA（英文版）
4. [ ] Hacker News Show HN（英文标题突出 "Chinese text chunking with exact character offsets"）
5. [ ] 提 PR 进入 awesome-rag / awesome-chinese-nlp 等清单

## 上游贡献（发布周内）

- [ ] 给 chonkie 提 issue：询问中文 chunker 计划，附 status_quo 数据（礼貌、贡献姿态——若官方有计划则转向做其生态的中文适配层）
- [ ] LangChain 中文文档社区：提交一条 RecursiveCharacterTextSplitter 中文分隔符的文档改进 PR（引用 status_quo 实验）

## 发布后 48 小时

- [ ] 回复所有 issue（响应速度是早期用户转化为 star 传播者的关键）
- [ ] 记录渠道引流数据（哪个渠道带来 star/clone），决定第二篇文章方向
- [ ] 第二篇选题储备：《LLM 判分 vs BM25 召回：两种分块评测方法的一致性分析》（用 docs/eval_guide.md 的实跑数据）
