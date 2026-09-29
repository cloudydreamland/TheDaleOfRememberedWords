# 红队评审 #2（2026-09-27 凌晨）

> 换角色再打一轮：安全工程师 + 运维/SRE。第一轮见 red_team_1.md。

## 角色一：安全工程师

**S1 正则回溯爆炸（ReDoS）。** `_HEADING_RE = ^(#{1,6})\s+(.*?)\s*#*\s*$` 含惰性量词，markdown 文件可能来自不可信上传。
→ 实测：三条病态输入（5 万个 `#`、20 万字符行、10 层 `#` 前缀）全部 < 10ms——单一惰性量词无嵌套，不构成回溯爆炸。**已加回归测试锁定**（`test_pathological_markdown_lines_no_redos`）。其余正则均为固定字符类，安全。

**S2 提示词注入。** qa_gen/LLMJudge 把文档内容拼进 prompt——恶意文档可操纵生成/判分输出。
→ 定性：这是 LLM 评测方法论的行业级通病，不是本库能单独解决的；但**必须让用户知道**。已加 `docs/eval_guide.md` 安全注意事项：评测对象内容视为不可信，判分结果用于聚合统计而非单条信任。

**S3 供应链。** 核心零依赖（无可投毒的传递依赖面）；jieba/langchain/tiktoken 为可选 extra。urllib 直连的 base_url 由用户显式配置，不存在隐式外呼。

**S4 无 eval/exec、无动态导入用户输入、路径仅用户自己传给本地 CLI。** 无发现。

## 角色二：运维/SRE

**O1 jieba 首次加载往 stderr 打"Building prefix dict"。** 数据管线里污染日志。
→ 已修：`jieba.setLogLevel(60)` 在导入时抑制（retrieval.py）。jieba 全局状态副作用已注释说明。

**O2 内存：`chunk()` 全量物化。** 1GB 文本会 OOM。
→ 定性：当前 API 契约就是"单文档入内存"（README/架构文档已写明）；GB 级流式 API 进 backlog，不伪装成已支持。

**O3 跨平台确定性。** Windows 开发机 + Linux CI：换行符 \r\n 差异会进块文本。
→ 事实核查：`\r` 在句界/卫生逻辑中按空白处理（`_hygiene_pass` 的 whitespace 跳过、classify_end 的换行扫描都含 \r）；CI 已含 windows-latest 矩阵，18 个 offset fuzz 测试双平台跑。

**O4 输出非确定性。** 已系统性排查：哈希→crc32（R1 修复）、dict 保序（3.7+）、set 只做成员判断不做有序遍历输出、评测不依赖进程内随机。**无发现。**

## 本轮修复清单

| 发现 | 处置 |
|---|---|
| S1 ReDoS 疑点 | 实测排除 + 回归测试锁定 |
| O1 jieba 日志噪音 | 已修（setLogLevel） |
| S2 提示词注入 | eval_guide 增安全注意事项 |
| O2 大文件内存 | 文档化 + backlog（不伪装） |
| O3/O4 确定性 | 排查无发现，CI windows 矩阵保障 |
