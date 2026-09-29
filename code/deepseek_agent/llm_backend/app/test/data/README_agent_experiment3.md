# 实验三：正式运行说明

本实验对应简历项目内容3中的两个指标：

- 系统整体问答准确率 88%；
- 平均响应时间 3.5 秒。

它不是旧 `run_text2cypher_experiments.py` 中名为 `experiment_3` 的 Text2Cypher 测试。旧测试只覆盖 Text2Cypher 工作流成功，不得用于项目内容3的指标。

## 全链路边界

每个计分请求必须经过与 `/api/langgraph/query` 相同的生产图：

`FastAPI → Router → Guardrails → Planner → Tool Selection → Predefined_Cypher/Text2Cypher/GraphRAG → Summarize → Final Answer`

实验专用端点只增加结构化 trace 和耗时信息，不替换任何生产节点。端点默认关闭，并要求 token。

## 数据集与计分

数据集为 `agent_experiment3_v1.json`，共50题：

- 预定义结构化查询：8题；
- 动态结构化查询：12题；
- GraphRAG 非结构化查询：15题；
- 结构化与非结构化混合查询：10题；
- 综合多要求查询：5题。

每题独立运行3轮，共150个计分请求。回答评分不使用 LLM judge，而是检查冻结的必答事实：结构化事实来自只读 Neo4j 参考 Cypher，非结构化事实来自三份冻结源文档。必须同时满足正确路由、正确工具集合、工作流完成以及全部必答事实，才记为一次正确回答。

主口径为请求级正确率，至少132/150才达到88%；同时报告按题三轮多数通过口径，至少44/50。平均响应时间为完成三次不计分预热后的150个端到端请求算术平均值，并同时报告P50、P95、最大值和各分类均值。

## 正式运行前的强制门禁

1. 构建独立 GraphRAG 索引：

   ```bash
   NUMBA_CACHE_DIR=/tmp/codex-numba-cache \
   MPLCONFIGDIR=/tmp/codex-mpl-cache \
   conda run -n intelligent-customer-service-agent \
   python app/test/build_experiment3_graphrag_index.py --skip-connectivity-validation
   ```

2. 检查 `documents.parquet` 确实包含 `faq_service.txt`、`user_experience.txt`、`user_reviews.txt`，并执行全部只读参考查询：

   ```bash
   conda run -n intelligent-customer-service-agent \
   python app/test/validate_agent_experiment3_v1.py \
   --neo4j --require-index --freeze
   ```

3. 只有生成状态为 `ready_for_formal_run` 的 `agent_experiment3_preflight_manifest_v1.json` 后，运行器才会放行。它会锁定题集、协议、源文档、Neo4j Schema/参考结果、GraphRAG 全索引树和关键运行时代码的 SHA-256。

4. 启动后端并打开 token 保护的实验端点，然后先执行：

   ```bash
   conda run -n intelligent-customer-service-agent \
   python app/test/run_agent_experiment3_backend.py \
   --base-url http://127.0.0.1:8000 --preflight-only
   ```

5. 去掉 `--preflight-only` 才会正式运行。结果采用逐请求 checkpoint，可在冻结哈希完全一致时恢复；任何哈希漂移都会拒绝续跑。

## 结果文件

- 题集质量：`../results/agent_experiment3_v1_quality.json`；
- 冻结清单：`../results/agent_experiment3_preflight_manifest_v1.json`；
- 逐请求证据：`../results/agent_experiment3_backend_v1_raw.json`；
- 汇总报告：`../results/agent_experiment3_backend_v1_report.md`。

