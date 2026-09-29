# Text2Cypher Benchmark v1.1

本数据集用于评测“自然语言 → Cypher → 五层验证 → 修正 → 执行”链路，不能作为 Few-shot 示例重新写入向量库。

## 设计原则

- `generation_cases` 是独立测试集，共 40 题；使用题目要求的参考结果投影作为语义标准，不用字符串完全相等判断正确性。候选可以返回额外辅助列，但不能缺少题目要求的信息。
- 难度配比为 Easy 12、Medium 18、Hard 10；覆盖单实体查询、过滤、排序、聚合、时间条件、空结果、多跳和销售分析。
- 数据域覆盖当前真实 Neo4j 中的 8 个业务标签和 8 种业务关系。
- 原始 `Product`、`Review` 的两套属性族已通过 `CatalogProduct/CatalogReview` 与 `GraphProduct/GraphReview` 规范标签隔离；v1测试集保留旧字段参考查询用于兼容性复验。
- `validation_cases` 共 22 题，覆盖语法、写操作、关系方向、LLM 语义和 Schema 五层验证，以及组合错误。
- 所有参考 Cypher 都必须只读、通过 Schema 校验，并能在当前项目 Neo4j 上执行。
- 测试问题不得与 Few-shot 向量库中的问题完全重复，防止数据泄漏。

## 后续指标口径

- 生成准确率：生成 Cypher 必须覆盖参考 Cypher 在同一数据库快照上的必要结果；只有题目明确要求排序或Top-K时才比较顺序。
- 成功执行率：通过完整工作流后能执行且没有 `errors` 的样本数 / 全部生成样本数。
- Few-shot 对照：同一模型、温度、提示词和数据快照，仅切换 `fewshot_k=0` 与 `fewshot_k=3`，每个条件至少重复 3 次并固定随机种子。
- 校验机制测试：按 `expected_layers` 检查对应层是否发现或自动修复问题；LLM 层样本需要人工复核语义标签。
- 40 条生成题使单题权重为 2.5 个百分点：65%、85%、90% 分别对应 26、34、36 条成功，指标不存在四舍五入歧义。

## 质量校验

在 `llm_backend` 目录运行：

```bash
/home/dhc/miniconda3/envs/intelligent-customer-service-agent/bin/python \
  -m app.test.validate_text2cypher_dataset
```

校验包括 JSON 结构、ID/问题去重、Few-shot 泄漏、难度和领域覆盖、只读安全、Schema、一致性、参考查询真实执行及五层验证覆盖。

## v1.1 评分修订

首次实验后发现v1.0把参考查询中仅用于稳定输出的排序和题目未要求的辅助列也当作强制答案，产生了系统性误判。v1.1只修订评分投影和排序标记，40条问题及全部模型原始输出均未改变。旧评分保留在原始实验记录的 `evaluation_v1_strict` 字段中。
