"""Regression tests for the live Text2Cypher workflow."""

import asyncio
import unittest
from typing import Any, List

from langchain_core.runnables import Runnable, RunnableLambda

from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.text2cypher.validation.models import (
    Property,
    ValidateCypherOutput,
)
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.text2cypher.generation.node import (
    create_text2cypher_generation_node,
)
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.text2cypher.correction.node import (
    create_text2cypher_correction_node,
)
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.text2cypher.query_context import (
    classify_text2cypher_question,
)
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.text2cypher.schema_context import (
    retrieve_relevant_schema_for_prompt,
)
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.text2cypher.validation.validators import (
    correct_cypher_query_relationship_direction,
    extract_string_equality_filters,
    validate_cypher_query_with_llm,
    validate_cypher_query_with_schema,
    validate_no_writes_in_cypher_query,
)
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.utils.utils import (
    retrieve_and_parse_schema_from_graph_for_prompts,
)
from app.lg_agent.kg_sub_graph.agentic_rag_agents.retrievers.cypher_examples.base import (
    BaseCypherExampleRetriever,
)
from app.lg_agent.kg_sub_graph.agentic_rag_agents.retrievers.cypher_examples.vector_store.factory import (
    _schema_capabilities,
    _seed_examples,
)
from app.core.config import settings
from app.lg_agent.kg_sub_graph.agentic_rag_agents.workflows.single_agent import (
    create_text2cypher_agent,
)


class EmptyExampleRetriever(BaseCypherExampleRetriever):
    def get_examples(self, query: str, k: int = 5, *args, **kwargs) -> str:
        return ""


class StructuredFakeChatModel(Runnable):
    def __init__(
        self,
        responses: List[str],
        validation_outputs: List[ValidateCypherOutput] | None = None,
    ) -> None:
        self.responses = responses
        self.response_index = 0
        self.validation_outputs = validation_outputs or [
            ValidateCypherOutput(errors=[], filters=[])
        ]
        self.validation_index = 0

    def _next_result(self) -> str:
        response = self.responses[self.response_index % len(self.responses)]
        self.response_index += 1
        return response

    def invoke(self, input, config=None, **kwargs) -> str:
        return self._next_result()

    async def ainvoke(self, input, config=None, **kwargs) -> str:
        return self._next_result()

    def with_structured_output(self, schema, **kwargs):
        async def valid_result(_: Any) -> ValidateCypherOutput:
            index = min(self.validation_index, len(self.validation_outputs) - 1)
            result = self.validation_outputs[index]
            self.validation_index += 1
            return result

        return RunnableLambda(valid_result)


class FakeGraph:
    get_schema = """
Node properties:
- Product: ProductName, UnitPrice
Relationship properties:
The relationships:
"""
    structured_schema = {
        "node_props": {
            "Product": [
                {"property": "ProductName", "type": "STRING"},
                {"property": "UnitPrice", "type": "FLOAT"},
            ]
        },
        "rel_props": {},
        "relationships": [],
    }

    def __init__(self) -> None:
        self.executed = []

    def query(self, statement, params=None):
        if statement.startswith("EXPLAIN "):
            if "NOT CYPHER" in statement or "CREATE " in statement:
                raise RuntimeError("invalid Cypher")
            return []
        self.executed.append(statement)
        return [{"product": "智能音箱"}]


class Text2CypherWorkflowTests(unittest.TestCase):
    def run_workflow(self, workflow, state):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(workflow.ainvoke(state))
        finally:
            loop.close()

    def test_invalid_generation_is_corrected_then_revalidated(self):
        graph = FakeGraph()
        llm = StructuredFakeChatModel(responses=["NOT CYPHER"])

        async def correct_once(state):
            return {
                "statement": "MATCH (p:Product) RETURN p.ProductName AS product",
                "errors": [],
                "mapping_errors": [],
                "steps": ["correct_cypher"],
            }

        workflow = create_text2cypher_agent(
            llm=llm,
            graph=graph,
            cypher_example_retriever=EmptyExampleRetriever(),
            max_retries=3,
            correction_node=correct_once,
        )

        result = self.run_workflow(workflow, {"task": "查询产品"})

        cypher = result["cyphers"][0]
        self.assertEqual(cypher["attempts"], 2)
        self.assertEqual(cypher["retries"], 1)
        self.assertEqual(len(graph.executed), 1)
        self.assertIn("correct_cypher", cypher["steps"])
        self.assertEqual(
            cypher["validation_layers"],
            [
                "syntax",
                "write_operation",
                "relationship_direction",
                "llm",
                "schema",
            ],
        )
        self.assertEqual(len(cypher["validation_trace"]), 2)
        self.assertEqual(cypher["validation_trace"][0]["attempt"], 1)
        self.assertEqual(cypher["validation_trace"][0]["next_action"], "correct_cypher")
        self.assertEqual(cypher["validation_trace"][1]["attempt"], 2)
        self.assertEqual(cypher["validation_trace"][1]["next_action"], "execute_cypher")
        self.assertEqual(
            [item["layer"] for item in cypher["validation_trace"][0]["layers"]],
            [
                "syntax",
                "write_operation",
                "relationship_direction",
                "llm",
                "schema",
            ],
        )

    def test_retry_limit_stops_before_invalid_query_execution(self):
        graph = FakeGraph()
        llm = StructuredFakeChatModel(responses=["NOT CYPHER"])

        async def keep_invalid(state):
            return {
                "statement": "NOT CYPHER",
                "errors": [],
                "mapping_errors": [],
                "steps": ["correct_cypher"],
            }

        workflow = create_text2cypher_agent(
            llm=llm,
            graph=graph,
            cypher_example_retriever=EmptyExampleRetriever(),
            max_retries=3,
            correction_node=keep_invalid,
        )

        result = self.run_workflow(workflow, {"task": "查询产品"})

        cypher = result["cyphers"][0]
        self.assertEqual(cypher["attempts"], 4)
        self.assertEqual(cypher["retries"], 3)
        self.assertTrue(cypher["errors"])
        self.assertEqual(graph.executed, [])
        self.assertEqual(len(cypher["validation_trace"]), 4)
        self.assertEqual(
            [attempt["next_action"] for attempt in cypher["validation_trace"]],
            [
                "correct_cypher",
                "correct_cypher",
                "correct_cypher",
                "finalize_failure",
            ],
        )

    def test_retry_limit_cannot_exceed_three(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 3"):
            create_text2cypher_agent(
                llm=StructuredFakeChatModel(responses=["NOT CYPHER"]),
                graph=FakeGraph(),
                cypher_example_retriever=EmptyExampleRetriever(),
                max_retries=4,
            )

    def test_llm_operational_failure_fails_closed_without_correction(self):
        class ValidationFailureChatModel(StructuredFakeChatModel):
            def with_structured_output(self, schema, **kwargs):
                async def fail_validation(_: Any):
                    raise RuntimeError("provider timeout")

                return RunnableLambda(fail_validation)

        async def correction_must_not_run(state):
            raise AssertionError("operational failure must not trigger correction")

        graph = FakeGraph()
        workflow = create_text2cypher_agent(
            llm=ValidationFailureChatModel(
                responses=["MATCH (p:Product) RETURN p.ProductName AS product"]
            ),
            graph=graph,
            cypher_example_retriever=EmptyExampleRetriever(),
            max_retries=3,
            correction_node=correction_must_not_run,
        )

        result = self.run_workflow(workflow, {"task": "查询产品"})

        cypher = result["cyphers"][0]
        self.assertEqual(cypher["attempts"], 1)
        self.assertEqual(cypher["retries"], 0)
        self.assertEqual(graph.executed, [])
        self.assertIn("provider timeout", cypher["errors"][0])
        attempt = cypher["validation_trace"][0]
        self.assertTrue(attempt["operational_failure"])
        self.assertEqual(attempt["next_action"], "finalize_failure")

    def test_write_query_is_never_executed_after_retry_limit(self):
        graph = FakeGraph()

        async def keep_write_query(state):
            return {
                "statement": "MATCH (p:Product) DELETE p",
                "errors": [],
                "mapping_errors": [],
                "steps": ["correct_cypher"],
            }

        workflow = create_text2cypher_agent(
            llm=StructuredFakeChatModel(
                responses=["MATCH (p:Product) DELETE p"]
            ),
            graph=graph,
            cypher_example_retriever=EmptyExampleRetriever(),
            max_retries=3,
            correction_node=keep_write_query,
        )

        result = self.run_workflow(workflow, {"task": "删除产品"})

        cypher = result["cyphers"][0]
        self.assertEqual(cypher["retries"], 3)
        self.assertEqual(graph.executed, [])
        self.assertEqual(len(cypher["validation_trace"]), 4)
        for attempt in cypher["validation_trace"]:
            write_layer = attempt["layers"][1]
            self.assertEqual(write_layer["layer"], "write_operation")
            self.assertEqual(write_layer["status"], "failed")

    def test_production_correction_history_is_retained(self):
        graph = FakeGraph()
        node = create_text2cypher_correction_node(
            llm=StructuredFakeChatModel(
                responses=["MATCH (p:Product) RETURN p.ProductName AS product"]
            ),
            graph=graph,
        )
        state = {
                "task": "查询产品",
                "statement": "NOT CYPHER",
                "errors": ["Invalid input"],
                "mapping_errors": [],
                "retries": 0,
                "schema_version": settings.CYPHER_SCHEMA_VERSION,
        }
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(node(state))
        finally:
            loop.close()

        self.assertEqual(len(result["correction_history"]), 1)
        self.assertEqual(result["correction_history"][0]["retry"], 1)
        self.assertEqual(
            result["correction_history"][0]["statement_before"], "NOT CYPHER"
        )
        self.assertIn(
            "MATCH (p:Product)", result["correction_history"][0]["statement_after"]
        )

    def test_relationship_direction_auto_correction_is_traced(self):
        class GraphWithRelationship(FakeGraph):
            structured_schema = {
                **FakeGraph.structured_schema,
                "node_props": {
                    **FakeGraph.structured_schema["node_props"],
                    "Category": [
                        {"property": "CategoryName", "type": "STRING"}
                    ],
                },
                "relationships": [
                    {"start": "Product", "type": "BELONGS_TO", "end": "Category"}
                ],
            }

        graph = GraphWithRelationship()
        workflow = create_text2cypher_agent(
            llm=StructuredFakeChatModel(
                responses=[
                    "MATCH (p:Product)<-[:BELONGS_TO]-(c:Category) "
                    "RETURN p.ProductName"
                ]
            ),
            graph=graph,
            cypher_example_retriever=EmptyExampleRetriever(),
            max_retries=3,
        )

        result = self.run_workflow(workflow, {"task": "查询产品及类别"})

        cypher = result["cyphers"][0]
        direction_layer = cypher["validation_trace"][0]["layers"][2]
        self.assertEqual(direction_layer["layer"], "relationship_direction")
        self.assertEqual(direction_layer["status"], "corrected")
        self.assertEqual(cypher["retries"], 0)
        self.assertIn("(p:Product)-[:BELONGS_TO]->(c:Category)", cypher["statement"])
        self.assertEqual(len(graph.executed), 1)

    def test_unknown_relationship_is_owned_by_schema_not_direction(self):
        graph = FakeGraph()
        workflow = create_text2cypher_agent(
            llm=StructuredFakeChatModel(
                responses=[
                    "MATCH (p:Product)-[:UNKNOWN_REL]->(q:Product) "
                    "RETURN p.ProductName"
                ]
            ),
            graph=graph,
            cypher_example_retriever=EmptyExampleRetriever(),
            max_retries=0,
        )

        result = self.run_workflow(workflow, {"task": "查询有关联的产品"})

        cypher = result["cyphers"][0]
        layers = {
            item["layer"]: item for item in cypher["validation_trace"][0]["layers"]
        }
        self.assertEqual(layers["relationship_direction"]["status"], "passed")
        self.assertEqual(layers["schema"]["status"], "failed")
        self.assertEqual(graph.executed, [])

    def test_missing_value_mapping_triggers_correction(self):
        class MissingValueGraph(FakeGraph):
            def query(self, statement, params=None):
                if "toLower(n." in statement:
                    return []
                return super().query(statement, params)

        graph = MissingValueGraph()
        llm = StructuredFakeChatModel(
            responses=[
                "MATCH (p:Product) WHERE p.ProductName = '不存在' RETURN p.ProductName"
            ],
            validation_outputs=[
                ValidateCypherOutput(
                    errors=[],
                    filters=[
                        Property(
                            node_label="Product",
                            property_key="ProductName",
                            property_value="不存在",
                        )
                    ],
                ),
                ValidateCypherOutput(errors=[], filters=[]),
            ],
        )

        async def remove_invalid_filter(state):
            return {
                "statement": "MATCH (p:Product) RETURN p.ProductName",
                "errors": [],
                "mapping_errors": [],
                "steps": ["correct_cypher"],
            }

        workflow = create_text2cypher_agent(
            llm=llm,
            graph=graph,
            cypher_example_retriever=EmptyExampleRetriever(),
            max_retries=3,
            correction_node=remove_invalid_filter,
        )

        result = self.run_workflow(workflow, {"task": "查询不存在的产品"})

        cypher = result["cyphers"][0]
        self.assertEqual(cypher["attempts"], 2)
        self.assertEqual(cypher["retries"], 1)
        self.assertEqual(len(graph.executed), 1)

    def test_generation_node_k_zero_skips_vector_retrieval(self):
        class RetrieverThatMustNotRun(BaseCypherExampleRetriever):
            def get_examples(self, query: str, k: int = 5, *args, **kwargs) -> str:
                raise AssertionError("k=0 must not call the vector retriever")

        node = create_text2cypher_generation_node(
            llm=StructuredFakeChatModel(
                responses=["MATCH (p:Product) RETURN p.ProductName"]
            ),
            graph=FakeGraph(),
            cypher_example_retriever=RetrieverThatMustNotRun(),
            fewshot_k=0,
        )

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(node({"task": "查询产品"}))
        finally:
            loop.close()

        self.assertEqual(result["fewshot_examples"], "")
        self.assertEqual(result["fewshot_k"], 0)

    def test_generation_passes_domain_query_type_and_schema_version(self):
        class RecordingRetriever(BaseCypherExampleRetriever):
            calls: list = []

            def get_examples(self, query: str, k: int = 5, *args, **kwargs) -> str:
                self.calls.append({"query": query, "k": k, **kwargs})
                return "Question: 示例\nCypher:\nMATCH (p:CatalogProduct) RETURN p.name"

        retriever = RecordingRetriever()
        node = create_text2cypher_generation_node(
            llm=StructuredFakeChatModel(
                responses=["MATCH (p:CatalogProduct) RETURN p.name"]
            ),
            graph=FakeGraph(),
            cypher_example_retriever=retriever,
            fewshot_k=3,
        )
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                node({"task": "列出小米品牌的应用商品"})
            )
        finally:
            loop.close()

        self.assertEqual(retriever.calls[0]["domain"], "app_catalog")
        self.assertEqual(retriever.calls[0]["query_type"], "filter")
        self.assertEqual(
            retriever.calls[0]["schema_version"], settings.CYPHER_SCHEMA_VERSION
        )
        self.assertEqual(retriever.calls[0]["allowed_labels"], {"CatalogProduct"})
        self.assertEqual(result["retrieval_domain"], "app_catalog")


class Text2CypherValidationTests(unittest.TestCase):
    def run_async(self, awaitable):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(awaitable)
        finally:
            loop.close()

    def test_string_equality_filters_are_extracted_from_cypher(self):
        filters = extract_string_equality_filters(
            FakeGraph(),
            "MATCH (p:Product {ProductName: '智能音箱'}) "
            "WHERE p.ProductName = '智能音箱' RETURN p.ProductName",
        )

        self.assertEqual(
            filters,
            [
                Property(
                    node_label="Product",
                    property_key="ProductName",
                    property_value="智能音箱",
                )
            ],
        )

    def test_noncanonical_llm_filter_is_warning_not_blocking_error(self):
        chain = RunnableLambda(
            lambda _: {"errors": [], "filters": ["p.ProductName = '智能音箱'"]}
        )

        result = self.run_async(
            validate_cypher_query_with_llm(
                chain,
                "查询智能音箱",
                FakeGraph(),
                "MATCH (p:Product) WHERE p.ProductName = '智能音箱' "
                "RETURN p.ProductName",
            )
        )

        self.assertEqual(result["errors"], [])
        self.assertEqual(result["mapping_errors"], [])
        self.assertTrue(result["warnings"])

    def test_mapping_uses_cypher_even_when_llm_returns_no_filters(self):
        class MissingValueGraph(FakeGraph):
            def query(self, statement, params=None):
                if "toLower(n." in statement:
                    return []
                return super().query(statement, params)

        result = self.run_async(
            validate_cypher_query_with_llm(
                RunnableLambda(lambda _: {"errors": [], "filters": []}),
                "查询不存在的产品",
                MissingValueGraph(),
                "MATCH (p:Product) WHERE p.ProductName = '不存在' "
                "RETURN p.ProductName",
            )
        )

        self.assertEqual(result["errors"], [])
        self.assertEqual(len(result["mapping_errors"]), 1)
        self.assertIn("ProductName", result["mapping_errors"][0])

    def test_null_predicate_from_llm_is_ignored_without_losing_semantic_errors(self):
        result = self.run_async(
            validate_cypher_query_with_llm(
                RunnableLambda(
                    lambda _: {
                        "errors": "The query returns price instead of product name",
                        "filters": ["p.UnitPrice IS NOT NULL"],
                    }
                ),
                "查询产品名称",
                FakeGraph(),
                "MATCH (p:Product) WHERE p.UnitPrice IS NOT NULL "
                "RETURN p.UnitPrice",
            )
        )

        self.assertEqual(
            result["errors"],
            ["The query returns price instead of product name"],
        )
        self.assertEqual(result["mapping_errors"], [])
        self.assertTrue(result["warnings"])

    def test_vector_store_label_is_removed_from_prompt_schema(self):
        class GraphWithVectorMetadata(FakeGraph):
            get_schema = """Node properties:
Product [ProductName: STRING]
CypherQuery [question: STRING, questionEmbedding: LIST]
Relationship properties:
"""

        schema = retrieve_and_parse_schema_from_graph_for_prompts(
            GraphWithVectorMetadata()
        )

        self.assertIn("Product", schema)
        self.assertNotIn("CypherQuery", schema)

    def test_schema_validation_checks_labels_relationships_and_properties(self):
        graph = FakeGraph()
        statement = "MATCH (p:Missing)-[:UNKNOWN]->(q:Product) RETURN q.BadProperty"

        errors = validate_cypher_query_with_schema(graph, statement)

        self.assertTrue(any("Missing" in error for error in errors))
        self.assertTrue(any("UNKNOWN" in error for error in errors))
        self.assertTrue(any("BadProperty" in error for error in errors))

    def test_write_validator_matches_clauses_not_property_substrings(self):
        self.assertTrue(validate_no_writes_in_cypher_query("MATCH (n) DELETE n"))
        self.assertTrue(validate_no_writes_in_cypher_query("DROP INDEX product_index"))
        self.assertTrue(
            validate_no_writes_in_cypher_query(
                "CALL apoc.create.node(['Product'], {}) YIELD node RETURN node"
            )
        )
        self.assertFalse(
            validate_no_writes_in_cypher_query(
                "MATCH (p:Product) RETURN p.AssetCode"
            )
        )
        self.assertFalse(
            validate_no_writes_in_cypher_query(
                "RETURN 'DELETE and CREATE are words' AS explanation"
            )
        )
        self.assertFalse(
            validate_no_writes_in_cypher_query(
                "MATCH (n) // DELETE is only a comment\nRETURN n"
            )
        )

    def test_relationship_direction_is_corrected_from_schema(self):
        class GraphWithRelationship(FakeGraph):
            structured_schema = {
                **FakeGraph.structured_schema,
                "node_props": {
                    **FakeGraph.structured_schema["node_props"],
                    "Category": [
                        {"property": "CategoryName", "type": "STRING"}
                    ],
                },
                "relationships": [
                    {"start": "Product", "type": "BELONGS_TO", "end": "Category"}
                ],
            }

        statement = (
            "MATCH (p:Product)<-[:BELONGS_TO]-(c:Category) "
            "RETURN p.ProductName"
        )

        corrected = correct_cypher_query_relationship_direction(
            GraphWithRelationship(), statement
        )

        self.assertIn("(p:Product)-[:BELONGS_TO]->(c:Category)", corrected)

    def test_curated_examples_are_available_for_vector_ingestion(self):
        examples = _seed_examples()
        self.assertEqual(len(examples), 45)
        self.assertTrue(all(item["question"] and item["cql"] for item in examples))
        self.assertGreaterEqual(
            sum(item["difficulty"] == "hard" for item in examples), 20
        )
        self.assertTrue(
            all(item["schema_version"] == settings.CYPHER_SCHEMA_VERSION for item in examples)
        )

    def test_domain_classifier_separates_catalog_and_relationship_graph(self):
        self.assertEqual(
            classify_text2cypher_question("应用商品的平均价格是多少").domain,
            "app_catalog",
        )
        self.assertEqual(
            classify_text2cypher_question("按名称列出全部商品类别").domain,
            "legacy_graph",
        )
        self.assertEqual(
            classify_text2cypher_question("按订单明细计算销售额").query_type,
            "multi_hop_aggregation",
        )

    def test_question_aware_schema_excludes_the_other_domain(self):
        class CanonicalGraph(FakeGraph):
            structured_schema = {
                "node_props": {
                    "CatalogProduct": [
                        {"property": "name", "type": "STRING"},
                        {"property": "price", "type": "FLOAT"},
                    ],
                    "GraphProduct": [
                        {"property": "name", "type": "STRING"},
                    ],
                    "Order": [
                        {"property": "orderDate", "type": "LOCAL_DATE_TIME"},
                    ],
                },
                "rel_props": {
                    "CONTAINS": [
                        {"property": "unitPrice", "type": "FLOAT"},
                        {"property": "quantity", "type": "INTEGER"},
                        {"property": "discount", "type": "FLOAT"},
                    ]
                },
                "relationships": [
                    {"start": "Order", "type": "CONTAINS", "end": "GraphProduct"}
                ],
            }

        graph = CanonicalGraph()
        app_schema = retrieve_relevant_schema_for_prompt(
            graph, "应用商品价格排行", "app_catalog", settings.CYPHER_SCHEMA_VERSION
        )
        graph_schema = retrieve_relevant_schema_for_prompt(
            graph, "按订单明细汇总图谱商品销售额", "legacy_graph", settings.CYPHER_SCHEMA_VERSION
        )

        self.assertIn("CatalogProduct", app_schema)
        self.assertNotIn("GraphProduct", app_schema)
        self.assertNotIn("Order", app_schema)
        self.assertIn("GraphProduct", graph_schema)
        self.assertIn("Order", graph_schema)
        self.assertIn("CONTAINS [unitPrice: FLOAT", graph_schema)

    def test_schema_capabilities_are_qualified(self):
        capabilities = _schema_capabilities(FakeGraph())
        self.assertIn("Product", capabilities["labels"])
        self.assertIn("Product.UnitPrice", capabilities["properties"])

    def test_schema_validation_can_enforce_a_canonical_slice(self):
        graph = FakeGraph()
        errors = validate_cypher_query_with_schema(
            graph,
            "MATCH (p:Product) RETURN p.ProductName",
            allowed_labels={"CatalogProduct"},
            allowed_relationships=set(),
            allowed_properties={"CatalogProduct.name"},
        )
        self.assertTrue(any("Product" in error for error in errors))

    def test_text2cypher_temperature_is_deterministic(self):
        self.assertEqual(settings.TEXT2CYPHER_GENERATION_TEMPERATURE, 0.0)

    def test_catalog_price_phrasing_selects_catalog_schema(self):
        self.assertEqual(
            classify_text2cypher_question("鹿客智能锁多少钱？").domain,
            "app_catalog",
        )
        self.assertEqual(
            classify_text2cypher_question("目录中一共有几款智能音箱？").domain,
            "app_catalog",
        )


if __name__ == "__main__":
    unittest.main()
