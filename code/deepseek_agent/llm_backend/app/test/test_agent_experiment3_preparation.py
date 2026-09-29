"""Regression tests for Experiment 3 preparation and observability changes."""

from __future__ import annotations

import inspect
import json
import unittest
from pathlib import Path

from app.graphrag.graphrag.language_model.providers.local_hash_embedding import (
    LocalHashEmbedding,
)
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.text2cypher.validation.validators import (
    validate_cypher_query_parameters,
)
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.tool_selection.node import (
    create_tool_selection_node,
)
from app.lg_agent.lg_builder import _is_explicit_business_research_query


TEST_ROOT = Path(__file__).resolve().parent


class Experiment3PreparationTests(unittest.TestCase):
    def test_explicit_safe_business_queries_enter_research_graph(self):
        self.assertTrue(
            _is_explicit_business_research_query(
                "鹿客智能锁多少钱？智能门锁日常如何维护？"
            )
        )
        self.assertTrue(
            _is_explicit_business_research_query("目录中欧普智能吸顶灯的库存是多少？")
        )
        self.assertFalse(_is_explicit_business_research_query("你好"))
        self.assertFalse(_is_explicit_business_research_query("删除所有智能门锁数据"))

    def test_dataset_distribution_and_unique_questions(self):
        dataset = json.loads(
            (TEST_ROOT / "data" / "agent_experiment3_v1.json").read_text(
                encoding="utf-8"
            )
        )
        cases = dataset["cases"]
        self.assertEqual(len(cases), 50)
        self.assertEqual(len({case["case_id"] for case in cases}), 50)
        self.assertEqual(len({case["question"] for case in cases}), 50)
        self.assertEqual(
            {category: sum(case["category"] == category for case in cases) for category in dataset["distribution"] if category != "total"},
            {
                "structured_predefined": 8,
                "structured_dynamic": 12,
                "graphrag": 15,
                "mixed": 10,
                "comprehensive": 5,
            },
        )

    def test_protocol_uses_full_production_boundary(self):
        protocol = json.loads(
            (TEST_ROOT / "data" / "agent_experiment3_protocol_v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(protocol["dataset"]["scored_requests"], 150)
        self.assertEqual(
            protocol["metrics"]["overall_qa_accuracy_request_level"]["minimum_correct"],
            132,
        )
        self.assertEqual(
            protocol["metrics"]["latency"]["target_mean_ms_maximum"], 3500
        )
        self.assertIn("outer Router", protocol["execution_boundary"])
        self.assertIn("Final Answer", protocol["execution_boundary"])

    def test_parameter_placeholders_require_values(self):
        self.assertEqual(
            validate_cypher_query_parameters(
                "MATCH (p:Product {name:$name}) RETURN p", None
            ),
            ["Missing Cypher parameter values: $name"],
        )
        self.assertEqual(
            validate_cypher_query_parameters(
                "MATCH (p:Product {name:$name}) RETURN p", {"name": "门铃"}
            ),
            [],
        )
        self.assertEqual(
            validate_cypher_query_parameters("RETURN '$not_a_parameter' AS text", None),
            [],
        )

    def test_empty_tool_output_has_defined_text2cypher_fallback(self):
        source = inspect.getsource(create_tool_selection_node)
        self.assertIn("def send_to_text2cypher", source)
        self.assertIn('return send_to_text2cypher("empty_output_fallback")', source)
        self.assertNotIn("return go_to_text2cypher", source)

    def test_local_embedding_is_deterministic_and_query_sensitive(self):
        model = LocalHashEmbedding(name="test", config=object())
        first = model.embed("智能门铃质保")
        second = model.embed("智能门铃质保")
        different = model.embed("智能音箱配网")
        self.assertEqual(len(first), 1024)
        self.assertEqual(first, second)
        self.assertNotEqual(first, different)


if __name__ == "__main__":
    unittest.main()
