"""Validate the frozen ecommerce-v2 oracle before running Experiment 1."""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List

from langchain_neo4j import Neo4jGraph

from app.core.config import settings
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.text2cypher.query_context import (
    classify_text2cypher_question,
)
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.text2cypher.schema_context import (
    relevant_schema_capabilities,
)
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.text2cypher.validation.validators import (
    validate_cypher_query_syntax,
    validate_cypher_query_with_schema,
    validate_no_writes_in_cypher_query,
)
from app.lg_agent.kg_sub_graph.agentic_rag_agents.retrievers.cypher_examples.vector_store.factory import (
    _seed_examples,
)


TEST_DIR = Path(__file__).parent
DATASET_PATH = TEST_DIR / "data" / "text2cypher_experiment1_v2.json"
SOURCE_PATH = TEST_DIR / "data" / "text2cypher_benchmark_v1.json"
OUTPUT_PATH = TEST_DIR / "results" / "text2cypher_experiment1_v2_quality.json"


def _normalize(value: str) -> str:
    return re.sub(r"[^\w]+", "", value, flags=re.UNICODE).lower()


def _graph() -> Neo4jGraph:
    return Neo4jGraph(
        url=settings.NEO4J_URL,
        username=settings.NEO4J_USERNAME,
        password=settings.NEO4J_PASSWORD,
        database=settings.NEO4J_DATABASE,
    )


def main() -> int:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    cases = dataset.get("generation_cases", [])
    source_cases = source.get("generation_cases", [])
    errors: List[str] = []
    records: List[Dict[str, Any]] = []

    if dataset.get("version") != "2.0.0":
        errors.append("dataset version must be 2.0.0")
    if dataset.get("schema_version") != settings.CYPHER_SCHEMA_VERSION:
        errors.append("dataset and runtime Schema versions differ")
    if len(cases) != 40:
        errors.append(f"expected 40 cases, found {len(cases)}")
    if [(row["id"], row["question"]) for row in cases] != [
        (row["id"], row["question"]) for row in source_cases
    ]:
        errors.append("v2 must retain the exact v1 question IDs and wording")

    seed_questions = [_normalize(row["question"]) for row in _seed_examples()]
    for case in cases:
        question = case["question"]
        normalized = _normalize(question)
        if normalized in seed_questions:
            errors.append(f"{case['id']}: exact Few-shot question leakage")
        closest = max(
            SequenceMatcher(None, normalized, seed).ratio()
            for seed in seed_questions
        )
        if closest >= 0.86:
            errors.append(f"{case['id']}: near Few-shot leakage ratio={closest:.3f}")
        context = classify_text2cypher_question(question)
        if context.domain != case["domain"]:
            errors.append(
                f"{case['id']}: classified domain {context.domain}, expected {case['domain']}"
            )
        if context.query_type != case["query_type"]:
            errors.append(
                f"{case['id']}: classified type {context.query_type}, expected {case['query_type']}"
            )
        labels = set(re.findall(r":\s*([A-Za-z_][A-Za-z0-9_]*)", case["reference_cypher"]))
        if labels & {"Product", "Review"}:
            errors.append(f"{case['id']}: legacy base labels remain: {sorted(labels)}")

    graph = _graph()
    try:
        for case in cases:
            statement = case["reference_cypher"]
            context = classify_text2cypher_question(case["question"])
            capabilities = relevant_schema_capabilities(
                graph, case["question"], context.domain
            )
            case_errors = []
            case_errors.extend(validate_no_writes_in_cypher_query(statement))
            case_errors.extend(validate_cypher_query_syntax(graph, statement))
            case_errors.extend(
                validate_cypher_query_with_schema(
                    graph,
                    statement,
                    allowed_labels=capabilities["labels"],
                    allowed_relationships=capabilities["relationships"],
                    allowed_properties=capabilities["properties"],
                )
            )
            rows: List[Dict[str, Any]] = []
            if not case_errors:
                try:
                    rows = graph.query(statement)
                except Exception as exc:
                    case_errors.append(str(exc))
            if case.get("expected_non_empty") and not rows:
                case_errors.append("expected a non-empty oracle result")
            if rows and set(rows[0]) != set(case["expected_columns"]):
                case_errors.append(
                    f"columns {sorted(rows[0])} != {sorted(case['expected_columns'])}"
                )
            if case_errors:
                errors.extend(f"{case['id']}: {error}" for error in case_errors)
            records.append(
                {
                    "id": case["id"],
                    "row_count": len(rows),
                    "errors": case_errors,
                }
            )
    finally:
        graph.close()

    report = {
        "status": "PASS" if not errors else "FAIL",
        "dataset": str(DATASET_PATH),
        "schema_version": dataset.get("schema_version"),
        "cases": len(cases),
        "valid_and_non_empty": sum(
            not row["errors"] and row["row_count"] > 0 for row in records
        ),
        "errors": errors,
        "records": records,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "records"},
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"output={OUTPUT_PATH}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
