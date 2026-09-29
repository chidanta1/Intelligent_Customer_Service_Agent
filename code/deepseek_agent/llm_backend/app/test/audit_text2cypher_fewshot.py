"""Audit the live Few-shot seed examples against the current Neo4j graph."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from langchain_neo4j import Neo4jGraph

from app.core.config import settings
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.text2cypher.validation.validators import (
    validate_cypher_query_syntax,
    validate_cypher_query_with_schema,
    validate_no_writes_in_cypher_query,
)
from app.lg_agent.kg_sub_graph.agentic_rag_agents.retrievers.cypher_examples.vector_store.factory import (
    _seed_examples,
)


OUTPUT_PATH = Path(__file__).parent / "results" / "text2cypher_fewshot_seed_audit.json"


def main() -> int:
    graph = Neo4jGraph(
        url=settings.NEO4J_URL,
        username=settings.NEO4J_USERNAME,
        password=settings.NEO4J_PASSWORD,
        database=settings.NEO4J_DATABASE,
    )
    records: List[Dict[str, Any]] = []
    ids = set()
    try:
        for index, example in enumerate(_seed_examples(), 1):
            statement = example["cql"]
            syntax_errors = validate_cypher_query_syntax(graph, statement)
            schema_errors = validate_cypher_query_with_schema(graph, statement)
            write_errors = validate_no_writes_in_cypher_query(statement)
            executed = False
            execution_error = None
            result_count = 0
            metadata_errors = []
            example_id = example.get("id")
            if not example_id or example_id in ids:
                metadata_errors.append("missing or duplicate id")
            ids.add(example_id)
            if example.get("domain") not in {"app_catalog", "legacy_graph"}:
                metadata_errors.append("invalid domain")
            if not example.get("query_type"):
                metadata_errors.append("missing query_type")
            if example.get("schema_version") != settings.CYPHER_SCHEMA_VERSION:
                metadata_errors.append("schema_version mismatch")
            for field in (
                "required_labels",
                "required_relationships",
                "required_properties",
            ):
                if not isinstance(example.get(field), list):
                    metadata_errors.append(f"{field} must be a list")
            if not syntax_errors and not schema_errors and not write_errors:
                try:
                    result_count = len(graph.query(statement))
                    executed = True
                except Exception as exc:
                    execution_error = str(exc)
            records.append(
                {
                    "index": index,
                    "example_id": example_id,
                    "question": example["question"],
                    "domain": example.get("domain"),
                    "query_type": example.get("query_type"),
                    "difficulty": example.get("difficulty"),
                    "schema_version": example.get("schema_version"),
                    "metadata_errors": metadata_errors,
                    "syntax_errors": syntax_errors,
                    "schema_errors": schema_errors,
                    "write_errors": write_errors,
                    "executed": executed,
                    "result_count": result_count,
                    "execution_error": execution_error,
                }
            )
    finally:
        graph.close()

    report = {
        "total": len(records),
        "syntax_valid": sum(not row["syntax_errors"] for row in records),
        "schema_valid": sum(not row["schema_errors"] for row in records),
        "metadata_valid": sum(not row["metadata_errors"] for row in records),
        "fully_valid_and_executed": sum(row["executed"] for row in records),
        "non_empty_results": sum(row["result_count"] > 0 for row in records),
        "records": records,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, ensure_ascii=False, indent=2))
    print(f"output={OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
