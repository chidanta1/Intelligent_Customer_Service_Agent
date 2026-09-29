"""Quality checks for the held-out Text2Cypher benchmark."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List

from langchain_neo4j import Neo4jGraph

from app.core.config import settings
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.text2cypher.validation.validators import (
    correct_cypher_query_relationship_direction,
    validate_cypher_query_syntax,
    validate_cypher_query_with_schema,
    validate_no_writes_in_cypher_query,
)
from app.lg_agent.kg_sub_graph.agentic_rag_agents.retrievers.cypher_examples.vector_store.factory import (
    _seed_examples,
)


DATASET_PATH = (
    Path(__file__).parent / "data" / "text2cypher_benchmark_v1.json"
)
ID_PATTERN = re.compile(r"^T2C-[GV]\d{3}$")
DIFFICULTY_TARGET = {"easy": 12, "medium": 18, "hard": 10}
REQUIRED_LABELS = {
    "Category",
    "Customer",
    "Employee",
    "Order",
    "Product",
    "Review",
    "Shipper",
    "Supplier",
}
REQUIRED_RELATIONSHIPS = {
    "ABOUT",
    "BELONGS_TO",
    "CONTAINS",
    "PLACED",
    "PROCESSED",
    "SHIPPED_VIA",
    "SUPPLIED_BY",
    "WROTE",
}
REQUIRED_VALIDATION_LAYERS = {
    "syntax",
    "write_operation",
    "relationship_direction",
    "llm",
    "schema",
}
GENERATION_REQUIRED_FIELDS = {
    "id",
    "question",
    "reference_cypher",
    "difficulty",
    "domain",
    "query_type",
    "expected_non_empty",
    "expected_columns",
    "ordered",
    "tags",
}
VALIDATION_REQUIRED_FIELDS = {
    "id",
    "question",
    "candidate_cypher",
    "corrected_reference_cypher",
    "expected_layers",
    "expected_outcome",
    "issue_type",
    "manual_review_required",
}


def _normalize_text(value: str) -> str:
    return re.sub(r"[^\w]+", "", value, flags=re.UNICODE).lower()


def _normalize_cypher(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def _duplicates(values: Iterable[str]) -> List[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def _make_graph() -> Neo4jGraph:
    return Neo4jGraph(
        url=settings.NEO4J_URL,
        username=settings.NEO4J_USERNAME,
        password=settings.NEO4J_PASSWORD,
        database=settings.NEO4J_DATABASE,
    )


def validate_dataset(online: bool = True) -> Dict[str, Any]:
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    generation = data.get("generation_cases", [])
    validation = data.get("validation_cases", [])
    errors: List[str] = []
    warnings: List[str] = []

    if data.get("split") != "test_only":
        errors.append("dataset split must be test_only")
    if data.get("evaluation_policy", {}).get("fewshot_isolation") != "never_ingest_this_dataset":
        errors.append("Few-shot isolation policy is missing")
    if len(generation) != 40:
        errors.append(f"expected 40 generation cases, found {len(generation)}")
    if len(validation) != 22:
        errors.append(f"expected 22 validation cases, found {len(validation)}")
    expected_metric_counts = {
        "65_percent": 26,
        "85_percent": 34,
        "90_percent": 36,
    }
    if (
        data.get("evaluation_policy", {}).get("target_metric_counts_on_40_cases")
        != expected_metric_counts
    ):
        errors.append(
            "target metric counts must map 65%, 85%, and 90% to 26, 34, and 36 cases"
        )

    all_cases = generation + validation
    ids = [case.get("id", "") for case in all_cases]
    questions = [_normalize_text(case.get("question", "")) for case in all_cases]
    if any(not ID_PATTERN.fullmatch(case_id) for case_id in ids):
        errors.append("one or more case IDs do not match T2C-[GV]NNN")
    if _duplicates(ids):
        errors.append(f"duplicate IDs: {_duplicates(ids)}")
    if _duplicates(questions):
        errors.append(f"duplicate normalized questions: {_duplicates(questions)}")

    for case in generation:
        missing = GENERATION_REQUIRED_FIELDS - set(case)
        if missing:
            errors.append(f"{case.get('id')}: missing fields {sorted(missing)}")
        if not case.get("question", "").strip() or not case.get("reference_cypher", "").strip():
            errors.append(f"{case.get('id')}: empty question or reference Cypher")
        if case.get("ordered") and "ORDER BY" not in case.get("reference_cypher", "").upper():
            errors.append(f"{case.get('id')}: ordered=true but reference has no ORDER BY")

    for case in validation:
        missing = VALIDATION_REQUIRED_FIELDS - set(case)
        if missing:
            errors.append(f"{case.get('id')}: missing fields {sorted(missing)}")
        unknown_layers = set(case.get("expected_layers", [])) - REQUIRED_VALIDATION_LAYERS
        if unknown_layers:
            errors.append(f"{case.get('id')}: unknown validation layers {sorted(unknown_layers)}")
        if "llm" in case.get("expected_layers", []) and not case.get("manual_review_required"):
            errors.append(f"{case.get('id')}: LLM semantic case must require manual review")

    difficulty_counts = Counter(case.get("difficulty") for case in generation)
    if dict(difficulty_counts) != DIFFICULTY_TARGET:
        errors.append(
            f"difficulty distribution must be {DIFFICULTY_TARGET}, found {dict(difficulty_counts)}"
        )
    domain_counts = Counter(case.get("domain") for case in generation)
    query_type_counts = Counter(case.get("query_type") for case in generation)
    covered_tags = {tag for case in generation for tag in case.get("tags", [])}
    if not REQUIRED_LABELS <= covered_tags:
        errors.append(f"missing label coverage: {sorted(REQUIRED_LABELS - covered_tags)}")
    if not REQUIRED_RELATIONSHIPS <= covered_tags:
        errors.append(
            f"missing relationship coverage: {sorted(REQUIRED_RELATIONSHIPS - covered_tags)}"
        )

    layer_counts = Counter(
        layer for case in validation for layer in case.get("expected_layers", [])
    )
    if not REQUIRED_VALIDATION_LAYERS <= set(layer_counts):
        errors.append(
            f"missing validation layer coverage: {sorted(REQUIRED_VALIDATION_LAYERS - set(layer_counts))}"
        )

    seed_examples = _seed_examples()
    seed_questions = {_normalize_text(item["question"]) for item in seed_examples}
    seed_cyphers = {_normalize_cypher(item["cql"]) for item in seed_examples}
    leaked_questions = [
        case["id"]
        for case in generation
        if _normalize_text(case["question"]) in seed_questions
    ]
    leaked_cyphers = [
        case["id"]
        for case in generation
        if _normalize_cypher(case["reference_cypher"]) in seed_cyphers
    ]
    if leaked_questions:
        errors.append(f"exact Few-shot question leakage: {leaked_questions}")
    if leaked_cyphers:
        errors.append(f"exact Few-shot Cypher leakage: {leaked_cyphers}")

    seed_question_list = [item["question"] for item in seed_examples]
    near_leaks = []
    for case in generation:
        closest = max(
            seed_question_list,
            key=lambda seed: SequenceMatcher(
                None, _normalize_text(case["question"]), _normalize_text(seed)
            ).ratio(),
        )
        ratio = SequenceMatcher(
            None, _normalize_text(case["question"]), _normalize_text(closest)
        ).ratio()
        if ratio >= 0.86:
            near_leaks.append(
                {"case_id": case["id"], "seed_question": closest, "ratio": round(ratio, 3)}
            )
    if near_leaks:
        warnings.append(f"near-duplicate Few-shot questions need review: {near_leaks}")

    app_only_properties = {"name", "price", "stock", "brand", "category"}
    legacy_only_properties = {
        "ProductName",
        "UnitPrice",
        "UnitsInStock",
        "Rating",
        "ReviewText",
        "ReviewDate",
    }
    property_pattern = re.compile(r"\.\s*`?([A-Za-z_][A-Za-z0-9_]*)`?")
    for case in generation:
        props = set(property_pattern.findall(case["reference_cypher"]))
        if case["domain"] == "app_catalog" and props & legacy_only_properties:
            errors.append(
                f"{case['id']}: app_catalog query mixes legacy properties {sorted(props & legacy_only_properties)}"
            )
        if case["domain"] == "legacy_graph" and props & app_only_properties:
            errors.append(
                f"{case['id']}: legacy_graph query mixes app properties {sorted(props & app_only_properties)}"
            )

    graph = None
    executed_generation = 0
    executed_corrections = 0
    detected_layer_counts: Counter[str] = Counter()
    try:
        if online:
            graph = _make_graph()

            actual_labels = {
                row["label"]
                for row in graph.query(
                    "CALL db.labels() YIELD label WHERE label <> 'CypherQuery' RETURN label"
                )
            }
            actual_relationships = {
                row["relationshipType"]
                for row in graph.query(
                    "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
                )
            }
            if not REQUIRED_LABELS <= actual_labels:
                errors.append(
                    f"database is missing labels: {sorted(REQUIRED_LABELS - actual_labels)}"
                )
            if not REQUIRED_RELATIONSHIPS <= actual_relationships:
                errors.append(
                    "database is missing relationships: "
                    f"{sorted(REQUIRED_RELATIONSHIPS - actual_relationships)}"
                )

            for case in generation:
                case_id = case["id"]
                reference = case["reference_cypher"]
                write_errors = validate_no_writes_in_cypher_query(reference)
                syntax_errors = validate_cypher_query_syntax(graph, reference)
                schema_errors = validate_cypher_query_with_schema(graph, reference)
                if write_errors:
                    errors.append(f"{case_id}: reference is not read-only: {write_errors}")
                if syntax_errors:
                    errors.append(f"{case_id}: reference syntax errors: {syntax_errors}")
                if schema_errors:
                    errors.append(f"{case_id}: reference Schema errors: {schema_errors}")
                if write_errors or syntax_errors or schema_errors:
                    continue
                try:
                    records = graph.query(reference)
                    executed_generation += 1
                    if case["expected_non_empty"] and not records:
                        errors.append(f"{case_id}: expected non-empty result, got zero rows")
                    if records:
                        actual_columns = set(records[0])
                        expected_columns = set(case["expected_columns"])
                        if actual_columns != expected_columns:
                            errors.append(
                                f"{case_id}: expected columns {sorted(expected_columns)}, "
                                f"got {sorted(actual_columns)}"
                            )
                except Exception as exc:
                    errors.append(f"{case_id}: reference execution failed: {exc}")

            for case in validation:
                case_id = case["id"]
                candidate = case["candidate_cypher"]
                detected = set()
                if validate_cypher_query_syntax(graph, candidate):
                    detected.add("syntax")
                if validate_no_writes_in_cypher_query(candidate):
                    detected.add("write_operation")
                try:
                    corrected = correct_cypher_query_relationship_direction(
                        graph, candidate
                    )
                    if corrected and _normalize_cypher(corrected) != _normalize_cypher(candidate):
                        detected.add("relationship_direction")
                except Exception:
                    pass
                if validate_cypher_query_with_schema(graph, candidate):
                    detected.add("schema")
                detected_layer_counts.update(detected)

                expected_static = set(case["expected_layers"]) - {"llm"}
                missing_static = expected_static - detected
                if missing_static:
                    errors.append(
                        f"{case_id}: expected static layers not detected {sorted(missing_static)}; "
                        f"detected={sorted(detected)}"
                    )

                corrected_reference = case["corrected_reference_cypher"]
                if validate_no_writes_in_cypher_query(corrected_reference):
                    errors.append(f"{case_id}: corrected reference is not read-only")
                    continue
                syntax_errors = validate_cypher_query_syntax(graph, corrected_reference)
                schema_errors = validate_cypher_query_with_schema(
                    graph, corrected_reference
                )
                if syntax_errors or schema_errors:
                    errors.append(
                        f"{case_id}: corrected reference invalid; "
                        f"syntax={syntax_errors}, schema={schema_errors}"
                    )
                    continue
                try:
                    graph.query(corrected_reference)
                    executed_corrections += 1
                except Exception as exc:
                    errors.append(f"{case_id}: corrected reference execution failed: {exc}")
    finally:
        if graph is not None:
            graph.close()

    return {
        "status": "PASS" if not errors else "FAIL",
        "dataset": str(DATASET_PATH),
        "stats": {
            "generation_cases": len(generation),
            "validation_cases": len(validation),
            "total_cases": len(all_cases),
            "difficulty": dict(sorted(difficulty_counts.items())),
            "domains": dict(sorted(domain_counts.items())),
            "query_type_count": len(query_type_counts),
            "validation_layers": dict(sorted(layer_counts.items())),
            "fewshot_seed_examples": len(seed_examples),
            "target_metric_counts": expected_metric_counts,
            "exact_question_leaks": len(leaked_questions),
            "exact_cypher_leaks": len(leaked_cyphers),
            "near_leak_warnings": len(near_leaks),
            "online_checks": online,
            "generation_references_executed": executed_generation,
            "corrected_references_executed": executed_corrections,
            "statically_detected_layers": dict(sorted(detected_layer_counts.items())),
        },
        "errors": errors,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip Neo4j Schema and execution checks.",
    )
    args = parser.parse_args()
    report = validate_dataset(online=not args.offline)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
