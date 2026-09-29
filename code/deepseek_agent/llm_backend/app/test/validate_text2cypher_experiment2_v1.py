"""Validate the frozen inputs for Text2Cypher Experiment 2 without calling an LLM.

Part A candidates are checked only with the deterministic production validators.
Write candidates are never executed.  All reference queries and clean controls are
executed against the real Neo4j snapshot.  Part B is linked by SHA-256 to the
already validated Experiment 1 v2 question/oracle set.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Set

from langchain_neo4j import Neo4jGraph

from app.core.config import settings
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.text2cypher.query_context import (
    classify_text2cypher_question,
)
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.text2cypher.schema_context import (
    relevant_schema_capabilities,
)
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.text2cypher.validation.validators import (
    correct_cypher_query_relationship_direction,
    validate_cypher_query_syntax,
    validate_cypher_query_with_schema,
    validate_no_writes_in_cypher_query,
)
from app.lg_agent.kg_sub_graph.agentic_rag_agents.retrievers.cypher_examples.vector_store.factory import (
    _seed_examples,
)


TEST_DIR = Path(__file__).parent
DATASET_PATH = TEST_DIR / "data" / "text2cypher_experiment2_v1.json"
OUTPUT_PATH = TEST_DIR / "results" / "text2cypher_experiment2_v1_quality.json"
STATIC_LAYERS = {"syntax", "write_operation", "relationship_direction", "schema"}
ALL_LAYERS = STATIC_LAYERS | {"llm"}


def _normalize_text(value: str) -> str:
    return re.sub(r"[^\w]+", "", value, flags=re.UNICODE).lower()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _graph() -> Neo4jGraph:
    return Neo4jGraph(
        url=settings.NEO4J_URL,
        username=settings.NEO4J_USERNAME,
        password=settings.NEO4J_PASSWORD,
        database=settings.NEO4J_DATABASE,
    )


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, str):
        stripped = value.strip()
        if re.fullmatch(r"-?\d+", stripped):
            return int(stripped)
        if re.fullmatch(r"-?(?:\d+\.\d*|\d*\.\d+)", stripped):
            return round(float(stripped), 6)
        return value
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return str(value)


def _row_signature(row: Dict[str, Any]) -> str:
    return json.dumps(_canonical(row), ensure_ascii=False, sort_keys=True)


def _results_equal(
    candidate: List[Dict[str, Any]],
    reference: List[Dict[str, Any]],
    ordered: bool,
) -> bool:
    candidate_signatures = [_row_signature(row) for row in candidate]
    reference_signatures = [_row_signature(row) for row in reference]
    if not ordered:
        candidate_signatures.sort()
        reference_signatures.sort()
    return candidate_signatures == reference_signatures


def _schema_errors(
    graph: Neo4jGraph, question: str, statement: str
) -> List[str]:
    context = classify_text2cypher_question(question)
    capabilities = relevant_schema_capabilities(graph, question, context.domain)
    return validate_cypher_query_with_schema(
        graph,
        statement,
        allowed_labels=capabilities["labels"],
        allowed_relationships=capabilities["relationships"],
        allowed_properties=capabilities["properties"],
    )


def _static_layer_evidence(
    graph: Neo4jGraph, question: str, statement: str
) -> Dict[str, Dict[str, Any]]:
    syntax_errors: List[str] = []
    try:
        syntax_errors = validate_cypher_query_syntax(graph, statement)
    except Exception as exc:
        syntax_errors = [f"Syntax validation failed: {exc}"]
    write_errors = validate_no_writes_in_cypher_query(statement)

    corrected = statement
    direction_errors: List[str] = []
    direction_status = "passed"
    try:
        result = correct_cypher_query_relationship_direction(graph, statement)
        if result:
            corrected = result
            if result.strip() != statement.strip():
                direction_status = "corrected"
        elif re.search(r"\[[^\]]*:[^\]]+\]", statement):
            declared_relationships = set(
                re.findall(
                    r"\[[^\]]*:\s*`?([A-Za-z_][A-Za-z0-9_]*)`?",
                    statement,
                )
            )
            known_relationships = {
                str(item.get("type"))
                for item in graph.structured_schema.get("relationships", [])
                if item.get("type")
            }
            known_relationships.update(graph.structured_schema.get("rel_props", {}).keys())
            if declared_relationships and declared_relationships.issubset(known_relationships):
                direction_errors = [
                    "Relationship direction does not match the provided Schema"
                ]
                direction_status = "failed"
    except Exception as exc:
        direction_errors = [f"Relationship direction validation failed: {exc}"]
        direction_status = "failed"

    try:
        schema_errors = _schema_errors(graph, question, corrected)
    except Exception as exc:
        schema_errors = [f"Schema validation failed: {exc}"]

    return {
        "syntax": {
            "hit": bool(syntax_errors),
            "status": "failed" if syntax_errors else "passed",
            "errors": syntax_errors,
        },
        "write_operation": {
            "hit": bool(write_errors),
            "status": "failed" if write_errors else "passed",
            "errors": write_errors,
        },
        "relationship_direction": {
            "hit": direction_status != "passed",
            "status": direction_status,
            "errors": direction_errors,
            "statement_after": corrected,
        },
        "schema": {
            "hit": bool(schema_errors),
            "status": "failed" if schema_errors else "passed",
            "errors": schema_errors,
        },
    }


def _check_structure(dataset: Dict[str, Any], errors: List[str]) -> None:
    cases = dataset.get("part_a", {}).get("cases", [])
    if dataset.get("version") != "1.0.0":
        errors.append("dataset version must be 1.0.0")
    if dataset.get("status") not in {"candidate", "frozen"}:
        errors.append("dataset status must be candidate or frozen")
    if dataset.get("status") == "frozen" and dataset.get("manual_review", {}).get("status") != "completed":
        errors.append("frozen dataset must record a completed manual review")
    if dataset.get("schema_version") != settings.CYPHER_SCHEMA_VERSION:
        errors.append("dataset and runtime Schema versions differ")
    if len(cases) != 35:
        errors.append(f"Part A must contain 35 cases, found {len(cases)}")

    type_counts = Counter(case.get("case_type") for case in cases)
    expected_type_counts = {
        "isolated_fault": 25,
        "combined_fault": 5,
        "clean_control": 5,
    }
    if dict(type_counts) != expected_type_counts:
        errors.append(f"Part A type counts differ: {dict(type_counts)}")

    identifiers = [str(case.get("id", "")) for case in cases]
    questions = [str(case.get("question", "")).strip() for case in cases]
    candidates = [str(case.get("candidate_cypher", "")).strip() for case in cases]
    for label, values in (
        ("IDs", identifiers),
        ("questions", questions),
        ("candidate queries", candidates),
    ):
        if any(not value for value in values):
            errors.append(f"Part A contains an empty value among {label}")
        if len(set(values)) != len(values):
            duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
            errors.append(f"Part A contains duplicate {label}: {duplicates}")

    isolated_counts = Counter(
        case["expected_layers"][0]
        for case in cases
        if case.get("case_type") == "isolated_fault"
        and len(case.get("expected_layers", [])) == 1
    )
    if isolated_counts != Counter({layer: 5 for layer in ALL_LAYERS}):
        errors.append(f"isolated layer counts differ: {dict(isolated_counts)}")

    for case in cases:
        case_id = case.get("id", "unknown")
        expected = case.get("expected_layers")
        if not isinstance(expected, list) or not set(expected).issubset(ALL_LAYERS):
            errors.append(f"{case_id}: invalid expected_layers")
        if case.get("case_type") == "isolated_fault" and len(expected or []) != 1:
            errors.append(f"{case_id}: isolated fault must expect exactly one layer")
        if case.get("case_type") == "combined_fault" and len(expected or []) < 2:
            errors.append(f"{case_id}: combined fault must expect at least two layers")
        if case.get("case_type") == "clean_control" and expected:
            errors.append(f"{case_id}: clean control must have no expected layers")
        if bool("llm" in (expected or []) or case.get("case_type") == "clean_control") != bool(
            case.get("manual_review_required")
        ):
            errors.append(f"{case_id}: manual_review_required is inconsistent")

    part_b = dataset.get("part_b", {})
    if part_b.get("question_count") != 40 or part_b.get("repeats") != 3:
        errors.append("Part B must freeze 40 questions and three repeats")
    if part_b.get("fewshot_k") != 3 or not part_b.get("llm_validation"):
        errors.append("Part B must use Few-shot k=3 and LLM validation")
    if part_b.get("max_correction_retries") != 3:
        errors.append("Part B must allow exactly three correction retries")


def main() -> int:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    cases = dataset.get("part_a", {}).get("cases", [])
    errors: List[str] = []
    records: List[Dict[str, Any]] = []
    _check_structure(dataset, errors)

    part_b = dataset.get("part_b", {})
    part_b_path = DATASET_PATH.parent / str(part_b.get("source_file", ""))
    if not part_b_path.is_file():
        errors.append(f"Part B source does not exist: {part_b_path}")
        part_b_cases: List[Dict[str, Any]] = []
    else:
        source_sha = _sha256(part_b_path)
        if source_sha != part_b.get("source_sha256"):
            errors.append(
                f"Part B source SHA differs: {source_sha} != {part_b.get('source_sha256')}"
            )
        part_b_dataset = json.loads(part_b_path.read_text(encoding="utf-8"))
        part_b_cases = part_b_dataset.get(str(part_b.get("case_field", "")), [])
        if part_b_dataset.get("version") != part_b.get("source_version"):
            errors.append("Part B source version differs")
        if len(part_b_cases) != part_b.get("question_count"):
            errors.append("Part B source case count differs")

    seed_examples = _seed_examples()
    seed_questions = [_normalize_text(row["question"]) for row in seed_examples]
    seed_cyphers = [_normalize_text(row["cql"]) for row in seed_examples]
    leakage: List[Dict[str, Any]] = []
    for group, rows, cypher_field in (
        ("part_a", cases, "corrected_reference_cypher"),
        ("part_b", part_b_cases, "reference_cypher"),
    ):
        for case in rows:
            normalized_question = _normalize_text(case["question"])
            normalized_cypher = _normalize_text(case[cypher_field])
            closest = max(
                SequenceMatcher(None, normalized_question, seed).ratio()
                for seed in seed_questions
            )
            exact_question = normalized_question in seed_questions
            exact_cypher = normalized_cypher in seed_cyphers
            candidate_cypher = _normalize_text(case.get("candidate_cypher", ""))
            exact_candidate = bool(candidate_cypher) and candidate_cypher in seed_cyphers
            # Reference queries are a hidden scoring oracle and never enter a
            # provider request.  Record exact oracle matches for audit, but only
            # actual model inputs (question and Part A candidate) can leak.
            if exact_question or exact_candidate or closest >= 0.86:
                errors.append(
                    f"{case['id']}: Few-shot leakage "
                    f"exact_question={exact_question}, exact_candidate={exact_candidate}, "
                    f"question_similarity={closest:.3f}"
                )
            leakage.append(
                {
                    "group": group,
                    "id": case["id"],
                    "exact_question": exact_question,
                    "exact_candidate": exact_candidate,
                    "exact_hidden_oracle": exact_cypher,
                    "max_question_similarity": round(closest, 4),
                }
            )

    graph = _graph()
    try:
        for case in cases:
            case_id = case["id"]
            case_errors: List[str] = []
            reference = case["corrected_reference_cypher"]
            reference_validation: List[str] = []
            reference_validation.extend(validate_no_writes_in_cypher_query(reference))
            reference_validation.extend(validate_cypher_query_syntax(graph, reference))
            reference_validation.extend(_schema_errors(graph, case["question"], reference))
            direction_reference = correct_cypher_query_relationship_direction(graph, reference)
            if direction_reference.strip() != reference.strip():
                reference_validation.append("reference relationship direction is not canonical")

            reference_rows: List[Dict[str, Any]] = []
            if not reference_validation:
                try:
                    reference_rows = graph.query(reference)
                except Exception as exc:
                    reference_validation.append(f"reference execution failed: {exc}")
            if case.get("expected_non_empty") and not reference_rows:
                reference_validation.append("reference result must be non-empty")
            if reference_rows and set(reference_rows[0]) != set(case["expected_columns"]):
                reference_validation.append(
                    f"reference columns {sorted(reference_rows[0])} != "
                    f"{sorted(case['expected_columns'])}"
                )
            case_errors.extend(reference_validation)

            evidence = _static_layer_evidence(
                graph, case["question"], case["candidate_cypher"]
            )
            actual_static: Set[str] = {
                layer for layer, result in evidence.items() if result["hit"]
            }
            expected_static = set(case["expected_layers"]) & STATIC_LAYERS
            if actual_static != expected_static:
                case_errors.append(
                    f"static layer hits {sorted(actual_static)} != expected "
                    f"{sorted(expected_static)}"
                )

            candidate_rows: List[Dict[str, Any]] = []
            if case.get("case_type") == "clean_control" and not actual_static:
                try:
                    candidate_rows = graph.query(case["candidate_cypher"])
                except Exception as exc:
                    case_errors.append(f"clean control execution failed: {exc}")
                if not _results_equal(
                    candidate_rows, reference_rows, bool(case.get("ordered"))
                ):
                    case_errors.append("clean control result differs from reference")

            errors.extend(f"{case_id}: {error}" for error in case_errors)
            records.append(
                {
                    "id": case_id,
                    "case_type": case["case_type"],
                    "expected_layers": case["expected_layers"],
                    "actual_static_layers": sorted(actual_static),
                    "reference_row_count": len(reference_rows),
                    "candidate_executed": case.get("case_type") == "clean_control",
                    "candidate_row_count": len(candidate_rows),
                    "static_evidence": evidence,
                    "errors": case_errors,
                }
            )
    finally:
        graph.close()

    report = {
        "status": "PASS" if not errors else "FAIL",
        "dataset": str(DATASET_PATH),
        "dataset_sha256": _sha256(DATASET_PATH),
        "schema_version": dataset.get("schema_version"),
        "part_a_cases": len(cases),
        "part_a_valid": sum(not row["errors"] for row in records),
        "part_b_cases": len(part_b_cases),
        "write_candidates_executed": 0,
        "errors": errors,
        "leakage_review": leakage,
        "records": records,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key not in {"records", "leakage_review"}},
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"output={OUTPUT_PATH}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
