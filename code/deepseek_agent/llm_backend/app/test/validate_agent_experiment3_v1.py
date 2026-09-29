"""Validate and optionally freeze the held-out Experiment 3 assets."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


TEST_ROOT = Path(__file__).resolve().parent
BACKEND_ROOT = TEST_ROOT.parents[1]
sys.path.insert(0, str(BACKEND_ROOT))
DATASET_PATH = TEST_ROOT / "data" / "agent_experiment3_v1.json"
PROTOCOL_PATH = TEST_ROOT / "data" / "agent_experiment3_protocol_v1.json"
RESULT_PATH = TEST_ROOT / "results" / "agent_experiment3_v1_quality.json"
MANIFEST_PATH = TEST_ROOT / "results" / "agent_experiment3_preflight_manifest_v1.json"
GRAPHRAG_ROOT = BACKEND_ROOT / "app" / "graphrag" / "data"
INPUT_ROOT = GRAPHRAG_ROOT / "input"
INDEX_ROOT = GRAPHRAG_ROOT / "output_experiment3_v1"
EXPECTED_COUNTS = {
    "structured_predefined": 8,
    "structured_dynamic": 12,
    "graphrag": 15,
    "mixed": 10,
    "comprehensive": 5,
}
INDEX_FILES = [
    "documents.parquet",
    "text_units.parquet",
    "entities.parquet",
    "relationships.parquet",
    "communities.parquet",
    "community_reports.parquet",
]
KG_RUNTIME_ROOT = BACKEND_ROOT / "app" / "lg_agent" / "kg_sub_graph"
RUNTIME_FILES = sorted({
    BACKEND_ROOT / "main.py",
    BACKEND_ROOT / "app" / "core" / "config.py",
    BACKEND_ROOT / "app" / "api" / "agent_experiment3.py",
    BACKEND_ROOT / "app" / "api" / "text2cypher_experiment.py",
    BACKEND_ROOT / "app" / "lg_agent" / "lg_builder.py",
    BACKEND_ROOT / "app" / "lg_agent" / "lg_prompts.py",
    BACKEND_ROOT / "app" / "lg_agent" / "lg_states.py",
    *KG_RUNTIME_ROOT.rglob("*.py"),
    BACKEND_ROOT
    / "app"
    / "graphrag"
    / "graphrag"
    / "language_model"
    / "providers"
    / "local_hash_embedding.py",
    GRAPHRAG_ROOT / "settings.yaml",
    TEST_ROOT / "run_agent_experiment3_backend.py",
}, key=str)
RUNTIME_DISTRIBUTIONS = [
    "fastapi",
    "graphrag",
    "httpx",
    "lancedb",
    "langchain-core",
    "langchain-openai",
    "langgraph",
    "neo4j",
    "openai",
    "pandas",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def runtime_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in RUNTIME_DISTRIBUTIONS:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def runtime_config_snapshot() -> dict[str, Any]:
    from app.core.config import settings

    return {
        "model": settings.DEEPSEEK_MODEL,
        "agent_temperature": settings.AGENT_WORKFLOW_TEMPERATURE,
        "text2cypher_temperature": settings.TEXT2CYPHER_GENERATION_TEMPERATURE,
        "schema_version": settings.CYPHER_SCHEMA_VERSION,
        "cypher_example_embedding_model": settings.CYPHER_EXAMPLE_EMBEDDING_MODEL,
        "cypher_example_vector_index": settings.CYPHER_EXAMPLE_VECTOR_INDEX,
        "cypher_example_top_k": settings.CYPHER_EXAMPLE_TOP_K,
        "cypher_example_min_score": settings.CYPHER_EXAMPLE_MIN_SCORE,
        "graphrag_query_type": settings.GRAPHRAG_QUERY_TYPE,
        "graphrag_response_type": settings.GRAPHRAG_RESPONSE_TYPE,
        "graphrag_community_level": settings.GRAPHRAG_COMMUNITY_LEVEL,
        "graphrag_dynamic_community": settings.GRAPHRAG_DYNAMIC_COMMUNITY,
    }


def normalize(text: Any) -> str:
    value = unicodedata.normalize("NFKC", str(text)).lower()
    return re.sub(r"[\s\W_]+", "", value, flags=re.UNICODE)


def extract_questions(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "question" and isinstance(item, str):
                found.append(item)
            else:
                found.extend(extract_questions(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(extract_questions(item))
    return found


def old_dataset_questions() -> list[tuple[str, str]]:
    questions: list[tuple[str, str]] = []
    for path in sorted((TEST_ROOT / "data").glob("*.json")):
        if path in {DATASET_PATH, PROTOCOL_PATH}:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        questions.extend((path.name, question) for question in extract_questions(payload))
    return questions


def validate_static(dataset: dict[str, Any]) -> tuple[list[str], list[str], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    cases = dataset.get("cases", [])
    ids = [case.get("case_id") for case in cases]
    questions = [case.get("question", "") for case in cases]
    counts = Counter(case.get("category") for case in cases)

    if len(cases) != 50:
        errors.append(f"Expected 50 cases, found {len(cases)}")
    if len(set(ids)) != len(ids):
        errors.append("Duplicate case_id detected")
    if len(set(map(normalize, questions))) != len(questions):
        errors.append("Duplicate normalized question detected")
    if dict(counts) != EXPECTED_COUNTS:
        errors.append(f"Unexpected category distribution: {dict(counts)}")

    source_cache: dict[str, str] = {}
    for case in cases:
        case_id = case.get("case_id", "<missing>")
        category = case.get("category")
        expected_tools = case.get("expected_tools", [])
        if not case.get("question", "").strip():
            errors.append(f"{case_id}: empty question")
        if not expected_tools or any(
            tool not in {
                "predefined_cypher",
                "cypher_query",
                "microsoft_graphrag_query",
            }
            for tool in expected_tools
        ):
            errors.append(f"{case_id}: invalid expected_tools={expected_tools}")
        if category in {"mixed", "comprehensive"} and set(expected_tools) != {
            "cypher_query",
            "microsoft_graphrag_query",
        }:
            errors.append(f"{case_id}: mixed case must require both tool families")
        facts = case.get("answer_facts", [])
        if not facts:
            errors.append(f"{case_id}: no answer_facts")
        fact_ids = [fact.get("id") for fact in facts]
        if len(set(fact_ids)) != len(fact_ids):
            errors.append(f"{case_id}: duplicate fact id")

        source_files = case.get("reference", {}).get("source_files", [])
        source_text = ""
        for filename in source_files:
            path = INPUT_ROOT / filename
            if not path.is_file():
                errors.append(f"{case_id}: missing source file {filename}")
                continue
            source_cache.setdefault(filename, path.read_text(encoding="utf-8"))
            source_text += source_cache[filename]
        for fact in facts:
            alternatives = fact.get("any_of", [])
            if not alternatives or any(not str(item).strip() for item in alternatives):
                errors.append(f"{case_id}/{fact.get('id')}: empty fact alternatives")
                continue
            if fact.get("origin") == "source" and not any(
                normalize(alternative) in normalize(source_text)
                for alternative in alternatives
            ):
                errors.append(
                    f"{case_id}/{fact.get('id')}: no alternative appears in frozen sources"
                )

        cypher = case.get("reference", {}).get("cypher")
        if cypher:
            upper = f" {cypher.upper()} "
            for clause in [" CREATE ", " MERGE ", " DELETE ", " SET ", " REMOVE ", " DROP "]:
                if clause in upper:
                    errors.append(f"{case_id}: reference query contains write clause {clause.strip()}")

    prior = old_dataset_questions()
    nearest: list[dict[str, Any]] = []
    for case in cases:
        question = normalize(case["question"])
        best_source, best_question, best_ratio = "", "", 0.0
        for source, old_question in prior:
            ratio = SequenceMatcher(None, question, normalize(old_question)).ratio()
            if ratio > best_ratio:
                best_source, best_question, best_ratio = source, old_question, ratio
        nearest.append(
            {
                "case_id": case["case_id"],
                "similarity": round(best_ratio, 4),
                "source": best_source,
                "question": best_question,
            }
        )
        if best_ratio >= 0.92:
            errors.append(
                f"{case['case_id']}: likely leakage from {best_source} ({best_ratio:.3f})"
            )
        elif best_ratio >= 0.85:
            warnings.append(
                f"{case['case_id']}: review similarity to {best_source} ({best_ratio:.3f})"
            )

    return errors, warnings, {
        "case_count": len(cases),
        "distribution": dict(counts),
        "source_sha256": {
            name: sha256_file(INPUT_ROOT / name) for name in sorted(source_cache)
        },
        "heldout_nearest": nearest,
        "heldout_max_similarity": max((item["similarity"] for item in nearest), default=0),
    }


def validate_neo4j(dataset: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.text2cypher.validation.validators import (
        validate_cypher_query_syntax,
    )
    from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.utils.utils import (
        retrieve_and_parse_schema_from_graph_for_prompts,
    )
    from app.lg_agent.kg_sub_graph.kg_neo4j_conn import get_neo4j_graph

    errors: list[str] = []
    graph = get_neo4j_graph()
    schema = retrieve_and_parse_schema_from_graph_for_prompts(graph)
    references: dict[str, Any] = {}
    for case in dataset["cases"]:
        cypher = case.get("reference", {}).get("cypher")
        if not cypher:
            continue
        case_id = case["case_id"]
        syntax_errors = validate_cypher_query_syntax(graph, cypher)
        if syntax_errors:
            errors.append(f"{case_id}: reference syntax errors: {syntax_errors}")
            continue
        try:
            records = graph.query(cypher)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{case_id}: reference execution failed: {type(exc).__name__}: {exc}")
            continue
        minimum = int(case.get("reference", {}).get("min_rows", 1))
        if len(records) < minimum:
            errors.append(f"{case_id}: expected at least {minimum} rows, got {len(records)}")
        serialized = json.loads(json.dumps(records, ensure_ascii=False, default=str))
        references[case_id] = serialized
        record_text = normalize(json.dumps(serialized, ensure_ascii=False))
        for fact in case.get("answer_facts", []):
            if fact.get("origin") != "neo4j":
                continue
            alternatives = fact.get("any_of", [])
            direct = any(normalize(item) in record_text for item in alternatives)
            numeric = any(
                any(number in record_text for number in re.findall(r"\d+(?:\.\d+)?", str(item)))
                for item in alternatives
            )
            if not (direct or numeric):
                errors.append(
                    f"{case_id}/{fact.get('id')}: Neo4j fact not supported by reference rows"
                )
    return errors, {
        "schema_sha256": hashlib.sha256(schema.encode("utf-8")).hexdigest(),
        "reference_records": references,
        "reference_snapshot_sha256": stable_hash(references),
    }


def validate_index() -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    paths = [INDEX_ROOT / name for name in INDEX_FILES]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        errors.append(f"GraphRAG index files missing: {missing}")
        return errors, {"ready": False, "missing": missing}

    import pandas as pd

    documents = pd.read_parquet(INDEX_ROOT / "documents.parquet")
    titles = sorted(set(str(value) for value in documents.get("title", [])))
    required_titles = {"faq_service.txt", "user_experience.txt", "user_reviews.txt"}
    if set(titles) != required_titles or len(documents) != len(required_titles):
        errors.append(
            "GraphRAG documents must contain exactly the three frozen sources: "
            f"rows={len(documents)}, titles={titles}"
        )
    if {"title", "text"}.issubset(documents.columns):
        for row in documents[["title", "text"]].itertuples(index=False):
            source = INPUT_ROOT / str(row.title)
            if source.is_file() and str(row.text) != source.read_text(encoding="utf-8"):
                errors.append(f"GraphRAG document content differs from source: {row.title}")
    vector_tables = [
        "default-text_unit-text.lance",
        "default-community-full_content.lance",
        "default-entity-description.lance",
    ]
    missing_vectors = [
        name
        for name in vector_tables
        if not any((INDEX_ROOT / "lancedb" / name).rglob("*.lance"))
        and not any((INDEX_ROOT / "lancedb" / name).rglob("*.arrow"))
    ]
    if missing_vectors:
        errors.append(f"GraphRAG vector tables are incomplete: {missing_vectors}")
    combined_hash = hashlib.sha256()
    all_index_files = sorted(
        (path for path in INDEX_ROOT.rglob("*") if path.is_file()),
        key=lambda item: str(item.relative_to(INDEX_ROOT)),
    )
    for path in all_index_files:
        combined_hash.update(str(path.relative_to(INDEX_ROOT)).encode("utf-8"))
        combined_hash.update(path.read_bytes())
    return errors, {
        "ready": not errors,
        "titles": titles,
        "row_count": len(documents),
        "index_sha256": combined_hash.hexdigest(),
        "files": {
            str(path.relative_to(INDEX_ROOT)): sha256_file(path)
            for path in all_index_files
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--neo4j", action="store_true")
    parser.add_argument("--require-index", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if args.freeze and not (args.neo4j and args.require_index):
        raise SystemExit("--freeze requires --neo4j --require-index")

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    errors, warnings, static = validate_static(dataset)
    neo4j: dict[str, Any] = {"checked": False}
    if args.neo4j:
        neo4j_errors, neo4j = validate_neo4j(dataset)
        neo4j["checked"] = True
        errors.extend(neo4j_errors)
    index: dict[str, Any] = {"checked": False}
    if args.require_index:
        index_errors, index = validate_index()
        index["checked"] = True
        errors.extend(index_errors)

    report = {
        "status": "PASS" if not errors else "FAIL",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(DATASET_PATH),
        "dataset_sha256": sha256_file(DATASET_PATH),
        "protocol": str(PROTOCOL_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "static": static,
        "neo4j": neo4j,
        "graphrag_index": index,
        "runtime_config": runtime_config_snapshot(),
        "runtime_versions": runtime_versions(),
        "warnings": warnings,
        "errors": errors,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.freeze and not errors:
        manifest = {
            "status": "ready_for_formal_run",
            "protocol": protocol["protocol"],
            "frozen_at": datetime.now(timezone.utc).isoformat(),
            "dataset_sha256": report["dataset_sha256"],
            "protocol_sha256": report["protocol_sha256"],
            "source_sha256": static["source_sha256"],
            "neo4j_schema_sha256": neo4j["schema_sha256"],
            "neo4j_reference_snapshot_sha256": neo4j[
                "reference_snapshot_sha256"
            ],
            "graphrag_index_sha256": index["index_sha256"],
            "runtime_config": report["runtime_config"],
            "runtime_versions": report["runtime_versions"],
            "runtime_sha256": {
                str(path.relative_to(BACKEND_ROOT)): sha256_file(path)
                for path in RUNTIME_FILES
            },
        }
        MANIFEST_PATH.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(json.dumps({
        "status": report["status"],
        "errors": errors,
        "warnings": warnings,
        "quality_report": str(RESULT_PATH),
        "manifest": str(MANIFEST_PATH) if args.freeze and not errors else None,
    }, ensure_ascii=False, indent=2))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
