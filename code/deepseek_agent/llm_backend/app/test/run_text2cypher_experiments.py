"""Run the three resume-claim Text2Cypher experiments against real services.

The runner is resumable. It stores every generated statement, validation trace,
execution outcome and semantic result comparison in a raw JSON artifact.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek import ChatDeepSeek
from langchain_neo4j import Neo4jGraph

from app.core.config import settings
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.text2cypher import (
    create_text2cypher_correction_node,
    create_text2cypher_execution_node,
    create_text2cypher_validation_node,
)
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.text2cypher.generation.prompts import (
    create_text2cypher_generation_prompt_template,
)
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.text2cypher.validation.validators import (
    correct_cypher_query_relationship_direction,
    validate_cypher_query_syntax,
    validate_cypher_query_with_schema,
    validate_no_writes_in_cypher_query,
)
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.utils.utils import (
    retrieve_and_parse_schema_from_graph_for_prompts,
)
from app.lg_agent.kg_sub_graph.agentic_rag_agents.retrievers.cypher_examples.vector_store import (
    create_neo4j_vector_example_retriever,
)
from app.lg_agent.kg_sub_graph.agentic_rag_agents.workflows.single_agent import (
    create_text2cypher_agent,
)


TEST_DIR = Path(__file__).parent
DATASET_PATH = TEST_DIR / "data" / "text2cypher_benchmark_v1.json"
RESULTS_DIR = TEST_DIR / "results"
RAW_RESULTS_PATH = RESULTS_DIR / "text2cypher_experiments_raw.json"
REPORT_PATH = RESULTS_DIR / "text2cypher_experiments_report.md"
FEWSHOT_AUDIT_PATH = RESULTS_DIR / "text2cypher_fewshot_seed_audit.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strip_cypher_fence(text: str) -> str:
    value = text.strip()
    value = re.sub(r"^```(?:cypher)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value)
    return value.strip()


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if re.fullmatch(r"-?\d+", stripped):
            return int(stripped)
        if re.fullmatch(r"-?(?:\d+\.\d*|\d*\.\d+)", stripped):
            return round(float(stripped), 6)
        return value
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def _row_values(record: Dict[str, Any]) -> List[str]:
    return sorted(
        (
            json.dumps(_json_value(value), ensure_ascii=False, sort_keys=True)
            for value in record.values()
        )
    )


def _contains_required_values(candidate: Dict[str, Any], reference: Dict[str, Any]) -> bool:
    available = _row_values(candidate)
    for required in _row_values(reference):
        try:
            available.remove(required)
        except ValueError:
            return False
    return True


def _results_equivalent(
    candidate: List[Dict[str, Any]],
    reference: List[Dict[str, Any]],
    expected_columns: List[str],
    ordered: bool,
) -> bool:
    if len(candidate) != len(reference):
        return False
    if reference and len(reference[0]) != len(expected_columns):
        return False
    if ordered:
        return all(
            _contains_required_values(candidate_row, reference_row)
            for candidate_row, reference_row in zip(candidate, reference)
        )

    unmatched = list(candidate)
    for reference_row in reference:
        for index, candidate_row in enumerate(unmatched):
            if _contains_required_values(candidate_row, reference_row):
                unmatched.pop(index)
                break
        else:
            return False
    return not unmatched


def _load_dataset() -> Dict[str, Any]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def _dataset_sha256() -> str:
    return hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest()


def _new_results(dataset: Dict[str, Any], repeats: int) -> Dict[str, Any]:
    return {
        "metadata": {
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "dataset": str(DATASET_PATH),
            "dataset_sha256": _dataset_sha256(),
            "dataset_version": dataset["version"],
            "database_snapshot": dataset["database_snapshot"],
            "model": settings.DEEPSEEK_MODEL,
            "provider": "DeepSeek",
            "temperature": 0,
            "fewshot_k": 3,
            "experiment_1_repeats": repeats,
            "max_correction_retries": 3,
            "semantic_match": "execution_result_equivalence",
        },
        "experiment_1": {"records": []},
        "experiment_2": {"records": []},
        "experiment_3": {"records": []},
    }


def _load_or_create_results(
    dataset: Dict[str, Any], repeats: int, allow_dataset_revision: bool = False
) -> Dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_RESULTS_PATH.exists():
        results = json.loads(RAW_RESULTS_PATH.read_text(encoding="utf-8"))
        metadata = results.get("metadata", {})
        if (
            metadata.get("dataset_sha256") != _dataset_sha256()
            and not allow_dataset_revision
        ):
            raise RuntimeError(
                "Dataset changed after an experiment started. Move the old result file "
                "before running a new benchmark."
            )
        if metadata.get("model") != settings.DEEPSEEK_MODEL:
            raise RuntimeError("Configured model differs from the checkpoint model")
        if metadata.get("experiment_1_repeats") != repeats:
            raise RuntimeError("--repeats differs from the checkpoint configuration")
        return results
    results = _new_results(dataset, repeats)
    _save_results(results)
    return results


def rescore_results(
    dataset: Dict[str, Any], results: Dict[str, Any], graph: Neo4jGraph
) -> None:
    """Re-score immutable model outputs after a documented rubric correction."""

    generation_cases = {case["id"]: case for case in dataset["generation_cases"]}
    generation_reference = _reference_records(
        graph, dataset["generation_cases"], "reference_cypher"
    )
    for record in results["experiment_1"]["records"]:
        record.setdefault("evaluation_v1_strict", record["evaluation"])
        case = generation_cases[record["case_id"]]
        record["evaluation"] = _evaluate_statement(
            graph,
            case,
            record["generated_cypher"],
            generation_reference[record["case_id"]],
        )

    validation_cases = {case["id"]: case for case in dataset["validation_cases"]}
    validation_reference = _reference_records(
        graph, dataset["validation_cases"], "corrected_reference_cypher"
    )
    for record in results["experiment_2"]["records"]:
        record.setdefault(
            "semantically_correct_v1_strict",
            record["semantically_correct_after_correction"],
        )
        case = validation_cases[record["case_id"]]
        semantic_correct = False
        if not record.get("final_errors") and record.get("final_statement"):
            try:
                final_records = graph.query(record["final_statement"])
                reference_records = validation_reference[record["case_id"]]
                semantic_correct = _results_equivalent(
                    final_records,
                    reference_records,
                    list(reference_records[0]) if reference_records else [],
                    False,
                )
            except Exception:
                semantic_correct = False
        record["semantically_correct_after_correction"] = semantic_correct

    for record in results["experiment_3"]["records"]:
        record.setdefault("semantic_correct_v1_strict", record["semantic_correct"])
        case = generation_cases[record["case_id"]]
        semantic_correct = False
        if record.get("execution_success") and record.get("final_cypher"):
            try:
                final_records = graph.query(record["final_cypher"])
                semantic_correct = _results_equivalent(
                    final_records,
                    generation_reference[record["case_id"]],
                    case["expected_columns"],
                    case["ordered"],
                )
            except Exception:
                semantic_correct = False
        record["semantic_correct"] = semantic_correct

    metadata = results["metadata"]
    history = metadata.setdefault("evaluation_history", [])
    if not any(item.get("dataset_version") == "1.0.0" for item in history):
        history.append(
            {
                "dataset_version": "1.0.0",
                "dataset_sha256": metadata.get("dataset_sha256"),
                "semantic_match": metadata.get("semantic_match"),
                "reason_for_revision": (
                    "The strict rubric treated deterministic reference ordering and "
                    "unrequested helper columns as mandatory answer content."
                ),
            }
        )
    metadata["dataset_sha256"] = _dataset_sha256()
    metadata["dataset_version"] = dataset["version"]
    metadata["semantic_match"] = "required_reference_projection_equivalence"
    metadata["rescored_at"] = _utc_now()


def _save_results(results: Dict[str, Any]) -> None:
    results["metadata"]["updated_at"] = _utc_now()
    temporary = RAW_RESULTS_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(RAW_RESULTS_PATH)


def _make_graph() -> Neo4jGraph:
    return Neo4jGraph(
        url=settings.NEO4J_URL,
        username=settings.NEO4J_USERNAME,
        password=settings.NEO4J_PASSWORD,
        database=settings.NEO4J_DATABASE,
    )


def _make_model() -> ChatDeepSeek:
    return ChatDeepSeek(
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_BASE_URL,
        model_name=settings.DEEPSEEK_MODEL,
        temperature=0,
        max_tokens=1024,
        max_retries=3,
        request_timeout=120,
    )


def _reference_records(
    graph: Neo4jGraph, cases: Iterable[Dict[str, Any]], field: str
) -> Dict[str, List[Dict[str, Any]]]:
    return {case["id"]: graph.query(case[field]) for case in cases}


def _evaluate_statement(
    graph: Neo4jGraph,
    case: Dict[str, Any],
    statement: str,
    reference_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    outcome: Dict[str, Any] = {
        "read_only": False,
        "syntax_valid": False,
        "schema_valid": False,
        "executed": False,
        "semantic_correct": False,
        "failure_type": None,
        "errors": [],
        "record_count": 0,
    }
    write_errors = validate_no_writes_in_cypher_query(statement)
    if write_errors:
        outcome["failure_type"] = "write_operation"
        outcome["errors"] = write_errors
        return outcome
    outcome["read_only"] = True

    try:
        syntax_errors = validate_cypher_query_syntax(graph, statement)
    except Exception as exc:
        syntax_errors = [str(exc)]
    if syntax_errors:
        outcome["failure_type"] = "syntax"
        outcome["errors"] = syntax_errors
        return outcome
    outcome["syntax_valid"] = True

    try:
        schema_errors = validate_cypher_query_with_schema(graph, statement)
    except Exception as exc:
        schema_errors = [str(exc)]
    if schema_errors:
        outcome["failure_type"] = "schema"
        outcome["errors"] = schema_errors
        return outcome
    outcome["schema_valid"] = True

    try:
        records = graph.query(statement)
    except Exception as exc:
        outcome["failure_type"] = "execution"
        outcome["errors"] = [str(exc)]
        return outcome
    outcome["executed"] = True
    outcome["record_count"] = len(records)

    if case.get("ordered") and "ORDER BY" not in statement.upper():
        outcome["failure_type"] = "missing_required_order"
        return outcome
    outcome["semantic_correct"] = _results_equivalent(
        records,
        reference_records,
        case.get("expected_columns", []),
        case.get("ordered", False),
    )
    if not outcome["semantic_correct"]:
        outcome["failure_type"] = "result_mismatch"
    return outcome


async def _run_limited(
    jobs: List[Any], concurrency: int, label: str, on_result: Any
) -> List[Dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)
    total = len(jobs)
    completed = 0

    async def wrapped(job: Any) -> Dict[str, Any]:
        nonlocal completed
        async with semaphore:
            result = await job
        on_result(result)
        completed += 1
        print(f"[{label}] {completed}/{total} {result.get('case_id', '')}", flush=True)
        return result

    return await asyncio.gather(*(wrapped(job) for job in jobs))


async def run_experiment_1(
    dataset: Dict[str, Any],
    results: Dict[str, Any],
    graph: Neo4jGraph,
    model: ChatDeepSeek,
    repeats: int,
    concurrency: int,
) -> None:
    """Compare raw generation with k=0 and k=3, without downstream correction."""

    cases = dataset["generation_cases"]
    reference = _reference_records(graph, cases, "reference_cypher")
    schema = retrieve_and_parse_schema_from_graph_for_prompts(graph)
    retriever = create_neo4j_vector_example_retriever(graph)
    prompt = create_text2cypher_generation_prompt_template()
    chain = prompt | model | StrOutputParser()
    fewshot_by_case = {
        case["id"]: retriever.get_examples(case["question"], k=3) for case in cases
    }
    existing = {
        (record["case_id"], record["variant"], record["repeat"])
        for record in results["experiment_1"]["records"]
    }

    async def one(case: Dict[str, Any], variant: str, repeat: int) -> Dict[str, Any]:
        started = time.perf_counter()
        examples = "" if variant == "no_fewshot" else fewshot_by_case[case["id"]]
        try:
            generated = await chain.ainvoke(
                {
                    "question": case["question"],
                    "fewshot_examples": examples,
                    "schema": schema,
                }
            )
            statement = _strip_cypher_fence(generated)
            evaluation = _evaluate_statement(
                graph, case, statement, reference[case["id"]]
            )
            api_error = None
        except Exception as exc:
            statement = ""
            evaluation = {
                "read_only": False,
                "syntax_valid": False,
                "schema_valid": False,
                "executed": False,
                "semantic_correct": False,
                "failure_type": "generation_api",
                "errors": [str(exc)],
                "record_count": 0,
            }
            api_error = str(exc)
        return {
            "case_id": case["id"],
            "variant": variant,
            "repeat": repeat,
            "question": case["question"],
            "difficulty": case["difficulty"],
            "domain": case["domain"],
            "retrieved_examples": re.findall(r"^Question:\s*(.+)$", examples, re.MULTILINE),
            "generated_cypher": statement,
            "evaluation": evaluation,
            "api_error": api_error,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    jobs = []
    for variant in ("no_fewshot", "with_fewshot"):
        for repeat in range(1, repeats + 1):
            for case in cases:
                if (case["id"], variant, repeat) not in existing:
                    jobs.append(one(case, variant, repeat))
    if not jobs:
        print("[experiment_1] checkpoint already complete", flush=True)
        return
    def checkpoint(record: Dict[str, Any]) -> None:
        results["experiment_1"]["records"].append(record)
        _save_results(results)

    await _run_limited(jobs, concurrency, "experiment_1", checkpoint)


def _static_layer_evidence(graph: Neo4jGraph, statement: str) -> Dict[str, Any]:
    syntax_errors = []
    try:
        syntax_errors = validate_cypher_query_syntax(graph, statement)
    except Exception as exc:
        syntax_errors = [str(exc)]
    write_errors = validate_no_writes_in_cypher_query(statement)
    schema_errors = []
    try:
        schema_errors = validate_cypher_query_with_schema(graph, statement)
    except Exception as exc:
        schema_errors = [str(exc)]
    corrected_direction = ""
    try:
        corrected_direction = correct_cypher_query_relationship_direction(
            graph, statement
        )
    except Exception:
        pass
    direction_changed = bool(
        corrected_direction
        and re.sub(r"\s+", " ", corrected_direction).strip()
        != re.sub(r"\s+", " ", statement).strip()
    )
    return {
        "syntax_errors": syntax_errors,
        "write_errors": write_errors,
        "schema_errors": schema_errors,
        "direction_changed": direction_changed,
        "direction_corrected_cypher": corrected_direction if direction_changed else None,
    }


def _apply_state_update(state: Dict[str, Any], update: Dict[str, Any]) -> None:
    for key, value in update.items():
        if key == "steps":
            state[key] = list(state.get(key, [])) + list(value)
        else:
            state[key] = value


async def run_experiment_2(
    dataset: Dict[str, Any],
    results: Dict[str, Any],
    graph: Neo4jGraph,
    model: ChatDeepSeek,
    concurrency: int,
) -> None:
    """Inject known faults and exercise real validation/correction nodes."""

    cases = dataset["validation_cases"]
    existing = {record["case_id"] for record in results["experiment_2"]["records"]}
    validation_node = create_text2cypher_validation_node(
        graph=graph, llm=model, llm_validation=True, max_retries=3
    )
    correction_node = create_text2cypher_correction_node(llm=model, graph=graph)
    execution_node = create_text2cypher_execution_node(graph=graph)
    reference = _reference_records(graph, cases, "corrected_reference_cypher")

    async def one(case: Dict[str, Any]) -> Dict[str, Any]:
        started = time.perf_counter()
        initial_statement = case["candidate_cypher"]
        static = _static_layer_evidence(graph, initial_statement)
        static_errors = {
            str(error)
            for error in (
                static["syntax_errors"]
                + static["write_errors"]
                + static["schema_errors"]
            )
        }
        detected_layers = {
            layer
            for layer, present in (
                ("syntax", bool(static["syntax_errors"])),
                ("write_operation", bool(static["write_errors"])),
                ("relationship_direction", static["direction_changed"]),
                ("schema", bool(static["schema_errors"])),
            )
            if present
        }
        state: Dict[str, Any] = {
            "task": case["question"],
            "statement": initial_statement,
            "parameters": None,
            "errors": [],
            "mapping_errors": [],
            "records": [],
            "next_action_cypher": "validate_cypher",
            "attempts": 0,
            "retries": 0,
            "validation_layers": [],
            "steps": ["inject_candidate"],
        }
        trace = []
        output: Optional[Dict[str, Any]] = None
        api_error = None
        unsafe_candidate_executed = False
        try:
            for _ in range(4):
                before = state["statement"]
                command = await validation_node(state)
                update = dict(command.update or {})
                goto = str(command.goto)
                _apply_state_update(state, update)
                if not trace:
                    llm_only_errors = [
                        error
                        for error in state.get("errors", [])
                        if str(error) not in static_errors
                    ]
                    if llm_only_errors or state.get("mapping_errors"):
                        detected_layers.add("llm")
                trace.append(
                    {
                        "attempt": state.get("attempts"),
                        "statement_before": before,
                        "statement_after_validation": state.get("statement"),
                        "errors": list(state.get("errors", [])),
                        "mapping_errors": list(state.get("mapping_errors", [])),
                        "goto": goto,
                    }
                )
                if goto == "correct_cypher":
                    correction_update = await correction_node(state)
                    _apply_state_update(state, correction_update)
                    trace[-1]["corrected_cypher"] = state["statement"]
                    continue
                if goto == "execute_cypher":
                    if state["statement"] == initial_statement and validate_no_writes_in_cypher_query(initial_statement):
                        unsafe_candidate_executed = True
                    execution_result = await execution_node(state)
                    output = execution_result["cyphers"][0]
                    break
                output = {
                    "statement": state.get("statement", ""),
                    "records": [],
                    "errors": list(state.get("errors", []))
                    + list(state.get("mapping_errors", [])),
                    "attempts": state.get("attempts", 0),
                    "retries": state.get("retries", 0),
                    "steps": state.get("steps", []) + ["finalize_failure"],
                }
                break
        except Exception as exc:
            api_error = str(exc)
            output = output or {
                "statement": state.get("statement", ""),
                "records": [],
                "errors": [str(exc)],
                "attempts": state.get("attempts", 0),
                "retries": state.get("retries", 0),
                "steps": state.get("steps", []),
            }

        output = output or {
            "statement": state.get("statement", ""),
            "records": [],
            "errors": ["workflow ended without output"],
            "attempts": state.get("attempts", 0),
            "retries": state.get("retries", 0),
            "steps": state.get("steps", []),
        }
        expected_layers = set(case["expected_layers"])
        layers_matched = expected_layers <= detected_layers
        records = output.get("records", [])
        semantically_correct = bool(
            not output.get("errors")
            and isinstance(records, list)
            and _results_equivalent(
                records,
                reference[case["id"]],
                list(reference[case["id"]][0]) if reference[case["id"]] else [],
                False,
            )
        )
        passed = bool(
            layers_matched
            and semantically_correct
            and not unsafe_candidate_executed
            and api_error is None
        )
        return {
            "case_id": case["id"],
            "question": case["question"],
            "issue_type": case["issue_type"],
            "expected_layers": sorted(expected_layers),
            "detected_layers": sorted(detected_layers),
            "layers_matched": layers_matched,
            "initial_candidate": initial_statement,
            "static_evidence": static,
            "trace": trace,
            "final_statement": output.get("statement"),
            "final_errors": output.get("errors", []),
            "attempts": output.get("attempts", 0),
            "retries": output.get("retries", 0),
            "semantically_correct_after_correction": semantically_correct,
            "unsafe_candidate_executed": unsafe_candidate_executed,
            "passed": passed,
            "api_error": api_error,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    jobs = [one(case) for case in cases if case["id"] not in existing]
    if not jobs:
        print("[experiment_2] checkpoint already complete", flush=True)
        return
    def checkpoint(record: Dict[str, Any]) -> None:
        results["experiment_2"]["records"].append(record)
        _save_results(results)

    await _run_limited(jobs, concurrency, "experiment_2", checkpoint)


async def run_experiment_3(
    dataset: Dict[str, Any],
    results: Dict[str, Any],
    graph: Neo4jGraph,
    model: ChatDeepSeek,
    concurrency: int,
) -> None:
    """Run the complete four-stage workflow and measure successful execution."""

    cases = dataset["generation_cases"]
    existing = {record["case_id"] for record in results["experiment_3"]["records"]}
    reference = _reference_records(graph, cases, "reference_cypher")
    retriever = create_neo4j_vector_example_retriever(graph)
    workflow = create_text2cypher_agent(
        llm=model,
        graph=graph,
        cypher_example_retriever=retriever,
        llm_cypher_validation=True,
        max_retries=3,
        fewshot_k=3,
    )

    async def one(case: Dict[str, Any]) -> Dict[str, Any]:
        started = time.perf_counter()
        try:
            result = await workflow.ainvoke({"task": case["question"]})
            output = result["cyphers"][0]
            steps = list(output.get("steps", []))
            records = output.get("records", [])
            execution_success = bool(
                "execute_cypher" in steps and not output.get("errors")
            )
            semantic_correct = bool(
                execution_success
                and isinstance(records, list)
                and _results_equivalent(
                    records,
                    reference[case["id"]],
                    case["expected_columns"],
                    case["ordered"],
                )
            )
            return {
                "case_id": case["id"],
                "question": case["question"],
                "difficulty": case["difficulty"],
                "domain": case["domain"],
                "final_cypher": output.get("statement", ""),
                "errors": output.get("errors", []),
                "attempts": output.get("attempts", 0),
                "retries": output.get("retries", 0),
                "steps": steps,
                "record_count": len(records) if isinstance(records, list) else 0,
                "execution_success": execution_success,
                "semantic_correct": semantic_correct,
                "api_error": None,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        except Exception as exc:
            return {
                "case_id": case["id"],
                "question": case["question"],
                "difficulty": case["difficulty"],
                "domain": case["domain"],
                "final_cypher": "",
                "errors": [str(exc)],
                "attempts": 0,
                "retries": 0,
                "steps": [],
                "record_count": 0,
                "execution_success": False,
                "semantic_correct": False,
                "api_error": str(exc),
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }

    jobs = [one(case) for case in cases if case["id"] not in existing]
    if not jobs:
        print("[experiment_3] checkpoint already complete", flush=True)
        return
    def checkpoint(record: Dict[str, Any]) -> None:
        results["experiment_3"]["records"].append(record)
        _save_results(results)

    await _run_limited(jobs, concurrency, "experiment_3", checkpoint)


def _percent(numerator: int, denominator: int) -> float:
    return round(100 * numerator / denominator, 2) if denominator else 0.0


def _fewshot_cluster_bootstrap_ci(
    records: List[Dict[str, Any]], samples: int = 20_000
) -> tuple[float, float]:
    lookup = {
        (record["case_id"], record["repeat"], record["variant"]): int(
            record["evaluation"]["semantic_correct"]
        )
        for record in records
    }
    case_ids = sorted({record["case_id"] for record in records})
    repeats = sorted({record["repeat"] for record in records})
    deltas = {
        case_id: 100
        * sum(
            lookup[(case_id, repeat, "with_fewshot")]
            - lookup[(case_id, repeat, "no_fewshot")]
            for repeat in repeats
        )
        / len(repeats)
        for case_id in case_ids
    }
    rng = random.Random(20260903)
    bootstrap = []
    for _ in range(samples):
        selected = [case_ids[rng.randrange(len(case_ids))] for _ in case_ids]
        bootstrap.append(sum(deltas[case_id] for case_id in selected) / len(selected))
    bootstrap.sort()
    return (
        round(bootstrap[int(samples * 0.025)], 2),
        round(bootstrap[int(samples * 0.975)], 2),
    )


def reclassify_experiment_2(results: Dict[str, Any]) -> None:
    """Derive layer hits from immutable evidence without treating parser failure as a hit."""

    for record in results["experiment_2"]["records"]:
        static = record.get("static_evidence", {})
        detected_layers = {
            layer
            for layer, present in (
                ("syntax", bool(static.get("syntax_errors"))),
                ("write_operation", bool(static.get("write_errors"))),
                ("relationship_direction", bool(static.get("direction_changed"))),
                ("schema", bool(static.get("schema_errors"))),
            )
            if present
        }
        initial_trace = (record.get("trace") or [{}])[0]
        initial_errors = [str(error) for error in initial_trace.get("errors", [])]
        llm_parser_failed = any(
            error.startswith("LLM validation failed:") for error in initial_errors
        )
        static_errors = {
            str(error)
            for error in (
                list(static.get("syntax_errors", []))
                + list(static.get("write_errors", []))
                + list(static.get("schema_errors", []))
            )
        }
        llm_semantic_evidence = [
            error
            for error in initial_errors
            if error not in static_errors
            and not error.startswith("LLM validation failed:")
        ]
        if not llm_parser_failed and (
            llm_semantic_evidence or initial_trace.get("mapping_errors")
        ):
            detected_layers.add("llm")

        expected_layers = set(record.get("expected_layers", []))
        record["detected_layers"] = sorted(detected_layers)
        record["llm_validation_operational"] = not llm_parser_failed
        record["llm_semantic_evidence"] = llm_semantic_evidence
        record["layers_matched"] = expected_layers <= detected_layers
        record["passed"] = bool(
            record["layers_matched"]
            and record.get("semantically_correct_after_correction")
            and not record.get("unsafe_candidate_executed")
            and record.get("api_error") is None
        )


def write_report(results: Dict[str, Any]) -> None:
    exp1 = results["experiment_1"]["records"]
    exp2 = results["experiment_2"]["records"]
    exp3 = results["experiment_3"]["records"]
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in exp1:
        grouped[record["variant"]].append(record)

    lines = [
        "# Text2Cypher 三项真实实验报告",
        "",
        f"- 模型：`{results['metadata']['model']}`",
        f"- 温度：{results['metadata']['temperature']}",
        f"- 数据集：`{results['metadata']['dataset_version']}`（40条生成题、22条验证题）",
        f"- 原始结果：`{RAW_RESULTS_PATH.name}`",
    ]
    old_exp1 = [record for record in exp1 if "evaluation_v1_strict" in record]
    if old_exp1:
        old_no = [record for record in old_exp1 if record["variant"] == "no_fewshot"]
        old_with = [record for record in old_exp1 if record["variant"] == "with_fewshot"]
        old_exp3_semantic = sum(
            record.get("semantic_correct_v1_strict", False) for record in exp3
        )
        lines.extend(
            [
                "",
                "## 评分规范审计",
                "",
                "v1.0曾把参考查询的稳定排序和题目未要求的辅助列当作强制答案。审计后发布v1.1，仅重算已保存输出，未重新请求模型。",
                "",
                f"- v1.0严格口径：无Few-shot {sum(r['evaluation_v1_strict']['semantic_correct'] for r in old_no)}/{len(old_no)}；有Few-shot {sum(r['evaluation_v1_strict']['semantic_correct'] for r in old_with)}/{len(old_with)}；完整流程语义正确 {old_exp3_semantic}/{len(exp3)}。",
                "- 每条v1.0评分仍保存在原始JSON中，可复算、可追溯。",
            ]
        )
    lines.extend(
        [
        "",
        "## 实验一：Few-shot 对生成准确率的影响",
        "",
        "| 条件 | 语义正确 | 总运行数 | 生成准确率 | 可执行 | 可执行率 | API错误 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for variant in ("no_fewshot", "with_fewshot"):
        records = grouped.get(variant, [])
        correct = sum(item["evaluation"]["semantic_correct"] for item in records)
        executed = sum(item["evaluation"]["executed"] for item in records)
        api_errors = sum(bool(item.get("api_error")) for item in records)
        lines.append(
            f"| {variant} | {correct} | {len(records)} | "
            f"{_percent(correct, len(records))}% | {executed} | "
            f"{_percent(executed, len(records))}% | {api_errors} |"
        )
    if grouped.get("no_fewshot") and grouped.get("with_fewshot"):
        no_rate = _percent(
            sum(item["evaluation"]["semantic_correct"] for item in grouped["no_fewshot"]),
            len(grouped["no_fewshot"]),
        )
        with_rate = _percent(
            sum(item["evaluation"]["semantic_correct"] for item in grouped["with_fewshot"]),
            len(grouped["with_fewshot"]),
        )
        lines.extend(
            [
                "",
                f"观察提升：**{round(with_rate - no_rate, 2)} 个百分点**。",
                f"简历65%→85%是否复现：**{'是' if no_rate == 65 and with_rate == 85 else '否'}**。",
                "",
                "三轮语义准确率：",
            ]
        )
        for variant in ("no_fewshot", "with_fewshot"):
            per_run = []
            for repeat in range(1, results["metadata"]["experiment_1_repeats"] + 1):
                run_records = [
                    item for item in grouped[variant] if item["repeat"] == repeat
                ]
                correct = sum(
                    item["evaluation"]["semantic_correct"] for item in run_records
                )
                per_run.append(f"第{repeat}轮 {correct}/40（{_percent(correct, len(run_records))}%）")
            lines.append(f"- {variant}: " + "；".join(per_run))
        ci_low, ci_high = _fewshot_cluster_bootstrap_ci(exp1)
        lines.append(
            f"- 按40个问题聚类Bootstrap（固定种子、20,000次）的提升95%区间：{ci_low}—{ci_high}个百分点。"
        )

    layer_expected = Counter(layer for row in exp2 for layer in row["expected_layers"])
    layer_detected = Counter(
        layer
        for row in exp2
        for layer in row["expected_layers"]
        if layer in row["detected_layers"]
    )
    exp2_pass = sum(row["passed"] for row in exp2)
    correction_success = sum(
        row["semantically_correct_after_correction"] for row in exp2
    )
    unsafe = sum(row["unsafe_candidate_executed"] for row in exp2)
    llm_operational = sum(
        row.get("llm_validation_operational", False)
        for row in exp2
        if "llm" in row["expected_layers"]
    )
    llm_expected = sum("llm" in row["expected_layers"] for row in exp2)
    lines.extend(
        [
            "",
            "## 实验二：五层校验与自动修正",
            "",
            f"- 完整通过：{exp2_pass}/{len(exp2)}（{_percent(exp2_pass, len(exp2))}%）",
            f"- 修正后结果与参考答案等价：{correction_success}/{len(exp2)}（{_percent(correction_success, len(exp2))}%）",
            f"- LLM语义校验正常返回结构化结果：{llm_operational}/{llm_expected}",
            f"- 非法写查询被执行：{unsafe}",
            "",
            "| 校验层 | 命中 | 应命中 |",
            "| --- | ---: | ---: |",
        ]
    )
    for layer in ("syntax", "write_operation", "relationship_direction", "llm", "schema"):
        lines.append(
            f"| {layer} | {layer_detected[layer]} | {layer_expected[layer]} |"
        )

    success = sum(row["execution_success"] for row in exp3)
    semantic = sum(row["semantic_correct"] for row in exp3)
    retries = sum(row["retries"] > 0 for row in exp3)
    exp3_failure_reasons = Counter()
    for row in exp3:
        if row["execution_success"]:
            continue
        errors = " ".join(str(error) for error in row.get("errors", []))
        if row.get("api_error"):
            exp3_failure_reasons["API异常"] += 1
        elif "LLM validation failed" in errors:
            exp3_failure_reasons["LLM校验输出解析失败"] += 1
        elif not row.get("final_cypher") or "Unexpected end of input" in errors:
            exp3_failure_reasons["修正后Cypher为空或语法无效"] += 1
        else:
            exp3_failure_reasons["其他验证失败"] += 1
    lines.extend(
        [
            "",
            "## 实验三：完整工作流成功执行率",
            "",
            f"- 成功执行：{success}/{len(exp3)}（**{_percent(success, len(exp3))}%**）",
            f"- 执行结果语义正确：{semantic}/{len(exp3)}（{_percent(semantic, len(exp3))}%）",
            f"- 发生自动重试：{retries}/{len(exp3)}",
            f"- 简历90%是否复现：**{'是' if _percent(success, len(exp3)) == 90 else '否'}**",
            "- 未执行原因："
            + "；".join(
                f"{reason} {count}条"
                for reason, count in exp3_failure_reasons.most_common()
            ),
            "",
            "## 判定说明",
            "",
            "- 生成准确率按题目要求的参考结果投影判定；允许额外辅助列，明确要求排序/Top-K时比较顺序。",
            "- 成功执行要求完整工作流进入 `execute_cypher` 且没有错误。",
            "- 所有失败样本、生成语句、校验错误及重试轨迹均保存在原始结果 JSON 中。",
        ]
    )
    if FEWSHOT_AUDIT_PATH.exists():
        audit = json.loads(FEWSHOT_AUDIT_PATH.read_text(encoding="utf-8"))
        lines.extend(
            [
                "",
                "## 关键根因审计",
                "",
                f"- Few-shot种子：{audit['fully_valid_and_executed']}/{audit['total']}条与当前Schema一致且可执行；{audit['total'] - audit['fully_valid_and_executed']}条失效。",
                f"- 其中Schema有效：{audit['schema_valid']}/{audit['total']}；语法有效：{audit['syntax_valid']}/{audit['total']}。",
                "- 主要失效原因：旧字段 `OrderID`/`CustomerID` 与当前 `orderId`/`customerId` 不一致、`REPORTS_TO`关系不存在，以及订单明细数值以字符串存储却直接参与乘法/求和。",
                "- LLM校验层的JSON解析失败会被当作验证错误，反复触发修正，是完整流程成功率下降的直接原因。",
            ]
        )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def async_main(args: argparse.Namespace) -> int:
    dataset = _load_dataset()
    results = _load_or_create_results(
        dataset, args.repeats, allow_dataset_revision=args.rescore
    )
    if args.rescore:
        graph = _make_graph()
        try:
            rescore_results(dataset, results, graph)
            reclassify_experiment_2(results)
            _save_results(results)
            write_report(results)
            print(f"raw_results={RAW_RESULTS_PATH}", flush=True)
            print(f"report={REPORT_PATH}", flush=True)
            return 0
        finally:
            graph.close()
    reclassify_experiment_2(results)
    if args.report_only:
        _save_results(results)
        write_report(results)
        print(f"raw_results={RAW_RESULTS_PATH}", flush=True)
        print(f"report={REPORT_PATH}", flush=True)
        return 0
    graph = _make_graph()
    model = _make_model()
    try:
        selected = {"1", "2", "3"} if args.experiment == "all" else {args.experiment}
        if "1" in selected:
            await run_experiment_1(
                dataset, results, graph, model, args.repeats, args.concurrency
            )
        if "2" in selected:
            await run_experiment_2(
                dataset, results, graph, model, args.concurrency
            )
        if "3" in selected:
            await run_experiment_3(
                dataset, results, graph, model, args.concurrency
            )
        reclassify_experiment_2(results)
        _save_results(results)
        write_report(results)
        print(f"raw_results={RAW_RESULTS_PATH}", flush=True)
        print(f"report={REPORT_PATH}", flush=True)
        return 0
    finally:
        graph.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["1", "2", "3", "all"], default="all")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Preserve model outputs and recompute scores after a rubric revision.",
    )
    args = parser.parse_args()
    if args.repeats < 1 or args.concurrency < 1:
        parser.error("--repeats and --concurrency must be positive")
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
