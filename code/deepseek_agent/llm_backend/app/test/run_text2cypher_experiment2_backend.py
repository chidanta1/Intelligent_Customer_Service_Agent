"""Run Text2Cypher Experiment 2A and 2B through the local FastAPI backend.

The driver never constructs an LLM and never calls DeepSeek directly.  Part A
injects frozen candidates into the production validation/correction/execution
nodes exposed by the backend.  Part B sends natural-language questions to the
complete production Text2Cypher LangGraph.  Neo4j is used here only as an
independent read-only oracle for semantic scoring.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx
from langchain_neo4j import Neo4jGraph

from app.core.config import settings
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.text2cypher.validation.validators import (
    validate_no_writes_in_cypher_query,
)


PROTOCOL = "text2cypher-exp2-backend-v2"
EXPECTED_LAYERS = [
    "syntax",
    "write_operation",
    "relationship_direction",
    "llm",
    "schema",
]
TEST_DIR = Path(__file__).parent
DATASET_PATH = TEST_DIR / "data" / "text2cypher_experiment2_v1.json"
PROTOCOL_PATH = TEST_DIR / "data" / "text2cypher_experiment2_protocol_v2.json"
RESULTS_DIR = TEST_DIR / "results"
RAW_RESULTS_PATH = RESULTS_DIR / "text2cypher_experiment2_backend_v2_raw.json"
REPORT_PATH = RESULTS_DIR / "text2cypher_experiment2_backend_v2_report.md"
RANDOMIZATION_SEED = 20260911


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        json.dumps(_json_value(value), ensure_ascii=False, sort_keys=True)
        for value in record.values()
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
    if reference and set(reference[0]) != set(expected_columns):
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


def _graph() -> Neo4jGraph:
    return Neo4jGraph(
        url=settings.NEO4J_URL,
        username=settings.NEO4J_USERNAME,
        password=settings.NEO4J_PASSWORD,
        database=settings.NEO4J_DATABASE,
    )


def _new_results(
    dataset: Dict[str, Any], base_url: str, health: Dict[str, Any]
) -> Dict[str, Any]:
    part_b_path = DATASET_PATH.parent / dataset["part_b"]["source_file"]
    return {
        "metadata": {
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "completed_at": None,
            "experiment": "Experiment 2A/2B: production validation and full workflow",
            "protocol": PROTOCOL,
            "protocol_file_sha256": _sha256(PROTOCOL_PATH),
            "provider_call_owner": "backend",
            "driver_provider_sdk_calls": 0,
            "base_url": base_url,
            "dataset": str(DATASET_PATH),
            "dataset_version": dataset["version"],
            "dataset_sha256": _sha256(DATASET_PATH),
            "part_b_source_sha256": _sha256(part_b_path),
            "database_snapshot": dataset["database_snapshot"],
            "model": health["model"],
            "temperature": health["temperature"],
            "schema_version": health["schema_version"],
            "schema_sha256": health["schema_sha256"],
            "validation_layers": health["validation_layers"],
            "fewshot_k": health["fewshot_k"],
            "llm_validation": health["llm_validation"],
            "max_correction_retries": health["max_correction_retries"],
            "force_execute_after_retry_limit": health[
                "force_execute_after_retry_limit"
            ],
            "part_a_requests": len(dataset["part_a"]["cases"]),
            "part_b_questions": dataset["part_b"]["question_count"],
            "part_b_repeats": dataset["part_b"]["repeats"],
            "part_b_requests": dataset["part_b"]["question_count"]
            * dataset["part_b"]["repeats"],
            "randomization_seed": RANDOMIZATION_SEED,
            "semantic_match": "required_reference_projection_equivalence",
        },
        "part_a_records": [],
        "part_b_records": [],
        "integrity_violations": [],
    }


def _save(results: Dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results["metadata"]["updated_at"] = _utc_now()
    temporary = RAW_RESULTS_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(RAW_RESULTS_PATH)


def _load_or_create(
    dataset: Dict[str, Any], base_url: str, health: Dict[str, Any]
) -> Dict[str, Any]:
    if not RAW_RESULTS_PATH.exists():
        results = _new_results(dataset, base_url, health)
        _save(results)
        return results
    results = json.loads(RAW_RESULTS_PATH.read_text(encoding="utf-8"))
    metadata = results.get("metadata", {})
    expected = _new_results(dataset, base_url, health)["metadata"]
    keys = (
        "protocol",
        "protocol_file_sha256",
        "dataset_version",
        "dataset_sha256",
        "part_b_source_sha256",
        "model",
        "temperature",
        "schema_version",
        "schema_sha256",
        "validation_layers",
        "fewshot_k",
        "llm_validation",
        "max_correction_retries",
        "force_execute_after_retry_limit",
        "part_a_requests",
        "part_b_questions",
        "part_b_repeats",
        "part_b_requests",
        "randomization_seed",
    )
    mismatches = {
        key: (metadata.get(key), expected.get(key))
        for key in keys
        if metadata.get(key) != expected.get(key)
    }
    if mismatches:
        raise RuntimeError(f"Existing Experiment 2 checkpoint is incompatible: {mismatches}")
    return results


async def _health(client: httpx.AsyncClient, token: str) -> Dict[str, Any]:
    response = await client.get(
        "/api/experiments/text2cypher/experiment2/health",
        headers={"X-Experiment-Token": token},
    )
    response.raise_for_status()
    health = response.json()
    expected = {
        "protocol": PROTOCOL,
        "provider_call_owner": "backend",
        "temperature": 0.0,
        "schema_version": "ecommerce-v2",
        "validation_layers": EXPECTED_LAYERS,
        "fewshot_k": 3,
        "llm_validation": True,
        "max_correction_retries": 3,
        "force_execute_after_retry_limit": False,
    }
    mismatches = {
        key: (health.get(key), value)
        for key, value in expected.items()
        if health.get(key) != value
    }
    boundaries = health.get("execution_boundaries", {})
    if boundaries.get("part_a") != "fastapi-to-production-validation-correction-execution-nodes":
        mismatches["part_a_boundary"] = boundaries.get("part_a")
    if boundaries.get("part_b") != "fastapi-to-production-text2cypher-langgraph":
        mismatches["part_b_boundary"] = boundaries.get("part_b")
    if not re.fullmatch(r"[0-9a-f]{64}", str(health.get("schema_sha256", ""))):
        mismatches["schema_sha256"] = health.get("schema_sha256")
    if mismatches:
        raise RuntimeError(f"Experiment 2 backend preflight failed: {mismatches}")
    return health


def _response_invariant_errors(
    body: Dict[str, Any], health: Dict[str, Any], expected: Dict[str, Any]
) -> List[str]:
    errors = []
    fixed = {
        "protocol": PROTOCOL,
        "model": health["model"],
        "temperature": health["temperature"],
        "schema_version": health["schema_version"],
        "schema_sha256": health["schema_sha256"],
        **expected,
    }
    for key, value in fixed.items():
        if body.get(key) != value:
            errors.append(f"{key}={body.get(key)!r}, expected {value!r}")
    return errors


def _trace_errors(body: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    trace = body.get("validation_trace", [])
    attempts = body.get("attempts")
    retries = body.get("retries")
    if not isinstance(attempts, int) or not 1 <= attempts <= 4:
        errors.append(f"attempts out of range: {attempts}")
    if not isinstance(retries, int) or not 0 <= retries <= 3:
        errors.append(f"retries out of range: {retries}")
    if isinstance(attempts, int) and len(trace) != attempts:
        errors.append(f"trace length {len(trace)} != attempts {attempts}")
    if isinstance(attempts, int) and isinstance(retries, int) and attempts != retries + 1:
        errors.append(f"attempts {attempts} != retries+1 {retries + 1}")
    if len(body.get("correction_history", [])) != retries:
        errors.append("correction_history length differs from retries")
    for index, attempt in enumerate(trace, start=1):
        if attempt.get("attempt") != index or attempt.get("retry") != index - 1:
            errors.append(f"invalid trace numbering at attempt {index}")
        layer_names = [layer.get("layer") for layer in attempt.get("layers", [])]
        if layer_names != EXPECTED_LAYERS:
            errors.append(f"attempt {index} layer order differs: {layer_names}")
    if body.get("executed"):
        if not trace or trace[-1].get("next_action") != "execute_cypher":
            errors.append("execution occurred without an execute_cypher validation decision")
        if body.get("errors"):
            errors.append("executed response still contains errors")
    elif trace and trace[-1].get("next_action") == "execute_cypher":
        errors.append("validation allowed execution but execution did not occur")
    if body.get("unsafe_executed"):
        errors.append("unsafe write query was executed")
    if body.get("executed_with_blocking_validation_errors"):
        errors.append("query executed with blocking validation errors")
    return errors


def _failed_record(
    *, part: str, case: Dict[str, Any], repeat: Optional[int], exc: Exception,
    started: float,
) -> Dict[str, Any]:
    exception_text = f"{type(exc).__name__}: {exc}".rstrip()
    record = {
        "case_id": case["id"],
        "question": case["question"],
        "backend_request_id": None,
        "http_status": getattr(getattr(exc, "response", None), "status_code", None),
        "api_error": exception_text,
        "integrity_errors": [exception_text],
        "executed": False,
        "unsafe_executed": False,
        "executed_with_blocking_validation_errors": False,
        "semantic_correct": False,
        "attempts": 0,
        "retries": 0,
        "validation_trace": [],
        "correction_history": [],
        "records": [],
        "driver_roundtrip_latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }
    if part == "a":
        record.update(
            case_type=case["case_type"],
            issue_type=case["issue_type"],
            expected_layers=case["expected_layers"],
            initially_detected_layers=[],
            all_expected_layers_detected=False,
            clean_false_positive=False,
        )
    else:
        record.update(
            repeat=repeat,
            difficulty=case["difficulty"],
            domain=case["domain"],
            query_type=case["query_type"],
            successful_execution=False,
            initial_validation_pass=False,
            rescued_by_correction=False,
        )
    return record


async def _run_part_a_request(
    client: httpx.AsyncClient,
    token: str,
    semaphore: asyncio.Semaphore,
    case: Dict[str, Any],
    reference: List[Dict[str, Any]],
    health: Dict[str, Any],
) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        async with semaphore:
            response = await client.post(
                "/api/experiments/text2cypher/experiment2/inject",
                headers={"X-Experiment-Token": token},
                json={
                    "case_id": case["id"],
                    "question": case["question"],
                    "candidate_cypher": case["candidate_cypher"],
                },
            )
        response.raise_for_status()
        body = response.json()
        invariant_errors = _response_invariant_errors(
            body,
            health,
            {
                "case_id": case["id"],
                "question": case["question"],
                "initial_cypher": case["candidate_cypher"],
            },
        )
        trace_errors = _trace_errors(body)
        first_layers = body.get("validation_trace", [{}])[0].get("layers", [])
        detected_layers = [
            item["layer"]
            for item in first_layers
            if item.get("status") in {"failed", "corrected"}
        ]
        expected_layers = case["expected_layers"]
        all_detected = set(expected_layers).issubset(detected_layers)
        missing_order = bool(
            case.get("ordered") and "ORDER BY" not in body.get("statement", "").upper()
        )
        semantic_correct = bool(
            body.get("executed")
            and not body.get("errors")
            and not missing_order
            and _results_equivalent(
                body.get("records", []),
                reference,
                case["expected_columns"],
                bool(case.get("ordered")),
            )
        )
        return {
            "case_id": case["id"],
            "case_type": case["case_type"],
            "issue_type": case["issue_type"],
            "question": case["question"],
            "candidate_cypher": case["candidate_cypher"],
            "expected_layers": expected_layers,
            "initially_detected_layers": detected_layers,
            "all_expected_layers_detected": all_detected,
            "unexpected_initial_layers": sorted(set(detected_layers) - set(expected_layers)),
            "clean_false_positive": bool(
                case["case_type"] == "clean_control" and detected_layers
            ),
            "direction_deterministically_corrected": bool(
                case.get("allow_direction_auto_correct")
                and "relationship_direction" in detected_layers
                and body.get("retries") == 0
                and body.get("executed")
            ),
            "backend_request_id": body.get("request_id"),
            "initial_cypher": body.get("initial_cypher", ""),
            "final_cypher": body.get("statement", ""),
            "errors": body.get("errors", []),
            "records": body.get("records", []),
            "record_count": len(body.get("records", [])),
            "attempts": body.get("attempts", 0),
            "retries": body.get("retries", 0),
            "validation_trace": body.get("validation_trace", []),
            "correction_history": body.get("correction_history", []),
            "steps": body.get("steps", []),
            "executed": bool(body.get("executed")),
            "unsafe_executed": bool(body.get("unsafe_executed")),
            "executed_with_blocking_validation_errors": bool(
                body.get("executed_with_blocking_validation_errors")
            ),
            "semantic_correct": semantic_correct,
            "missing_required_order": missing_order,
            "http_status": response.status_code,
            "api_error": None,
            "integrity_errors": invariant_errors + trace_errors,
            "backend_total_latency_ms": body.get("total_latency_ms"),
            "driver_roundtrip_latency_ms": round(
                (time.perf_counter() - started) * 1000, 2
            ),
        }
    except Exception as exc:
        return _failed_record(part="a", case=case, repeat=None, exc=exc, started=started)


async def _run_part_b_request(
    client: httpx.AsyncClient,
    token: str,
    semaphore: asyncio.Semaphore,
    case: Dict[str, Any],
    repeat: int,
    reference: List[Dict[str, Any]],
    health: Dict[str, Any],
) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        async with semaphore:
            response = await client.post(
                "/api/experiments/text2cypher/experiment2/full-workflow",
                headers={"X-Experiment-Token": token},
                json={
                    "case_id": case["id"],
                    "question": case["question"],
                    "repeat": repeat,
                },
            )
        response.raise_for_status()
        body = response.json()
        invariant_errors = _response_invariant_errors(
            body,
            health,
            {
                "case_id": case["id"],
                "repeat": repeat,
                "question": case["question"],
                "fewshot_k": 3,
                "llm_validation": True,
                "max_correction_retries": 3,
            },
        )
        trace_errors = _trace_errors(body)
        statement = body.get("statement", "")
        read_only = not validate_no_writes_in_cypher_query(statement)
        successful_execution = bool(
            body.get("executed")
            and not body.get("errors")
            and read_only
            and not body.get("unsafe_executed")
            and not body.get("executed_with_blocking_validation_errors")
        )
        trace = body.get("validation_trace", [])
        initial_pass = bool(trace and trace[0].get("next_action") == "execute_cypher")
        missing_order = bool(case.get("ordered") and "ORDER BY" not in statement.upper())
        semantic_correct = bool(
            successful_execution
            and not missing_order
            and _results_equivalent(
                body.get("records", []),
                reference,
                case["expected_columns"],
                bool(case.get("ordered")),
            )
        )
        return {
            "case_id": case["id"],
            "repeat": repeat,
            "question": case["question"],
            "difficulty": case["difficulty"],
            "domain": case["domain"],
            "query_type": case["query_type"],
            "backend_request_id": body.get("request_id"),
            "initial_cypher": trace[0].get("statement_before", "") if trace else "",
            "final_cypher": statement,
            "errors": body.get("errors", []),
            "records": body.get("records", []),
            "record_count": len(body.get("records", [])),
            "attempts": body.get("attempts", 0),
            "retries": body.get("retries", 0),
            "validation_trace": trace,
            "correction_history": body.get("correction_history", []),
            "steps": body.get("steps", []),
            "executed": bool(body.get("executed")),
            "successful_execution": successful_execution,
            "unsafe_executed": bool(body.get("unsafe_executed")),
            "executed_with_blocking_validation_errors": bool(
                body.get("executed_with_blocking_validation_errors")
            ),
            "initial_validation_pass": initial_pass,
            "rescued_by_correction": bool(not initial_pass and successful_execution),
            "semantic_correct": semantic_correct,
            "missing_required_order": missing_order,
            "http_status": response.status_code,
            "api_error": None,
            "integrity_errors": invariant_errors + trace_errors,
            "backend_total_latency_ms": body.get("total_latency_ms"),
            "driver_roundtrip_latency_ms": round(
                (time.perf_counter() - started) * 1000, 2
            ),
        }
    except Exception as exc:
        return _failed_record(part="b", case=case, repeat=repeat, exc=exc, started=started)


def _integrity_violations(
    results: Dict[str, Any], part_a_count: int, part_b_count: int
) -> List[str]:
    errors: List[str] = []
    part_a = results["part_a_records"]
    part_b = results["part_b_records"]
    if len(part_a) != part_a_count:
        errors.append(f"Part A records {len(part_a)} != {part_a_count}")
    if len(part_b) != part_b_count:
        errors.append(f"Part B records {len(part_b)} != {part_b_count}")
    if len({row["case_id"] for row in part_a}) != part_a_count:
        errors.append("Part A has duplicate or missing case IDs")
    if len({(row["case_id"], row["repeat"]) for row in part_b}) != part_b_count:
        errors.append("Part B has duplicate or missing case/repeat keys")
    all_rows = part_a + part_b
    request_ids = [row.get("backend_request_id") for row in all_rows]
    if any(not value for value in request_ids):
        errors.append("At least one request lacks a backend request ID")
    elif len(set(request_ids)) != len(request_ids):
        errors.append("Backend request IDs are not unique")
    api_errors = [row for row in all_rows if row.get("api_error")]
    if api_errors:
        errors.append(f"There are {len(api_errors)} HTTP/API errors")
    trace_errors = [row for row in all_rows if row.get("integrity_errors")]
    if trace_errors:
        errors.append(f"There are {len(trace_errors)} response/trace invariant failures")
    unsafe = [row for row in all_rows if row.get("unsafe_executed")]
    if unsafe:
        errors.append(f"There are {len(unsafe)} unsafe write executions")
    blocked = [
        row for row in all_rows if row.get("executed_with_blocking_validation_errors")
    ]
    if blocked:
        errors.append(f"There are {len(blocked)} executions with blocking errors")
    if any(row.get("retries", 0) > 3 for row in all_rows):
        errors.append("At least one request exceeded three retries")
    return errors


def _ratio(numerator: int, denominator: int) -> str:
    if not denominator:
        return "0/0（N/A）"
    return f"{numerator}/{denominator}（{numerator / denominator * 100:.2f}%）"


def _percentile(values: Iterable[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _cluster_bootstrap_ci(records: List[Dict[str, Any]]) -> Tuple[float, float]:
    by_case: Dict[str, List[int]] = defaultdict(list)
    for row in records:
        by_case[row["case_id"]].append(int(row["successful_execution"]))
    cases = sorted(by_case)
    rng = random.Random(RANDOMIZATION_SEED)
    samples = []
    for _ in range(20_000):
        draw = [rng.choice(cases) for _ in cases]
        samples.append(
            sum(statistics.mean(by_case[case_id]) for case_id in draw) / len(draw)
        )
    samples.sort()
    return samples[499] * 100, samples[19_499] * 100


def _write_report(results: Dict[str, Any]) -> None:
    part_a = results["part_a_records"]
    part_b = results["part_b_records"]
    faults = [row for row in part_a if row["case_type"] != "clean_control"]
    controls = [row for row in part_a if row["case_type"] == "clean_control"]
    lines = [
        "# 实验二：五层校验、自动修正与Cypher成功执行率",
        "",
        "## 实验边界与证据",
        "",
        "- A组通过本机 FastAPI 将冻结候选注入生产 `validation -> correction -> execution` 节点；B组从自然语言进入生产 Text2Cypher LangGraph。",
        "- 实验驱动器不创建大模型、不调用 DeepSeek；所有模型请求均由后端发出。",
        f"- 协议：`{results['metadata']['protocol']}`；数据集 SHA-256：`{results['metadata']['dataset_sha256']}`。",
        f"- 模型：`{results['metadata']['model']}`；温度：{results['metadata']['temperature']}；Schema SHA-256：`{results['metadata']['schema_sha256']}`。",
        "- 五层顺序：语法、写操作、关系方向、LLM、Schema；最大自动修正次数3；达到上限后禁止强制执行。",
        "",
        "## 实验二A：受控故障注入",
        "",
        "### 检出结果",
        "",
        "| 校验层 | 预期故障 | 首轮检出 | 检出率 | 正确对照误报 |",
        "|---|---:|---:|---:|---:|",
    ]
    for layer in EXPECTED_LAYERS:
        expected_rows = [row for row in faults if layer in row["expected_layers"]]
        detected = sum(layer in row["initially_detected_layers"] for row in expected_rows)
        false_positive = sum(layer in row["initially_detected_layers"] for row in controls)
        lines.append(
            f"| `{layer}` | {len(expected_rows)} | {detected} | "
            f"{detected / len(expected_rows) * 100:.2f}% | {_ratio(false_positive, len(controls))} |"
        )
    full_detection = sum(row["all_expected_layers_detected"] for row in faults)
    semantic_repair = sum(row["semantic_correct"] for row in faults)
    clean_pass = sum(
        not row["clean_false_positive"] and row["semantic_correct"] for row in controls
    )
    direction_rows = [
        row for row in faults if "relationship_direction" in row["expected_layers"]
    ]
    direction_fixed = sum(row["direction_deterministically_corrected"] for row in direction_rows)
    retry_counts = Counter(row["retries"] for row in part_a)
    unsafe_count = sum(row["unsafe_executed"] for row in part_a)
    blocking_count = sum(row["executed_with_blocking_validation_errors"] for row in part_a)
    lines.extend(
        [
            "",
            "### 修正、安全与执行",
            "",
            f"- 首轮覆盖全部预期层：{_ratio(full_detection, len(faults))}。",
            f"- 故障样本最终语义修复：{_ratio(semantic_repair, len(faults))}。",
            f"- 关系方向确定性修正且零LLM重试执行：{_ratio(direction_fixed, len(direction_rows))}。",
            f"- 正确对照无误报且结果正确：{_ratio(clean_pass, len(controls))}。",
            "- 重试分布：" + "，".join(f"{retry}次={retry_counts[retry]}" for retry in range(4)) + "。",
            f"- 危险写查询执行：{unsafe_count}；带阻断校验错误执行：{blocking_count}。",
            f"- HTTP/API错误：{sum(bool(row['api_error']) for row in part_a)}。",
            "",
            "## 实验二B：完整工作流成功执行率",
            "",
        ]
    )
    success = sum(row["successful_execution"] for row in part_b)
    semantic = sum(row["semantic_correct"] for row in part_b)
    api_ok_rows = [row for row in part_b if not row["api_error"]]
    conditional_success = sum(row["successful_execution"] for row in api_ok_rows)
    initial_pass = sum(row["initial_validation_pass"] for row in part_b)
    initial_failures = [row for row in part_b if not row["initial_validation_pass"]]
    rescued = sum(row["rescued_by_correction"] for row in initial_failures)
    ci_low, ci_high = _cluster_bootstrap_ci(part_b)
    by_case = defaultdict(list)
    for row in part_b:
        by_case[row["case_id"]].append(bool(row["successful_execution"]))
    majority = sum(sum(values) >= 2 for values in by_case.values())
    latencies = [
        row["backend_total_latency_ms"]
        for row in part_b
        if row.get("backend_total_latency_ms") is not None
    ]
    lines.extend(
        [
            f"- 主成功执行率（API异常计入分母）：{_ratio(success, len(part_b))}。",
            f"- 排除API异常后的条件执行率：{_ratio(conditional_success, len(api_ok_rows))}。",
            f"- 按题多数票成功率：{_ratio(majority, len(by_case))}。",
            f"- 按40题聚类Bootstrap的执行率95%区间：{ci_low:.2f}%～{ci_high:.2f}%（20,000次，固定种子）。",
            f"- 首轮校验直接通过：{_ratio(initial_pass, len(part_b))}；初轮未通过样本中被自动修正救回：{_ratio(rescued, len(initial_failures))}。",
            f"- 最终语义正确率（与独立参考投影等价）：{_ratio(semantic, len(part_b))}。",
            f"- 后端总耗时：P50={_percentile(latencies, 0.5):.2f}ms，P95={_percentile(latencies, 0.95):.2f}ms。",
            "",
            "### 分轮结果",
            "",
            "| 轮次 | 执行成功率 | 语义正确率 |",
            "|---:|---:|---:|",
        ]
    )
    for repeat in range(1, results["metadata"]["part_b_repeats"] + 1):
        rows = [row for row in part_b if row["repeat"] == repeat]
        lines.append(
            f"| {repeat} | {_ratio(sum(row['successful_execution'] for row in rows), len(rows))} | "
            f"{_ratio(sum(row['semantic_correct'] for row in rows), len(rows))} |"
        )
    lines.extend(
        [
            "",
            "### 分层结果",
            "",
            "| 维度 | 执行成功率 | 语义正确率 |",
            "|---|---:|---:|",
        ]
    )
    for field in ("difficulty", "domain"):
        for value in sorted({row[field] for row in part_b}):
            rows = [row for row in part_b if row[field] == value]
            lines.append(
                f"| {field}=`{value}` | "
                f"{_ratio(sum(row['successful_execution'] for row in rows), len(rows))} | "
                f"{_ratio(sum(row['semantic_correct'] for row in rows), len(rows))} |"
            )

    failure_reasons = Counter()
    for row in part_b:
        if row["successful_execution"]:
            continue
        if row["api_error"]:
            failure_reasons["HTTP/API异常"] += 1
        elif not row["executed"]:
            failure_reasons["校验/重试后未执行"] += 1
        elif row.get("errors"):
            failure_reasons["执行异常"] += 1
        else:
            failure_reasons["安全或完整性条件失败"] += 1
    integrity = results.get("integrity_violations", [])
    point_target = success >= 108
    majority_target = majority >= 36
    safety_target = not any(row["unsafe_executed"] for row in part_a + part_b)
    claim_supported = point_target and majority_target and safety_target and not integrity
    lines.extend(
        [
            "",
            "## 失败与完整性",
            "",
            "- B组执行失败原因："
            + ("；".join(f"{key}{value}次" for key, value in failure_reasons.items()) or "无")
            + "。",
            "- 完整性违规：" + ("；".join(integrity) if integrity else "0项") + "。",
            "- 阶段级耗时未由当前生产 LangGraph 输出，故本报告只给出可审计的后端总耗时，不伪造阶段拆分。",
            "",
            "## 结论",
            "",
            f"- 120次点估计至少108次成功：{'通过' if point_target else '未通过'}。",
            f"- 40题多数票至少36题成功：{'通过' if majority_target else '未通过'}。",
            f"- 五层安全硬约束（危险执行=0）：{'通过' if safety_target else '未通过'}。",
            f"- 因此，简历中“Cypher查询成功执行率达90%”在本实验固定口径下：{'得到支持' if claim_supported else '未得到支持'}。",
            "- 执行成功不等同于语义正确；语义正确率已作为独立次指标完整报告。",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def async_main(args: argparse.Namespace) -> int:
    token = os.environ.get("TEXT2CYPHER_EXPERIMENT_TOKEN", "")
    if not token:
        raise RuntimeError("TEXT2CYPHER_EXPERIMENT_TOKEN is required")
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if dataset.get("status") != "frozen" or protocol.get("status") != "frozen":
        raise RuntimeError("Experiment 2 dataset and protocol must be frozen")
    if protocol.get("dataset", {}).get("sha256") != _sha256(DATASET_PATH):
        raise RuntimeError("Protocol dataset SHA-256 does not match the frozen dataset")
    quality_path = RESULTS_DIR / "text2cypher_experiment2_v1_quality.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    if quality.get("status") != "PASS" or quality.get("dataset_sha256") != _sha256(DATASET_PATH):
        raise RuntimeError("The frozen dataset lacks a matching PASS quality report")

    part_a_cases = dataset["part_a"]["cases"]
    part_b_path = DATASET_PATH.parent / dataset["part_b"]["source_file"]
    part_b_dataset = json.loads(part_b_path.read_text(encoding="utf-8"))
    part_b_cases = part_b_dataset[dataset["part_b"]["case_field"]]
    repeats = dataset["part_b"]["repeats"]
    timeout = httpx.Timeout(connect=15, read=300, write=30, pool=30)
    limits = httpx.Limits(
        max_connections=args.concurrency,
        max_keepalive_connections=args.concurrency,
    )
    async with httpx.AsyncClient(
        base_url=args.base_url,
        timeout=timeout,
        limits=limits,
        trust_env=False,
    ) as client:
        health = await _health(client, token)
        print(
            f"[backend-exp2] preflight PASS model={health['model']} "
            f"schema={health['schema_sha256']}",
            flush=True,
        )
        if args.preflight_only:
            return 0
        results = _load_or_create(dataset, args.base_url, health)
        graph = _graph()
        try:
            part_a_references = {
                case["id"]: graph.query(case["corrected_reference_cypher"])
                for case in part_a_cases
            }
            part_b_references = {
                case["id"]: graph.query(case["reference_cypher"])
                for case in part_b_cases
            }
        finally:
            graph.close()

        semaphore = asyncio.Semaphore(args.concurrency)
        existing_a = {row["case_id"] for row in results["part_a_records"]}
        ordered_a = list(part_a_cases)
        random.Random(RANDOMIZATION_SEED).shuffle(ordered_a)
        jobs_a = [case for case in ordered_a if case["id"] not in existing_a]
        if args.max_new_a is not None:
            jobs_a = jobs_a[: args.max_new_a]
        tasks_a = [
            asyncio.create_task(
                _run_part_a_request(
                    client,
                    token,
                    semaphore,
                    case,
                    part_a_references[case["id"]],
                    health,
                )
            )
            for case in jobs_a
        ]
        completed = 0
        for future in asyncio.as_completed(tasks_a):
            record = await future
            results["part_a_records"].append(record)
            _save(results)
            completed += 1
            print(
                f"[backend-exp2:A] {completed}/{len(tasks_a)} {record['case_id']} "
                f"detected={record.get('initially_detected_layers')} "
                f"retries={record.get('retries')} executed={record.get('executed')}",
                flush=True,
            )

        existing_b = {
            (row["case_id"], row["repeat"]) for row in results["part_b_records"]
        }
        ordered_b = [
            (case, repeat)
            for repeat in range(1, repeats + 1)
            for case in part_b_cases
        ]
        random.Random(RANDOMIZATION_SEED).shuffle(ordered_b)
        jobs_b = [
            (case, repeat)
            for case, repeat in ordered_b
            if (case["id"], repeat) not in existing_b
        ]
        if args.max_new_b is not None:
            jobs_b = jobs_b[: args.max_new_b]
        tasks_b = [
            asyncio.create_task(
                _run_part_b_request(
                    client,
                    token,
                    semaphore,
                    case,
                    repeat,
                    part_b_references[case["id"]],
                    health,
                )
            )
            for case, repeat in jobs_b
        ]
        completed = 0
        for future in asyncio.as_completed(tasks_b):
            record = await future
            results["part_b_records"].append(record)
            _save(results)
            completed += 1
            print(
                f"[backend-exp2:B] {completed}/{len(tasks_b)} {record['case_id']} "
                f"r{record['repeat']} retries={record.get('retries')} "
                f"success={record.get('successful_execution')}",
                flush=True,
            )

    expected_a = len(part_a_cases)
    expected_b = len(part_b_cases) * repeats
    if (
        len(results["part_a_records"]) < expected_a
        or len(results["part_b_records"]) < expected_b
    ):
        results["integrity_violations"] = []
        _save(results)
        print(
            "checkpoint_pending="
            f"A:{len(results['part_a_records'])}/{expected_a},"
            f"B:{len(results['part_b_records'])}/{expected_b}",
            flush=True,
        )
        return 0

    results["integrity_violations"] = _integrity_violations(
        results,
        expected_a,
        expected_b,
    )
    results["metadata"]["completed_at"] = _utc_now()
    _save(results)
    _write_report(results)
    print(f"integrity_violations={results['integrity_violations']}", flush=True)
    print(f"raw_results={RAW_RESULTS_PATH}", flush=True)
    print(f"report={REPORT_PATH}", flush=True)
    return 0 if not results["integrity_violations"] else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--max-new-a", type=int)
    parser.add_argument("--max-new-b", type=int)
    return asyncio.run(async_main(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
