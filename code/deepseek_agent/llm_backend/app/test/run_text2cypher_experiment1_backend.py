"""Run Experiment 1 through the local FastAPI backend only.

The system under test is the token-protected backend endpoint. This driver does
not instantiate an LLM or call any provider API. It sends a frozen question set
over HTTP, queries Neo4j only for the independent reference oracle, and writes a
new result artifact that does not reuse the former component-level experiment.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx
from langchain_neo4j import Neo4jGraph

from app.core.config import settings


PROTOCOL = "text2cypher-exp1-backend-v2"
TEST_DIR = Path(__file__).parent
DATASET_PATH = TEST_DIR / "data" / "text2cypher_experiment1_v2.json"
RESULTS_DIR = TEST_DIR / "results"
RAW_RESULTS_PATH = RESULTS_DIR / "text2cypher_experiment1_backend_v2_raw.json"
REPORT_PATH = RESULTS_DIR / "text2cypher_experiment1_backend_v2_report.md"
RANDOMIZATION_SEED = 20260910


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dataset_sha256() -> str:
    return hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest()


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


def _make_graph() -> Neo4jGraph:
    return Neo4jGraph(
        url=settings.NEO4J_URL,
        username=settings.NEO4J_USERNAME,
        password=settings.NEO4J_PASSWORD,
        database=settings.NEO4J_DATABASE,
    )


def _new_results(
    dataset: Dict[str, Any], repeats: int, base_url: str, health: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "metadata": {
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "completed_at": None,
            "experiment": "Experiment 1: production-node Few-shot A/B via FastAPI",
            "protocol": PROTOCOL,
            "system_under_test": "local FastAPI backend -> production Text2Cypher generation node",
            "provider_call_owner": "backend",
            "driver_provider_sdk_calls": 0,
            "base_url": base_url,
            "dataset": str(DATASET_PATH),
            "dataset_version": dataset["version"],
            "dataset_sha256": _dataset_sha256(),
            "database_snapshot": dataset["database_snapshot"],
            "model": health["model"],
            "temperature": health["temperature"],
            "schema_version": health["schema_version"],
            "schema_strategy": health["schema_strategy"],
            "schema_sha256": health["schema_sha256"],
            "variants": {"no_fewshot": 0, "with_fewshot": 3},
            "repeats": repeats,
            "randomization_seed": RANDOMIZATION_SEED,
            "semantic_match": "required_reference_projection_equivalence",
        },
        "records": [],
    }


def _save_results(results: Dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results["metadata"]["updated_at"] = _utc_now()
    temporary = RAW_RESULTS_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(RAW_RESULTS_PATH)


def _load_or_create_results(
    dataset: Dict[str, Any], repeats: int, base_url: str, health: Dict[str, Any]
) -> Dict[str, Any]:
    if not RAW_RESULTS_PATH.exists():
        results = _new_results(dataset, repeats, base_url, health)
        _save_results(results)
        return results
    results = json.loads(RAW_RESULTS_PATH.read_text(encoding="utf-8"))
    metadata = results.get("metadata", {})
    required = {
        "protocol": PROTOCOL,
        "dataset_sha256": _dataset_sha256(),
        "model": health["model"],
        "temperature": health["temperature"],
        "schema_version": health["schema_version"],
        "schema_strategy": health["schema_strategy"],
        "schema_sha256": health["schema_sha256"],
        "repeats": repeats,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in required.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Existing checkpoint is incompatible: {mismatches}")
    return results


async def _health(
    client: httpx.AsyncClient, token: str
) -> Dict[str, Any]:
    response = await client.get(
        "/api/experiments/text2cypher/health",
        headers={"X-Experiment-Token": token},
    )
    response.raise_for_status()
    health = response.json()
    if health.get("protocol") != PROTOCOL:
        raise RuntimeError(f"Unexpected backend experiment protocol: {health}")
    if health.get("execution_boundary") != "fastapi-to-production-generation-node":
        raise RuntimeError(f"Backend did not confirm the production boundary: {health}")
    if health.get("allowed_fewshot_k") != [0, 3]:
        raise RuntimeError(f"Backend does not expose the required A/B variants: {health}")
    return health


async def _one_request(
    client: httpx.AsyncClient,
    token: str,
    semaphore: asyncio.Semaphore,
    case: Dict[str, Any],
    variant: str,
    fewshot_k: int,
    repeat: int,
    reference: List[Dict[str, Any]],
    expected_backend: Dict[str, Any],
) -> Dict[str, Any]:
    started = time.perf_counter()
    payload = {
        "question": case["question"],
        "fewshot_k": fewshot_k,
        "case_id": case["id"],
        "repeat": repeat,
    }
    try:
        async with semaphore:
            response = await client.post(
                "/api/experiments/text2cypher/generate",
                headers={"X-Experiment-Token": token},
                json=payload,
            )
        response.raise_for_status()
        body = response.json()
        invariant_fields = {
            "protocol": PROTOCOL,
            "case_id": case["id"],
            "repeat": repeat,
            "question": case["question"],
            "fewshot_k": fewshot_k,
            "model": expected_backend["model"],
            "temperature": expected_backend["temperature"],
            "schema_version": expected_backend["schema_version"],
        }
        violations = {
            key: (body.get(key), expected)
            for key, expected in invariant_fields.items()
            if body.get(key) != expected
        }
        if violations:
            raise RuntimeError(f"Backend response invariant failure: {violations}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(body.get("schema_sha256", ""))):
            raise RuntimeError("Backend response has an invalid prompt Schema hash")
        if fewshot_k == 0 and body.get("retrieved_examples"):
            raise RuntimeError("No-Few-shot response unexpectedly contains examples")
        if fewshot_k == 3 and len(body.get("retrieved_examples", [])) != 3:
            raise RuntimeError("Few-shot response did not contain exactly three examples")

        statement = body["generated_cypher"]
        records = body["records"]
        validation = body["validation"]
        missing_required_order = bool(
            case.get("ordered") and "ORDER BY" not in statement.upper()
        )
        semantic_correct = bool(
            validation.get("executed")
            and not missing_required_order
            and _results_equivalent(
                records,
                reference,
                case.get("expected_columns", []),
                case.get("ordered", False),
            )
        )
        return {
            "case_id": case["id"],
            "variant": variant,
            "fewshot_k": fewshot_k,
            "repeat": repeat,
            "question": case["question"],
            "difficulty": case["difficulty"],
            "domain": case["domain"],
            "backend_request_id": body["request_id"],
            "retrieved_examples": body["retrieved_examples"],
            "retrieval_domain": body["retrieval_domain"],
            "retrieval_query_type": body["retrieval_query_type"],
            "prompt_schema_sha256": body["schema_sha256"],
            "generated_cypher": statement,
            "validation": validation,
            "record_count": len(records),
            "semantic_correct": semantic_correct,
            "missing_required_order": missing_required_order,
            "http_status": response.status_code,
            "api_error": None,
            "backend_generation_latency_ms": body["generation_latency_ms"],
            "backend_total_latency_ms": body["total_latency_ms"],
            "driver_roundtrip_latency_ms": round(
                (time.perf_counter() - started) * 1000, 2
            ),
        }
    except Exception as exc:
        return {
            "case_id": case["id"],
            "variant": variant,
            "fewshot_k": fewshot_k,
            "repeat": repeat,
            "question": case["question"],
            "difficulty": case["difficulty"],
            "domain": case["domain"],
            "backend_request_id": None,
            "retrieved_examples": [],
            "generated_cypher": "",
            "validation": {
                "read_only": False,
                "syntax_valid": False,
                "schema_valid": False,
                "executed": False,
                "failure_type": "backend_http",
                "errors": [str(exc)],
            },
            "record_count": 0,
            "semantic_correct": False,
            "missing_required_order": False,
            "http_status": getattr(getattr(exc, "response", None), "status_code", None),
            "api_error": str(exc),
            "backend_generation_latency_ms": None,
            "backend_total_latency_ms": None,
            "driver_roundtrip_latency_ms": round(
                (time.perf_counter() - started) * 1000, 2
            ),
        }


def _bootstrap_ci(records: List[Dict[str, Any]]) -> Tuple[float, float]:
    lookup = {
        (row["case_id"], row["repeat"], row["variant"]): int(
            row["semantic_correct"]
        )
        for row in records
    }
    cases = sorted({row["case_id"] for row in records})
    repeats = sorted({row["repeat"] for row in records})
    deltas = {
        case_id: sum(
            lookup[(case_id, repeat, "with_fewshot")]
            - lookup[(case_id, repeat, "no_fewshot")]
            for repeat in repeats
        )
        / len(repeats)
        for case_id in cases
    }
    rng = random.Random(RANDOMIZATION_SEED)
    samples = []
    for _ in range(20_000):
        draw = [rng.choice(cases) for _ in cases]
        samples.append(sum(deltas[case_id] for case_id in draw) / len(draw))
    samples.sort()
    return samples[499] * 100, samples[19_499] * 100


def _integrity_check(
    results: Dict[str, Any], case_count: int, repeats: int
) -> None:
    records = results["records"]
    expected = case_count * repeats * 2
    if len(records) != expected:
        raise RuntimeError(f"Expected {expected} records, found {len(records)}")
    keys = {
        (row["case_id"], row["variant"], row["repeat"]) for row in records
    }
    if len(keys) != expected:
        raise RuntimeError("Experiment contains duplicate or missing A/B keys")
    request_ids = [row["backend_request_id"] for row in records]
    if any(not request_id for request_id in request_ids):
        raise RuntimeError("At least one record lacks a backend request id")
    if len(set(request_ids)) != expected:
        raise RuntimeError("Backend request ids are not unique")
    errors = [row for row in records if row["api_error"]]
    if errors:
        raise RuntimeError(f"Experiment contains {len(errors)} backend/API errors")
    no_fewshot = [row for row in records if row["variant"] == "no_fewshot"]
    with_fewshot = [row for row in records if row["variant"] == "with_fewshot"]
    if any(row["retrieved_examples"] for row in no_fewshot):
        raise RuntimeError("No-Few-shot arm contains retrieved examples")
    if any(len(row["retrieved_examples"]) != 3 for row in with_fewshot):
        raise RuntimeError("Few-shot arm does not consistently contain three examples")


def _ratio(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator}（{numerator / denominator * 100:.2f}%）"


def _write_report(results: Dict[str, Any]) -> None:
    records = results["records"]
    grouped = {
        variant: [row for row in records if row["variant"] == variant]
        for variant in ("no_fewshot", "with_fewshot")
    }
    no_correct = sum(row["semantic_correct"] for row in grouped["no_fewshot"])
    with_correct = sum(row["semantic_correct"] for row in grouped["with_fewshot"])
    no_total = len(grouped["no_fewshot"])
    with_total = len(grouped["with_fewshot"])
    no_rate = no_correct / no_total * 100
    with_rate = with_correct / with_total * 100
    improvement = with_rate - no_rate
    ci_low, ci_high = _bootstrap_ci(records)

    lines = [
        "# 实验一：后端生产路径 Few-shot A/B 实验报告",
        "",
        "## 实验边界",
        "",
        "- 本报告完全替代此前的组件级实验一，不复用此前实验一的模型输出或统计结果。",
        "- 所有测试问题均通过本机 FastAPI HTTP 接口进入后端。",
        "- 后端调用项目实际的 `create_text2cypher_generation_node`；实验驱动器不创建大模型，也不调用 DeepSeek API。",
        "- 后端完成 Few-shot 向量检索、Prompt 生成、模型调用、Cypher 校验与 Neo4j 执行；驱动器只使用参考查询作独立语义判分。",
        "",
        "## 固定条件",
        "",
        f"- 协议：`{results['metadata']['protocol']}`",
        f"- 数据集：{results['metadata']['dataset_version']}（SHA-256 `{results['metadata']['dataset_sha256']}`）",
        f"- 测试问题：{len({row['case_id'] for row in records})}条",
        f"- 重复次数：{results['metadata']['repeats']}次/配置",
        f"- 模型：`{results['metadata']['model']}`",
        f"- 生产温度：{results['metadata']['temperature']}",
        f"- Schema SHA-256：`{results['metadata']['schema_sha256']}`",
        f"- 随机化种子：{results['metadata']['randomization_seed']}",
        "- 唯一实验变量：`fewshot_k=0` 与 `fewshot_k=3`。",
        "",
        "## 结果",
        "",
        "| 配置 | 语义正确率 | 可执行率 | API成功率 |",
        "|---|---:|---:|---:|",
    ]
    for variant, title in (
        ("no_fewshot", "无 Few-shot（k=0）"),
        ("with_fewshot", "有 Few-shot（k=3）"),
    ):
        rows = grouped[variant]
        correct = sum(row["semantic_correct"] for row in rows)
        executed = sum(row["validation"]["executed"] for row in rows)
        api_ok = sum(not row["api_error"] and row["http_status"] == 200 for row in rows)
        lines.append(
            f"| {title} | {_ratio(correct, len(rows))} | "
            f"{_ratio(executed, len(rows))} | {_ratio(api_ok, len(rows))} |"
        )

    lines.extend(
        [
            "",
            f"- Few-shot准确率变化：{no_rate:.2f}% → {with_rate:.2f}%（{improvement:+.2f}个百分点）。",
            f"- 按40个问题聚类Bootstrap 95%置信区间：{ci_low:.2f}～{ci_high:.2f}个百分点（20,000次，固定种子）。",
            "",
            "## 分轮结果",
            "",
            "| 配置 | 第1轮 | 第2轮 | 第3轮 |",
            "|---|---:|---:|---:|",
        ]
    )
    repeats = range(1, results["metadata"]["repeats"] + 1)
    for variant, title in (
        ("no_fewshot", "无 Few-shot"),
        ("with_fewshot", "有 Few-shot"),
    ):
        cells = []
        for repeat in repeats:
            rows = [row for row in grouped[variant] if row["repeat"] == repeat]
            cells.append(_ratio(sum(row["semantic_correct"] for row in rows), len(rows)))
        lines.append(f"| {title} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## 分层结果",
            "",
            "| 维度 | 无 Few-shot | 有 Few-shot | 变化 |",
            "|---|---:|---:|---:|",
        ]
    )
    for field, labels in (
        ("difficulty", ("easy", "medium", "hard")),
        ("domain", ("app_catalog", "legacy_graph")),
    ):
        for label in labels:
            no_rows = [row for row in grouped["no_fewshot"] if row[field] == label]
            with_rows = [row for row in grouped["with_fewshot"] if row[field] == label]
            no_hits = sum(row["semantic_correct"] for row in no_rows)
            with_hits = sum(row["semantic_correct"] for row in with_rows)
            no_layer_rate = no_hits / len(no_rows) * 100
            with_layer_rate = with_hits / len(with_rows) * 100
            lines.append(
                f"| {field}={label} | {_ratio(no_hits, len(no_rows))} | "
                f"{_ratio(with_hits, len(with_rows))} | "
                f"{with_layer_rate - no_layer_rate:+.2f}个百分点 |"
            )

    paired = defaultdict(int)
    paired_rows: Dict[Tuple[str, int], Dict[str, bool]] = defaultdict(dict)
    for row in records:
        paired_rows[(row["case_id"], row["repeat"])][row["variant"]] = bool(
            row["semantic_correct"]
        )
    for values in paired_rows.values():
        paired[(values["no_fewshot"], values["with_fewshot"])] += 1
    no_execution_failures = no_total - sum(
        row["validation"]["executed"] for row in grouped["no_fewshot"]
    )
    with_execution_failures = with_total - sum(
        row["validation"]["executed"] for row in grouped["with_fewshot"]
    )

    reproduced = no_rate == 65.0 and with_rate == 85.0
    if ci_low > 0:
        confidence_conclusion = "95%置信区间完全高于0，支持Few-shot存在稳定正向提升。"
    elif ci_high < 0:
        confidence_conclusion = "95%置信区间完全低于0，表明Few-shot造成稳定负向影响。"
    else:
        confidence_conclusion = "95%置信区间跨过0，尚不能确认Few-shot存在稳定提升。"

    domain_observations = []
    for domain in ("app_catalog", "legacy_graph"):
        no_rows = [
            row for row in grouped["no_fewshot"] if row["domain"] == domain
        ]
        with_rows = [
            row for row in grouped["with_fewshot"] if row["domain"] == domain
        ]
        no_domain_rate = (
            sum(row["semantic_correct"] for row in no_rows) / len(no_rows) * 100
        )
        with_domain_rate = (
            sum(row["semantic_correct"] for row in with_rows)
            / len(with_rows)
            * 100
        )
        domain_observations.append(
            f"`{domain}`：{no_domain_rate:.2f}% → {with_domain_rate:.2f}%"
        )
    lines.extend(
        [
            "",
            "## 配对与失败观察",
            "",
            f"- 120个同题同轮配对中：两组均正确{paired[(True, True)]}组、均错误{paired[(False, False)]}组、仅Few-shot正确{paired[(False, True)]}组、仅无Few-shot正确{paired[(True, False)]}组。",
            f"- 无Few-shot有{no_execution_failures}次未执行成功；有Few-shot有{with_execution_failures}次未执行成功。",
            "- 分领域变化：" + "；".join(domain_observations) + "。",
            "",
            "## 结论",
            "",
            f"- 简历中的“65%→85%”：{'得到复现' if reproduced else '未得到复现'}。",
            f"- {confidence_conclusion}",
            "- 该结论只描述首次Cypher生成能力，不包含验证、自动修正和重试带来的结果。",
            "- 每条原始记录均保存后端请求ID、检索示例、生成Cypher、校验状态、执行状态和耗时，可追溯到真实HTTP请求。",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def async_main(args: argparse.Namespace) -> int:
    token = os.environ.get("TEXT2CYPHER_EXPERIMENT_TOKEN", "")
    if not token:
        raise RuntimeError("TEXT2CYPHER_EXPERIMENT_TOKEN is required")
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    cases = dataset["generation_cases"]
    if len(cases) != 40:
        raise RuntimeError(f"Frozen benchmark must contain 40 cases, found {len(cases)}")

    timeout = httpx.Timeout(connect=10, read=180, write=30, pool=30)
    limits = httpx.Limits(
        max_connections=args.concurrency, max_keepalive_connections=args.concurrency
    )
    async with httpx.AsyncClient(
        base_url=args.base_url,
        timeout=timeout,
        limits=limits,
        trust_env=False,
    ) as client:
        health = await _health(client, token)
        results = _load_or_create_results(dataset, args.repeats, args.base_url, health)
        existing = {
            (row["case_id"], row["variant"], row["repeat"])
            for row in results["records"]
        }
        graph = _make_graph()
        try:
            references = {
                case["id"]: graph.query(case["reference_cypher"]) for case in cases
            }
        finally:
            graph.close()

        jobs: List[Tuple[Dict[str, Any], str, int, int]] = []
        for repeat in range(1, args.repeats + 1):
            for case in cases:
                for variant, fewshot_k in (("no_fewshot", 0), ("with_fewshot", 3)):
                    if (case["id"], variant, repeat) not in existing:
                        jobs.append((case, variant, fewshot_k, repeat))
        random.Random(RANDOMIZATION_SEED).shuffle(jobs)
        semaphore = asyncio.Semaphore(args.concurrency)
        tasks = [
            asyncio.create_task(
                _one_request(
                    client,
                    token,
                    semaphore,
                    case,
                    variant,
                    fewshot_k,
                    repeat,
                    references[case["id"]],
                    health,
                )
            )
            for case, variant, fewshot_k, repeat in jobs
        ]
        completed = 0
        for future in asyncio.as_completed(tasks):
            record = await future
            results["records"].append(record)
            _save_results(results)
            completed += 1
            print(
                f"[backend-exp1] {completed}/{len(tasks)} "
                f"{record['case_id']} {record['variant']} r{record['repeat']}",
                flush=True,
            )

    _integrity_check(results, len(cases), args.repeats)
    results["metadata"]["completed_at"] = _utc_now()
    _save_results(results)
    _write_report(results)
    print(f"raw_results={RAW_RESULTS_PATH}", flush=True)
    print(f"report={REPORT_PATH}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=4)
    return asyncio.run(async_main(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
