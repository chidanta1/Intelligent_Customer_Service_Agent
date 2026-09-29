"""Run the frozen full-agent Experiment 3 through its backend endpoint."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import re
import statistics
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


TEST_ROOT = Path(__file__).resolve().parent
BACKEND_ROOT = TEST_ROOT.parents[1]
DATASET_PATH = TEST_ROOT / "data" / "agent_experiment3_v1.json"
PROTOCOL_PATH = TEST_ROOT / "data" / "agent_experiment3_protocol_v1.json"
MANIFEST_PATH = TEST_ROOT / "results" / "agent_experiment3_preflight_manifest_v1.json"
RAW_PATH = TEST_ROOT / "results" / "agent_experiment3_backend_v1_raw.json"
REPORT_PATH = TEST_ROOT / "results" / "agent_experiment3_backend_v1_report.md"
GRAPHRAG_INPUT = BACKEND_ROOT / "app" / "graphrag" / "data" / "input"
WARMUPS = [
    ("WARMUP_STRUCTURED", "目录中欧普智能吸顶灯的库存是多少？"),
    ("WARMUP_GRAPHRAG", "智能门锁的指纹传感器平时应该怎样维护？"),
    ("WARMUP_MIXED", "鹿客智能锁多少钱？智能门锁日常如何维护？"),
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).lower()
    return re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def verify_local_freeze(dataset: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        raise RuntimeError(
            "Preflight manifest is missing. Build GraphRAG, validate, then run "
            "validate_agent_experiment3_v1.py --neo4j --require-index --freeze."
        )
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("status") != "ready_for_formal_run":
        raise RuntimeError(f"Preflight manifest is not ready: {manifest.get('status')}")
    if dataset.get("status") != "frozen" or dataset.get("version") != "1.0.0":
        raise RuntimeError("Dataset must have status=frozen and version=1.0.0")
    checks = {
        "dataset_sha256": sha256_file(DATASET_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
    }
    for key, current in checks.items():
        if manifest.get(key) != current:
            raise RuntimeError(f"Frozen {key} changed")
    for filename, expected in manifest.get("source_sha256", {}).items():
        current = sha256_file(GRAPHRAG_INPUT / filename)
        if current != expected:
            raise RuntimeError(f"Frozen source changed: {filename}")
    for relative, expected in manifest.get("runtime_sha256", {}).items():
        current = sha256_file(BACKEND_ROOT / relative)
        if current != expected:
            raise RuntimeError(f"Frozen runtime file changed: {relative}")
    for distribution, expected in manifest.get("runtime_versions", {}).items():
        try:
            current = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            current = "not-installed"
        if current != expected:
            raise RuntimeError(
                f"Frozen dependency changed: {distribution}={current}, expected={expected}"
            )
    if manifest.get("protocol") != protocol.get("protocol"):
        raise RuntimeError("Manifest protocol version mismatch")
    return manifest


def score_response(case: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    answer = response.get("final_answer", "")
    normalized_answer = normalize(answer)
    fact_results = []
    for fact in case.get("answer_facts", []):
        matched = next(
            (
                alternative
                for alternative in fact["any_of"]
                if normalize(alternative) in normalized_answer
            ),
            None,
        )
        fact_results.append(
            {
                "id": fact["id"],
                "origin": fact["origin"],
                "passed": matched is not None,
                "matched": matched,
            }
        )
    expected_tools = set(case.get("expected_tools", []))
    selected_tools = set(response.get("selected_tools", []))
    tool_correct = selected_tools == expected_tools
    route_correct = bool(response.get("outer_route_passed"))
    workflow_correct = response.get("research_status") == "completed"
    facts_correct = all(item["passed"] for item in fact_results)
    unsafe_write = False
    tool_errors: list[str] = []
    for output in response.get("tool_outputs", []):
        statement = str(output.get("statement", ""))
        if re.search(
            r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD\s+CSV)\b",
            statement,
            flags=re.IGNORECASE,
        ):
            unsafe_write = True
        tool_errors.extend(str(error) for error in output.get("errors", []) if error)
    correct = (
        route_correct
        and workflow_correct
        and tool_correct
        and facts_correct
        and not unsafe_write
    )
    return {
        "correct": correct,
        "route_correct": route_correct,
        "workflow_correct": workflow_correct,
        "tool_correct": tool_correct,
        "expected_tools": sorted(expected_tools),
        "selected_tools": sorted(selected_tools),
        "facts_correct": facts_correct,
        "fact_results": fact_results,
        "unsafe_write": unsafe_write,
        "tool_errors": tool_errors,
    }


def new_results(manifest: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    return {
        "metadata": {
            "experiment": "Experiment 3: production full-agent QA and latency",
            "protocol": protocol["protocol"],
            "started_at": datetime.now(timezone.utc).isoformat(),
            "manifest": manifest,
            "dataset": str(DATASET_PATH),
            "protocol_file": str(PROTOCOL_PATH),
        },
        "health": None,
        "warmups": [],
        "records": [],
    }


def save_results(results: dict[str, Any]) -> None:
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = RAW_PATH.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(RAW_PATH)


def load_or_initialize(
    manifest: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    if not RAW_PATH.is_file():
        return new_results(manifest, protocol)
    results = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    metadata = results.get("metadata", {})
    if metadata.get("protocol") != protocol["protocol"]:
        raise RuntimeError("Existing raw result uses another protocol")
    old_manifest = metadata.get("manifest", {})
    for key in [
        "dataset_sha256",
        "protocol_sha256",
        "neo4j_schema_sha256",
        "neo4j_reference_snapshot_sha256",
        "graphrag_index_sha256",
    ]:
        if old_manifest.get(key) != manifest.get(key):
            raise RuntimeError(f"Cannot resume: frozen {key} differs")
    return results


def aggregate(results: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
    records = results["records"]
    latencies = [float(record["total_latency_ms"]) for record in records]
    correct = sum(bool(record["score"]["correct"]) for record in records)
    tool_correct = sum(bool(record["score"]["tool_correct"]) for record in records)
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    case_map = {case["case_id"]: case for case in dataset["cases"]}
    for record in records:
        by_case[record["case_id"]].append(record)
        by_category[case_map[record["case_id"]]["category"]].append(record)
    majority = sum(
        sum(bool(item["score"]["correct"]) for item in case_records) >= 2
        for case_records in by_case.values()
    )
    category_metrics = {}
    for category, category_records in sorted(by_category.items()):
        category_latency = [float(item["total_latency_ms"]) for item in category_records]
        category_metrics[category] = {
            "requests": len(category_records),
            "correct": sum(item["score"]["correct"] for item in category_records),
            "accuracy": sum(item["score"]["correct"] for item in category_records)
            / len(category_records),
            "mean_latency_ms": statistics.fmean(category_latency),
        }
    return {
        "requests": len(records),
        "correct": correct,
        "request_accuracy": correct / len(records) if records else 0,
        "tool_selection_accuracy": tool_correct / len(records) if records else 0,
        "majority_correct_cases": majority,
        "case_majority_accuracy": majority / len(by_case) if by_case else 0,
        "mean_latency_ms": statistics.fmean(latencies) if latencies else math.nan,
        "median_latency_ms": statistics.median(latencies) if latencies else math.nan,
        "p95_latency_ms": percentile(latencies, 0.95),
        "max_latency_ms": max(latencies) if latencies else math.nan,
        "category_metrics": category_metrics,
    }


def write_report(results: dict[str, Any], dataset: dict[str, Any]) -> None:
    metrics = aggregate(results, dataset)
    complete = metrics["requests"] == 150
    accuracy_pass = complete and metrics["request_accuracy"] >= 0.88
    majority_pass = complete and metrics["case_majority_accuracy"] >= 0.88
    latency_pass = complete and metrics["mean_latency_ms"] <= 3500
    failed = [record for record in results["records"] if not record["score"]["correct"]]
    lines = [
        "# 实验三：全链路问答准确率与响应时间",
        "",
        f"- 状态：{'完成' if complete else '未完成'}（{metrics['requests']}/150）",
        f"- 请求级整体问答准确率：{metrics['correct']}/{metrics['requests']}（{metrics['request_accuracy']:.2%}），88%目标：{'通过' if accuracy_pass else '未通过'}",
        f"- 按题三轮多数通过：{metrics['majority_correct_cases']}/50（{metrics['case_majority_accuracy']:.2%}），44/50目标：{'通过' if majority_pass else '未通过'}",
        f"- 工具选择准确率：{metrics['tool_selection_accuracy']:.2%}",
        f"- 平均端到端响应时间：{metrics['mean_latency_ms'] / 1000:.3f}s，3.5s目标：{'通过' if latency_pass else '未通过'}",
        f"- P50/P95/最大响应时间：{metrics['median_latency_ms'] / 1000:.3f}s / {metrics['p95_latency_ms'] / 1000:.3f}s / {metrics['max_latency_ms'] / 1000:.3f}s",
        "",
        "## 分类结果",
        "",
        "| 分类 | 请求数 | 正确数 | 准确率 | 平均耗时(s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for category, item in metrics["category_metrics"].items():
        lines.append(
            f"| {category} | {item['requests']} | {item['correct']} | "
            f"{item['accuracy']:.2%} | {item['mean_latency_ms'] / 1000:.3f} |"
        )
    lines.extend(["", "## 失败样例", ""])
    if not failed:
        lines.append("无。")
    else:
        for record in failed[:20]:
            score = record["score"]
            missed = [item["id"] for item in score["fact_results"] if not item["passed"]]
            lines.append(
                f"- {record['case_id']} 第{record['repeat']}轮："
                f"route={score['route_correct']}，tool={score['tool_correct']}，"
                f"workflow={score['workflow_correct']}，缺失事实={missed}"
            )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "只有在150个计分请求全部完成后才可引用最终结论；未完成报告不得用于简历指标背书。",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def main_async(args: argparse.Namespace) -> None:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    manifest = verify_local_freeze(dataset, protocol)
    token = args.token or os.environ.get("EXPERIMENT_API_TOKEN", "")
    if not token:
        raise RuntimeError("Provide --token or EXPERIMENT_API_TOKEN")
    headers = {"x-experiment-token": token}
    timeout = httpx.Timeout(100.0)
    async with httpx.AsyncClient(base_url=args.base_url, headers=headers, timeout=timeout) as client:
        health_response = await client.get("/api/experiments/agent/experiment3/health")
        health_response.raise_for_status()
        health = health_response.json()
        if health.get("status") != "ok" or health.get("protocol") != protocol["protocol"]:
            raise RuntimeError(f"Backend is not ready: {health}")
        if health.get("schema_sha256") != manifest.get("neo4j_schema_sha256"):
            raise RuntimeError("Backend Neo4j schema differs from frozen manifest")
        if health.get("neo4j_reference_snapshot_sha256") != manifest.get(
            "neo4j_reference_snapshot_sha256"
        ):
            raise RuntimeError("Backend Neo4j reference data differs from frozen manifest")
        if health.get("runtime_config") != manifest.get("runtime_config"):
            raise RuntimeError("Backend runtime configuration differs from frozen manifest")
        if health.get("graphrag", {}).get("index_sha256") != manifest.get(
            "graphrag_index_sha256"
        ):
            raise RuntimeError("Backend GraphRAG index differs from frozen manifest")
        if args.preflight_only:
            print(json.dumps(health, ensure_ascii=False, indent=2))
            return

        results = load_or_initialize(manifest, protocol)
        results["health"] = health
        if not results["warmups"]:
            for case_id, question in WARMUPS:
                started = datetime.now(timezone.utc).isoformat()
                response = await client.post(
                    "/api/experiments/agent/experiment3/full-workflow",
                    json={"case_id": case_id, "question": question, "repeat": 1},
                )
                response.raise_for_status()
                results["warmups"].append({"started_at": started, **response.json()})
                save_results(results)

        existing = {(item["case_id"], item["repeat"]) for item in results["records"]}
        for repeat in range(1, 4):
            for case in dataset["cases"]:
                key = (case["case_id"], repeat)
                if key in existing:
                    continue
                request_started = time.perf_counter()
                try:
                    response = await client.post(
                        "/api/experiments/agent/experiment3/full-workflow",
                        json={
                            "case_id": case["case_id"],
                            "question": case["question"],
                            "repeat": repeat,
                        },
                    )
                    response.raise_for_status()
                    payload = response.json()
                    if payload.get("schema_sha256") != manifest.get(
                        "neo4j_schema_sha256"
                    ):
                        raise RuntimeError("Schema hash drift during run")
                    if payload.get("graphrag_index_sha256") != manifest.get(
                        "graphrag_index_sha256"
                    ):
                        raise RuntimeError("GraphRAG index hash drift during run")
                except (httpx.HTTPError, json.JSONDecodeError) as exc:
                    payload = {
                        "protocol": protocol["protocol"],
                        "request_id": None,
                        "case_id": case["case_id"],
                        "repeat": repeat,
                        "question": case["question"],
                        "outer_route_passed": False,
                        "research_status": "api_error",
                        "selected_tools": [],
                        "tool_outputs": [],
                        "workflow_steps": [],
                        "node_timings": [],
                        "final_answer": "",
                        "total_latency_ms": round(
                            (time.perf_counter() - request_started) * 1000, 2
                        ),
                        "api_error": f"{type(exc).__name__}: {exc}",
                    }
                score = score_response(case, payload)
                record = {**payload, "category": case["category"], "score": score}
                results["records"].append(record)
                save_results(results)
                write_report(results, dataset)
                print(
                    f"[{len(results['records']):03d}/150] {case['case_id']} "
                    f"repeat={repeat} correct={score['correct']} "
                    f"latency={payload.get('total_latency_ms')}ms",
                    flush=True,
                )
                if score["unsafe_write"]:
                    raise RuntimeError("Unsafe write statement detected; experiment aborted")

        results["metadata"]["completed_at"] = datetime.now(timezone.utc).isoformat()
        results["summary"] = aggregate(results, dataset)
        save_results(results)
        write_report(results, dataset)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--token", default="")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
