"""Token-protected full-agent endpoint for the held-out Experiment 3."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Header
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from app.api.text2cypher_experiment import _authorize, _json_value
from app.core.config import ROOT_DIR, settings
from app.core.logger import get_logger
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.customer_tools.node import (
    GraphRAGAPI,
)
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.utils.utils import (
    retrieve_and_parse_schema_from_graph_for_prompts,
)
from app.lg_agent.kg_sub_graph.kg_neo4j_conn import get_neo4j_graph
from app.lg_agent.lg_builder import graph as production_graph
from app.lg_agent.lg_states import InputState


PROTOCOL_VERSION = "agent-exp3-backend-v1"
router = APIRouter(prefix="/experiments/agent/experiment3")
logger = get_logger(service="agent_experiment3_api")


class FullWorkflowRequest(BaseModel):
    case_id: str = Field(min_length=1, max_length=64)
    question: str = Field(min_length=1, max_length=3000)
    repeat: int = Field(ge=1, le=3)


def _graphrag_data_root() -> Path:
    configured = Path(settings.GRAPHRAG_PROJECT_DIR)
    candidates = [
        configured / settings.GRAPHRAG_DATA_DIR,
        Path.cwd() / configured / settings.GRAPHRAG_DATA_DIR,
        ROOT_DIR / "app" / "graphrag" / settings.GRAPHRAG_DATA_DIR,
    ]
    for candidate in candidates:
        if (candidate / "settings.yaml").is_file():
            return candidate.resolve()
    return candidates[-1].resolve()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        (item for item in root.rglob("*") if item.is_file()),
        key=lambda item: str(item.relative_to(root)),
    ):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _schema_sha256() -> str:
    neo4j_graph = get_neo4j_graph()
    schema = retrieve_and_parse_schema_from_graph_for_prompts(neo4j_graph)
    return hashlib.sha256(schema.encode("utf-8")).hexdigest()


def _reference_snapshot_sha256() -> str:
    """Fingerprint every frozen structured reference result without exposing rows."""

    dataset_path = ROOT_DIR / "app" / "test" / "data" / "agent_experiment3_v1.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    neo4j_graph = get_neo4j_graph()
    references: Dict[str, Any] = {}
    for case in dataset["cases"]:
        cypher = case.get("reference", {}).get("cypher")
        if cypher:
            rows = neo4j_graph.query(cypher)
            references[case["case_id"]] = json.loads(
                json.dumps(rows, ensure_ascii=False, default=str)
            )
    encoded = json.dumps(
        references, ensure_ascii=False, sort_keys=True, default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _runtime_config() -> Dict[str, Any]:
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


def _index_evidence() -> Dict[str, Any]:
    data_root = _graphrag_data_root()
    output_root = data_root / "output_experiment3_v1"
    required = [
        output_root / "documents.parquet",
        output_root / "text_units.parquet",
        output_root / "entities.parquet",
        output_root / "relationships.parquet",
        output_root / "communities.parquet",
        output_root / "community_reports.parquet",
    ]
    existing = [path for path in required if path.is_file()]
    return {
        "data_root": str(data_root),
        "output_root": str(output_root),
        "ready": len(existing) == len(required),
        "required_files": [path.name for path in required],
        "present_files": [path.name for path in existing],
        "index_sha256": _tree_sha256(output_root) if len(existing) == len(required) else None,
    }


def _selected_tools(trace: Dict[str, Any]) -> list[str]:
    prefix = "tool_selection:"
    return [
        step[len(prefix) :]
        for step in trace.get("steps", [])
        if isinstance(step, str)
        and step.startswith(prefix)
        and not step.startswith("tool_selection_reason:")
    ]


def _node_timings(trace: Dict[str, Any]) -> list[Dict[str, Any]]:
    timings: list[Dict[str, Any]] = []
    for step in trace.get("steps", []):
        if not isinstance(step, str) or not step.startswith("timing:"):
            continue
        _, node, milliseconds = step.split(":", 2)
        try:
            timings.append({"node": node, "latency_ms": float(milliseconds)})
        except ValueError:
            continue
    return timings


@router.get("/health", include_in_schema=False)
async def experiment3_health(
    x_experiment_token: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    _authorize(x_experiment_token)
    neo4j_graph = get_neo4j_graph()
    neo4j_graph.query("RETURN 1 AS ready")
    index = _index_evidence()
    if index["ready"]:
        # Load all required GraphRAG tables without issuing any model request.
        await GraphRAGAPI().initialize()
    return {
        "status": "ok" if index["ready"] else "not_ready",
        "protocol": PROTOCOL_VERSION,
        "execution_boundary": "fastapi-to-production-outer-langgraph-and-multi-tool-subgraph",
        "provider_call_owner": "backend",
        "model": settings.DEEPSEEK_MODEL,
        "agent_temperature": settings.AGENT_WORKFLOW_TEMPERATURE,
        "text2cypher_temperature": settings.TEXT2CYPHER_GENERATION_TEMPERATURE,
        "schema_version": settings.CYPHER_SCHEMA_VERSION,
        "schema_sha256": _schema_sha256(),
        "neo4j_reference_snapshot_sha256": _reference_snapshot_sha256(),
        "runtime_config": _runtime_config(),
        "graphrag": index,
        "timeout_seconds": 90,
    }


@router.post("/full-workflow", include_in_schema=False)
async def full_workflow_for_experiment(
    request: FullWorkflowRequest,
    x_experiment_token: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    """Run one held-out question through the same outer graph as the user API."""

    _authorize(x_experiment_token)
    started = time.perf_counter()
    request_id = str(uuid.uuid4())
    thread_config = {
        "configurable": {
            "thread_id": f"experiment3-{request.case_id}-{request.repeat}-{request_id}",
            "user_id": 0,
            "image_path": None,
        }
    }
    result = await asyncio.wait_for(
        production_graph.ainvoke(
            InputState(messages=[HumanMessage(content=request.question)]),
            config=thread_config,
        ),
        timeout=90.0,
    )
    messages = list(result.get("messages", []))
    final_answer = ""
    for message in reversed(messages):
        content = getattr(message, "content", "")
        if content and not isinstance(message, HumanMessage):
            final_answer = content if isinstance(content, str) else str(content)
            break
    trace = _json_value(result.get("research_trace", {}))
    outer_router = _json_value(result.get("router", {}))
    response = {
        "protocol": PROTOCOL_VERSION,
        "request_id": request_id,
        "case_id": request.case_id,
        "repeat": request.repeat,
        "question": request.question,
        "model": settings.DEEPSEEK_MODEL,
        "agent_temperature": settings.AGENT_WORKFLOW_TEMPERATURE,
        "text2cypher_temperature": settings.TEXT2CYPHER_GENERATION_TEMPERATURE,
        "schema_version": settings.CYPHER_SCHEMA_VERSION,
        "schema_sha256": _schema_sha256(),
        "graphrag_index_sha256": _index_evidence()["index_sha256"],
        "outer_router": outer_router,
        "outer_router_latency_ms": result.get("outer_router_latency_ms", 0.0),
        "outer_route_passed": outer_router.get("type") == "graphrag-query",
        "planned_tasks": trace.get("tasks", []),
        "selected_tools": _selected_tools(trace),
        "tool_outputs": trace.get("tool_outputs", []),
        "workflow_steps": trace.get("steps", []),
        "node_timings": _node_timings(trace),
        "research_status": trace.get("status", "not_entered"),
        "final_answer": final_answer,
        "total_latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }
    logger.info(
        "Experiment 3 request completed: "
        f"request_id={request_id}, case_id={request.case_id}, "
        f"repeat={request.repeat}, tools={response['selected_tools']}"
    )
    return response
