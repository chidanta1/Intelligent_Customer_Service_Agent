"""Token-protected production-path endpoints for Text2Cypher Experiment 2."""

from __future__ import annotations

import hashlib
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, Header
from pydantic import BaseModel, Field

from app.api.text2cypher_experiment import _authorize, _get_resources, _json_value
from app.core.config import settings
from app.core.logger import get_logger
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.text2cypher.correction.node import (
    create_text2cypher_correction_node,
)
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.text2cypher.execution.node import (
    create_text2cypher_execution_node,
)
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.text2cypher.query_context import (
    classify_text2cypher_question,
)
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.text2cypher.validation.node import (
    create_text2cypher_validation_node,
)
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.text2cypher.validation.validators import (
    validate_no_writes_in_cypher_query,
)
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.utils.utils import (
    retrieve_and_parse_schema_from_graph_for_prompts,
)
from app.lg_agent.kg_sub_graph.agentic_rag_agents.constants import (
    MAX_TEXT2CYPHER_CORRECTION_RETRIES,
)
from app.lg_agent.kg_sub_graph.agentic_rag_agents.workflows.single_agent.text2cypher import (
    create_text2cypher_agent,
)


PROTOCOL_VERSION = "text2cypher-exp2-backend-v2"
VALIDATION_LAYERS = [
    "syntax",
    "write_operation",
    "relationship_direction",
    "llm",
    "schema",
]

router = APIRouter(prefix="/experiments/text2cypher/experiment2")
logger = get_logger(service="text2cypher_experiment2_api")


class InjectionRequest(BaseModel):
    case_id: str = Field(min_length=1, max_length=64)
    question: str = Field(min_length=1, max_length=2000)
    candidate_cypher: str = Field(min_length=1, max_length=10000)


class FullWorkflowRequest(BaseModel):
    case_id: str = Field(min_length=1, max_length=64)
    question: str = Field(min_length=1, max_length=2000)
    repeat: int = Field(ge=1, le=3)


def _schema_hash(graph: Any) -> str:
    schema = retrieve_and_parse_schema_from_graph_for_prompts(graph)
    return hashlib.sha256(schema.encode("utf-8")).hexdigest()


def _merge_state(state: Dict[str, Any], update: Optional[Dict[str, Any]]) -> None:
    """Apply LangGraph reducer semantics while invoking production nodes manually."""

    if not update:
        return
    for key, value in update.items():
        if key in {"steps", "validation_trace", "correction_history"}:
            state[key] = list(state.get(key, [])) + list(value or [])
        else:
            state[key] = value


def _output_summary(output: Dict[str, Any], request_id: str) -> Dict[str, Any]:
    statement = str(output.get("statement", ""))
    errors = list(output.get("errors", []))
    steps = list(output.get("steps", []))
    trace = list(output.get("validation_trace", []))
    executed = "execute_cypher" in steps
    last_trace = trace[-1] if trace else {}
    blocking_at_execution = bool(
        executed
        and (last_trace.get("errors") or last_trace.get("mapping_errors"))
    )
    return {
        "request_id": request_id,
        "statement": statement,
        "errors": errors,
        "records": _json_value(output.get("records", [])),
        "attempts": int(output.get("attempts", 0)),
        "retries": int(output.get("retries", 0)),
        "validation_layers": list(output.get("validation_layers", [])),
        "validation_trace": _json_value(trace),
        "correction_history": _json_value(output.get("correction_history", [])),
        "steps": steps,
        "executed": executed,
        "unsafe_executed": bool(
            executed and validate_no_writes_in_cypher_query(statement)
        ),
        "executed_with_blocking_validation_errors": blocking_at_execution,
    }


@router.get("/health", include_in_schema=False)
async def experiment2_health(
    x_experiment_token: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    _authorize(x_experiment_token)
    graph, _, _ = await _get_resources()
    graph.query("RETURN 1 AS ready")
    return {
        "status": "ok",
        "protocol": PROTOCOL_VERSION,
        "execution_boundaries": {
            "part_a": "fastapi-to-production-validation-correction-execution-nodes",
            "part_b": "fastapi-to-production-text2cypher-langgraph",
        },
        "provider_call_owner": "backend",
        "model": settings.DEEPSEEK_MODEL,
        "temperature": settings.TEXT2CYPHER_GENERATION_TEMPERATURE,
        "schema_version": settings.CYPHER_SCHEMA_VERSION,
        "schema_sha256": _schema_hash(graph),
        "validation_layers": VALIDATION_LAYERS,
        "fewshot_k": settings.CYPHER_EXAMPLE_TOP_K,
        "llm_validation": True,
        "max_correction_retries": MAX_TEXT2CYPHER_CORRECTION_RETRIES,
        "force_execute_after_retry_limit": False,
    }


@router.post("/inject", include_in_schema=False)
async def inject_for_experiment(
    request: InjectionRequest,
    x_experiment_token: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    """Run an injected candidate through the production validation pipeline."""

    _authorize(x_experiment_token)
    started = time.perf_counter()
    request_id = str(uuid.uuid4())
    graph, _, model = await _get_resources()
    context = classify_text2cypher_question(request.question)
    validate = create_text2cypher_validation_node(
        graph=graph,
        llm=model,
        llm_validation=True,
        max_retries=MAX_TEXT2CYPHER_CORRECTION_RETRIES,
        attempt_cypher_execution_on_final_attempt=False,
    )
    correct = create_text2cypher_correction_node(llm=model, graph=graph)
    execute = create_text2cypher_execution_node(graph=graph)
    state: Dict[str, Any] = {
        "task": request.question,
        "statement": request.candidate_cypher,
        "parameters": None,
        "errors": [],
        "mapping_errors": [],
        "records": [],
        "next_action_cypher": "validate_cypher",
        "attempts": 0,
        "retries": 0,
        "validation_layers": [],
        "validation_trace": [],
        "correction_history": [],
        "fewshot_examples": "",
        "fewshot_k": 0,
        "retrieval_domain": context.domain,
        "retrieval_query_type": context.query_type,
        "schema_version": settings.CYPHER_SCHEMA_VERSION,
        "steps": [],
    }

    output: Dict[str, Any]
    while True:
        command = await validate(state)
        _merge_state(state, command.update)
        action = str(command.goto)
        if action == "correct_cypher":
            correction_update = await correct(state)
            _merge_state(state, correction_update)
            continue
        if action == "execute_cypher":
            execution_result = await execute(state)
            output = dict(execution_result["cyphers"][0])
            break
        output = {
            "task": request.question,
            "statement": state.get("statement", ""),
            "parameters": None,
            "errors": list(state.get("errors", []))
            + list(state.get("mapping_errors", [])),
            "records": [],
            "attempts": state.get("attempts", 0),
            "retries": state.get("retries", 0),
            "validation_layers": state.get("validation_layers", []),
            "validation_trace": state.get("validation_trace", []),
            "correction_history": state.get("correction_history", []),
            "steps": list(state.get("steps", [])) + ["finalize_failure"],
        }
        break

    response = {
        "protocol": PROTOCOL_VERSION,
        "case_id": request.case_id,
        "question": request.question,
        "initial_cypher": request.candidate_cypher,
        "model": settings.DEEPSEEK_MODEL,
        "temperature": settings.TEXT2CYPHER_GENERATION_TEMPERATURE,
        "schema_version": settings.CYPHER_SCHEMA_VERSION,
        "schema_sha256": _schema_hash(graph),
        "retrieval_domain": context.domain,
        **_output_summary(output, request_id),
        "total_latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }
    logger.info(
        "Text2Cypher Experiment 2A completed: "
        f"request_id={request_id}, case_id={request.case_id}, "
        f"attempts={response['attempts']}, executed={response['executed']}"
    )
    return response


@router.post("/full-workflow", include_in_schema=False)
async def full_workflow_for_experiment(
    request: FullWorkflowRequest,
    x_experiment_token: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    """Run a natural-language question through the production Text2Cypher graph."""

    _authorize(x_experiment_token)
    started = time.perf_counter()
    request_id = str(uuid.uuid4())
    graph, retriever, model = await _get_resources()
    agent = create_text2cypher_agent(
        llm=model,
        graph=graph,
        cypher_example_retriever=retriever,
        llm_cypher_validation=True,
        max_retries=MAX_TEXT2CYPHER_CORRECTION_RETRIES,
        fewshot_k=settings.CYPHER_EXAMPLE_TOP_K,
        attempt_cypher_execution_on_final_attempt=False,
    )
    result = await agent.ainvoke({"task": request.question})
    cyphers = list(result.get("cyphers", []))
    output = dict(cyphers[0]) if cyphers else {
        "task": request.question,
        "statement": "",
        "errors": ["Text2Cypher workflow returned no Cypher output"],
        "records": [],
        "attempts": 0,
        "retries": 0,
        "validation_layers": [],
        "validation_trace": [],
        "correction_history": [],
        "steps": list(result.get("steps", [])),
    }
    context = classify_text2cypher_question(request.question)
    response = {
        "protocol": PROTOCOL_VERSION,
        "case_id": request.case_id,
        "repeat": request.repeat,
        "question": request.question,
        "model": settings.DEEPSEEK_MODEL,
        "temperature": settings.TEXT2CYPHER_GENERATION_TEMPERATURE,
        "schema_version": settings.CYPHER_SCHEMA_VERSION,
        "schema_sha256": _schema_hash(graph),
        "retrieval_domain": context.domain,
        "retrieval_query_type": context.query_type,
        "fewshot_k": settings.CYPHER_EXAMPLE_TOP_K,
        "llm_validation": True,
        "max_correction_retries": MAX_TEXT2CYPHER_CORRECTION_RETRIES,
        **_output_summary(output, request_id),
        "total_latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }
    logger.info(
        "Text2Cypher Experiment 2B completed: "
        f"request_id={request_id}, case_id={request.case_id}, "
        f"repeat={request.repeat}, attempts={response['attempts']}, "
        f"executed={response['executed']}"
    )
    return response
