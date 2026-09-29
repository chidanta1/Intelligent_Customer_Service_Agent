"""Token-protected local API for production-path Text2Cypher experiments."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import re
import time
import uuid
from datetime import date, datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastapi import APIRouter, Header, HTTPException
from langchain_deepseek import ChatDeepSeek
from langchain_neo4j import Neo4jGraph
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logger import get_logger
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.text2cypher.generation.node import (
    create_text2cypher_generation_node,
)
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.text2cypher.schema_context import (
    relevant_schema_capabilities,
    retrieve_relevant_schema_for_prompt,
)
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.text2cypher.validation.validators import (
    validate_cypher_query_syntax,
    validate_cypher_query_with_schema,
    validate_no_writes_in_cypher_query,
)
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.utils.utils import (
    retrieve_and_parse_schema_from_graph_for_prompts,
)
from app.lg_agent.kg_sub_graph.agentic_rag_agents.retrievers.cypher_examples.base import (
    BaseCypherExampleRetriever,
)
from app.lg_agent.kg_sub_graph.agentic_rag_agents.retrievers.cypher_examples.vector_store import (
    create_neo4j_vector_example_retriever,
)
from app.lg_agent.kg_sub_graph.kg_neo4j_conn import get_neo4j_graph


PROTOCOL_VERSION = "text2cypher-exp1-backend-v2"
PRODUCTION_TEMPERATURE = settings.TEXT2CYPHER_GENERATION_TEMPERATURE

router = APIRouter(prefix="/experiments/text2cypher")
logger = get_logger(service="text2cypher_experiment_api")

_resource_lock = asyncio.Lock()
_resources: Optional[Tuple[Neo4jGraph, BaseCypherExampleRetriever, ChatDeepSeek]] = None


class GenerationExperimentRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    fewshot_k: Literal[0, 3]
    case_id: str = Field(min_length=1, max_length=64)
    repeat: int = Field(ge=1, le=20)


class GenerationExperimentResponse(BaseModel):
    protocol: str
    request_id: str
    case_id: str
    repeat: int
    question: str
    fewshot_k: int
    model: str
    temperature: float
    schema_version: str
    schema_sha256: str
    retrieval_domain: str
    retrieval_query_type: str
    retrieved_examples: List[str]
    generated_cypher: str
    validation: Dict[str, Any]
    records: List[Dict[str, Any]]
    generation_latency_ms: float
    total_latency_ms: float


def _authorize(token: Optional[str]) -> None:
    configured = settings.EXPERIMENT_API_TOKEN
    if not settings.ENABLE_EXPERIMENT_ENDPOINTS:
        raise HTTPException(status_code=404, detail="Not found")
    if not configured:
        raise HTTPException(
            status_code=503,
            detail="Experiment endpoint is enabled without a configured token",
        )
    if not token or not hmac.compare_digest(token, configured):
        raise HTTPException(status_code=403, detail="Invalid experiment token")


async def _get_resources() -> Tuple[
    Neo4jGraph, BaseCypherExampleRetriever, ChatDeepSeek
]:
    global _resources
    if _resources is not None:
        return _resources
    async with _resource_lock:
        if _resources is None:
            graph = get_neo4j_graph()
            retriever = create_neo4j_vector_example_retriever(graph)
            model = ChatDeepSeek(
                api_key=settings.DEEPSEEK_API_KEY,
                model_name=settings.DEEPSEEK_MODEL,
                temperature=PRODUCTION_TEMPERATURE,
                tags=["text2cypher_generation", "backend_experiment"],
            )
            _resources = (graph, retriever, model)
    return _resources


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def _retrieved_questions(examples: str) -> List[str]:
    return re.findall(r"^Question:\s*(.+)$", examples, flags=re.MULTILINE)


def _validate_and_execute(
    graph: Neo4jGraph,
    statement: str,
    question: str,
    domain: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    validation: Dict[str, Any] = {
        "read_only": False,
        "syntax_valid": False,
        "schema_valid": False,
        "executed": False,
        "failure_type": None,
        "errors": [],
    }
    write_errors = validate_no_writes_in_cypher_query(statement)
    if write_errors:
        validation.update(failure_type="write_operation", errors=write_errors)
        return validation, []
    validation["read_only"] = True

    try:
        syntax_errors = validate_cypher_query_syntax(graph, statement)
    except Exception as exc:
        syntax_errors = [str(exc)]
    if syntax_errors:
        validation.update(failure_type="syntax", errors=syntax_errors)
        return validation, []
    validation["syntax_valid"] = True

    try:
        capabilities = relevant_schema_capabilities(graph, question, domain)
        schema_errors = validate_cypher_query_with_schema(
            graph,
            statement,
            allowed_labels=capabilities["labels"],
            allowed_relationships=capabilities["relationships"],
            allowed_properties=capabilities["properties"],
        )
    except Exception as exc:
        schema_errors = [str(exc)]
    if schema_errors:
        validation.update(failure_type="schema", errors=schema_errors)
        return validation, []
    validation["schema_valid"] = True

    try:
        records = graph.query(statement)
    except Exception as exc:
        validation.update(failure_type="execution", errors=[str(exc)])
        return validation, []
    validation["executed"] = True
    return validation, _json_value(records)


@router.get("/health", include_in_schema=False)
async def experiment_health(
    x_experiment_token: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    _authorize(x_experiment_token)
    graph, _, _ = await _get_resources()
    graph.query("RETURN 1 AS ready")
    schema = retrieve_and_parse_schema_from_graph_for_prompts(graph)
    return {
        "status": "ok",
        "protocol": PROTOCOL_VERSION,
        "execution_boundary": "fastapi-to-production-generation-node",
        "model": settings.DEEPSEEK_MODEL,
        "temperature": PRODUCTION_TEMPERATURE,
        "schema_version": settings.CYPHER_SCHEMA_VERSION,
        "schema_strategy": "question_aware_canonical_slice",
        "allowed_fewshot_k": [0, settings.CYPHER_EXAMPLE_TOP_K],
        "schema_sha256": hashlib.sha256(schema.encode("utf-8")).hexdigest(),
    }


@router.post(
    "/generate",
    response_model=GenerationExperimentResponse,
    include_in_schema=False,
)
async def generate_for_experiment(
    request: GenerationExperimentRequest,
    x_experiment_token: Optional[str] = Header(default=None),
) -> GenerationExperimentResponse:
    """Invoke the same generation node used by the live Text2Cypher workflow."""

    _authorize(x_experiment_token)
    request_id = str(uuid.uuid4())
    total_started = time.perf_counter()
    graph, retriever, model = await _get_resources()
    node = create_text2cypher_generation_node(
        llm=model,
        graph=graph,
        cypher_example_retriever=retriever,
        fewshot_k=request.fewshot_k,
    )
    generation_started = time.perf_counter()
    generated = await node({"task": request.question})
    generation_latency_ms = (time.perf_counter() - generation_started) * 1000
    statement = str(generated.get("statement", ""))
    examples = str(generated.get("fewshot_examples", ""))
    retrieval_domain = str(generated.get("retrieval_domain", "legacy_graph"))
    validation, records = _validate_and_execute(
        graph,
        statement,
        question=request.question,
        domain=retrieval_domain,
    )
    schema = retrieve_relevant_schema_for_prompt(
        graph=graph,
        question=request.question,
        domain=retrieval_domain,
        schema_version=settings.CYPHER_SCHEMA_VERSION,
    )

    logger.info(
        "Text2Cypher experiment request completed: "
        f"request_id={request_id}, case_id={request.case_id}, "
        f"repeat={request.repeat}, fewshot_k={request.fewshot_k}, "
        f"executed={validation['executed']}"
    )
    return GenerationExperimentResponse(
        protocol=PROTOCOL_VERSION,
        request_id=request_id,
        case_id=request.case_id,
        repeat=request.repeat,
        question=request.question,
        fewshot_k=request.fewshot_k,
        model=settings.DEEPSEEK_MODEL,
        temperature=PRODUCTION_TEMPERATURE,
        schema_version=settings.CYPHER_SCHEMA_VERSION,
        schema_sha256=hashlib.sha256(schema.encode("utf-8")).hexdigest(),
        retrieval_domain=retrieval_domain,
        retrieval_query_type=str(generated.get("retrieval_query_type", "")),
        retrieved_examples=_retrieved_questions(examples),
        generated_cypher=statement,
        validation=validation,
        records=records,
        generation_latency_ms=round(generation_latency_ms, 2),
        total_latency_ms=round((time.perf_counter() - total_started) * 1000, 2),
    )
