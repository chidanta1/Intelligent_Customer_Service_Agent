"""Adapter that exposes the four-stage Text2Cypher subgraph as a tool node."""

import time
from typing import Any, Callable, Coroutine, Dict

from langchain_core.language_models import BaseChatModel
from langchain_neo4j import Neo4jGraph

from app.lg_agent.kg_sub_graph.agentic_rag_agents.retrievers.cypher_examples.base import (
    BaseCypherExampleRetriever,
)
from app.lg_agent.kg_sub_graph.agentic_rag_agents.workflows.single_agent import (
    create_text2cypher_agent,
)


def create_cypher_query_node(
    llm: BaseChatModel,
    graph: Neo4jGraph,
    cypher_example_retriever: BaseCypherExampleRetriever,
    llm_cypher_validation: bool = True,
    max_retries: int = 3,
    fewshot_k: int = 3,
) -> Callable[[Dict[str, Any]], Coroutine[Any, Any, Dict[str, Any]]]:
    """Create the live Text2Cypher tool backed by the LangGraph sub-workflow."""

    text2cypher_workflow = create_text2cypher_agent(
        llm=llm,
        graph=graph,
        cypher_example_retriever=cypher_example_retriever,
        llm_cypher_validation=llm_cypher_validation,
        max_retries=max_retries,
        fewshot_k=fewshot_k,
        attempt_cypher_execution_on_final_attempt=False,
    )

    async def cypher_query(state: Dict[str, Any]) -> Dict[str, Any]:
        started = time.perf_counter()
        task = state.get("task", "")
        if isinstance(task, list):
            task = task[0] if task else ""
        if not task:
            return {
                "cyphers": [
                    {
                        "task": "",
                        "statement": "",
                        "parameters": None,
                        "errors": ["未提供查询文本"],
                        "records": [],
                        "attempts": 0,
                        "retries": 0,
                        "validation_layers": [],
                        "validation_trace": [],
                        "correction_history": [],
                        "steps": ["text2cypher_failed"],
                    }
                ],
                "steps": ["text2cypher_failed"],
            }
        result = await text2cypher_workflow.ainvoke({"task": task})
        return {
            **result,
            "steps": list(state.get("steps", []))
            + list(result.get("steps", []))
            + [f"timing:cypher_query:{(time.perf_counter() - started) * 1000:.2f}"],
        }

    return cypher_query
