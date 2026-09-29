from typing import Any, Callable, Coroutine, Dict, Optional

from langchain_core.language_models import BaseChatModel
from langchain_neo4j import Neo4jGraph
from langgraph.constants import END, START
from langgraph.graph.state import CompiledStateGraph, StateGraph

from ...components.text2cypher import (
    create_text2cypher_correction_node,
    create_text2cypher_execution_node,
    create_text2cypher_generation_node,
    create_text2cypher_validation_node,
)
from ...components.text2cypher.state import (
    CypherAgentOutputState,
    CypherInputState,
    CypherState,
)
from ...retrievers.cypher_examples.base import BaseCypherExampleRetriever
from ...constants import MAX_TEXT2CYPHER_CORRECTION_RETRIES


def create_text2cypher_agent(
    llm: BaseChatModel,
    graph: Neo4jGraph,
    cypher_example_retriever: BaseCypherExampleRetriever,
    llm_cypher_validation: bool = True,
    max_retries: int = 3,
    fewshot_k: int = 3,
    attempt_cypher_execution_on_final_attempt: bool = False,
    correction_node: Optional[
        Callable[[CypherState], Coroutine[Any, Any, Dict[str, Any]]]
    ] = None,
) -> CompiledStateGraph:
    """
    Create a Text2Cypher agent using LangGraph.
    This agent contains only Text2cypher components with no guardrails, query parser or summarizer.
    This agent may be used as an independent workflow or a node in a larger LangGraph workflow.

    Parameters
    ----------
    graph : Neo4jGraph
        The Neo4j graph wrapper.
    llm : BaseChatModel
        The LLM to use for processing.
    cypher_example_retriever: BaseCypherExampleRetriever
        The retriever used to collect Cypher examples for few shot prompting.
    llm_cypher_validation : bool, optional
        Whether to perform LLM validation with the provided LLM, by default True
    max_retries: int, optional
        The maximum number of correction retries after initial generation, by default 3
    attempt_cypher_execution_on_final_attempt, bool, optional
        THIS MAY BE DANGEROUS.
        Whether to attempt Cypher execution on the last attempt, regardless of if the Cypher contains errors, by default False

    Returns
    -------
    CompiledStateGraph
        The workflow.
    """

    if not isinstance(max_retries, int) or isinstance(max_retries, bool):
        raise TypeError("max_retries must be an integer")
    if not 0 <= max_retries <= MAX_TEXT2CYPHER_CORRECTION_RETRIES:
        raise ValueError(
            "max_retries must be between 0 and "
            f"{MAX_TEXT2CYPHER_CORRECTION_RETRIES}"
        )

    # 1. 根据自定义的 Cypher 示例，引导大模型生成 当前输入 问题的 Cypher 查询语句
    generate_cypher = create_text2cypher_generation_node(
        llm=llm,
        graph=graph,
        cypher_example_retriever=cypher_example_retriever,
        fewshot_k=fewshot_k,
    )
    # 2. 验证生成的 Cypher 查询语句是否正确
    validate_cypher = create_text2cypher_validation_node(
        llm=llm,
        graph=graph,
        llm_validation=llm_cypher_validation,
        max_retries=max_retries,
        attempt_cypher_execution_on_final_attempt=attempt_cypher_execution_on_final_attempt,
    )
    correct_cypher = correction_node or create_text2cypher_correction_node(
        llm=llm,
        graph=graph,
    )
    execute_cypher = create_text2cypher_execution_node(graph=graph)

    text2cypher_graph_builder = StateGraph(
        CypherState, input=CypherInputState, output=CypherAgentOutputState
    )

    async def finalize_failure(state: CypherState) -> dict:
        """Return a structured failure after all three correction attempts."""

        errors = list(state.get("errors", [])) + list(
            state.get("mapping_errors", [])
        )
        return {
            "cyphers": [
                {
                    "task": state.get("task", ""),
                    "statement": state.get("statement", ""),
                    "parameters": None,
                    "errors": errors,
                    "records": [],
                    "attempts": state.get("attempts", max_retries + 1),
                    "retries": state.get("retries", max_retries),
                    "validation_layers": state.get("validation_layers", []),
                    "validation_trace": state.get("validation_trace", []),
                    "correction_history": state.get("correction_history", []),
                    "steps": state.get("steps", []) + ["finalize_failure"],
                }
            ],
            "steps": ["text2cypher_failed"],
        }

    text2cypher_graph_builder.add_node(generate_cypher)
    text2cypher_graph_builder.add_node(
        validate_cypher,
        destinations=("correct_cypher", "execute_cypher", "finalize_failure"),
    )
    text2cypher_graph_builder.add_node("correct_cypher", correct_cypher)
    text2cypher_graph_builder.add_node(execute_cypher)
    text2cypher_graph_builder.add_node("finalize_failure", finalize_failure)

    text2cypher_graph_builder.add_edge(START, "generate_cypher")
    text2cypher_graph_builder.add_edge("generate_cypher", "validate_cypher")
    text2cypher_graph_builder.add_edge("correct_cypher", "validate_cypher")
    text2cypher_graph_builder.add_edge("execute_cypher", END)
    text2cypher_graph_builder.add_edge("finalize_failure", END)

    return text2cypher_graph_builder.compile()
