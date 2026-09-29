"""
This code is based on content found in the LangGraph documentation: https://python.langchain.com/docs/tutorials/graph/#advanced-implementation-with-langgraph
"""

from typing import Any, Callable, Coroutine, Dict

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_neo4j import Neo4jGraph

from app.core.config import settings
from ....components.text2cypher.correction.prompts import (
    create_text2cypher_correction_prompt_template,
)
from ..query_context import classify_text2cypher_question
from ..schema_context import retrieve_relevant_schema_for_prompt
from ..state import CypherState

correction_cypher_prompt = create_text2cypher_correction_prompt_template()


def create_text2cypher_correction_node(
    llm: BaseChatModel, graph: Neo4jGraph
) -> Callable[[CypherState], Coroutine[Any, Any, dict[str, Any]]]:
    """
    Create a Text2Cypher query correction node for a LangGraph workflow.

    Parameters
    ----------
    llm : BaseChatModel
        The LLM to use for processing.
    graph : Neo4jGraph
        The Neo4j graph wrapper.

    Returns
    -------
    Callable[[CypherState], CypherState]
        The LangGraph node.
    """

    correct_cypher_chain = correction_cypher_prompt | llm | StrOutputParser()

    async def correct_cypher(state: CypherState) -> Dict[str, Any]:
        """
        Correct the Cypher statement based on the provided errors.
        """

        task = state.get("task", "")
        question = task[0] if isinstance(task, list) else task
        domain = state.get("retrieval_domain") or classify_text2cypher_question(
            question
        ).domain
        validation_errors = list(state.get("errors", [])) + list(
            state.get("mapping_errors", [])
        )
        statement_before = str(state.get("statement", ""))
        corrected_cypher = await correct_cypher_chain.ainvoke(
            {
                "question": question,
                "errors": validation_errors,
                "cypher": statement_before,
                "schema": retrieve_relevant_schema_for_prompt(
                    graph=graph,
                    question=question,
                    domain=domain,
                    schema_version=state.get(
                        "schema_version", settings.CYPHER_SCHEMA_VERSION
                    ),
                ),
            }
        )
        corrected_cypher = corrected_cypher.strip().removeprefix("```cypher")
        corrected_cypher = corrected_cypher.removeprefix("```").removesuffix("```").strip()

        return {
            "next_action_cypher": "validate_cypher",
            "statement": corrected_cypher,
            "errors": [],
            "mapping_errors": [],
            "correction_history": [
                {
                    "retry": state.get("retries", 0) + 1,
                    "statement_before": statement_before,
                    "statement_after": corrected_cypher,
                    "errors": validation_errors,
                }
            ],
            "steps": ["correct_cypher"],
        }

    return correct_cypher
