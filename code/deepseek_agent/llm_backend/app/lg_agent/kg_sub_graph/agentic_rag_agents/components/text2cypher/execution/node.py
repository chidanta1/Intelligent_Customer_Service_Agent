"""
This code is based on content found in the LangGraph documentation: https://python.langchain.com/docs/tutorials/graph/#advanced-implementation-with-langgraph
"""

from typing import Any, Callable, Coroutine, Dict, List

from langchain_neo4j import Neo4jGraph

from ....constants import NO_CYPHER_RESULTS
from ..state import CypherOutputState, CypherState


def create_text2cypher_execution_node(
    graph: Neo4jGraph,
) -> Callable[
    [CypherState], Coroutine[Any, Any, Dict[str, List[CypherOutputState] | List[str]]]
]:
    """
    Create a Text2Cypher execution node for a LangGraph workflow.

    Parameters
    ----------
    graph : Neo4jGraph
        The Neo4j graph wrapper.

    Returns
    -------
    Callable[[CypherState], Dict[str, List[CypherOutputState] | List[str]]]
        The LangGraph node.
    """

    async def execute_cypher(
        state: CypherState,
    ) -> Dict[str, List[CypherOutputState] | List[str]]:
        """
        Executes the given Cypher statement.
        """
        statement = state.get("statement", "").strip()
        errors = list(state.get("errors", []))
        try:
            records = graph.query(statement)
        except Exception as exc:
            records = []
            errors.append(f"Cypher execution failed: {exc}")

        steps = state.get("steps", []) + ["execute_cypher"]

        return {
            "cyphers": [
                CypherOutputState(
                    **{
                        "task": state.get("task", ""),
                        "statement": statement,
                        "parameters": None,
                        "errors": errors,
                        "records": records if records else NO_CYPHER_RESULTS,
                        "attempts": state.get("attempts", 1),
                        "retries": state.get("retries", 0),
                        "validation_layers": state.get("validation_layers", []),
                        "validation_trace": state.get("validation_trace", []),
                        "correction_history": state.get("correction_history", []),
                        "steps": steps,
                    }
                )
            ],
            "steps": ["text2cypher"],
        }

    return execute_cypher
