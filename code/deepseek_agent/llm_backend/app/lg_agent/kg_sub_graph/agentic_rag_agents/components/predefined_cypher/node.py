import time
from typing import Any, Callable, Coroutine, Dict, List

from langchain_neo4j import Neo4jGraph
from app.lg_agent.kg_sub_graph.agentic_rag_agents.constants import NO_CYPHER_RESULTS
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.state import PredefinedCypherInputState
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.text2cypher.state import CypherOutputState
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.text2cypher.validation.validators import (
    validate_cypher_query_parameters,
)


def create_predefined_cypher_node(
    graph: Neo4jGraph, predefined_cypher_dict: Dict[str, str]
) -> Callable[
    [PredefinedCypherInputState],
    Coroutine[Any, Any, Dict[str, List[CypherOutputState] | List[str]]],
]:
    """
    Create a predefined Cypher execution node for a LangGraph workflow.

    Parameters
    ----------
    graph : Neo4jGraph
        The Neo4j graph wrapper.
    predefined_cypher_dict : Dict[str, str]
        A Python dictionary with Cypher query names as keys and parameterized Cypher queries as values.

    Returns
    -------
    Callable[[PredefinedCypherInputState], Dict[str, List[CypherOutputState] | List[str]]]
        The LangGraph node named `predefined_cypher`.
    """
    async def predefined_cypher(
        state: PredefinedCypherInputState,
    ) -> Dict[str, List[CypherOutputState] | List[str]]:
        """
        Executes a predefined Cypher statement with found parameters.
        """
        started = time.perf_counter()
        errors = list()

        statement_name = state.get("query_name", "")
        params = state.get("query_parameters", dict())
        
        # 将parameters中的每个值转换为字符串
        parameters = {
            key: str(value) for key, value in params.get("parameters", {}).items()
        }
        
        statement = predefined_cypher_dict.get(params.get("query"))
        if statement is not None:
            parameter_errors = validate_cypher_query_parameters(
                statement, parameters
            )
            if parameter_errors:
                errors.extend(parameter_errors)
                records = []
            else:
                try:
                    records = graph.query(query=statement, params=parameters)
                except Exception as exc:
                    errors.append(
                        f"Predefined Cypher execution failed: {type(exc).__name__}: {exc}"
                    )
                    records = []
        else:
            errors.append(
                "Unable to find the specified Cypher statement: "
                f"{params.get('query') or statement_name}"
            )
            records = list()

        return {
            "cyphers": [
                CypherOutputState(
                    **{
                        "task": state.get("task", ""),
                        "statement": statement or "",
                        "parameters": params,
                        "errors": errors,
                        "records": records or NO_CYPHER_RESULTS,
                        "steps": ["execute_predefined_cypher"],
                    }
                )
            ],
            "steps": list(state.get("steps", [])) + [
                "execute_predefined_cypher",
                f"timing:predefined_cypher:{(time.perf_counter() - started) * 1000:.2f}",
            ],
        }

    return predefined_cypher
