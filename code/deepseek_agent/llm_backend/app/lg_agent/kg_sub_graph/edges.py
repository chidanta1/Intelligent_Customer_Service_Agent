"""
LangGraph conditional edges for multi-tool workflow.
"""
from typing import Literal, Dict, Any, List
from app.lg_agent.kg_sub_graph.kg_states import OverallState


def guardrails_conditional_edge(
    state: OverallState,
) -> Literal["planner", "final_answer"]:
    """
    Conditional edge from guardrails node.

    Routes to:
    - "planner": if the question is within scope
    - "final_answer": if the question is out of scope or should end
    """
    next_action = state.get("next_action")

    if next_action == "final_answer" or next_action == "end":
        return "final_answer"
    elif next_action == "planner":
        return "planner"
    else:
        # Default to planner if next_action is not set
        return "planner"


def map_reduce_planner_to_tool_selection(
    state: OverallState,
) -> List[str]:
    """
    Map reduce edge from planner to tool_selection nodes.

    Creates one tool_selection node invocation per task.
    """
    tasks = state.get("tasks", [])

    # Return a list of "tool_selection" for each task
    # LangGraph will invoke tool_selection once per task in parallel
    return ["tool_selection" for _ in tasks]
