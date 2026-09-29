import time
from typing import Any, Callable, Coroutine, Dict
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables.base import Runnable
from app.core.logger import get_logger

# 获取日志记录器
logger = get_logger(service="planner_node")

from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.models import Task
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.planner.models import PlannerOutput
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.planner.prompts import create_planner_prompt_template
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.state import InputState


# 定义planner prompt
planner_prompt = create_planner_prompt_template()

def create_planner_node(
    llm: BaseChatModel, ignore_node: bool = False, next_action: str = "tool_selection"
) -> Callable[[InputState], Coroutine[Any, Any, Dict[str, Any]]]:
    """
    Create a planner node to be used in a LangGraph workflow.

    Parameters
    ----------
    llm : BaseChatModel
        The LLM used to process data.
    ignore_node : bool, optional
        Whether to ignore this node in the workflow, by default False

    Returns
    -------
    Callable[[InputState], OverallState]
        The LangGraph node.
    """

    # 检查是否是 DeepSeek 模型
    model_name = getattr(llm, "model_name", "").lower()
    is_deepseek = "deepseek" in model_name

    if is_deepseek:
        # DeepSeek 不支持 with_structured_output，使用 JSON 模式
        logger.info("Using DeepSeek model in planner, switching to JSON parsing mode")
        from langchain_core.output_parsers import JsonOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        # 修改 prompt 以要求 JSON 输出
        json_prompt = ChatPromptTemplate.from_messages([
            ("system", planner_prompt.messages[0].prompt.template + "\n\nPlease respond with a JSON object with field: tasks (array of objects, each with fields: question, parent_task, requires_visualization (boolean, default false))."),
            ("human", "{question}")
        ])

        planner_chain: Runnable[Dict[str, Any], Any] = (
            json_prompt | llm | JsonOutputParser()
        )
    else:
        # 创建planner chain
        planner_chain: Runnable[Dict[str, Any], Any] = (
            planner_prompt | llm.with_structured_output(PlannerOutput)
        )

    async def planner(state: InputState) -> Dict[str, Any]:
        """
        Break user query into chunks, if appropriate.
        """

        started = time.perf_counter()
        if not ignore_node:
            planner_result = await planner_chain.ainvoke(
                {"question": state.get("question", "")}
            )

            # 处理不同的输出格式
            if is_deepseek:
                # JSON 输出，需要转换为 Task 对象
                tasks_data = planner_result.get("tasks", [])
                tasks = []
                question = state.get("question", "")
                for task_dict in tasks_data:
                    tasks.append(Task(
                        question=task_dict.get("question", question),
                        parent_task=task_dict.get("parent_task") or question,  # 如果为 None，使用原问题
                        requires_visualization=task_dict.get("requires_visualization", False)
                    ))
                logger.info(f"DeepSeek planner output: {len(tasks)} tasks")
            else:
                # PlannerOutput 对象
                tasks = planner_result.tasks
        else:
            tasks = []

        planner_task_decomposition = {
            "next_action": next_action,
            "tasks": tasks
            or [
                Task(
                    question=state.get("question", ""),
                    parent_task=state.get("question", ""),
                )
            ],
            "steps": [
                "planner",
                f"timing:planner:{(time.perf_counter() - started) * 1000:.2f}",
            ],
        }

        # 日志打印格式，分别打印每个任务
        logger.info(f"Total Sub Task: {len(planner_task_decomposition['tasks'])}")

        for i, task in enumerate(planner_task_decomposition['tasks']):
            logger.info(f"Sub Task[{i+1}]: {task.question}")

        return planner_task_decomposition

    return planner
