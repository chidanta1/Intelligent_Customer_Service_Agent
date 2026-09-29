from app.lg_agent.lg_states import AgentState, Router
from app.lg_agent.lg_prompts import (
    ROUTER_SYSTEM_PROMPT,
    GET_ADDITIONAL_SYSTEM_PROMPT,
    GENERAL_QUERY_SYSTEM_PROMPT,
    GET_IMAGE_SYSTEM_PROMPT,
    GUARDRAILS_SYSTEM_PROMPT,
    RAGSEARCH_SYSTEM_PROMPT,
    CHECK_HALLUCINATIONS,
    GENERATE_QUERIES_SYSTEM_PROMPT
)
from langchain_core.runnables import RunnableConfig
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama
from langchain_anthropic import ChatAnthropic
from app.core.config import settings, ServiceType
from app.core.logger import get_logger
from typing import cast, Literal, TypedDict, List, Dict, Any
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from app.lg_agent.lg_states import AgentState, InputState, Router, GradeHallucinations
from app.lg_agent.kg_sub_graph.agentic_rag_agents.retrievers.cypher_examples.vector_store import (
    create_neo4j_vector_example_retriever,
)
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.planner.node import create_planner_node
from app.lg_agent.kg_sub_graph.agentic_rag_agents.workflows.multi_agent.multi_tool import create_multi_tool_workflow
from app.lg_agent.kg_sub_graph.kg_neo4j_conn import get_neo4j_graph
from pydantic import BaseModel
from typing import Dict, List
from langchain_core.messages import AIMessage
from langchain_core.runnables.base import Runnable
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.utils.utils import retrieve_and_parse_schema_from_graph_for_prompts
from langchain_core.prompts import ChatPromptTemplate
import base64
import os
import aiohttp
import asyncio
import json
import re
import time
from pathlib import Path


from typing import Literal
from pydantic import BaseModel, Field


class AdditionalGuardrailsOutput(BaseModel):
    """
    格式化输出，用于判断用户的问题是否与图谱内容相关
    """
    decision: Literal["end", "continue"] = Field(
        description="Decision on whether the question is related to the graph contents."
    )


# 构建日志记录器
logger = get_logger(service="lg_builder")


_UNSAFE_QUERY_PATTERN = re.compile(
    r"删除|清空|修改(?:价格|库存|订单|数据库)|更新数据库|导出(?:数据库|数据)|"
    r"执行(?:SQL|sql|命令|脚本)|密码|API\s*密钥|token|管理员权限|系统权限|提升权限"
)
_BUSINESS_RESEARCH_PATTERN = re.compile(
    r"智能(?:门铃|门锁|音箱|灯|照明|家居|设备|开关|空调|冰箱|马桶)|"
    r"订单\s*\d+|商品目录|目录|客户|供应商|员工|物流|售后|质保|保修|退货|退款"
)
_RESEARCH_INTENT_PATTERN = re.compile(
    r"查询|查一下|查找|列出|统计|哪些|哪款|多少|几款|价格|售价|多少钱|库存|"
    r"下单|送达|发出|职位|入职|评价|反馈|优点|缺点|问题|改进|维护|使用|配网|"
    r"WiFi|wifi|故障|退换|退款|运费|期限|多久|时限|语音助手|开锁|没电"
)


def _is_explicit_business_research_query(question: str) -> bool:
    """Route clear, safe business lookups directly to the production research graph.

    The LLM remains responsible for genuinely ambiguous turns.  This rule prevents
    concrete multi-part lookups from being mistaken for a request for more details.
    """

    compact = re.sub(r"\s+", "", question or "")
    if not compact or _UNSAFE_QUERY_PATTERN.search(compact):
        return False
    return bool(
        _BUSINESS_RESEARCH_PATTERN.search(compact)
        and _RESEARCH_INTENT_PATTERN.search(compact)
    )


def _trace_json_value(value: Any) -> Any:
    """Convert LangGraph/Pydantic/Neo4j values into stable JSON-safe evidence."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, BaseModel):
        return _trace_json_value(value.model_dump())
    if isinstance(value, dict):
        return {str(key): _trace_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_trace_json_value(item) for item in value]
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except (TypeError, ValueError):
            pass
    return str(value)

async def analyze_and_route_query(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, Router]:
    """Analyze the user's query and determine the appropriate routing.

    This function uses a language model to classify the user's query and decide how to route it
    within the conversation flow.

    Args:
        state (AgentState): The current state of the agent, including conversation history.
        config (RunnableConfig): Configuration with the model used for query analysis.

    Returns:
        dict[str, Router]: A dictionary containing the 'router' key with the classification result (classification type and logic).
    """
    router_started = time.perf_counter()
    latest_question = (
        str(state.messages[-1].content) if state.messages else ""
    )
    if _is_explicit_business_research_query(latest_question):
        response = Router(
            type="graphrag-query",
            question=latest_question,
            logic="规则识别为信息充分且安全的业务检索请求",
        )
        logger.info(
            "Rule-based router sent explicit business lookup to research workflow"
        )
        return {
            "router": response,
            "outer_router_latency_ms": round(
                (time.perf_counter() - router_started) * 1000, 2
            ),
        }
    # 选择模型实例，通过.env文件中的AGENT_SERVICE参数选择
    if settings.AGENT_SERVICE == ServiceType.CLAUDE:
        # Claude 不支持流式结构化输出，需要禁用流式
        model = ChatAnthropic(
            api_key=settings.ANTHROPIC_API_KEY,
            base_url=settings.ANTHROPIC_BASE_URL,
            model=settings.ANTHROPIC_MODEL,
            temperature=settings.AGENT_WORKFLOW_TEMPERATURE,
            streaming=False  # 禁用流式输出以支持结构化输出
        )
        logger.info(f"Using Claude model: {settings.ANTHROPIC_MODEL}")
        use_structured_output = True
    elif settings.AGENT_SERVICE == ServiceType.DEEPSEEK:
        model = ChatDeepSeek(api_key=settings.DEEPSEEK_API_KEY, model_name=settings.DEEPSEEK_MODEL, temperature=settings.AGENT_WORKFLOW_TEMPERATURE, tags=["router"])
        logger.info(f"Using DeepSeek model: {settings.DEEPSEEK_MODEL}")
        use_structured_output = False  # DeepSeek thinking mode 不支持结构化输出
    else:
        model = ChatOllama(model=settings.OLLAMA_AGENT_MODEL, base_url=settings.OLLAMA_BASE_URL, temperature=settings.AGENT_WORKFLOW_TEMPERATURE, tags=["router"])
        logger.info(f"Using Ollama model: {settings.OLLAMA_AGENT_MODEL}")
        use_structured_output = True

    # 拼接提示模版 + 用户的实时问题（包含历史上下文对话）
    json_instruction = "\n\n请以 JSON 格式返回你的分析结果，格式如下：\n{\"type\": \"类型\", \"question\": \"用户问题\", \"logic\": \"分类逻辑\"}\n其中type必须是以下之一：general-query, additional-query, graphrag-query, image-query, file-query"

    messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT + (json_instruction if not use_structured_output else "")}
    ] + state.messages
    logger.info("-----Analyze user query type-----")
    logger.info(f"History messages: {state.messages}")

    # 根据模型选择使用结构化输出或JSON解析
    if use_structured_output:
        response = cast(
            Router, await model.with_structured_output(Router).ainvoke(messages)
        )
    else:
        # 对于不支持结构化输出的模型，使用 JSON 解析
        ai_response = await model.ainvoke(messages)
        content = ai_response.content
        logger.info(f"Raw response from model: {content}")

        # 尝试提取 JSON
        try:
            # 如果响应包含 markdown 代码块，提取JSON部分
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_str = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                json_str = content[json_start:json_end].strip()
            else:
                json_str = content.strip()

            parsed = json.loads(json_str)
            response = Router(
                type=parsed.get("type", "general-query"),
                question=parsed.get("question", state.messages[-1].content if state.messages else ""),
                logic=parsed.get("logic", "")
            )
        except Exception as e:
            logger.error(f"Failed to parse JSON response: {e}, content: {content}")
            # 降级处理：默认为 general-query
            response = Router(
                type="general-query",
                question=state.messages[-1].content if state.messages else "",
                logic="Failed to parse model response, defaulting to general query"
            )

    logger.info(f"Analyze user query type completed, result: {response}")
    return {
        "router": response,
        "outer_router_latency_ms": round(
            (time.perf_counter() - router_started) * 1000, 2
        ),
    }

def route_query(
    state: AgentState,
) -> Literal["respond_to_general_query", "get_additional_info", "create_research_plan", "create_image_query", "create_file_query"]:
    """根据查询分类确定下一步操作。

    Args:
        state (AgentState): 当前代理状态，包括路由器的分类。

    Returns:
        Literal["respond_to_general_query", "get_additional_info", "create_research_plan", "create_image_query", "create_file_query"]: 下一步操作。
    """
    _type = state.router["type"]
    
    # 检查配置中是否有图片路径，如果有，优先处理为图片查询
    if hasattr(state, "config") and state.config and state.config.get("configurable", {}).get("image_path"):
        logger.info("检测到图片路径，转为图片查询处理")
        return "create_image_query"

    if _type == "general-query":
        return "respond_to_general_query"
    elif _type == "additional-query":
        return "get_additional_info"
    elif _type == "graphrag-query":
        return "create_research_plan"
    elif _type == "image-query":
        return "create_image_query"
    elif _type == "file-query":
        return "create_file_query"
    else:
        raise ValueError(f"Unknown router type {_type}")
    
async def respond_to_general_query(
    state: AgentState, *, config: RunnableConfig
) -> Dict[str, List[BaseMessage]]:
    """生成对一般查询的响应，完全基于大模型，不会触发任何外部服务的调用，包括自定义工具、知识库查询等。

    当路由器将查询分类为一般问题时，将调用此节点。

    Args:
        state (AgentState): 当前代理状态，包括对话历史和路由逻辑。
        config (RunnableConfig): 用于配置响应生成的模型。

    Returns:
        Dict[str, List[BaseMessage]]: 包含'messages'键的字典，其中包含生成的响应。
    """
    logger.info("-----generate general-query response-----")
    
    # 使用大模型生成回复
    if settings.AGENT_SERVICE == ServiceType.CLAUDE:
        model = ChatAnthropic(api_key=settings.ANTHROPIC_API_KEY, base_url=settings.ANTHROPIC_BASE_URL, model=settings.ANTHROPIC_MODEL, temperature=0.7)
    elif settings.AGENT_SERVICE == ServiceType.DEEPSEEK:
        model = ChatDeepSeek(api_key=settings.DEEPSEEK_API_KEY, model_name=settings.DEEPSEEK_MODEL, temperature=0.7, tags=["general_query"])
    else:
        model = ChatOllama(model=settings.OLLAMA_AGENT_MODEL, base_url=settings.OLLAMA_BASE_URL, temperature=0.7, tags=["general_query"])
    
    system_prompt = GENERAL_QUERY_SYSTEM_PROMPT.format(
        logic=state.router["logic"]
    )
    
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    response = await model.ainvoke(messages)
    return {"messages": [response]}

async def get_additional_info(
    state: AgentState, *, config: RunnableConfig
) -> Dict[str, List[BaseMessage]]:
    """生成一个响应，要求用户提供更多信息。

    当路由确定需要从用户那里获取更多信息时，将调用此函数。

    Args:
        state (AgentState): 当前代理状态，包括对话历史和路由逻辑。
        config (RunnableConfig): 用于配置响应生成的模型。

    Returns:
        Dict[str, List[BaseMessage]]: 包含'messages'键的字典，其中包含生成的响应。
    """
    logger.info("------continue to get additional info------")
    
    # 使用大模型生成回复
    if settings.AGENT_SERVICE == ServiceType.CLAUDE:
        model = ChatAnthropic(api_key=settings.ANTHROPIC_API_KEY, base_url=settings.ANTHROPIC_BASE_URL, model=settings.ANTHROPIC_MODEL, temperature=0.7)
    elif settings.AGENT_SERVICE == ServiceType.DEEPSEEK:
        model = ChatDeepSeek(api_key=settings.DEEPSEEK_API_KEY, model_name=settings.DEEPSEEK_MODEL, temperature=0.7, tags=["additional_info"])
    else:
        model = ChatOllama(model=settings.OLLAMA_AGENT_MODEL, base_url=settings.OLLAMA_BASE_URL, temperature=0.7, tags=["additional_info"])

    # 如果用户的问题是电商相关，但与自己的业务无关，则需要返回"无关问题"

    # 首先连接 Neo4j 图数据库
    try:
        neo4j_graph = get_neo4j_graph()
        logger.info("success to get Neo4j graph database connection")
    except Exception as e:
        logger.error(f"failed to get Neo4j graph database connection: {e}")

    # 定义电商经营范围
    scope_description = """
    个人电商经营范围：智能家居产品，包括但不限于：
    - 智能照明（灯泡、灯带、开关）
    - 智能安防（摄像头、门锁、传感器）
    - 智能控制（温控器、遥控器、集线器）
    - 智能音箱（语音助手、音响）
    - 智能厨电（电饭煲、冰箱、洗碗机）
    - 智能清洁（扫地机器人、洗衣机）
    
    不包含：服装、鞋类、体育用品、化妆品、食品等非智能家居产品。
    """

    scope_context = (
        f"参考此范围描述来决策:\n{scope_description}"
        if scope_description is not None
        else ""
    )

    # 动态从 Neo4j 图表中获取图表结构
    graph_context = (
        f"\n参考图表结构来回答:\n{retrieve_and_parse_schema_from_graph_for_prompts(neo4j_graph)}"
        if neo4j_graph is not None
        else ""
    )

    message = scope_context + graph_context + "\nQuestion: {question}"

    # 拼接提示模版
    full_system_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                GUARDRAILS_SYSTEM_PROMPT,
            ),
            (
                "human",
                (message),
            ),
        ]
    )

    # 构建格式化输出的 Chain， 如果匹配，返回 continue，否则返回 end
    # DeepSeek thinking 模式不支持 structured output，使用 JSON 解析
    if settings.AGENT_SERVICE == ServiceType.DEEPSEEK:
        # 添加 JSON 格式要求到 system prompt (使用双花括号转义)
        json_instruction = '\n\n请以 JSON 格式返回你的判断，格式如下：\n{{"decision": "continue" 或 "end", "reason": "判断理由"}}'
        full_system_prompt_with_json = ChatPromptTemplate.from_messages(
            [
                ("system", GUARDRAILS_SYSTEM_PROMPT + json_instruction),
                ("human", message),
            ]
        )
        response = await (full_system_prompt_with_json | model).ainvoke(
            {"question": state.messages[-1].content if state.messages else ""}
        )

        # 解析 JSON 响应
        try:
            content = response.content
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_str = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                json_str = content[json_start:json_end].strip()
            else:
                json_str = content.strip()

            parsed = json.loads(json_str)
            guardrails_output = AdditionalGuardrailsOutput(
                decision=parsed.get("decision", "continue"),
                reason=parsed.get("reason", "")
            )
        except Exception as e:
            logger.warning(f"Failed to parse JSON from DeepSeek: {e}, defaulting to continue")
            guardrails_output = AdditionalGuardrailsOutput(decision="continue", reason="JSON parse failed")
    else:
        # Claude 和 Ollama 支持 structured output
        guardrails_chain = full_system_prompt | model.with_structured_output(AdditionalGuardrailsOutput)
        guardrails_output = await guardrails_chain.ainvoke(
                {"question": state.messages[-1].content if state.messages else ""}
            )

    # 根据格式化输出的结果，返回不同的响应
    if guardrails_output.decision == "end":
        logger.info("-----Fail to pass guardrails check-----")
        return {"messages": [AIMessage(content="抱歉，我家暂时没有这方面的商品，可以在别家看看哦~")]}
    else:
        logger.info("-----Pass guardrails check-----")
        system_prompt = GET_ADDITIONAL_SYSTEM_PROMPT.format(
            logic=state.router["logic"]
        )
        messages = [{"role": "system", "content": system_prompt}] + state.messages
        response = await model.ainvoke(messages)
        return {"messages": [response]}

async def create_image_query(
    state: AgentState, *, config: RunnableConfig
) -> Dict[str, List[BaseMessage]]:
    """处理图片查询并生成描述回复
    
    Args:
        state (AgentState): 当前代理状态，包括对话历史
        config (RunnableConfig): 配置参数，包含线程ID等配置信息
        
    Returns:
        Dict[str, List[BaseMessage]]: 包含'messages'键的字典，其中包含生成的响应
    """
    logger.info("-----Found User Upload Image-----")    
    image_path = config.get("configurable", {}).get("image_path", None)

    if not image_path or not Path(image_path).exists():
        logger.warning(f"User Upload Image Not Found: {image_path}")
        return {"messages": [AIMessage(content="抱歉，我无法查看这张图片，请重新上传。")]}
    
    # 获取视觉模型配置
    api_key = settings.VISION_API_KEY
    base_url = settings.VISION_BASE_URL
    vision_model = settings.VISION_MODEL
    
    if not api_key or not base_url or not vision_model:
        logger.error("Vision Model Configuration Not Complete")
        return {"messages": [AIMessage(content="抱歉，我无法查看这张图片，请重新上传。")]}
    
    logger.info(f"Using Vision Model: {vision_model} to process image: {image_path}")
    
    try:
        # 导入图片处理库
        from PIL import Image
        import io
        
        # 读取并压缩图片
        with Image.open(image_path) as img:
            # 设置最大尺寸
            max_size = 1024
            # 计算缩放比例
            width, height = img.size
            ratio = min(max_size / width, max_size / height)
            
            # 如果图片尺寸已经小于最大尺寸，不需要缩放
            if width <= max_size and height <= max_size:
                resized_img = img
            else:
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # 转换为JPEG格式，并调整质量
            img_byte_arr = io.BytesIO()
            if resized_img.mode != 'RGB':
                resized_img = resized_img.convert('RGB')
            resized_img.save(img_byte_arr, format='JPEG', quality=85)
            img_byte_arr.seek(0)
            
            # 转换为base64
            image_data = base64.b64encode(img_byte_arr.read()).decode('utf-8')
            
            logger.info(f"Image Compressed, Original Size: {width}x{height}, New Size: {resized_img.width}x{resized_img.height}")
        
        # 构建API请求
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": vision_model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个专业的图像分析助手。请详细分析图片中的内容，特别关注产品细节、品牌、型号等信息。"
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 4000,
            "temperature": 0.7
        }
        
        # 发送API请求
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60  # 增加超时时间
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    image_description = result["choices"][0]["message"]["content"]
                    logger.info(f"Successfully processed image and generated description")
                    # 使用图片描述和用户问题生成最终回复
                    # 从lg_prompts导入电商客服模板
                    
                    # 构建回复请求
                    if settings.AGENT_SERVICE == ServiceType.DEEPSEEK:
                        model = ChatDeepSeek(api_key=settings.DEEPSEEK_API_KEY, model_name=settings.DEEPSEEK_MODEL, temperature=0.7, tags=["image_query"])
                    else:
                        model = ChatOllama(model=settings.OLLAMA_AGENT_MODEL, base_url=settings.OLLAMA_BASE_URL, temperature=0.7, tags=["image_query"])
                    # 使用专门的图片查询提示模板
                    system_prompt = GET_IMAGE_SYSTEM_PROMPT.format(
                        image_description=image_description
                    )
                    messages = [{"role": "system", "content": system_prompt}] + state.messages
                    response = await model.ainvoke(messages)
                    return {"messages": [response]}    
        
                else:
                    error_text = await response.text()
                    logger.error(f"Vision API Request Failed: {response.status} - {error_text}")
                    return {"messages": [AIMessage(content=f"抱歉，我无法查看这张图片，请重新上传。")]}





    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return {"messages": [AIMessage(content=f"抱歉，我无法查看这张图片，请重新上传。")]}

async def create_file_query(
    state: AgentState, *, config: RunnableConfig
) -> Dict[str, List[BaseMessage]]:
    """Create a file query."""
    
    # TODO

async def create_research_plan(
    state: AgentState, *, config: RunnableConfig
) -> Dict[str, List[str] | str]:
    """通过查询本地知识库回答客户问题，执行任务分解，创建分布查询计划。

    Args:
        state (AgentState): 当前代理状态，包括对话历史。
        config (RunnableConfig): 用于配置计划生成的模型。

    Returns:
        Dict[str, List[str] | str]: 包含'steps'键的字典，其中包含研究步骤列表。
    """
    logger.info("------execute local knowledge base query------")

    # 使用大模型生成查询/多跳、并行查询计划
    if settings.AGENT_SERVICE == ServiceType.CLAUDE:
        model = ChatAnthropic(api_key=settings.ANTHROPIC_API_KEY, base_url=settings.ANTHROPIC_BASE_URL, model=settings.ANTHROPIC_MODEL, temperature=settings.AGENT_WORKFLOW_TEMPERATURE)
        text2cypher_model = ChatAnthropic(
            api_key=settings.ANTHROPIC_API_KEY,
            base_url=settings.ANTHROPIC_BASE_URL,
            model=settings.ANTHROPIC_MODEL,
            temperature=settings.TEXT2CYPHER_GENERATION_TEMPERATURE,
        )
    elif settings.AGENT_SERVICE == ServiceType.DEEPSEEK:
        model = ChatDeepSeek(api_key=settings.DEEPSEEK_API_KEY, model_name=settings.DEEPSEEK_MODEL, temperature=settings.AGENT_WORKFLOW_TEMPERATURE, tags=["research_plan"])
        text2cypher_model = ChatDeepSeek(
            api_key=settings.DEEPSEEK_API_KEY,
            model_name=settings.DEEPSEEK_MODEL,
            temperature=settings.TEXT2CYPHER_GENERATION_TEMPERATURE,
            tags=["text2cypher_generation"],
        )
    else:
        model = ChatOllama(model=settings.OLLAMA_AGENT_MODEL, base_url=settings.OLLAMA_BASE_URL, temperature=settings.AGENT_WORKFLOW_TEMPERATURE, tags=["research_plan"])
        text2cypher_model = ChatOllama(
            model=settings.OLLAMA_AGENT_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=settings.TEXT2CYPHER_GENERATION_TEMPERATURE,
            tags=["text2cypher_generation"],
        )
    
    # 初始化必要参数
    # 1. Neo4j图数据库连接 - 使用配置中的连接信息
    try:
        neo4j_graph = get_neo4j_graph()
        logger.info("success to get Neo4j graph database connection")
    except Exception as e:
        logger.error(f"failed to get Neo4j graph database connection: {e}")

    # 2. 将示例向量化存入 Neo4j，并通过向量索引检索 Top-K Few-shot 示例。
    try:
        cypher_retriever = create_neo4j_vector_example_retriever(neo4j_graph)
        logger.info(
            f"Cypher example vector retriever initialized: "
            f"index={settings.CYPHER_EXAMPLE_VECTOR_INDEX}, "
            f"top_k={settings.CYPHER_EXAMPLE_TOP_K}"
        )
    except Exception as e:
        logger.error(f"failed to initialize Cypher example vector retriever: {e}")
        return {
            "messages": [
                AIMessage(content=f"初始化Text2Cypher向量示例库失败: {str(e)}")
            ]
        }

    # step 3. 定义工具模式列表    
    from app.lg_agent.kg_sub_graph.kg_tools_list import cypher_query, predefined_cypher, microsoft_graphrag_query
    tool_schemas: List[type[BaseModel]] = [cypher_query, predefined_cypher, microsoft_graphrag_query]

    # 3. 预定义的Cypher查询 - 为电商场景定义有用的查询
    from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.predefined_cypher.cypher_dict import predefined_cypher_dict

    # 定义电商经营范围
    scope_description = """
    个人电商经营范围：智能家居产品，包括但不限于：
    - 智能照明（灯泡、灯带、开关）
    - 智能安防（摄像头、门锁、传感器）
    - 智能控制（温控器、遥控器、集线器）
    - 智能音箱（语音助手、音响）
    - 智能厨电（电饭煲、冰箱、洗碗机）
    - 智能清洁（扫地机器人、洗衣机）
    
    不包含：服装、鞋类、体育用品、化妆品、食品等非智能家居产品。
    """

    # 创建多工具工作流 - 使用 agentic_rag_agents 的完整实现
    logger.info("Creating multi-tool workflow using agentic_rag_agents...")
    try:
        from app.lg_agent.kg_sub_graph.agentic_rag_agents.workflows.multi_agent.multi_tool import create_multi_tool_workflow as create_agentic_workflow

        multi_tool_workflow = create_agentic_workflow(
            llm=model,
            text2cypher_llm=text2cypher_model,
            graph=neo4j_graph,
            tool_schemas=tool_schemas,
            predefined_cypher_dict=predefined_cypher_dict,
            cypher_example_retriever=cypher_retriever,
            scope_description=scope_description,
            llm_cypher_validation=True,
            max_retries=3,
            cypher_example_top_k=settings.CYPHER_EXAMPLE_TOP_K,
        )
        logger.info("Multi-tool workflow created successfully")
    except Exception as e:
        logger.error(f"Failed to create multi-tool workflow: {e}", exc_info=True)
        return {"messages": [AIMessage(content=f"创建工作流失败: {str(e)}")]}

    # 准备输入状态（不包含 data 字段）
    last_message = state.messages[-1].content if state.messages else ""
    input_state = {
        "question": last_message,
        "history": []
    }

    logger.info(f"Executing workflow with question: {last_message}")
    logger.info(f"Input state: {input_state}")
    # 执行工作流
    try:
        import asyncio
        workflow_started = time.perf_counter()
        logger.info("About to invoke workflow...")
        response = await asyncio.wait_for(
            multi_tool_workflow.ainvoke(input_state),
            timeout=60.0  # 60秒超时
        )
        logger.info(f"Workflow executed successfully, answer: {response.get('answer', 'N/A')[:100]}...")
        trace = {
            "status": "completed",
            "question": last_message,
            "tasks": _trace_json_value(response.get("tasks", [])),
            "next_action": response.get("next_action"),
            "steps": list(response.get("steps", [])),
            "tool_outputs": _trace_json_value(response.get("cyphers", [])),
            "summary": response.get("summary", ""),
            "answer": response.get("answer", ""),
            "subgraph_latency_ms": round(
                (time.perf_counter() - workflow_started) * 1000, 2
            ),
        }
        return {
            "messages": [AIMessage(content=response["answer"])],
            "research_trace": trace,
            "steps": ["create_research_plan"],
        }
    except asyncio.TimeoutError:
        logger.error("Workflow execution timed out after 60 seconds")
        return {
            "messages": [AIMessage(content="查询超时，请稍后重试")],
            "research_trace": {
                "status": "timeout",
                "question": last_message,
                "subgraph_latency_ms": round(
                    (time.perf_counter() - workflow_started) * 1000, 2
                ),
            },
            "steps": ["create_research_plan"],
        }
    except Exception as e:
        logger.error(f"Failed to execute workflow: {e}", exc_info=True)
        import traceback
        error_detail = traceback.format_exc()
        logger.error(f"Full traceback: {error_detail}")
        return {
            "messages": [AIMessage(content=f"执行查询失败: {str(e)}")],
            "research_trace": {
                "status": "error",
                "question": last_message,
                "error_type": type(e).__name__,
                "error": str(e),
                "subgraph_latency_ms": round(
                    (time.perf_counter() - workflow_started) * 1000, 2
                ),
            },
            "steps": ["create_research_plan"],
        }

async def check_hallucinations(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, Any]:
    """Analyze the user's query and checks if the response is supported by the set of facts based on the document retrieved,
    providing a binary score result.

    This function uses a language model to analyze the user's query and gives a binary score result.

    Args:
        state (AgentState): The current state of the agent, including conversation history.
        config (RunnableConfig): Configuration with the model used for query analysis.

    Returns:
        dict[str, Router]: A dictionary containing the 'router' key with the classification result (classification type and logic).
    """
    if settings.AGENT_SERVICE == ServiceType.CLAUDE:
        # Claude 不支持流式结构化输出，需要禁用流式
        model = ChatAnthropic(
            api_key=settings.ANTHROPIC_API_KEY,
            base_url=settings.ANTHROPIC_BASE_URL,
            model=settings.ANTHROPIC_MODEL,
            temperature=0.7,
            streaming=False
        )
    elif settings.AGENT_SERVICE == ServiceType.DEEPSEEK:
        model = ChatDeepSeek(api_key=settings.DEEPSEEK_API_KEY, model_name=settings.DEEPSEEK_MODEL, temperature=0.7, tags=["hallucinations"])
    else:
        model = ChatOllama(model=settings.OLLAMA_AGENT_MODEL, base_url=settings.OLLAMA_BASE_URL, temperature=0.7, tags=["hallucinations"])
    
    system_prompt = CHECK_HALLUCINATIONS.format(
        documents=state.documents,
        generation=state.messages[-1]
    )

    messages = [
        {"role": "system", "content": system_prompt}
    ] + state.messages

    logger.info("---CHECK HALLUCINATIONS---")
    
    response = cast(GradeHallucinations, await model.with_structured_output(GradeHallucinations).ainvoke(messages))
    
    return {"hallucination": response} 


# 定义持久化存储，也可以使用SQLiteSaver()、PostgresSaver()等
# LangGraph官方地址：https://langchain-ai.github.io/langgraph/how-tos/persistence/
checkpointer = MemorySaver()

# 定义状态图
builder = StateGraph(AgentState, input=InputState)
# 添加节点
builder.add_node(analyze_and_route_query)
builder.add_node(respond_to_general_query)
builder.add_node(get_additional_info)
builder.add_node("create_research_plan", create_research_plan)  # 这里是子图
builder.add_node(create_image_query)
builder.add_node(create_file_query)

# 添加边
builder.add_edge(START, "analyze_and_route_query")
builder.add_conditional_edges("analyze_and_route_query", route_query)


graph = builder.compile(checkpointer=checkpointer)

# from IPython.display import Image, display
# display(Image(graph.get_graph().draw_mermaid_png()))
