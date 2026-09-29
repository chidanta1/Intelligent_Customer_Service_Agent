from typing import Any, Callable, Coroutine, Dict, List
import asyncio
import os
import time
from pathlib import Path
from pydantic import BaseModel, Field

# 导入GraphRAG相关模块
import app.graphrag.graphrag.api as api
from app.graphrag.graphrag.config.load_config import load_config
from app.graphrag.graphrag.callbacks.noop_query_callbacks import NoopQueryCallbacks
from app.graphrag.graphrag.utils.storage import load_table_from_storage
from app.graphrag.graphrag.storage.file_pipeline_storage import FilePipelineStorage
from app.graphrag.graphrag.language_model.providers.local_hash_embedding import (
    LocalHashEmbedding,
)
from graphrag.language_model.factory import ModelFactory

# 导入配置
from app.core.config import ROOT_DIR, settings

# The backend normally resolves GraphRAG's absolute imports from the installed
# package, while the index builder resolves the vendored package.  Register the
# local provider in both processes before configuration validation.
if not ModelFactory.is_supported_embedding_model("local_hash_embedding"):
    ModelFactory.register_embedding(
        "local_hash_embedding", lambda **kwargs: LocalHashEmbedding(**kwargs)
    )


def _compact_context_evidence(value: Any) -> Any:
    """Keep retrieval evidence useful without serializing full DataFrames."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value if not isinstance(value, str) else value[:2000]
    if hasattr(value, "head") and hasattr(value, "to_dict"):
        try:
            return value.head(20).to_dict(orient="records")
        except (TypeError, ValueError):
            return str(value)[:2000]
    if isinstance(value, dict):
        return {
            str(key): _compact_context_evidence(item)
            for key, item in list(value.items())[:20]
        }
    if isinstance(value, (list, tuple)):
        return [_compact_context_evidence(item) for item in value[:20]]
    return str(value)[:2000]

# 定义GraphRAG查询的输入状态类型
class GraphRAGQueryInputState(BaseModel):
    task: str
    query: str
    steps: List[str]

# 定义GraphRAG查询的输出状态类型
class GraphRAGQueryOutputState(BaseModel):
    task: str
    query: str
    errors: List[str]
    records: Dict[str, Any]
    steps: List[str]

# 定义GraphRAG API包装器
class GraphRAGAPI:
    def __init__(self, project_dir: str = None, 
                 data_dir_name: str = None,
                 query_type: str = None,
                 response_type: str = None,
                 community_level: int = None,
                 dynamic_community_selection: bool = None):
        # 从环境变量获取配置，如果提供了参数则使用参数值
        self.project_dir = project_dir or settings.GRAPHRAG_PROJECT_DIR
        self.data_dir_name = data_dir_name or settings.GRAPHRAG_DATA_DIR
        self.query_type = query_type or settings.GRAPHRAG_QUERY_TYPE
        self.response_type = response_type or settings.GRAPHRAG_RESPONSE_TYPE
        self.community_level = community_level or settings.GRAPHRAG_COMMUNITY_LEVEL
        self.dynamic_community_selection = dynamic_community_selection if dynamic_community_selection is not None else settings.GRAPHRAG_DYNAMIC_COMMUNITY
        self.config = None
        self.storage = None
        self.entities = None
        self.text_units = None
        self.communities = None
        self.community_reports = None
        self.relationships = None
        self.covariates = None
        self.initialized = False
    
    async def initialize(self):
        """初始化GraphRAG API，加载必要的数据"""
        if self.initialized:
            return

        # Reuse the application's configured DeepSeek credential without
        # copying a secret into the GraphRAG project directory.
        if settings.DEEPSEEK_API_KEY:
            os.environ.setdefault("GRAPHRAG_API_KEY", settings.DEEPSEEK_API_KEY)

        # 构建完整项目路径。旧项目的 .env 可能仍保留 Windows 绝对路径，
        # 此时回退到当前后端内的实际 GraphRAG 数据目录。
        configured_directory = Path(self.project_dir) / self.data_dir_name
        local_directory = ROOT_DIR / "app" / "graphrag" / self.data_dir_name
        project_directory = (
            configured_directory
            if (configured_directory / "settings.yaml").is_file()
            else local_directory
        )
        
        # 加载配置
        self.config = load_config(Path(project_directory), None, None)
        
        # 创建存储路径
        output_dir = Path(self.config.output.base_dir)
        if not output_dir.is_absolute():
            output_dir = Path(project_directory) / output_dir
        
        # 创建FilePipelineStorage对象
        self.storage = FilePipelineStorage(root_dir=str(output_dir))
        
        # 加载必要的数据文件
        try:
            self.entities = await load_table_from_storage("entities", self.storage)
            self.text_units = await load_table_from_storage("text_units", self.storage)
            self.communities = await load_table_from_storage("communities", self.storage)
            self.community_reports = await load_table_from_storage("community_reports", self.storage)
            self.relationships = await load_table_from_storage("relationships", self.storage)
            
            # 尝试加载协变量数据（可能不存在）
            try:
                self.covariates = await load_table_from_storage("covariates", self.storage)
            except Exception:
                self.covariates = None
            
            self.initialized = True
        except Exception as e:
            raise Exception(f"加载GraphRAG数据文件时出错: {str(e)}")
    
    async def query_graphrag(self, query: str) -> Dict[str, Any]:
        """执行GraphRAG查询"""
        await self.initialize()
        
        # 创建回调对象
        callbacks = []
        context_data = {}
        
        def on_context(context):
            nonlocal context_data
            context_data = context
        
        local_callbacks = NoopQueryCallbacks()
        local_callbacks.on_context = on_context
        callbacks.append(local_callbacks)
        
        try:
            # 根据查询类型执行不同的查询
            if self.query_type.lower() == "local":
                response, context = await api.local_search(
                    config=self.config,
                    entities=self.entities,
                    communities=self.communities,
                    community_reports=self.community_reports,
                    text_units=self.text_units,
                    relationships=self.relationships,
                    covariates=self.covariates,
                    community_level=self.community_level,
                    response_type=self.response_type,
                    query=query,
                    callbacks=callbacks
                )
            
            elif self.query_type.lower() == "global":
                response, context = await api.global_search(
                    config=self.config,
                    entities=self.entities,
                    communities=self.communities,
                    community_reports=self.community_reports,
                    community_level=self.community_level,
                    dynamic_community_selection=self.dynamic_community_selection,
                    response_type=self.response_type,
                    query=query,
                    callbacks=callbacks
                )
            
            elif self.query_type.lower() == "drift":
                response, context = await api.drift_search(
                    config=self.config,
                    entities=self.entities,
                    communities=self.communities,
                    community_reports=self.community_reports,
                    text_units=self.text_units,
                    relationships=self.relationships,
                    community_level=self.community_level,
                    response_type=self.response_type,
                    query=query,
                    callbacks=callbacks
                )
            
            elif self.query_type.lower() == "basic":
                response, context = await api.basic_search(
                    config=self.config,
                    text_units=self.text_units,
                    query=query,
                    callbacks=callbacks
                )
            
            else:
                raise ValueError(f"不支持的查询类型: {self.query_type}")
            
            # 构建结果字典
            result = {
                "response": response,
                "context": context_data
            }
            
            return result
            
        except Exception as e:
            raise Exception(f"执行GraphRAG查询时出错: {str(e)}")

def create_graphrag_query_node(
) -> Callable[
    [GraphRAGQueryInputState],
    Coroutine[Any, Any, Dict[str, List[GraphRAGQueryOutputState] | List[str]]],
]:
    """
    创建GraphRAG查询节点，用于LangGraph工作流。

    返回
    -------
    Callable[[GraphRAGQueryInputState], Dict[str, List[GraphRAGQueryOutputState] | List[str]]]
        名为`graphrag_query`的LangGraph节点。
    """

    async def graphrag_query(
        state: Dict[str, Any],
    ) -> Dict[str, List[GraphRAGQueryOutputState] | List[str]]:
        """
        执行GraphRAG查询并返回结果。
        """
        started = time.perf_counter()
        errors = list()
        search_result = {}
        
        # 获取查询文本
        query = state.get("task", "")
        if not query:
            errors.append("未提供查询文本")
        else:
            try:
                from loguru import logger

                # 使用环境变量中的配置创建GraphRAGAPI实例
                graphrag_api = GraphRAGAPI()
                # 调用GraphRAG API获取数据
                search_result = await graphrag_api.query_graphrag(query)

                # 添加日志查看返回结果
                logger.info(f"GraphRAG Query - Result type: {type(search_result)}")
                logger.info(f"GraphRAG Query - Result keys: {search_result.keys() if isinstance(search_result, dict) else 'Not a dict'}")
                logger.info(f"GraphRAG Query - Result: {search_result}")

                # 安全地获取 response，兼容不同的返回格式
                if isinstance(search_result, dict):
                    response_data = search_result.get("response") or search_result.get("result") or search_result.get("answer") or str(search_result)
                else:
                    response_data = str(search_result)

            except Exception as e:
                errors.append(f"GraphRAG查询失败: {str(e)}")
                response_data = ""
                from loguru import logger
                logger.error(f"GraphRAG Query Error: {str(e)}")

            return {
                "cyphers": [
                    GraphRAGQueryOutputState(
                        **{
                            "task": state.get("task", ""),
                            "query": query,
                            "statement": "",
                            "parameters":"",
                            "errors": errors,
                            "records": {
                                "result": response_data,
                                "context_evidence": _compact_context_evidence(
                                    search_result.get("context")
                                    if isinstance(search_result, dict)
                                    else None
                                ),
                            },
                            "steps": ["execute_graphrag_query"],
                        }
                    )
                ],
                "steps": list(state.get("steps", [])) + [
                    "execute_graphrag_query",
                    f"timing:microsoft_graphrag_query:{(time.perf_counter() - started) * 1000:.2f}",
                ],
            }
  
    return graphrag_query

