from typing import List, Dict, AsyncGenerator, Callable, Optional
from anthropic import AsyncAnthropic
from app.core.config import settings
import json
from app.core.logger import get_logger
from app.core.database import AsyncSessionLocal
from app.models.conversation import Conversation, DialogueType
from app.models.message import Message
from app.services.redis_semantic_cache import RedisSemanticCache
import time
import asyncio

logger = get_logger(service="claude")

class ClaudeService:
    def __init__(self, model: str = "claude-sonnet-4-6"):
        logger.info("Initializing Claude Service")
        self.client = AsyncAnthropic(
            api_key=settings.ANTHROPIC_API_KEY,
            base_url=settings.ANTHROPIC_BASE_URL
        )
        # 优先使用配置中的 ANTHROPIC_MODEL，其次使用传入的 model
        self.model = settings.ANTHROPIC_MODEL or model
        self.cache = RedisSemanticCache(prefix="claude")

    async def _stream_response(self, messages: List[Dict]) -> AsyncGenerator[str, None]:
        """内部流式响应生成器"""
        try:
            # 转换消息格式：OpenAI 格式 -> Anthropic 格式
            anthropic_messages = []
            system_message = None

            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    anthropic_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })

            # 构建请求参数
            request_params = {
                "model": self.model,
                "messages": anthropic_messages,
                "max_tokens": 4096
            }

            if system_message:
                request_params["system"] = system_message

            async with self.client.messages.stream(**request_params) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            logger.error(f"Error in _stream_response: {str(e)}", exc_info=True)
            raise

    async def generate_stream(
        self,
        messages: List[Dict],
        user_id: Optional[int] = None,
        conversation_id: Optional[int] = None,
        on_complete: Optional[Callable] = None
    ) -> AsyncGenerator[str, None]:
        """流式生成回复（通过分块模拟流式输出）"""
        try:
            start_time = time.time()

            # 使用非流式 API 获取完整响应
            complete_response = await self.generate(messages)

            # 将响应分块以模拟流式输出
            chunk_size = 10  # 每次发送的字符数
            for i in range(0, len(complete_response), chunk_size):
                chunk = complete_response[i:i + chunk_size]
                data = json.dumps({"content": chunk}, ensure_ascii=False)
                yield f"data: {data}\n\n"
                # 添加小延迟以模拟流式效果
                await asyncio.sleep(0.05)

            # 发送结束标记
            yield "data: [DONE]\n\n"

            end_time = time.time()
            logger.info(f"Stream generation completed in {end_time - start_time:.4f} seconds")

            # 如果有回调，执行回调
            if on_complete and user_id is not None and conversation_id is not None:
                await on_complete(user_id, conversation_id, messages, complete_response)

        except Exception as e:
            logger.error(f"Error in generate_stream: {str(e)}", exc_info=True)
            error_msg = json.dumps(f"生成回复时出错: {str(e)}", ensure_ascii=False)
            yield f"data: {error_msg}\n\n"

    async def generate(self, messages: List[Dict]) -> str:
        """非流式生成回复"""
        try:
            # 转换消息格式
            anthropic_messages = []
            system_message = None

            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    anthropic_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })

            # 构建请求参数
            request_params = {
                "model": self.model,
                "messages": anthropic_messages,
                "max_tokens": 4096
            }

            if system_message:
                request_params["system"] = system_message

            logger.info(f"Sending request to Claude API with {len(anthropic_messages)} messages")
            response = await self.client.messages.create(**request_params)

            # 检查响应
            if hasattr(response, 'content') and len(response.content) > 0:
                if hasattr(response.content[0], 'text'):
                    return response.content[0].text
                else:
                    logger.error(f"Unexpected content format: {response.content[0]}")
                    return f"响应格式错误: {response.content[0]}"
            else:
                logger.error(f"Empty or invalid response: {response}")
                return "收到空响应"

        except Exception as e:
            logger.error(f"Generation error: {str(e)}", exc_info=True)
            return f"生成回复时出错: {str(e)}"
