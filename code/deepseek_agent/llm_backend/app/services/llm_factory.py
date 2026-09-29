from typing import Union
from app.core.config import settings, ServiceType
from app.services.claude_service import ClaudeService
from app.services.deepseek_service import DeepseekService
from app.services.ollama_service import OllamaService
from app.services.search_service import SearchService

class LLMFactory:
    @staticmethod
    def create_chat_service():
        """创建聊天服务实例"""
        if settings.CHAT_SERVICE == ServiceType.CLAUDE:
            return ClaudeService()
        elif settings.CHAT_SERVICE == ServiceType.DEEPSEEK:
            return DeepseekService()
        else:
            return OllamaService()

    @staticmethod
    def create_reasoner_service():
        """创建推理服务实例"""
        if settings.REASON_SERVICE == ServiceType.CLAUDE:
            return ClaudeService()
        elif settings.REASON_SERVICE == ServiceType.DEEPSEEK:
            return DeepseekService()
        else:
            return OllamaService()

    @staticmethod
    def create_search_service():
        """创建搜索服务实例"""
        return SearchService()