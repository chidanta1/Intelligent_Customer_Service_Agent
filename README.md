# Intelligent Customer Service Agent

一个基于 FastAPI、Vue 3、Agent 与 GraphRAG 构建的智能客服项目，支持 DeepSeek、Ollama、MySQL、Redis 和 Neo4j。

## 项目内容

- 主要源码：[code/deepseek_agent](code/deepseek_agent)
- 完整功能与开发说明：[code/deepseek_agent/README.md](code/deepseek_agent/README.md)
- 环境变量示例：[code/deepseek_agent/.env.example](code/deepseek_agent/.env.example)

## 快速开始

1. 进入 `code/deepseek_agent` 并安装 `requirements.txt` 中的依赖。
2. 将 `.env.example` 复制为 `llm_backend/.env`，然后填写自己的 API Key 和数据库连接信息。
3. 按照项目开发说明配置 MySQL、Redis、Neo4j 与模型服务。
4. 启动后端服务，默认访问地址为 `http://127.0.0.1:8000`。

## 安全说明

API Key、数据库文件、用户上传内容、缓存、日志和其他运行时数据不会提交到仓库。请勿将真实凭据写入示例配置或文档。

## License

项目包含的 GraphRAG 相关源码遵循其目录中的 MIT License。其他代码的授权信息请参阅项目内说明。
