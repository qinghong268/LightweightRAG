# rag_exceptions.py
class RAGException(Exception):
    """为 RAG 系统定义的自定义异常基类，便于统一错误处理。"""
    pass

class ModelConnectionError(RAGException):
    """当无法连接到 Ollama 或模型服务时抛出。"""
    pass

class BuildError(RAGException):
    """在构建知识库过程中发生错误时抛出。"""
    pass