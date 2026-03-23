# rag_exceptions.py
class RAGException(Exception):
    """自定义异常基类，便于统一错误处理。"""
    pass

class ModelConnectionError(RAGException):
    """当无法连接到Ollama或模型服务时抛出。"""
    pass

class BuildError(RAGException):
    """在构建知识库过程中发生错误时抛出。"""
    pass