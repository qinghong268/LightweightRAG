class RAGException(Exception):
    """Base exception for RAG-related failures."""


class ModelConnectionError(RAGException):
    """Raised when the Ollama service or model backend is unavailable."""


class BuildError(RAGException):
    """Raised when knowledge-base build steps fail."""


class SnapshotLoadError(RAGException):
    """Raised when the FAISS/metadata snapshot cannot be loaded consistently."""
