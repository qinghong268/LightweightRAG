# config_imports.py
import logging
from pathlib import Path
import numpy as np

# 导入配置
from config import (
    # Ollama
    OLLAMA_HOST,
    CHAT_MODEL,
    COMPRESSOR_MODEL,
    EMBEDDING_MODEL,
    RERANK_MODEL,
    # Files
    CACHE_FILE,
    DB_PATH,
    DOC_DIR,
    FAISS_INDEX_FILE,
    METADATA_FILE,
    # Build
    BATCH_SIZE_DOCS,
    CHUNK_SIZE_DEFAULT,
    CHUNK_OVERLAP_DEFAULT,
    LOCAL_EMBEDDING_MODEL_PATH,
    SEMANTIC_SPLITTER_MODEL_NAME,
    SEMANTIC_SPLITTER_THRESHOLD,
    SEMANTIC_SPLITTER_SEPARATORS,
    OLLAMA_EMBED_BATCH_SIZE_INTERNAL,
    # Query
    DEFAULT_TOP_K,
    DEFAULT_TOP_K_COMPRESSED,
    DEFAULT_THRESHOLD,
    MIN_RETRIEVE_KEEP,
    OLLAMA_CHAT_TEMPERATURE_DEFAULT,
    OLLAMA_COMPRESSOR_TEMPERATURE_DEFAULT,
)

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)