
import logging
from pathlib import Path
import numpy as np


from config import (

    OLLAMA_HOST,
    CHAT_MODEL,
    EMBEDDING_MODEL,
    RERANK_MODEL,

    CACHE_FILE,
    DB_PATH,
    DOC_DIR,
    FAISS_INDEX_FILE,
    METADATA_FILE,
    CONVERSATION_STATE_FILE,

    BATCH_SIZE_DOCS,
    CHUNK_SIZE_DEFAULT,
    CHUNK_OVERLAP_DEFAULT,
    LOCAL_EMBEDDING_MODEL_PATH,
    SEMANTIC_SPLITTER_MODEL_NAME,
    SEMANTIC_SPLITTER_THRESHOLD,
    SEMANTIC_SPLITTER_SEPARATORS,
    OLLAMA_EMBED_BATCH_SIZE_INTERNAL,

    DEFAULT_TOP_K,
    DEFAULT_TOP_K_COMPRESSED,
    DEFAULT_THRESHOLD,
    MIN_RETRIEVE_KEEP,
    RECENT_HISTORY_TOKEN_BUDGET,
    HISTORY_SUMMARY_TRIGGER_TOKENS,
    OLLAMA_CHAT_TEMPERATURE_DEFAULT,
)


logger = logging.getLogger("lightweightrag")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

for noisy_logger_name in (
    "httpx",
    "httpcore",
    "urllib3",
    "faiss",
    "faiss.loader",
):
    logging.getLogger(noisy_logger_name).setLevel(logging.WARNING)
