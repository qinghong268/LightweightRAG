
"""
闁板秶鐤嗛弬鍥︽閿涘本顒濋弬鍥︽鐎规矮绠熸禍鍡樺閺堝褰查柊宥囩枂閻ㄥ嫬寮弫鑸偓?
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


OLLAMA_HOST = "http://localhost:11434"
CHAT_MODEL = "deepseek-r1:8b"
EMBEDDING_MODEL = "bge-m3"
RERANK_MODEL = "bge-reranker-v2-m3"


CACHE_FILE = BASE_DIR / "embedding_cache.json"
DB_PATH = BASE_DIR / "knowledge_base.db"
DOC_DIR = BASE_DIR / "docs"
FAISS_INDEX_FILE = BASE_DIR / "faiss_index.bin"
METADATA_FILE = BASE_DIR / "metadata.json"
CONVERSATION_STATE_FILE = BASE_DIR / "conversation_state.json"

DEFAULT_TOP_K = 5
DEFAULT_TOP_K_COMPRESSED = 3
DEFAULT_THRESHOLD = 0.3
MIN_RETRIEVE_KEEP = 2
RECENT_HISTORY_TOKEN_BUDGET = 600
HISTORY_SUMMARY_TRIGGER_TOKENS = 240


BATCH_SIZE_DOCS = 10
CHUNK_SIZE_DEFAULT = 220
CHUNK_OVERLAP_DEFAULT = 40


LOCAL_EMBEDDING_MODEL_PATH = str(BASE_DIR / "models" / "bge-m3")
SEMANTIC_SPLITTER_MODEL_NAME = "BAAI/bge-m3"
SEMANTIC_SPLITTER_THRESHOLD = 0.85
SEMANTIC_SPLITTER_SEPARATORS = ["\n\n", "\n", "。", "？", "！", "；", "?", "!", ";", "..."]


OLLAMA_CHAT_TEMPERATURE_DEFAULT = 0.7
OLLAMA_EMBED_BATCH_SIZE_INTERNAL = 5
