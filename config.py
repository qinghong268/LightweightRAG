# config.py
"""
配置文件，此文件定义了所有可配置的参数。
"""

from pathlib import Path

#Ollama服务
OLLAMA_HOST = "http://localhost:11434"
CHAT_MODEL = "deepseek-r1:8b"  # 用于最终回答生成的本地模型
COMPRESSOR_MODEL = "qwen:7b"  # 用于上下文压缩和优化的本地模型
EMBEDDING_MODEL = "bge-m3" # 用于知识库文档索引的本地模型
RERANK_MODEL = "bge-reranker-v2-m3" # 用于检索结果重排序的模型

#文件路径
CACHE_FILE = Path("embedding_cache.json") # 用于存储嵌入向量的缓存文件
DB_PATH = Path("knowledge_base.db")       # 知识库数据库文件
DOC_DIR = Path("docs")                    # 知识库文档目录
FAISS_INDEX_FILE = Path("faiss_index.bin")      # 存储FAISS索引的文件
METADATA_FILE = Path("metadata.json")           # 存储向量对应的元数据 (ID, path, chunk_index, content)

#RAG逻辑
DEFAULT_TOP_K = 5 # 检索Top-K个片段，为压缩提供更多素材
DEFAULT_TOP_K_COMPRESSED = 3 # 压缩后最终使用的片段数
DEFAULT_THRESHOLD = 0.3 # 默认相似度阈值

#构建阶段
BATCH_SIZE_DOCS = 10 # 每次处理的文档数量，可根据内存情况调整
CHUNK_SIZE_DEFAULT = 400 # 文档切片的默认长度
CHUNK_OVERLAP_DEFAULT = 50 # 文档切片的默认重叠长度

#文本分割器
LOCAL_EMBEDDING_MODEL_PATH = "./models/bge-m3"
SEMANTIC_SPLITTER_MODEL_NAME = "BAAI/bge-m3"
SEMANTIC_SPLITTER_THRESHOLD = 0.75
SEMANTIC_SPLITTER_SEPARATORS = ["\n\n", "\n", "。", "？", "！", "；", "?", "!", ";", "...", " ", ""]

#Ollama API调用
OLLAMA_CHAT_TEMPERATURE_DEFAULT = 0.7
OLLAMA_COMPRESSOR_TEMPERATURE_DEFAULT = 0.3
OLLAMA_EMBED_BATCH_SIZE_INTERNAL = 5 # 内部向量化批次大小