# rag_helpers.py
import json
import sqlite3
import requests
import faiss
import numpy as np
import aiohttp
from typing import Iterable, List, Dict, Any, Tuple
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch
import asyncio

from .config_imports import (
    logger, OLLAMA_HOST, EMBEDDING_MODEL, CHAT_MODEL, COMPRESSOR_MODEL,
    CACHE_FILE, DB_PATH, FAISS_INDEX_FILE, METADATA_FILE,
    OLLAMA_CHAT_TEMPERATURE_DEFAULT, OLLAMA_COMPRESSOR_TEMPERATURE_DEFAULT
)
from .rag_exceptions import ModelConnectionError

class RAGHelpers:
    """包含所有静态辅助方法的混合类或工具集"""

    @staticmethod
    def _ollama_headers() -> dict:
        """获取 Ollama API 请求头。"""
        return {"Content-Type": "application/json"}

    @staticmethod
    def _chat_completion(ollama_host: str, messages: List[dict], model_name: str, temperature: float) -> str:
        """
        同步非流式聊天完成。
        """
        url = f"{ollama_host}/api/chat"
        data = {
            "model": model_name,
            "messages": messages,
            "options": {"temperature": temperature},
            "stream": False
        }
        try:
            response = requests.post(url, headers=RAGHelpers._ollama_headers(), json=data, timeout=60)
            response.raise_for_status()
            return response.json().get('message', {}).get('content', '')
        except requests.exceptions.RequestException as e:
            logger.error(f"聊天请求失败: {e}")
            raise ModelConnectionError(f"无法连接到 Ollama 服务: {e}")

    @staticmethod
    def _chat_completion_stream(ollama_host: str, messages: List[dict], model_name: str, temperature: float) -> Iterable[str]:
        """
        同步流式聊天完成。
        """
        url = f"{ollama_host}/api/chat"
        data = {
            "model": model_name,
            "messages": messages,
            "options": {"temperature": temperature},
            "stream": True
        }
        try:
            with requests.post(url, headers=RAGHelpers._ollama_headers(), json=data, stream=True, timeout=60) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line.decode('utf-8'))
                            if chunk.get('done'):
                                break
                            yield chunk['message']['content']
                        except json.JSONDecodeError:
                            continue
        except requests.exceptions.RequestException as e:
            logger.error(f"流式聊天请求失败: {e}")
            raise ModelConnectionError(f"无法连接到 Ollama 服务: {e}")

    @staticmethod
    def load_embedding_cache() -> Dict[str, List[float]]:
        """
        从文件加载嵌入缓存。
        """
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    logger.info(f"从 {CACHE_FILE} 加载了 {len(cache_data)} 个缓存向量。")
                    return cache_data
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.warning(f"缓存文件 {CACHE_FILE} 损坏或不存在: {e}，将创建新的缓存。")
        return {}

    @staticmethod
    def save_embedding_cache(cache: Dict[str, List[float]], cache_file_path: Path) -> None:
        """将当前缓存保存到指定文件。"""
        with open(cache_file_path, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        logger.info(f"嵌入缓存已保存到 {cache_file_path}。")

    # --- SQLite helpers ---
    @staticmethod
    def ensure_schema(conn: sqlite3.Connection) -> None:
        """确保 SQLite 数据库表结构存在。"""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                chunk_id INTEGER PRIMARY KEY, path TEXT, chunk_index INTEGER, content TEXT
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_path ON metadata(path);")
        conn.commit()

    @staticmethod
    def bulk_insert_metadata(conn: sqlite3.Connection, records: List[Tuple]) -> None:
        """批量插入元数据到数据库。"""
        conn.executemany("INSERT OR REPLACE INTO metadata (chunk_id, path, chunk_index, content) VALUES (?, ?, ?, ?)", records)
        conn.commit()

    @staticmethod
    def get_metadata_by_chunk_id(conn: sqlite3.Connection, chunk_id: int) -> Dict[str, Any]:
        """
        根据 chunk_id 从数据库获取元数据。
        """
        cursor = conn.cursor()
        cursor.execute("SELECT path, chunk_index, content FROM metadata WHERE chunk_id = ?", (chunk_id,))
        result = cursor.fetchone()
        if result:
            return {"path": result[0], "chunk_index": result[1], "content": result[2]}
        return {}

    # --- FAISS helpers ---
    @staticmethod
    def create_initial_faiss_index(dimension: int) -> faiss.Index:
        """
        创建一个内积（IP）类型的 FAISS 索引。
        """
        return faiss.IndexFlatIP(dimension)

    @staticmethod
    def add_vectors_to_faiss_index(index: faiss.Index, vectors: np.ndarray) -> None:
        """
        将向量添加到 FAISS 索引中。
        """
        if vectors.size > 0:
            faiss.normalize_L2(vectors)
        index.add(vectors)

    @staticmethod
    def save_faiss_index_and_metadata(index: faiss.Index, metadata_list: List[dict], faiss_file: Path, metadata_file: Path) -> None:
        """
        保存 FAISS 索引和元数据映射文件。
        """
        faiss.write_index(index, str(faiss_file))
        simplified_metadata = [{"chunk_id": m["chunk_id"]} for m in metadata_list]
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(simplified_metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"[构建] FAISS 索引和元数据文件已保存。")

    @staticmethod
    def load_faiss_index_and_metadata(faiss_file: Path, metadata_file: Path) -> Tuple[faiss.Index, List[dict]]:
        """
        加载 FAISS 索引和元数据映射文件。
        """
        index = faiss.read_index(str(faiss_file))
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        logger.info(f"[查询] FAISS 索引和元数据加载完成，维度: {index.d}, 总块数: {index.ntotal}")
        return index, metadata