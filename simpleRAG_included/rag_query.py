# rag_query.py
import requests
import numpy as np
import faiss
import torch
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer

from .config_imports import (
    logger, OLLAMA_HOST, CHAT_MODEL, COMPRESSOR_MODEL, DEFAULT_TOP_K, 
    DEFAULT_TOP_K_COMPRESSED, DEFAULT_THRESHOLD, OLLAMA_CHAT_TEMPERATURE_DEFAULT, 
    OLLAMA_COMPRESSOR_TEMPERATURE_DEFAULT, DB_PATH, FAISS_INDEX_FILE, METADATA_FILE
)
from .rag_helpers import RAGHelpers
from .rag_exceptions import ModelConnectionError
import prompts

try:
    from FlagEmbedding import FlagReranker
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    logger.warning("未安装 FlagEmbedding，重排功能将不可用。")

class RAGQuerier:
    def __init__(self, ollama_host: str, chat_model: str, compressor_model: str, reranker_model_name: str):
        self._ollama_host = ollama_host
        self._chat_model = chat_model
        self._compressor_model = compressor_model
        self._reranker_model_name = reranker_model_name
        self._reranker_model_path = f"./models/{reranker_model_name}"
        self._reranker = None
        self.embedding_model = None  # 将由 SimpleRAG 注入

    def set_embedding_model(self, model_instance: SentenceTransformer):
        self.embedding_model = model_instance
        logger.debug("RAGQuerier 已注入嵌入模型实例。")

    def _load_reranker(self):
        if self._reranker is None and RERANKER_AVAILABLE:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._reranker = FlagReranker(self._reranker_model_path, use_fp16=True, device=device)
                logger.info(f"重排模型加载成功：{self._reranker_model_path}")
            except Exception as e:
                logger.error(f"重排模型加载失败：{e}")

    def _rerank_results(self, query: str, results: list, top_k: int = 5) -> list:
        if not results or not query:
            return results
        if not RERANKER_AVAILABLE or self._reranker is None:
            self._load_reranker()
            if not self._reranker:
                return results[:top_k]

        pairs = [[query, item['content']] for item in results]
        try:
            scores = self._reranker.compute_score(pairs, normalize=True)
            if not isinstance(scores, list):
                scores = [scores] * len(results)
            
            for i, res in enumerate(results):
                res['rerank_score'] = float(scores[i])
            
            return sorted(results, key=lambda x: x['rerank_score'], reverse=True)[:top_k]
        except Exception as e:
            logger.error(f"重排失败：{e}")
            return results[:top_k]

    def search_similar_with_faiss(self, query_vec: List[float], top_k: int, score_threshold: float) -> List[Tuple[float, int]]:
        try:
            index, metadata_map = RAGHelpers.load_faiss_index_and_metadata(FAISS_INDEX_FILE, METADATA_FILE)
        except Exception as e:
            logger.error(f"加载 FAISS 失败：{e}")
            return []
            
        query_array = np.array([query_vec], dtype=np.float32)
        faiss.normalize_L2(query_array)
        scores, indices = index.search(query_array, top_k)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            score = scores[0][i]
            if score >= score_threshold and idx < len(metadata_map):
                chunk_id = metadata_map[idx]["chunk_id"]
                results.append((float(score), chunk_id))
        return results

    def compress_contexts(self, retrieved_results: List[Dict[str, Any]], compressor_model: str = None, temperature: float = OLLAMA_COMPRESSOR_TEMPERATURE_DEFAULT) -> str:
        if not retrieved_results:
            return ""
        if compressor_model is None:
            compressor_model = self._compressor_model
            
        messages = prompts.get_compress_prompt_template(retrieved_results)
        return RAGHelpers._chat_completion(self._ollama_host, messages, compressor_model, temperature)

    def prepare_final_prompt(self, question: str, contexts: List[Dict[str, Any]], compressed_context: str) -> List[dict]:
        return prompts.get_rag_prompt_template(compressed_context if compressed_context.strip() else "\n".join([c['content'] for c in contexts]), question)