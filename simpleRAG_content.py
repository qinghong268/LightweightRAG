# simpleRAG_content.py
"""
SimpleRAG模块
将RAG的核心流程（构建、检索、压缩、生成）封装成一个类。
"""

import sqlite3
import numpy as np
import faiss
from pathlib import Path
from typing import Iterable, List, Dict, Any, Tuple

# 引入拆分后的模块
from simpleRAG_included.config_imports import (
    logger, OLLAMA_HOST, CHAT_MODEL, COMPRESSOR_MODEL, EMBEDDING_MODEL,
    LOCAL_EMBEDDING_MODEL_PATH, DEFAULT_TOP_K, DEFAULT_TOP_K_COMPRESSED,
    DEFAULT_THRESHOLD, OLLAMA_CHAT_TEMPERATURE_DEFAULT, DB_PATH, CACHE_FILE,
    FAISS_INDEX_FILE, METADATA_FILE, RERANK_MODEL, OLLAMA_COMPRESSOR_TEMPERATURE_DEFAULT,
    MIN_RETRIEVE_KEEP
)
from simpleRAG_included.rag_exceptions import RAGException, ModelConnectionError, BuildError
from simpleRAG_included.rag_helpers import RAGHelpers
from simpleRAG_included.rag_build import RAGBuilder
from simpleRAG_included.rag_query import RAGQuerier
import prompts
import torch
from sentence_transformers import SentenceTransformer
import json
import hashlib
import re

class SimpleRAG:
    """
    一个集成了文档加载、向量化、索引、检索和生成的完整RAG系统。
    """
    ANSWER_REPLACE_MARKER = "__RAG_REPLACE_ANSWER__"

    def __init__(self):
        logger.info("正在初始化 SimpleRAG 实例...")
        
        # 处理缓存(增加模型指纹校验)
        self.cache = {}
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                    raw_cache = json.load(f)
                    if "__model_name__" in raw_cache:
                        if raw_cache["__model_name__"] != EMBEDDING_MODEL:
                            logger.warning(f"检测到模型变更({raw_cache['__model_name__']} -> {EMBEDDING_MODEL})，清空旧缓存。")
                            self.cache = {}
                        else:
                            self.cache = {k: v for k, v in raw_cache.items() if k != "__model_name__"}
                            logger.info(f"缓存加载成功，包含{len(self.cache)}个向量。")
                    else:
                        logger.warning("旧格式缓存，为避免维度错误，已清空。请重新构建知识库。")
                        self.cache = {}
            except Exception as e:
                logger.error(f"读取缓存失败：{e}")
                self.cache = {}

        # 预加载嵌入模型（单例）
        logger.info(f"正在预加载嵌入模型：{LOCAL_EMBEDDING_MODEL_PATH} ...")
        try:
            self.embedding_model = SentenceTransformer(
                LOCAL_EMBEDDING_MODEL_PATH,
                trust_remote_code=True,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            logger.info("嵌入模型加载成功。")
        except Exception as e:
            logger.error(f"嵌入模型加载失败：{e}")
            raise e

        self._ollama_host = OLLAMA_HOST
        self._chat_model = CHAT_MODEL
        self._compressor_model = COMPRESSOR_MODEL
        
        # 初始化子模块
        self._builder = RAGBuilder(self.cache, embedding_model_instance=self.embedding_model)
        self._querier = RAGQuerier(self._ollama_host, self._chat_model, self._compressor_model, RERANK_MODEL)
        self._querier.set_embedding_model(self.embedding_model)  # 注入主模型

    async def build_knowledge_base_async(self, source_dir: Path, chunk_size: int = None, overlap: int = None) -> None:
        from simpleRAG_included.config_imports import CHUNK_SIZE_DEFAULT, CHUNK_OVERLAP_DEFAULT
        if chunk_size is None: chunk_size = CHUNK_SIZE_DEFAULT
        if overlap is None: overlap = CHUNK_OVERLAP_DEFAULT
        
        self.cache["__model_name__"] = EMBEDDING_MODEL
        await self._builder.build_knowledge_base_async(source_dir, chunk_size, overlap)

        save_cache = {k: v for k, v in self.cache.items() if k != "__model_name__"}
        save_cache["__model_name__"] = EMBEDDING_MODEL
        RAGHelpers.save_embedding_cache(save_cache, CACHE_FILE)
        logger.info("知识库构建完成。")

    def _rewrite_query(self, original_query: str) -> str:
        prompt = (
            f"请将以下用户问题改写为更适合在知识库中进行文档检索的形式。\n"
            f"要求：\n"
            f"1.补充缺失的关键上下文。\n"
            f"2.表达更清晰完整。\n"
            f"3.【重要】只输出改写后的问题本身，严禁输出任何分析、解释、前言、后缀、标记或换行符。\n\n"
            f"原始问题：{original_query}\n\n"
            f"改写后的问题："
        )
        url = f"{self._ollama_host}/api/chat"
        data = {
            "model": self._chat_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.1}
        }
        try:
            import requests
            response = requests.post(url, headers={"Content-Type": "application/json"}, json=data, timeout=60)
            response.raise_for_status()
            content = response.json()['message']['content'].strip()
            lines = [l.strip() for l in content.split('\n') if l.strip()]
            final_query = ""
            for line in lines:
                if not any(k in line for k in ["分析", "说明", "风格", "---", "**"]):
                    return line
                if len(line) > 5:
                    final_query = line
                    break
            return lines[-1] if lines else original_query
        except Exception as e:
            logger.warning(f"Query rewriting failed: {e}")
            return original_query

    def _rerank_results(self, query: str, results: list, top_k: int = 5) -> list:
        return self._querier._rerank_results(query, results, top_k)

    def compress_contexts(self, retrieved_results: List[Dict[str, Any]], compressor_model: str = None, temperature: float = None) -> str:
        if temperature is None:
            temperature = OLLAMA_COMPRESSOR_TEMPERATURE_DEFAULT
        return self._querier.compress_contexts(retrieved_results, compressor_model, temperature)

    def prepare_final_prompt(self, question: str, contexts: List[Dict[str, Any]], compressed_context: str) -> List[dict]:
        return self._querier.prepare_final_prompt(question, contexts, compressed_context)

    def search_similar_with_faiss(self, query_vec: List[float], top_k: int, score_threshold: float) -> List[Tuple[float, int]]:
        return self._querier.search_similar_with_faiss(query_vec, top_k, score_threshold, min_keep=MIN_RETRIEVE_KEEP)

    def _validate_citations(self, text: str, contexts: List[Dict[str, Any]]) -> bool:
        if not text.strip():
            return False
        matches = re.findall(r"\[source=([^\]#]+)#chunk(\d+)\]", text, flags=re.IGNORECASE)
        if not matches:
            return False

        valid_sources = {(str(c.get("path", "")).strip(), str(c.get("chunk_index", "")).strip()) for c in contexts}
        if not valid_sources:
            return False

        return any((source_path.strip(), chunk_idx.strip()) in valid_sources for source_path, chunk_idx in matches)

    def _build_raw_context(self, contexts: List[Dict[str, Any]]) -> str:
        return "\n".join(
            [f"[source={c['path']}#chunk{c['chunk_index']}] {c['content']}" for c in contexts]
        )

    def retrieve_contexts(self, question: str, top_k: int = DEFAULT_TOP_K, score_threshold: float = DEFAULT_THRESHOLD) -> List[Dict[str, Any]]:
        if not question.strip(): return []
        
        rewritten_query = self._rewrite_query(question)
        logger.debug(f"改写后查询：{rewritten_query}")
        
        with torch.no_grad():
            query_vec = self.embedding_model.encode(
                rewritten_query, 
                convert_to_numpy=True, 
                normalize_embeddings=True
            ).tolist()
            
        raw_results = self.search_similar_with_faiss(query_vec, top_k=top_k, score_threshold=score_threshold)
        
        conn = sqlite3.connect(DB_PATH)
        formatted_results = []
        for score, chunk_id in raw_results:
            metadata = RAGHelpers.get_metadata_by_chunk_id(conn, chunk_id)
            if metadata:
                formatted_results.append({"score": score, "chunk_id": chunk_id, **metadata})
        conn.close()
        return formatted_results

    def answer_question(self, question: str, top_k_retrieve: int = DEFAULT_TOP_K, top_k_compressed: int = DEFAULT_TOP_K_COMPRESSED, score_threshold: float = DEFAULT_THRESHOLD) -> str:
        rewritten_query = self._rewrite_query(question)
        retrieved_results = self.retrieve_contexts(question, top_k=top_k_retrieve, score_threshold=score_threshold)
        if not retrieved_results:
            return "检索未命中相关片段，请检查知识库或降低阈值。"

        logger.info(f"Reranker状态: {self._querier.get_reranker_status()}")
        reranked_results = self._rerank_results(rewritten_query, retrieved_results, top_k=top_k_compressed)
        compressed_context = self.compress_contexts(reranked_results)
        if not self._validate_citations(compressed_context, reranked_results):
            logger.warning("压缩上下文引用校验未通过，改为原始上下文。")
            compressed_context = self._build_raw_context(reranked_results)
        messages = self.prepare_final_prompt(question, reranked_results, compressed_context)
        answer = RAGHelpers._chat_completion(self._ollama_host, messages, model_name=self._chat_model, temperature=OLLAMA_CHAT_TEMPERATURE_DEFAULT)
        if self._validate_citations(answer, reranked_results):
            return answer

        logger.warning("回答引用校验未通过，降级为原始上下文重试。")
        raw_context = self._build_raw_context(reranked_results)
        fallback_messages = self.prepare_final_prompt(question, reranked_results, raw_context)
        return RAGHelpers._chat_completion(
            self._ollama_host,
            fallback_messages,
            model_name=self._chat_model,
            temperature=OLLAMA_CHAT_TEMPERATURE_DEFAULT
        )

    def answer_question_stream(self, question: str, top_k_retrieve: int = DEFAULT_TOP_K, top_k_compressed: int = DEFAULT_TOP_K_COMPRESSED, score_threshold: float = DEFAULT_THRESHOLD) -> Iterable[str]:
        if not question.strip():
            yield "问题不能为空。"
            return

        yield f"\n开始查询\n"
        try:
            fallback_retrieval_used = False
            citation_retry_used = False
            reranker_status = self._querier.get_reranker_status()
            index, metadata_list = RAGHelpers.load_faiss_index_and_metadata(FAISS_INDEX_FILE, METADATA_FILE)
            if index is None or len(metadata_list) == 0:
                yield "知识库为空或未正确加载。\n"
                return

            yield "步骤0:查询理解与改写\n"
            rewritten_query = self._rewrite_query(question)
            yield f"原始问题：{question}\n"
            yield f"改写后问题：{rewritten_query}\n\n"

            yield "步骤1:向量化问题\n"
            with torch.no_grad():
                question_vector = self.embedding_model.encode(
                    rewritten_query, 
                    convert_to_numpy=True, 
                    normalize_embeddings=True
                )
            question_vector = np.array([question_vector], dtype='float32')
            yield f"向量化完成(维度：{question_vector.shape[1]})\n"
            yield f"\n步骤2:在知识库中搜索Top-{top_k_retrieve}个相似片段 ---\n"
            raw_results = self.search_similar_with_faiss(
                question_vector[0].tolist(),
                top_k=top_k_retrieve,
                score_threshold=score_threshold
            )
            results_with_scores = []
            conn = sqlite3.connect(DB_PATH)
            for similarity_score, chunk_id in raw_results:
                full_metadata = RAGHelpers.get_metadata_by_chunk_id(conn, chunk_id)
                if full_metadata:
                    item = (similarity_score, full_metadata['path'], full_metadata['chunk_index'], full_metadata['content'])
                    results_with_scores.append(item)
            
            conn.close()
            if results_with_scores and len(results_with_scores) < MIN_RETRIEVE_KEEP:
                fallback_retrieval_used = True
                yield f"阈值后命中不足，已按保底策略返回{len(results_with_scores)}个结果。\n"
            results_with_scores.sort(key=lambda x: x[0], reverse=True)
            if not results_with_scores:
                yield "未找到相关的文档片段。\n"
                return
            relevant_results = results_with_scores

            yield f"初始检索命中(Stage 1)\n"
            for i, (similarity, path, chunk_index, content) in enumerate(relevant_results[:3]):
                yield f"[Top{i+1}] 相似度：{similarity:.4f}\n来源：{path}（段落 #{chunk_index}）\n{content[:100]}...\n----------------------------------------\n"

            formatted_retrieved_results = [
                {"score": score, "chunk_id": i, "path": path, "chunk_index": chunk_idx, "content": content}
                for i, (score, path, chunk_idx, content) in enumerate(relevant_results)
            ]
            
            yield f"\n步骤3:使用交叉编码器重排Top-{top_k_compressed}个结果\n"
            reranked_results = self._rerank_results(rewritten_query, formatted_retrieved_results, top_k=top_k_compressed)
            reranker_status = self._querier.get_reranker_status()
            yield f"Reranker状态：{reranker_status}\n"
            
            if reranked_results and 'rerank_score' in reranked_results[0]:
                yield f"重排后Top-{top_k_compressed}结果\n"
                for i, item in enumerate(reranked_results):
                    yield f"[Rank{i+1}] 重排分数：{item['rerank_score']:.4f}\n来源：{item['path']}（段落 #{item['chunk_index']}）\n{item['content'][:100]}...\n----------------------------------------\n"
            else:
                logger.warning("重排失败，使用初始检索顺序。")
                yield f"重排失败，使用初始检索Top-{top_k_compressed}结果\n"
                for i, item in enumerate(reranked_results):
                    yield f"[Initial Rank{i+1}]相似度：{item['score']:.4f}\n来源：{item['path']}（段落 #{item['chunk_index']}）\n{item['content'][:100]}...\n----------------------------------------\n"
            
            yield "\n开始压缩上下文\n"
            yield f"输入片段数：{len(reranked_results)}\n\n"
            
            compressed_content = self.compress_contexts(reranked_results)
            if not self._validate_citations(compressed_content, reranked_results):
                yield "压缩上下文引用校验未通过，回退为原始上下文。\n"
                compressed_content = self._build_raw_context(reranked_results)
            yield f"调用压缩模型：{self._compressor_model}\n"
            yield f"压缩后的内容:\n{compressed_content}\n压缩完成\n\n"

            final_messages = self.prepare_final_prompt(question, reranked_results, compressed_content)
            yield f"调用回答生成模型\n"
            yield f"调用模型：{self._chat_model}\n"
            message_str = "\n".join([f"{msg['role']}: {msg['content'][:100]}..." for msg in final_messages])
            yield f"Prompt:\n{message_str}\n"

            yield "\n最终回答\n"
            full_answer = ""
            for token in RAGHelpers._chat_completion_stream(self._ollama_host, final_messages, model_name=self._chat_model, temperature=OLLAMA_CHAT_TEMPERATURE_DEFAULT):
                full_answer += token
                yield token

            if not self._validate_citations(full_answer, reranked_results):
                citation_retry_used = True
                yield "\n\n[提示]检测到回答引用不完整，正在使用原始上下文重试...\n"
                # 通知上层UI用重试答案替换首轮答案，避免双答案拼接
                yield self.ANSWER_REPLACE_MARKER
                raw_context = self._build_raw_context(reranked_results)
                fallback_messages = self.prepare_final_prompt(question, reranked_results, raw_context)
                for token in RAGHelpers._chat_completion_stream(
                    self._ollama_host,
                    fallback_messages,
                    model_name=self._chat_model,
                    temperature=OLLAMA_CHAT_TEMPERATURE_DEFAULT
                ):
                    yield token

            yield (
                f"\n\n诊断指标: reranker={reranker_status}, "
                f"retrieval_fallback={fallback_retrieval_used}, citation_retry={citation_retry_used}\n"
            )

        except Exception as e:
            logger.error(f"流式问答过程中发生错误：{e}", exc_info=True)
            yield f"\n[错误]在处理您的请求时发生了内部错误：{e}\n"
            return