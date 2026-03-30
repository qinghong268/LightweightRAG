import copy
import hashlib
import json
import math
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import requests
import torch
from sentence_transformers import SentenceTransformer

import prompts
from simpleRAG_included.config_imports import (
    CACHE_FILE,
    CHAT_MODEL,
    COMPRESSOR_MODEL,
    DEFAULT_THRESHOLD,
    DEFAULT_TOP_K,
    DEFAULT_TOP_K_COMPRESSED,
    EMBEDDING_MODEL,
    FAISS_INDEX_FILE,
    HISTORY_SUMMARY_TRIGGER_TOKENS,
    LOCAL_EMBEDDING_MODEL_PATH,
    METADATA_FILE,
    MIN_RETRIEVE_KEEP,
    OLLAMA_CHAT_TEMPERATURE_DEFAULT,
    OLLAMA_COMPRESSOR_TEMPERATURE_DEFAULT,
    OLLAMA_HOST,
    RECENT_HISTORY_TOKEN_BUDGET,
    RERANK_MODEL,
    logger,
)
from simpleRAG_included.rag_build import RAGBuilder
from simpleRAG_included.rag_exceptions import SnapshotLoadError
from simpleRAG_included.rag_helpers import RAGHelpers
from simpleRAG_included.rag_query import RAGQuerier


NON_PERSISTENT_ASSISTANT_PREFIXES = (
    "[System Error]:",
    "[系统错误]",
    "[Error]",
    "[错误]",
    "Cache hit:",
    "缓存命中：",
    "Searching and reasoning...",
    "正在检索并推理...",
    "Compressing retrieved context",
    "开始压缩召回上下文",
    "Start retrieval",
    "开始检索流程",
    "Step 0:",
    "Step 1:",
    "Step 2:",
    "Step 3:",
    "步骤 0：",
    "步骤 1：",
    "步骤 2：",
    "步骤 3：",
    "Original question:",
    "原始问题：",
    "Rewritten query:",
    "改写后的查询：",
    "Vectorized query dimension:",
    "查询向量维度：",
    "Reranker status:",
    "重排器状态：",
    "Compression model:",
    "压缩模型：",
    "Compressed context:",
    "压缩后的上下文：",
    "Compression complete",
    "上下文压缩完成",
    "Answer generation model:",
    "回答生成模型：",
    "Prompt:",
    "提示词：",
    "Final answer",
    "最终回答",
    "Diagnostics:",
    "诊断信息：",
    "[Notice]",
    "[提示]",
    "Threshold fallback kept only",
    "阈值回退后仅保留",
    "No relevant knowledge snippets were found.",
    "No relevant knowledge snippets were retrieved.",
    "未检索到相关知识片段。",
    "未检索到相关知识片段，请检查知识库或适当降低分数阈值。",
    "Knowledge base is empty or failed to load.",
    "Knowledge base failed to load:",
    "知识库为空或加载失败。",
    "知识库加载失败：",
    "I could not produce a grounded answer with valid citations for this question.",
    "无法基于当前索引文档生成带有效引用的可靠回答。",
)


def is_non_persistent_assistant_message(content: str) -> bool:
    content = str(content or "").strip()
    if not content:
        return True
    return any(content.startswith(prefix) for prefix in NON_PERSISTENT_ASSISTANT_PREFIXES)


def localize_runtime_status(status: str) -> str:
    status_map = {
        "pending": "待处理",
        "ready": "就绪",
        "not_loaded": "未加载",
        "unavailable": "不可用",
        "skipped_conversation_only": "仅对话历史",
        "unknown": "未知",
    }
    return status_map.get(str(status or "").strip(), str(status or "未知"))


class SimpleRAG:
    ANSWER_REPLACE_MARKER = "__RAG_REPLACE_ANSWER__"
    DEBUG_LOG_PREFIX = "__RAG_DEBUG__"
    SUMMARY_CACHE_MAX_ITEMS = 128
    QUERY_CACHE_MAX_ITEMS = 64
    CONVERSATION_META_MAX_MESSAGES = 24

    def __init__(self):
        logger.info("Initializing SimpleRAG instance")

        self.cache = {}
        self._summary_cache: "OrderedDict[str, str]" = OrderedDict()
        self._query_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._last_build_report: Dict[str, Any] = {}
        if CACHE_FILE.exists():
            try:
                raw_cache = RAGHelpers.load_embedding_cache(CACHE_FILE)
                if raw_cache.get("__model_name__") not in {None, EMBEDDING_MODEL}:
                    logger.warning(
                        f"Embedding model changed ({raw_cache.get('__model_name__')} -> {EMBEDDING_MODEL}); clearing old cache"
                    )
                    self.cache = {}
                else:
                    self.cache = {
                        key: value for key, value in raw_cache.items() if key != "__model_name__"
                    }
            except Exception as exc:
                logger.error(f"Failed to load embedding cache: {exc}")
                self.cache = {}

        try:
            self.embedding_model = SentenceTransformer(
                LOCAL_EMBEDDING_MODEL_PATH,
                trust_remote_code=True,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        except Exception as exc:
            logger.error(f"Failed to load embedding model: {exc}")
            raise

        self._ollama_host = OLLAMA_HOST
        self._chat_model = CHAT_MODEL
        self._compressor_model = COMPRESSOR_MODEL

        self._builder = RAGBuilder(self.cache, embedding_model_instance=self.embedding_model)
        self._querier = RAGQuerier(
            self._ollama_host,
            self._chat_model,
            self._compressor_model,
            RERANK_MODEL,
        )
        self._querier.set_embedding_model(self.embedding_model)

    async def build_knowledge_base_async(
        self,
        source_dir: Path,
        chunk_size: int = None,
        overlap: int = None,
    ) -> Dict[str, Any]:
        from simpleRAG_included.config_imports import CHUNK_OVERLAP_DEFAULT, CHUNK_SIZE_DEFAULT

        if chunk_size is None:
            chunk_size = CHUNK_SIZE_DEFAULT
        if overlap is None:
            overlap = CHUNK_OVERLAP_DEFAULT

        self.cache["__model_name__"] = EMBEDDING_MODEL
        build_report = await self._builder.build_knowledge_base_async(source_dir, chunk_size, overlap)

        save_cache = {key: value for key, value in self.cache.items() if key != "__model_name__"}
        save_cache["__model_name__"] = EMBEDDING_MODEL
        RAGHelpers.save_embedding_cache(save_cache, CACHE_FILE)
        self._last_build_report = build_report
        self._clear_query_cache()
        logger.info("Knowledge base build finished")
        return build_report

    def _debug_event(self, event: str, content: Any) -> str:
        payload = {"event": event, "content": content}
        return f"{self.DEBUG_LOG_PREFIX}{json.dumps(payload, ensure_ascii=False)}"

    def _clear_query_cache(self) -> None:
        self._query_cache.clear()

    def get_last_build_report(self) -> Dict[str, Any]:
        return copy.deepcopy(self._last_build_report)

    def get_runtime_cache_metrics(self) -> Dict[str, int]:
        return {
            "query_cache_entries": len(self._query_cache),
            "summary_cache_entries": len(self._summary_cache),
        }

    def _normalize_history(
        self,
        history: List[Dict[str, Any]] = None,
        max_turns: int = None,
    ) -> List[Dict[str, str]]:
        if not history:
            return []

        normalized_history: List[Dict[str, str]] = []
        for item in history:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = str(item.get("content", "")).strip()
            if role not in {"user", "assistant"} or not content:
                continue
            if role == "assistant" and is_non_persistent_assistant_message(content):
                continue
            normalized_history.append({"role": role, "content": content})

        if max_turns is not None:
            max_messages = max_turns * 2
            if len(normalized_history) > max_messages:
                normalized_history = normalized_history[-max_messages:]
        return normalized_history

    def _sanitize_history_content(self, content: str) -> str:
        content = re.sub(r"\[source=[^\]]+\]", "", content, flags=re.IGNORECASE)
        content = re.sub(r"\s+", " ", content).strip()
        return content

    def _history_messages_to_text(self, history: List[Dict[str, str]]) -> str:
        if not history:
            return ""

        lines = []
        for item in history:
            role_label = "User" if item["role"] == "user" else "Assistant"
            content = self._sanitize_history_content(item["content"])
            if content:
                lines.append(f"{role_label}: {content}")
        return "\n".join(lines)

    def _estimate_text_tokens(self, text: str) -> int:
        text = str(text or "")
        if not text:
            return 0

        cjk_chars = len(re.findall(r"[\u3400-\u9fff]", text))
        non_cjk_text = re.sub(r"[\u3400-\u9fff]", "", text)
        latin_estimate = math.ceil(len(non_cjk_text) / 4) if non_cjk_text else 0
        return max(1, cjk_chars + latin_estimate)

    def _estimate_message_tokens(self, message: Dict[str, str]) -> int:
        return self._estimate_text_tokens(message.get("content", "")) + 4

    def _select_recent_history_by_budget(
        self,
        normalized_history: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        if not normalized_history:
            return []

        recent_messages: List[Dict[str, str]] = []
        used_tokens = 0
        for item in reversed(normalized_history):
            item_tokens = self._estimate_message_tokens(item)
            if recent_messages and used_tokens + item_tokens > RECENT_HISTORY_TOKEN_BUDGET:
                break
            recent_messages.append(item)
            used_tokens += item_tokens

        recent_messages.reverse()
        return recent_messages or normalized_history[-1:]

    def _get_snapshot_fingerprint(self) -> str:
        parts = []
        for path_obj in (FAISS_INDEX_FILE, METADATA_FILE):
            if path_obj.exists():
                stat = path_obj.stat()
                parts.append(f"{path_obj.name}:{stat.st_mtime_ns}:{stat.st_size}")
            else:
                parts.append(f"{path_obj.name}:missing")
        return "|".join(parts)

    def _build_query_cache_key(
        self,
        question: str,
        history: List[Dict[str, Any]],
        top_k_retrieve: int,
        top_k_compressed: int,
        score_threshold: float,
    ) -> str:
        normalized_history = self._normalize_history(history, max_turns=None)
        cache_payload = {
            "question": question.strip(),
            "history": self._history_messages_to_text(normalized_history),
            "top_k_retrieve": int(top_k_retrieve),
            "top_k_compressed": int(top_k_compressed),
            "score_threshold": round(float(score_threshold), 4),
            "snapshot": self._get_snapshot_fingerprint(),
        }
        serialized = json.dumps(cache_payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _get_cached_query_entry(self, cache_key: str) -> Dict[str, Any]:
        cached = self._query_cache.get(cache_key)
        if cached is None:
            return {}
        self._query_cache.move_to_end(cache_key)
        return copy.deepcopy(cached)

    def _store_query_entry(self, cache_key: str, payload: Dict[str, Any]) -> None:
        self._query_cache[cache_key] = copy.deepcopy(payload)
        self._query_cache.move_to_end(cache_key)
        while len(self._query_cache) > self.QUERY_CACHE_MAX_ITEMS:
            self._query_cache.popitem(last=False)

    def _should_skip_rewrite(
        self,
        question: str,
        normalized_history: List[Dict[str, str]],
    ) -> bool:
        if not normalized_history:
            return True

        question = question.strip()
        if not question:
            return True

        follow_up_patterns = [
            r"\b(it|they|them|he|she|this|that|these|those|former|latter|continue|expand|elaborate)\b",
            r"(它|他|她|这个|那个|这些|那些|上述|上面|前面|刚才|继续|展开|详细说说|进一步|第二点|第三点|什么意思|为什么)",
        ]
        if any(re.search(pattern, question, flags=re.IGNORECASE) for pattern in follow_up_patterns):
            return False

        if len(question) >= 18:
            return True
        if re.search(r"[?？]", question) and len(question) >= 12:
            return True
        return False

    def _summarize_older_history(self, history_text: str) -> str:
        history_text = history_text.strip()
        if not history_text:
            return ""

        cache_key = hashlib.sha256(history_text.encode("utf-8")).hexdigest()
        cached = self._summary_cache.get(cache_key)
        if cached is not None:
            self._summary_cache.move_to_end(cache_key)
            return cached

        messages = prompts.get_conversation_summary_prompt_template(history_text)
        try:
            summary = RAGHelpers._chat_completion(
                self._ollama_host,
                messages,
                model_name=self._compressor_model,
                temperature=0.1,
            ).strip()
        except Exception as exc:
            logger.warning(f"Conversation summary generation failed: {exc}")
            summary = ""

        summary = self._sanitize_history_content(summary)
        if summary:
            self._summary_cache[cache_key] = summary
            self._summary_cache.move_to_end(cache_key)
            while len(self._summary_cache) > self.SUMMARY_CACHE_MAX_ITEMS:
                self._summary_cache.popitem(last=False)
        return summary

    def _build_history_views(self, history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        normalized_history = self._normalize_history(history, max_turns=None)
        recent_messages = self._select_recent_history_by_budget(normalized_history)
        older_messages = (
            normalized_history[: len(normalized_history) - len(recent_messages)]
            if recent_messages
            else []
        )

        recent_history_text = self._history_messages_to_text(recent_messages)
        older_history_text = self._history_messages_to_text(older_messages)

        summary_text = ""
        older_context_text = ""
        if older_history_text:
            if self._estimate_text_tokens(older_history_text) >= HISTORY_SUMMARY_TRIGGER_TOKENS:
                summary_text = self._summarize_older_history(older_history_text)
                older_context_text = summary_text or older_history_text
            else:
                older_context_text = older_history_text

        retrieval_sections = []
        if older_context_text:
            section_title = "Earlier conversation summary" if summary_text else "Earlier conversation"
            retrieval_sections.append(f"{section_title}:\n{older_context_text}")
        if recent_history_text:
            retrieval_sections.append(f"Recent conversation:\n{recent_history_text}")

        retrieval_history_text = "\n\n".join(retrieval_sections).strip()
        return {
            "normalized_history": normalized_history,
            "recent_history_text": recent_history_text,
            "summary_text": summary_text,
            "retrieval_history_text": retrieval_history_text,
            "summary_used": bool(summary_text),
        }

    def _extract_rewritten_query(self, raw_content: str, original_query: str) -> str:
        if not raw_content:
            return original_query

        lines = [line.strip().strip("\"'") for line in raw_content.splitlines() if line.strip()]
        if not lines:
            return original_query

        for line in lines:
            lowered = line.lower()
            if lowered.startswith("rewritten question:"):
                candidate = line.split(":", 1)[1].strip()
                if candidate:
                    return candidate
            if lowered.startswith("standalone question:"):
                candidate = line.split(":", 1)[1].strip()
                if candidate:
                    return candidate
            if len(line) > 3:
                return line
        return original_query

    def _is_conversation_meta_question(self, question: str) -> bool:
        question = question.strip().lower()
        if not question:
            return False

        patterns = [
            r"\b(last|previous)\s+(question|answer|message)\b",
            r"\bwhat did (i|we|you) (ask|say|mention)\b",
            r"\brecap\b",
            r"\bsummarize (our|the) conversation\b",
            r"上一个问题",
            r"前一个问题",
            r"上一条(消息|回答)",
            r"刚才(我|你|我们).*(说|问)",
            r"前面(我|你|我们).*(说|问)",
            r"总结一下(我们|刚才|前面).*(对话|聊天|内容)",
            r"回顾一下(我们|刚才|前面).*(对话|聊天|内容)",
        ]
        return any(re.search(pattern, question, flags=re.IGNORECASE) for pattern in patterns)

    def _answer_meets_policy(
        self,
        question: str,
        answer: str,
        contexts: List[Dict[str, Any]],
    ) -> bool:
        if self._is_conversation_meta_question(question):
            return bool(answer.strip())
        return self._validate_citations(answer, contexts)

    def _find_last_history_message(
        self,
        normalized_history: List[Dict[str, str]],
        role: str,
    ) -> str:
        for item in reversed(normalized_history):
            if item.get("role") == role:
                return item.get("content", "").strip()
        return ""

    def _answer_conversation_meta_question(
        self,
        question: str,
        normalized_history: List[Dict[str, str]],
    ) -> str:
        if not normalized_history:
            return "当前对话里还没有可参考的历史内容。"

        lowered_question = question.strip().lower()
        last_user_message = self._find_last_history_message(normalized_history, "user")
        last_assistant_message = self._find_last_history_message(normalized_history, "assistant")

        if re.search(r"\b(last|previous)\s+question\b|上一个问题|前一个问题", lowered_question, flags=re.IGNORECASE):
            if last_user_message:
                return f"上一个问题是：{last_user_message}"
            return "当前对话里还没有上一条用户问题。"

        if re.search(r"\b(last|previous)\s+answer\b|上一条回答|上一个回答", lowered_question, flags=re.IGNORECASE):
            if last_assistant_message:
                return f"上一条回答是：{last_assistant_message}"
            return "当前对话里还没有上一条助手回答。"

        history_for_answer = normalized_history[-self.CONVERSATION_META_MAX_MESSAGES :]
        history_text = self._history_messages_to_text(history_for_answer)
        messages = prompts.get_conversation_meta_prompt_template(question, history_text)
        try:
            answer = RAGHelpers._chat_completion(
                self._ollama_host,
                messages,
                model_name=self._chat_model,
                temperature=0.1,
            ).strip()
            if answer:
                return answer
        except Exception as exc:
            logger.warning(f"Conversation-meta answer generation failed: {exc}")

        return "我只能根据当前对话历史来回答，但暂时无法从现有历史中整理出可靠结果。"

    def _rewrite_query(self, original_query: str, history_text: str = "") -> str:
        if not history_text.strip():
            return original_query

        messages = prompts.get_query_rewrite_prompt_template(original_query, history_text)
        url = f"{self._ollama_host}/api/chat"
        data = {
            "model": self._chat_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.1},
        }
        try:
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json=data,
                timeout=60,
            )
            response.raise_for_status()
            content = response.json()["message"]["content"].strip()
            return self._extract_rewritten_query(content, original_query)
        except Exception as exc:
            logger.warning(f"Query rewriting failed: {exc}")
            return original_query

    def _rerank_results(self, query: str, results: list, top_k: int = 5) -> list:
        return self._querier._rerank_results(query, results, top_k)

    def compress_contexts(
        self,
        retrieved_results: List[Dict[str, Any]],
        compressor_model: str = None,
        temperature: float = None,
    ) -> str:
        if temperature is None:
            temperature = OLLAMA_COMPRESSOR_TEMPERATURE_DEFAULT
        return self._querier.compress_contexts(retrieved_results, compressor_model, temperature)

    def prepare_final_prompt(
        self,
        question: str,
        contexts: List[Dict[str, Any]],
        compressed_context: str,
        history_text: str = "",
    ) -> List[dict]:
        return self._querier.prepare_final_prompt(
            question,
            contexts,
            compressed_context,
            history_text,
        )

    def search_similar_with_faiss(
        self,
        query_vec: List[float],
        top_k: int,
        score_threshold: float,
    ) -> List[Dict[str, Any]]:
        return self._querier.search_similar_with_faiss(
            query_vec,
            top_k,
            score_threshold,
            min_keep=MIN_RETRIEVE_KEEP,
        )

    def _validate_citations(self, text: str, contexts: List[Dict[str, Any]]) -> bool:
        if not text.strip():
            return False
        matches = re.findall(r"\[source=([^\]#]+)#chunk(\d+)\]", text, flags=re.IGNORECASE)
        if not matches:
            return False

        valid_sources = {
            (str(item.get("path", "")).strip(), str(item.get("chunk_index", "")).strip())
            for item in contexts
        }
        if not valid_sources:
            return False

        return all(
            (source_path.strip(), chunk_idx.strip()) in valid_sources
            for source_path, chunk_idx in matches
        )

    def _build_raw_context(self, contexts: List[Dict[str, Any]]) -> str:
        return "\n".join(
            f"[source={item['path']}#chunk{item['chunk_index']}] {item['content']}"
            for item in contexts
        )

    def retrieve_contexts(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        score_threshold: float = DEFAULT_THRESHOLD,
        retrieval_history_text: str = "",
        rewritten_query: str = None,
    ) -> List[Dict[str, Any]]:
        if not question.strip():
            return []

        final_query = rewritten_query or self._rewrite_query(question, retrieval_history_text)
        logger.debug(f"Rewritten retrieval query: {final_query}")

        with torch.no_grad():
            query_vec = self.embedding_model.encode(
                final_query,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).tolist()

        return self.search_similar_with_faiss(
            query_vec,
            top_k=top_k,
            score_threshold=score_threshold,
        )

    def _prepare_conversation_state(
        self,
        question: str,
        history: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        history_views = self._build_history_views(history)
        rewrite_skipped = self._should_skip_rewrite(question, history_views["normalized_history"])
        rewritten_query = (
            question
            if rewrite_skipped
            else self._rewrite_query(question, history_views["retrieval_history_text"])
        )
        return {
            **history_views,
            "rewrite_skipped": rewrite_skipped,
            "rewritten_query": rewritten_query,
        }

    def answer_question(
        self,
        question: str,
        top_k_retrieve: int = DEFAULT_TOP_K,
        top_k_compressed: int = DEFAULT_TOP_K_COMPRESSED,
        score_threshold: float = DEFAULT_THRESHOLD,
        history: List[Dict[str, Any]] = None,
    ) -> str:
        conversation_state = self._prepare_conversation_state(question, history)
        cache_key = self._build_query_cache_key(
            question,
            history or [],
            top_k_retrieve,
            top_k_compressed,
            score_threshold,
        )
        cached_entry = self._get_cached_query_entry(cache_key)
        if cached_entry:
            return str(cached_entry.get("answer", ""))

        if self._is_conversation_meta_question(question):
            answer = self._answer_conversation_meta_question(
                question,
                conversation_state["normalized_history"],
            )
            self._store_query_entry(
                cache_key,
                {
                    "answer": answer,
                    "conversation_state": {
                        "summary_used": conversation_state["summary_used"],
                        "summary_text": conversation_state["summary_text"],
                        "recent_history_text": conversation_state["recent_history_text"],
                        "retrieval_history_text": conversation_state["retrieval_history_text"],
                        "rewrite_skipped": True,
                        "rewritten_query": question,
                        "history_message_count": len(conversation_state["normalized_history"]),
                    },
                    "retrieved_results": [],
                    "reranked_results": [],
                    "diagnostics": {
                        "reranker_status": "skipped_conversation_only",
                        "retrieval_fallback_used": False,
                        "citation_retry_used": False,
                        "compression_fallback_used": False,
                    },
                },
            )
            return answer

        try:
            retrieved_results = self.retrieve_contexts(
                question,
                top_k=top_k_retrieve,
                score_threshold=score_threshold,
                retrieval_history_text=conversation_state["retrieval_history_text"],
                rewritten_query=conversation_state["rewritten_query"],
            )
        except SnapshotLoadError as exc:
            logger.error(f"Knowledge base snapshot load failed: {exc}")
            return f"知识库加载失败：{exc}"
        if not retrieved_results:
            return "未检索到相关知识片段，请检查知识库或适当降低分数阈值。"

        reranked_results = self._rerank_results(
            conversation_state["rewritten_query"],
            retrieved_results,
            top_k=top_k_compressed,
        )
        compressed_context = self.compress_contexts(reranked_results)
        compression_fallback_used = False
        if not self._validate_citations(compressed_context, reranked_results):
            logger.warning("Compressed context failed citation validation; falling back to raw context")
            compressed_context = self._build_raw_context(reranked_results)
            compression_fallback_used = True

        messages = self.prepare_final_prompt(
            question,
            reranked_results,
            compressed_context,
            history_text=conversation_state["retrieval_history_text"],
        )
        answer = RAGHelpers._chat_completion(
            self._ollama_host,
            messages,
            model_name=self._chat_model,
            temperature=OLLAMA_CHAT_TEMPERATURE_DEFAULT,
        )
        citation_retry_used = False
        if self._answer_meets_policy(question, answer, reranked_results):
            self._store_query_entry(
                cache_key,
                {
                    "answer": answer,
                    "conversation_state": {
                        "summary_used": conversation_state["summary_used"],
                        "summary_text": conversation_state["summary_text"],
                        "recent_history_text": conversation_state["recent_history_text"],
                        "retrieval_history_text": conversation_state["retrieval_history_text"],
                        "rewrite_skipped": conversation_state["rewrite_skipped"],
                        "rewritten_query": conversation_state["rewritten_query"],
                        "history_message_count": len(conversation_state["normalized_history"]),
                    },
                    "retrieved_results": retrieved_results,
                    "reranked_results": reranked_results,
                    "diagnostics": {
                        "reranker_status": self._querier.get_reranker_status(),
                        "retrieval_fallback_used": bool(retrieved_results and len(retrieved_results) < MIN_RETRIEVE_KEEP),
                        "citation_retry_used": citation_retry_used,
                        "compression_fallback_used": compression_fallback_used,
                    },
                },
            )
            return answer

        logger.warning("Answer citation validation failed; retrying with raw context")
        citation_retry_used = True
        raw_context = self._build_raw_context(reranked_results)
        fallback_messages = self.prepare_final_prompt(
            question,
            reranked_results,
            raw_context,
            history_text=conversation_state["retrieval_history_text"],
        )
        fallback_answer = RAGHelpers._chat_completion(
            self._ollama_host,
            fallback_messages,
            model_name=self._chat_model,
            temperature=OLLAMA_CHAT_TEMPERATURE_DEFAULT,
        )
        if not self._answer_meets_policy(question, fallback_answer, reranked_results):
            logger.warning("Fallback answer still failed policy validation")
            return (
                "无法基于当前索引文档生成带有效引用的可靠回答。"
                "请尝试换一种问法，或确认答案能够从已索引文档中直接得到。"
            )
        self._store_query_entry(
            cache_key,
            {
                "answer": fallback_answer,
                "conversation_state": {
                    "summary_used": conversation_state["summary_used"],
                    "summary_text": conversation_state["summary_text"],
                    "recent_history_text": conversation_state["recent_history_text"],
                    "retrieval_history_text": conversation_state["retrieval_history_text"],
                    "rewrite_skipped": conversation_state["rewrite_skipped"],
                    "rewritten_query": conversation_state["rewritten_query"],
                    "history_message_count": len(conversation_state["normalized_history"]),
                },
                "retrieved_results": retrieved_results,
                "reranked_results": reranked_results,
                "diagnostics": {
                    "reranker_status": self._querier.get_reranker_status(),
                    "retrieval_fallback_used": bool(retrieved_results and len(retrieved_results) < MIN_RETRIEVE_KEEP),
                    "citation_retry_used": citation_retry_used,
                    "compression_fallback_used": compression_fallback_used,
                },
            },
        )
        return fallback_answer

    def answer_question_stream(
        self,
        question: str,
        top_k_retrieve: int = DEFAULT_TOP_K,
        top_k_compressed: int = DEFAULT_TOP_K_COMPRESSED,
        score_threshold: float = DEFAULT_THRESHOLD,
        history: List[Dict[str, Any]] = None,
    ) -> Iterable[str]:
        if not question.strip():
            yield "问题不能为空。\n"
            return

        yield "\n开始检索流程\n"
        try:
            fallback_retrieval_used = False
            citation_retry_used = False
            compression_fallback_used = False

            if not FAISS_INDEX_FILE.exists() or not METADATA_FILE.exists():
                yield "知识库为空或加载失败。\n"
                return

            conversation_state = self._prepare_conversation_state(question, history)
            cache_key = self._build_query_cache_key(
                question,
                history or [],
                top_k_retrieve,
                top_k_compressed,
                score_threshold,
            )
            cached_entry = self._get_cached_query_entry(cache_key)
            yield self._debug_event(
                "cache_status",
                {
                    "hit": bool(cached_entry),
                    "entries": len(self._query_cache),
                },
            )
            if cached_entry:
                cached_state = cached_entry.get("conversation_state", {})
                cached_diagnostics = cached_entry.get("diagnostics", {})
                yield self._debug_event(
                    "history_mode",
                    {
                        "summary_used": bool(cached_state.get("summary_used")),
                        "history_messages": int(cached_state.get("history_message_count", 0)),
                        "recent_token_budget": RECENT_HISTORY_TOKEN_BUDGET,
                    },
                )
                if cached_state.get("summary_text"):
                    yield self._debug_event("history_summary", cached_state["summary_text"])
                if cached_state.get("recent_history_text"):
                    yield self._debug_event("retrieval_history", cached_state["recent_history_text"])
                yield "步骤 0：结合对话理解并改写查询\n"
                yield self._debug_event(
                    "rewrite_mode",
                    "skipped" if cached_state.get("rewrite_skipped") else "used",
                )
                yield f"原始问题：{question}\n"
                yield f"改写后的查询：{cached_state.get('rewritten_query', question)}\n\n"
                yield self._debug_event(
                    "rewritten_query",
                    cached_state.get("rewritten_query", question),
                )
                yield "步骤 1：复用缓存的响应包\n"
                yield "缓存命中：已复用检索、重排和回答结果。\n"
                yield self._debug_event(
                    "retrieved_results",
                    cached_entry.get("retrieved_results", []),
                )
                yield self._debug_event(
                    "reranked_results",
                    cached_entry.get("reranked_results", []),
                )
                if cached_diagnostics.get("compression_fallback_used"):
                    yield "压缩结果未通过引用校验，已回退到原始上下文。\n"
                yield "\n最终回答\n"
                yield str(cached_entry.get("answer", ""))
                yield (
                    f"\n\n诊断信息：重排器={localize_runtime_status(cached_diagnostics.get('reranker_status', 'unknown'))}，"
                    f"召回回退={'是' if bool(cached_diagnostics.get('retrieval_fallback_used')) else '否'}，"
                    f"引用重试={'是' if bool(cached_diagnostics.get('citation_retry_used')) else '否'}\n"
                )
                return

            yield self._debug_event(
                "history_mode",
                {
                    "summary_used": conversation_state["summary_used"],
                    "history_messages": len(conversation_state["normalized_history"]),
                    "recent_token_budget": RECENT_HISTORY_TOKEN_BUDGET,
                },
            )
            if conversation_state["summary_text"]:
                yield self._debug_event("history_summary", conversation_state["summary_text"])
            if conversation_state["recent_history_text"]:
                yield self._debug_event("retrieval_history", conversation_state["recent_history_text"])

            if self._is_conversation_meta_question(question):
                answer = self._answer_conversation_meta_question(
                    question,
                    conversation_state["normalized_history"],
                )
                yield "步骤 0：直接根据对话历史回答\n"
                yield self._debug_event("rewrite_mode", "conversation_only")
                yield self._debug_event("retrieved_results", [])
                yield self._debug_event("reranked_results", [])
                yield "\n最终回答\n"
                yield answer
                self._store_query_entry(
                    cache_key,
                    {
                        "answer": answer,
                        "conversation_state": {
                            "summary_used": conversation_state["summary_used"],
                            "summary_text": conversation_state["summary_text"],
                            "recent_history_text": conversation_state["recent_history_text"],
                            "retrieval_history_text": conversation_state["retrieval_history_text"],
                            "rewrite_skipped": True,
                            "rewritten_query": question,
                            "history_message_count": len(conversation_state["normalized_history"]),
                        },
                        "retrieved_results": [],
                        "reranked_results": [],
                        "diagnostics": {
                            "reranker_status": "skipped_conversation_only",
                            "retrieval_fallback_used": False,
                            "citation_retry_used": False,
                            "compression_fallback_used": False,
                        },
                    },
                )
                yield "\n\n诊断信息：重排器=skipped_conversation_only，召回回退=否，引用重试=否\n"
                return

            yield "步骤 0：结合对话理解并改写查询\n"
            rewrite_mode = "skipped" if conversation_state["rewrite_skipped"] else "used"
            yield self._debug_event("rewrite_mode", rewrite_mode)
            yield f"原始问题：{question}\n"
            yield f"改写后的查询：{conversation_state['rewritten_query']}\n\n"
            yield self._debug_event("rewritten_query", conversation_state["rewritten_query"])

            yield "步骤 1：向量化改写后的查询\n"
            with torch.no_grad():
                question_vector = self.embedding_model.encode(
                    conversation_state["rewritten_query"],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
            question_vector = np.array([question_vector], dtype="float32")
            yield f"查询向量维度：{question_vector.shape[1]}\n"

            yield f"\n步骤 2：检索前 {top_k_retrieve} 条知识片段\n"
            retrieved_results = self.search_similar_with_faiss(
                question_vector[0].tolist(),
                top_k=top_k_retrieve,
                score_threshold=score_threshold,
            )
            if retrieved_results and len(retrieved_results) < MIN_RETRIEVE_KEEP:
                fallback_retrieval_used = True
                yield (
                    f"阈值回退后仅保留 {len(retrieved_results)} 条检索结果。\n"
                )
            if not retrieved_results:
                yield "未检索到相关知识片段。\n"
                return

            yield self._debug_event(
                "retrieved_results",
                [
                    {
                        "score": round(item["score"], 4),
                        "path": item["path"],
                        "chunk_index": item["chunk_index"],
                        "content": item["content"][:500],
                    }
                    for item in retrieved_results
                ],
            )

            for index_id, item in enumerate(retrieved_results[:3], start=1):
                yield (
                    f"[候选{index_id}] 相似度={item['score']:.4f}\n"
                    f"来源={item['path']}#chunk{item['chunk_index']}\n"
                    f"{item['content'][:100]}...\n"
                    "----------------------------------------\n"
                )

            yield f"\n步骤 3：重排前 {top_k_compressed} 条结果\n"
            reranked_results = self._rerank_results(
                conversation_state["rewritten_query"],
                retrieved_results,
                top_k=top_k_compressed,
            )
            yield self._debug_event(
                "reranked_results",
                [
                    {
                        "score": round(item.get("score", 0.0), 4),
                        "rerank_score": round(item.get("rerank_score", 0.0), 4),
                        "path": item["path"],
                        "chunk_index": item["chunk_index"],
                        "content": item["content"][:500],
                    }
                    for item in reranked_results
                ],
            )
            reranker_status = self._querier.get_reranker_status()
            yield f"重排器状态：{localize_runtime_status(reranker_status)}\n"

            if reranked_results and "rerank_score" in reranked_results[0]:
                for index_id, item in enumerate(reranked_results, start=1):
                    yield (
                        f"[重排{index_id}] 重排分数={item['rerank_score']:.4f}\n"
                        f"来源={item['path']}#chunk{item['chunk_index']}\n"
                        f"{item['content'][:100]}...\n"
                        "----------------------------------------\n"
                    )
            else:
                for index_id, item in enumerate(reranked_results, start=1):
                    yield (
                        f"[初排{index_id}] 相似度={item['score']:.4f}\n"
                        f"来源={item['path']}#chunk{item['chunk_index']}\n"
                        f"{item['content'][:100]}...\n"
                        "----------------------------------------\n"
                    )

            yield "\n开始压缩召回上下文\n"
            yield f"输入片段数量：{len(reranked_results)}\n\n"

            compressed_content = self.compress_contexts(reranked_results)
            if not self._validate_citations(compressed_content, reranked_results):
                yield "压缩结果未通过引用校验，已回退到原始上下文。\n"
                compressed_content = self._build_raw_context(reranked_results)
                compression_fallback_used = True
            yield f"压缩模型：{self._compressor_model}\n"
            yield f"压缩后的上下文：\n{compressed_content}\n上下文压缩完成\n\n"

            final_messages = self.prepare_final_prompt(
                question,
                reranked_results,
                compressed_content,
                history_text=conversation_state["retrieval_history_text"],
            )
            yield f"回答生成模型：{self._chat_model}\n"
            role_map = {
                "system": "系统",
                "user": "用户",
                "assistant": "助手",
            }
            message_str = "\n".join(
                f"{role_map.get(message['role'], message['role'])}: {message['content'][:100]}..."
                for message in final_messages
            )
            yield f"提示词：\n{message_str}\n"

            yield "\n最终回答\n"
            full_answer = ""
            for token in RAGHelpers._chat_completion_stream(
                self._ollama_host,
                final_messages,
                model_name=self._chat_model,
                temperature=OLLAMA_CHAT_TEMPERATURE_DEFAULT,
            ):
                full_answer += token
                yield token

            if not self._answer_meets_policy(question, full_answer, reranked_results):
                citation_retry_used = True
                yield "\n\n[提示] 回答中的引用不完整，正在基于原始上下文重试。\n"
                yield self.ANSWER_REPLACE_MARKER
                raw_context = self._build_raw_context(reranked_results)
                fallback_messages = self.prepare_final_prompt(
                    question,
                    reranked_results,
                    raw_context,
                    history_text=conversation_state["retrieval_history_text"],
                )
                fallback_answer = ""
                for token in RAGHelpers._chat_completion_stream(
                    self._ollama_host,
                    fallback_messages,
                    model_name=self._chat_model,
                    temperature=OLLAMA_CHAT_TEMPERATURE_DEFAULT,
                ):
                    fallback_answer += token
                    yield token
                full_answer = fallback_answer

                if not self._answer_meets_policy(question, full_answer, reranked_results):
                    logger.warning("Streaming fallback answer still failed policy validation")
                    yield self.ANSWER_REPLACE_MARKER
                    full_answer = (
                        "无法基于当前索引文档生成带有效引用的可靠回答。"
                        "请尝试换一种问法，或确认答案能够从已索引文档中直接得到。"
                    )
                    yield full_answer

            self._store_query_entry(
                cache_key,
                {
                    "answer": full_answer,
                    "conversation_state": {
                        "summary_used": conversation_state["summary_used"],
                        "summary_text": conversation_state["summary_text"],
                        "recent_history_text": conversation_state["recent_history_text"],
                        "retrieval_history_text": conversation_state["retrieval_history_text"],
                        "rewrite_skipped": conversation_state["rewrite_skipped"],
                        "rewritten_query": conversation_state["rewritten_query"],
                        "history_message_count": len(conversation_state["normalized_history"]),
                    },
                    "retrieved_results": [
                        {
                            "score": round(item["score"], 4),
                            "path": item["path"],
                            "chunk_index": item["chunk_index"],
                            "content": item["content"][:500],
                        }
                        for item in retrieved_results
                    ],
                    "reranked_results": [
                        {
                            "score": round(item.get("score", 0.0), 4),
                            "rerank_score": round(item.get("rerank_score", 0.0), 4),
                            "path": item["path"],
                            "chunk_index": item["chunk_index"],
                            "content": item["content"][:500],
                        }
                        for item in reranked_results
                    ],
                    "diagnostics": {
                        "reranker_status": self._querier.get_reranker_status(),
                        "retrieval_fallback_used": fallback_retrieval_used,
                        "citation_retry_used": citation_retry_used,
                        "compression_fallback_used": compression_fallback_used,
                    },
                },
            )

            yield (
                f"\n\n诊断信息：重排器={localize_runtime_status(self._querier.get_reranker_status())}，"
                f"召回回退={'是' if fallback_retrieval_used else '否'}，"
                f"引用重试={'是' if citation_retry_used else '否'}\n"
            )
        except SnapshotLoadError as exc:
            logger.error(f"Streaming QA snapshot load failed: {exc}")
            yield f"\n知识库加载失败：{exc}\n"
        except Exception as exc:
            logger.error(f"Streaming QA failed: {exc}", exc_info=True)
            yield f"\n[错误] 处理请求时发生内部错误：{exc}\n"
