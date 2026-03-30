import copy
import hashlib
import json
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
    LOCAL_EMBEDDING_MODEL_PATH,
    METADATA_FILE,
    MIN_RETRIEVE_KEEP,
    OLLAMA_CHAT_TEMPERATURE_DEFAULT,
    OLLAMA_COMPRESSOR_TEMPERATURE_DEFAULT,
    OLLAMA_HOST,
    RERANK_MODEL,
    logger,
)
from simpleRAG_included.rag_build import RAGBuilder
from simpleRAG_included.rag_exceptions import SnapshotLoadError
from simpleRAG_included.rag_helpers import RAGHelpers
from simpleRAG_included.rag_query import RAGQuerier


class SimpleRAG:
    ANSWER_REPLACE_MARKER = "__RAG_REPLACE_ANSWER__"
    DEBUG_LOG_PREFIX = "__RAG_DEBUG__"
    RECENT_HISTORY_TURNS = 3
    SUMMARY_TRIGGER_TURNS = 5
    SUMMARY_CACHE_MAX_ITEMS = 128
    QUERY_CACHE_MAX_ITEMS = 64

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
            if role == "assistant" and (
                content.startswith("[System Error]:")
                or content.startswith("[Error]")
                or content == "Searching and reasoning..."
            ):
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
        recent_messages = normalized_history[-self.RECENT_HISTORY_TURNS * 2 :]
        older_messages = normalized_history[: -self.RECENT_HISTORY_TURNS * 2] if len(normalized_history) > self.RECENT_HISTORY_TURNS * 2 else []

        recent_history_text = self._history_messages_to_text(recent_messages)
        older_history_text = self._history_messages_to_text(older_messages)

        summary_text = ""
        if len(normalized_history) > self.SUMMARY_TRIGGER_TURNS * 2 and older_history_text:
            summary_text = self._summarize_older_history(older_history_text)

        retrieval_sections = []
        if summary_text:
            retrieval_sections.append(f"Earlier conversation summary:\n{summary_text}")
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
            return f"Knowledge base failed to load: {exc}"
        if not retrieved_results:
            return "No relevant knowledge snippets were retrieved. Please check the knowledge base or lower the score threshold."

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
        if self._validate_citations(answer, reranked_results):
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
            yield "Question cannot be empty.\n"
            return

        yield "\nStart retrieval\n"
        try:
            fallback_retrieval_used = False
            citation_retry_used = False
            compression_fallback_used = False

            if not FAISS_INDEX_FILE.exists() or not METADATA_FILE.exists():
                yield "Knowledge base is empty or failed to load.\n"
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
                        "recent_turns": self.RECENT_HISTORY_TURNS,
                    },
                )
                if cached_state.get("summary_text"):
                    yield self._debug_event("history_summary", cached_state["summary_text"])
                if cached_state.get("recent_history_text"):
                    yield self._debug_event("retrieval_history", cached_state["recent_history_text"])
                yield "Step 0: interpret conversation and rewrite the query\n"
                yield self._debug_event(
                    "rewrite_mode",
                    "skipped" if cached_state.get("rewrite_skipped") else "used",
                )
                yield f"Original question: {question}\n"
                yield f"Rewritten query: {cached_state.get('rewritten_query', question)}\n\n"
                yield self._debug_event(
                    "rewritten_query",
                    cached_state.get("rewritten_query", question),
                )
                yield "Step 1: reuse cached response package\n"
                yield "Cache hit: reused retrieval, ranking, and answer synthesis outputs.\n"
                yield self._debug_event(
                    "retrieved_results",
                    cached_entry.get("retrieved_results", []),
                )
                yield self._debug_event(
                    "reranked_results",
                    cached_entry.get("reranked_results", []),
                )
                if cached_diagnostics.get("compression_fallback_used"):
                    yield "Compressed context failed citation validation; falling back to raw context.\n"
                yield "\nFinal answer\n"
                yield str(cached_entry.get("answer", ""))
                yield (
                    f"\n\nDiagnostics: reranker={cached_diagnostics.get('reranker_status', 'unknown')}, "
                    f"retrieval_fallback={bool(cached_diagnostics.get('retrieval_fallback_used'))}, "
                    f"citation_retry={bool(cached_diagnostics.get('citation_retry_used'))}\n"
                )
                return

            yield self._debug_event(
                "history_mode",
                {
                    "summary_used": conversation_state["summary_used"],
                    "history_messages": len(conversation_state["normalized_history"]),
                    "recent_turns": self.RECENT_HISTORY_TURNS,
                },
            )
            if conversation_state["summary_text"]:
                yield self._debug_event("history_summary", conversation_state["summary_text"])
            if conversation_state["recent_history_text"]:
                yield self._debug_event("retrieval_history", conversation_state["recent_history_text"])

            yield "Step 0: interpret conversation and rewrite the query\n"
            rewrite_mode = "skipped" if conversation_state["rewrite_skipped"] else "used"
            yield self._debug_event("rewrite_mode", rewrite_mode)
            yield f"Original question: {question}\n"
            yield f"Rewritten query: {conversation_state['rewritten_query']}\n\n"
            yield self._debug_event("rewritten_query", conversation_state["rewritten_query"])

            yield "Step 1: vectorize rewritten query\n"
            with torch.no_grad():
                question_vector = self.embedding_model.encode(
                    conversation_state["rewritten_query"],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
            question_vector = np.array([question_vector], dtype="float32")
            yield f"Vectorized query dimension: {question_vector.shape[1]}\n"

            yield f"\nStep 2: retrieve top-{top_k_retrieve} knowledge snippets\n"
            retrieved_results = self.search_similar_with_faiss(
                question_vector[0].tolist(),
                top_k=top_k_retrieve,
                score_threshold=score_threshold,
            )
            if retrieved_results and len(retrieved_results) < MIN_RETRIEVE_KEEP:
                fallback_retrieval_used = True
                yield (
                    f"Threshold fallback kept only {len(retrieved_results)} retrieval results.\n"
                )
            if not retrieved_results:
                yield "No relevant knowledge snippets were found.\n"
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
                    f"[Top{index_id}] similarity={item['score']:.4f}\n"
                    f"source={item['path']}#chunk{item['chunk_index']}\n"
                    f"{item['content'][:100]}...\n"
                    "----------------------------------------\n"
                )

            yield f"\nStep 3: rerank top-{top_k_compressed} results\n"
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
            yield f"Reranker status: {reranker_status}\n"

            if reranked_results and "rerank_score" in reranked_results[0]:
                for index_id, item in enumerate(reranked_results, start=1):
                    yield (
                        f"[Rank{index_id}] rerank_score={item['rerank_score']:.4f}\n"
                        f"source={item['path']}#chunk{item['chunk_index']}\n"
                        f"{item['content'][:100]}...\n"
                        "----------------------------------------\n"
                    )
            else:
                for index_id, item in enumerate(reranked_results, start=1):
                    yield (
                        f"[Initial Rank{index_id}] similarity={item['score']:.4f}\n"
                        f"source={item['path']}#chunk{item['chunk_index']}\n"
                        f"{item['content'][:100]}...\n"
                        "----------------------------------------\n"
                    )

            yield "\nCompressing retrieved context\n"
            yield f"Input snippet count: {len(reranked_results)}\n\n"

            compressed_content = self.compress_contexts(reranked_results)
            if not self._validate_citations(compressed_content, reranked_results):
                yield "Compressed context failed citation validation; falling back to raw context.\n"
                compressed_content = self._build_raw_context(reranked_results)
                compression_fallback_used = True
            yield f"Compression model: {self._compressor_model}\n"
            yield f"Compressed context:\n{compressed_content}\nCompression complete\n\n"

            final_messages = self.prepare_final_prompt(
                question,
                reranked_results,
                compressed_content,
                history_text=conversation_state["retrieval_history_text"],
            )
            yield f"Answer generation model: {self._chat_model}\n"
            message_str = "\n".join(
                f"{message['role']}: {message['content'][:100]}..."
                for message in final_messages
            )
            yield f"Prompt:\n{message_str}\n"

            yield "\nFinal answer\n"
            full_answer = ""
            for token in RAGHelpers._chat_completion_stream(
                self._ollama_host,
                final_messages,
                model_name=self._chat_model,
                temperature=OLLAMA_CHAT_TEMPERATURE_DEFAULT,
            ):
                full_answer += token
                yield token

            if not self._validate_citations(full_answer, reranked_results):
                citation_retry_used = True
                yield "\n\n[Notice] Answer citations were incomplete. Retrying with raw context.\n"
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
                f"\n\nDiagnostics: reranker={self._querier.get_reranker_status()}, "
                f"retrieval_fallback={fallback_retrieval_used}, citation_retry={citation_retry_used}\n"
            )
        except SnapshotLoadError as exc:
            logger.error(f"Streaming QA snapshot load failed: {exc}")
            yield f"\nKnowledge base failed to load: {exc}\n"
        except Exception as exc:
            logger.error(f"Streaming QA failed: {exc}", exc_info=True)
            yield f"\n[Error] Internal error while handling the request: {exc}\n"
