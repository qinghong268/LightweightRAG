import sqlite3
from typing import Any, Dict, List

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import prompts
from .config_imports import (
    DB_PATH,
    FAISS_INDEX_FILE,
    METADATA_FILE,
    MIN_RETRIEVE_KEEP,
    OLLAMA_COMPRESSOR_TEMPERATURE_DEFAULT,
    logger,
)
from .rag_exceptions import SnapshotLoadError
from .rag_helpers import RAGHelpers

try:
    from FlagEmbedding import FlagReranker

    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    logger.warning("FlagEmbedding is not installed; reranking will be unavailable.")


class RAGQuerier:
    def __init__(self, ollama_host: str, chat_model: str, compressor_model: str, reranker_model_name: str):
        self._ollama_host = ollama_host
        self._chat_model = chat_model
        self._compressor_model = compressor_model
        self._reranker_model_name = reranker_model_name
        self._reranker_model_path = f"./models/{reranker_model_name}"
        self._reranker = None
        self.embedding_model = None

    def set_embedding_model(self, model_instance: SentenceTransformer):
        self.embedding_model = model_instance
        logger.debug("RAGQuerier received the shared embedding model instance.")

    def _load_reranker(self):
        if self._reranker is None and RERANKER_AVAILABLE:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._reranker = FlagReranker(
                    self._reranker_model_path,
                    use_fp16=True,
                    device=device,
                )
                logger.info(f"Reranker model loaded from {self._reranker_model_path}")
            except Exception as exc:
                logger.error(f"Failed to load reranker model: {exc}")

    def get_reranker_status(self) -> str:
        if not RERANKER_AVAILABLE:
            return "unavailable"
        if self._reranker is None:
            return "not_loaded"
        return "ready"

    def _rerank_results(self, query: str, results: list, top_k: int = 5) -> list:
        if not results or not query:
            return results
        if not RERANKER_AVAILABLE or self._reranker is None:
            self._load_reranker()
            if not self._reranker:
                return results[:top_k]

        pairs = [[query, item["content"]] for item in results]
        try:
            scores = self._reranker.compute_score(pairs, normalize=True)
            if not isinstance(scores, list):
                scores = [scores] * len(results)

            for i, result in enumerate(results):
                result["rerank_score"] = float(scores[i])

            return sorted(results, key=lambda item: item["rerank_score"], reverse=True)[:top_k]
        except Exception as exc:
            logger.error(f"Reranking failed: {exc}")
            return results[:top_k]

    def search_similar_with_faiss(
        self,
        query_vec: List[float],
        top_k: int,
        score_threshold: float,
        min_keep: int = MIN_RETRIEVE_KEEP,
    ) -> List[Dict[str, Any]]:
        with RAGHelpers.SNAPSHOT_FILE_LOCK:
            try:
                index, metadata_map = RAGHelpers.load_faiss_index_and_metadata(
                    FAISS_INDEX_FILE,
                    METADATA_FILE,
                )
            except Exception as exc:
                logger.error(f"Failed to load FAISS snapshot: {exc}")
                raise SnapshotLoadError(f"Failed to load knowledge-base snapshot: {exc}") from exc

            if index is None:
                return []

            snapshot_chunk_ids = metadata_map.get("chunk_ids", []) if isinstance(metadata_map, dict) else []
            uses_vector_ids = bool(metadata_map.get("uses_vector_ids")) if isinstance(metadata_map, dict) else False
            if not uses_vector_ids and not snapshot_chunk_ids:
                return []

            query_array = np.array([query_vec], dtype=np.float32)
            faiss.normalize_L2(query_array)
            available_total = int(index.ntotal)
            candidate_k = min(available_total, max(top_k, top_k * 3, min_keep))
            if candidate_k <= 0:
                return []
            scores, indices = index.search(query_array, candidate_k)

            ranked_hits: List[Dict[str, Any]] = []
            selected_chunk_ids: List[int] = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                if idx < 0:
                    continue
                if uses_vector_ids:
                    chunk_id = int(idx)
                else:
                    if idx >= len(snapshot_chunk_ids):
                        continue
                    chunk_id = int(snapshot_chunk_ids[idx])
                if chunk_id < 0:
                    continue
                ranked_hits.append({"score": float(scores[0][i]), "chunk_id": chunk_id})
                selected_chunk_ids.append(chunk_id)

            if not ranked_hits:
                return []

            conn = sqlite3.connect(DB_PATH)
            try:
                metadata_lookup = RAGHelpers.get_metadata_by_chunk_ids(conn, selected_chunk_ids)
            finally:
                conn.close()

        complete_results = []
        for hit in ranked_hits:
            metadata = metadata_lookup.get(hit["chunk_id"])
            if not metadata:
                continue
            complete_results.append(
                {
                    **hit,
                    "path": metadata["path"],
                    "chunk_index": metadata["chunk_index"],
                    "content": metadata["content"],
                }
            )

        threshold_hits = [item for item in complete_results if item["score"] >= score_threshold]
        if len(threshold_hits) < int(min_keep) and complete_results:
            keep_n = max(1, min(int(min_keep), len(complete_results)))
            logger.info(f"Threshold hits were too few; falling back to Top-{keep_n} candidates.")
            return complete_results[:keep_n]
        return threshold_hits[:top_k]

    def compress_contexts(
        self,
        retrieved_results: List[Dict[str, Any]],
        compressor_model: str = None,
        temperature: float = OLLAMA_COMPRESSOR_TEMPERATURE_DEFAULT,
    ) -> str:
        if not retrieved_results:
            return ""
        if compressor_model is None:
            compressor_model = self._compressor_model

        messages = prompts.get_compress_prompt_template(retrieved_results)
        return RAGHelpers._chat_completion(self._ollama_host, messages, compressor_model, temperature)

    def prepare_final_prompt(
        self,
        question: str,
        contexts: List[Dict[str, Any]],
        compressed_context: str,
        history_text: str = "",
    ) -> List[dict]:
        context_text = (
            compressed_context
            if compressed_context.strip()
            else "\n".join([item["content"] for item in contexts])
        )
        return prompts.get_rag_prompt_template(context_text, question, history_text)
