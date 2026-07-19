import sqlite3
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import prompts
from config import BASE_DIR

try:
    from config import LOCAL_RERANK_MODEL_PATH
except ImportError:
    LOCAL_RERANK_MODEL_PATH = None

from .config_imports import (
    DB_PATH,
    FAISS_INDEX_FILE,
    METADATA_FILE,
    MIN_RETRIEVE_KEEP,
    logger,
)
from .rag_exceptions import SnapshotLoadError
from .rag_helpers import RAGHelpers

def _patch_transformers_for_flagembedding() -> None:
    """FlagEmbedding 1.3.x expects transformers v4 tokenizer APIs removed in v5."""
    import transformers.utils.import_utils as _tf_import_utils

    if not hasattr(_tf_import_utils, "is_torch_fx_available"):
        _tf_import_utils.is_torch_fx_available = lambda: True  # type: ignore[attr-defined]

    from transformers import PreTrainedTokenizerBase
    from transformers.tokenization_utils_base import (
        BatchEncoding,
        PaddingStrategy,
        TruncationStrategy,
    )

    if getattr(PreTrainedTokenizerBase, "_lwrag_v5_compat", False):
        return

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bos = self.cls_token_id if self.cls_token_id is not None else self.bos_token_id
        eos = self.sep_token_id if self.sep_token_id is not None else self.eos_token_id
        ids0 = list(token_ids_0)
        if bos is None or eos is None:
            return ids0 if token_ids_1 is None else ids0 + list(token_ids_1)
        if token_ids_1 is None:
            return [bos] + ids0 + [eos]
        # XLM-R / RoBERTa pair: <s> A </s> B </s>
        return [bos] + ids0 + [eos] + list(token_ids_1) + [eos]

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return [0] * (len(token_ids_0) + 2)
        return [0] * (len(token_ids_0) + len(token_ids_1) + 3)

    def truncate_sequences(
        self,
        ids,
        pair_ids=None,
        num_tokens_to_remove=0,
        truncation_strategy="longest_first",
        stride=0,
    ):
        if num_tokens_to_remove <= 0:
            return ids, pair_ids, []
        ids = list(ids)
        pair_ids = list(pair_ids) if pair_ids is not None else None
        overflowing = []
        strategy = truncation_strategy.value if hasattr(truncation_strategy, "value") else truncation_strategy
        if strategy == "only_second" and pair_ids is not None:
            while num_tokens_to_remove > 0 and pair_ids:
                overflowing.append(pair_ids.pop())
                num_tokens_to_remove -= 1
        elif strategy == "only_first":
            while num_tokens_to_remove > 0 and ids:
                overflowing.append(ids.pop())
                num_tokens_to_remove -= 1
        else:
            while num_tokens_to_remove > 0:
                if pair_ids is not None and len(pair_ids) >= len(ids):
                    overflowing.append(pair_ids.pop())
                elif ids:
                    overflowing.append(ids.pop())
                else:
                    break
                num_tokens_to_remove -= 1
        return ids, pair_ids, overflowing

    def prepare_for_model(
        self,
        ids,
        pair_ids=None,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        max_length=None,
        stride=0,
        pad_to_multiple_of=None,
        padding_side=None,
        return_tensors=None,
        return_token_type_ids=None,
        return_attention_mask=None,
        return_overflowing_tokens=False,
        return_special_tokens_mask=False,
        return_length=False,
        verbose=True,
        prepend_batch_axis=False,
        **kwargs,
    ):
        padding_strategy, truncation_strategy, max_length, _ = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        pair = pair_ids is not None
        num_special = self.num_special_tokens_to_add(pair=pair) if add_special_tokens else 0
        total_len = len(ids) + len(pair_ids or []) + num_special
        overflowing_tokens = []
        if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and max_length and total_len > max_length:
            ids, pair_ids, overflowing_tokens = self.truncate_sequences(
                ids,
                pair_ids=pair_ids,
                num_tokens_to_remove=total_len - max_length,
                truncation_strategy=truncation_strategy,
                stride=stride,
            )

        if add_special_tokens:
            sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
            token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
        else:
            sequence = list(ids) + (list(pair_ids) if pair_ids else [])
            token_type_ids = [0] * len(sequence)

        encoded_inputs = {"input_ids": sequence}
        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = token_type_ids
        if return_special_tokens_mask:
            encoded_inputs["special_tokens_mask"] = [0] * len(sequence)
        if return_overflowing_tokens and overflowing_tokens:
            encoded_inputs["overflowing_tokens"] = overflowing_tokens

        if padding_strategy != PaddingStrategy.DO_NOT_PAD or return_attention_mask:
            encoded_inputs = self.pad(
                encoded_inputs,
                max_length=max_length,
                padding=padding_strategy.value,
                pad_to_multiple_of=pad_to_multiple_of,
                padding_side=padding_side,
                return_attention_mask=return_attention_mask,
            )
        if return_length:
            encoded_inputs["length"] = len(encoded_inputs["input_ids"])
        return BatchEncoding(encoded_inputs, tensor_type=return_tensors, prepend_batch_axis=prepend_batch_axis)

    if not hasattr(PreTrainedTokenizerBase, "build_inputs_with_special_tokens"):
        PreTrainedTokenizerBase.build_inputs_with_special_tokens = build_inputs_with_special_tokens  # type: ignore[attr-defined]
    if not hasattr(PreTrainedTokenizerBase, "create_token_type_ids_from_sequences"):
        PreTrainedTokenizerBase.create_token_type_ids_from_sequences = create_token_type_ids_from_sequences  # type: ignore[attr-defined]
    if not hasattr(PreTrainedTokenizerBase, "truncate_sequences"):
        PreTrainedTokenizerBase.truncate_sequences = truncate_sequences  # type: ignore[attr-defined]
    if not hasattr(PreTrainedTokenizerBase, "prepare_for_model"):
        PreTrainedTokenizerBase.prepare_for_model = prepare_for_model  # type: ignore[attr-defined]

    PreTrainedTokenizerBase._lwrag_v5_compat = True  # type: ignore[attr-defined]


try:
    _patch_transformers_for_flagembedding()
    from FlagEmbedding import FlagReranker

    RERANKER_AVAILABLE = True
except Exception as exc:
    RERANKER_AVAILABLE = False
    logger.warning(
        "FlagEmbedding/reranker unavailable (%s); vector Top-K fallback will be used.",
        exc,
    )


class RAGQuerier:
    def __init__(self, ollama_host: str, chat_model: str, reranker_model_name: str):
        self._ollama_host = ollama_host
        self._chat_model = chat_model
        self._reranker_model_name = reranker_model_name
        if LOCAL_RERANK_MODEL_PATH is not None:
            self._reranker_model_path = LOCAL_RERANK_MODEL_PATH
        else:
            self._reranker_model_path = str(Path(BASE_DIR) / "models" / reranker_model_name)
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

    def prepare_final_prompt(
        self,
        question: str,
        contexts: List[Dict[str, Any]],
        context_text: str,
        history_text: str = "",
    ) -> List[dict]:
        context_text = context_text if context_text.strip() else "\n".join([item["content"] for item in contexts])
        return prompts.get_rag_prompt_template(context_text, question, history_text)
