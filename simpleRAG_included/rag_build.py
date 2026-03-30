import hashlib
import os
import sqlite3
import time
from contextlib import suppress
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from document_loader import batch_load_documents
from text_splitter import SmartTextSplitter
from .config_imports import (
    CHUNK_OVERLAP_DEFAULT,
    CHUNK_SIZE_DEFAULT,
    DB_PATH,
    FAISS_INDEX_FILE,
    LOCAL_EMBEDDING_MODEL_PATH,
    METADATA_FILE,
    SEMANTIC_SPLITTER_MODEL_NAME,
    SEMANTIC_SPLITTER_SEPARATORS,
    SEMANTIC_SPLITTER_THRESHOLD,
    logger,
)
from .rag_exceptions import BuildError
from .rag_helpers import RAGHelpers


class RAGBuilder:
    def __init__(self, cache: Dict[str, List[float]], embedding_model_instance=None):
        self.cache = cache
        self.embedding_model = embedding_model_instance
        self.splitter = SmartTextSplitter(
            model_path=LOCAL_EMBEDDING_MODEL_PATH,
            model_name=SEMANTIC_SPLITTER_MODEL_NAME,
            threshold=SEMANTIC_SPLITTER_THRESHOLD,
            base_splitter_params={
                "chunk_size": CHUNK_SIZE_DEFAULT,
                "chunk_overlap": CHUNK_OVERLAP_DEFAULT,
                "separators": SEMANTIC_SPLITTER_SEPARATORS,
            },
            load_timeout=60,
            model_instance=embedding_model_instance,
        )

    def _normalize_path(self, path: Path) -> str:
        return os.path.normcase(os.path.normpath(str(path)))

    def _path_belongs_to_source_dir(self, candidate_path: str, source_dir: Path) -> bool:
        candidate = Path(candidate_path).resolve(strict=False)
        source_root = Path(source_dir).resolve(strict=False)
        try:
            candidate.relative_to(source_root)
            return True
        except ValueError:
            return False

    def _load_documents_grouped_by_source(self, source_dir: Path) -> List[Tuple[str, List[str]]]:
        grouped: "OrderedDict[str, List[str]]" = OrderedDict()
        for doc in batch_load_documents(str(source_dir)):
            source_path = doc.metadata.get("source", "unknown")
            page_text = (doc.page_content or "").strip()
            if not page_text:
                continue
            grouped.setdefault(source_path, []).append(page_text)
        return list(grouped.items())

    def _split_source_texts(self, texts: List[str]) -> List[str]:
        chunks: List[str] = []
        for text in texts:
            if not text.strip():
                continue
            if self.splitter.use_semantic_splitting and self.splitter.model is not None:
                parts = self.splitter._perform_semantic_split(text)
            else:
                parts = self.splitter.base_splitter.split_text(text)
            chunks.extend(chunk.strip() for chunk in parts if chunk and chunk.strip())
        return chunks

    def _encode_texts_with_cache(self, texts: List[str], hash_keys: List[str]) -> List[List[float]]:
        if len(texts) != len(hash_keys):
            raise BuildError("Text and hash key counts do not match during embedding")

        model_to_use = self.embedding_model if self.embedding_model else self.splitter.model
        if model_to_use is None:
            raise BuildError("No embedding model available for knowledge base build")

        embeddings: List[List[float]] = [None] * len(texts)  # type: ignore[list-item]
        missing_indices: List[int] = []
        missing_texts: List[str] = []

        for idx, hash_key in enumerate(hash_keys):
            cached = self.cache.get(hash_key)
            if cached is not None:
                embeddings[idx] = cached
            else:
                missing_indices.append(idx)
                missing_texts.append(texts[idx])

        if missing_texts:
            with torch.no_grad():
                encoded = model_to_use.encode(
                    missing_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            encoded_list = encoded.tolist()
            for idx, vector in zip(missing_indices, encoded_list):
                embeddings[idx] = vector
                self.cache[hash_keys[idx]] = vector

        return embeddings

    def _rebuild_faiss_from_db(
        self,
        conn: sqlite3.Connection,
        faiss_file: Path = None,
        metadata_file: Path = None,
    ) -> int:
        faiss_file = faiss_file or FAISS_INDEX_FILE
        metadata_file = metadata_file or METADATA_FILE
        all_metadata = RAGHelpers.get_all_metadata(conn)
        if not all_metadata:
            if faiss_file.exists():
                faiss_file.unlink()
            if metadata_file.exists():
                metadata_file.unlink()
            logger.info("Knowledge base is empty after sync; removed stale index files")
            return 0

        texts = [item["content"] for item in all_metadata]
        hash_keys = []
        for item in all_metadata:
            chunk_hash = item.get("chunk_hash")
            if not chunk_hash:
                chunk_hash = hashlib.sha256(
                    f"{item['path']}::{item['content']}".encode("utf-8")
                ).hexdigest()
                item["chunk_hash"] = chunk_hash
            hash_keys.append(chunk_hash)

        embeddings = self._encode_texts_with_cache(texts, hash_keys)
        vec_array = np.array(embeddings, dtype=np.float32)
        index = RAGHelpers.create_initial_faiss_index(vec_array.shape[1])
        RAGHelpers.add_vectors_to_faiss_index(index, vec_array)
        RAGHelpers.save_faiss_index_and_metadata(
            index,
            all_metadata,
            faiss_file,
            metadata_file,
        )
        return len(all_metadata)

    async def build_knowledge_base_async(
        self,
        source_dir: Path,
        chunk_size: int = None,
        overlap: int = None,
    ) -> Dict[str, Any]:
        start_time = time.time()
        logger.info(f"Start building knowledge base from {source_dir}")

        if not source_dir.exists():
            raise BuildError(f"Source directory does not exist: {source_dir}")

        if chunk_size:
            self.splitter.base_splitter._chunk_size = chunk_size
        if overlap:
            self.splitter.base_splitter._chunk_overlap = overlap

        grouped_sources = self._load_documents_grouped_by_source(source_dir)
        current_paths = {path for path, _ in grouped_sources}
        discovered_documents = len(grouped_sources)
        empty_documents = 0

        conn = sqlite3.connect(DB_PATH)
        temp_faiss_file = FAISS_INDEX_FILE.with_suffix(f"{FAISS_INDEX_FILE.suffix}.tmp")
        temp_metadata_file = METADATA_FILE.with_suffix(f"{METADATA_FILE.suffix}.tmp")
        final_total = 0
        total_added = 0
        new_paths = set()
        refreshed_paths = set()
        stale_paths = set()
        try:
            RAGHelpers.ensure_schema(conn)
            conn.execute("BEGIN IMMEDIATE")

            existing_metadata = RAGHelpers.get_all_metadata(conn)
            existing_paths = {item["path"] for item in existing_metadata}
            existing_source_paths = {
                path
                for path in existing_paths
                if self._path_belongs_to_source_dir(path, source_dir)
            }
            stale_paths = {
                path
                for path in existing_source_paths
                if path not in current_paths
            }
            new_paths = current_paths - existing_source_paths
            refreshed_paths = current_paths & existing_source_paths
            paths_to_replace = sorted(current_paths | stale_paths)

            if paths_to_replace:
                RAGHelpers.delete_metadata_by_paths(conn, paths_to_replace, commit=False)
                logger.info(f"Removed stale metadata for {len(paths_to_replace)} source files")

            next_chunk_id = RAGHelpers.get_max_chunk_id(conn) + 1

            for file_index, (source_path, texts) in enumerate(grouped_sources, start=1):
                logger.info(f"Syncing document {file_index}/{len(grouped_sources)}: {source_path}")
                chunks = self._split_source_texts(texts)
                if not chunks:
                    logger.info(f"Skipped empty document after splitting: {source_path}")
                    empty_documents += 1
                    continue

                chunk_hashes = [
                    hashlib.sha256(f"{source_path}::{chunk}".encode("utf-8")).hexdigest()
                    for chunk in chunks
                ]
                embeddings = self._encode_texts_with_cache(chunks, chunk_hashes)

                records = []
                for chunk_index, (chunk_text, chunk_hash, embedding) in enumerate(
                    zip(chunks, chunk_hashes, embeddings)
                ):
                    if not embedding:
                        continue
                    records.append(
                        (
                            next_chunk_id,
                            source_path,
                            chunk_index,
                            chunk_text,
                            chunk_hash,
                        )
                    )
                    next_chunk_id += 1

                if records:
                    RAGHelpers.bulk_insert_metadata(conn, records, commit=False)
                    total_added += len(records)

            final_total = self._rebuild_faiss_from_db(
                conn,
                faiss_file=temp_faiss_file,
                metadata_file=temp_metadata_file,
            )
            with RAGHelpers.SNAPSHOT_FILE_LOCK:
                conn.commit()
                if temp_faiss_file.exists() and temp_metadata_file.exists():
                    os.replace(temp_faiss_file, FAISS_INDEX_FILE)
                    os.replace(temp_metadata_file, METADATA_FILE)
                else:
                    with suppress(FileNotFoundError):
                        FAISS_INDEX_FILE.unlink()
                    with suppress(FileNotFoundError):
                        METADATA_FILE.unlink()
        except Exception:
            conn.rollback()
            with suppress(FileNotFoundError):
                temp_faiss_file.unlink()
            with suppress(FileNotFoundError):
                temp_metadata_file.unlink()
            raise
        finally:
            conn.close()

        elapsed = time.time() - start_time
        logger.info(
            f"Knowledge base sync finished in {elapsed:.2f}s, rebuilt total chunks={final_total}, newly written chunks={total_added}"
        )
        return {
            "source_dir": str(source_dir.resolve(strict=False)),
            "discovered_documents": discovered_documents,
            "active_documents": len(current_paths),
            "new_documents": len(new_paths),
            "refreshed_documents": len(refreshed_paths),
            "removed_documents": len(stale_paths),
            "empty_documents": empty_documents,
            "written_chunks": total_added,
            "total_chunks": final_total,
            "duration_seconds": round(elapsed, 2),
            "snapshot_status": "cleared" if final_total == 0 else "updated",
        }
