import hashlib
import os
import sqlite3
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from document_loader import discover_supported_document_files, load_single_document
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

    def _path_belongs_to_source_dir(self, candidate_path: str, source_dir: Path) -> bool:
        candidate = Path(candidate_path).resolve(strict=False)
        source_root = Path(source_dir).resolve(strict=False)
        try:
            candidate.relative_to(source_root)
            return True
        except ValueError:
            return False

    def _calculate_file_fingerprint(self, file_path: Path) -> str:
        hasher = hashlib.sha256()
        with file_path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _discover_source_files_with_fingerprints(
        self,
        source_dir: Path,
    ) -> Tuple[List[str], Dict[str, Path], Dict[str, str]]:
        source_paths: List[str] = []
        path_lookup: Dict[str, Path] = {}
        fingerprints: Dict[str, str] = {}

        for file_path in discover_supported_document_files(source_dir):
            source_path = str(file_path)
            source_paths.append(source_path)
            path_lookup[source_path] = file_path
            fingerprints[source_path] = self._calculate_file_fingerprint(file_path)

        return source_paths, path_lookup, fingerprints

    def _load_document_texts(self, file_path: Path) -> List[str]:
        texts: List[str] = []
        for doc in load_single_document(str(file_path)):
            page_text = (doc.page_content or "").strip()
            if page_text:
                texts.append(page_text)
        return texts

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

        embeddings: List[List[float]] = [None] * len(texts)
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

    def _save_faiss_snapshot(
        self,
        index,
        conn: sqlite3.Connection,
        faiss_file: Path,
        metadata_file: Path,
    ) -> int:
        chunk_ids = RAGHelpers.get_all_chunk_ids(conn)
        if not chunk_ids:
            if faiss_file.exists():
                faiss_file.unlink()
            if metadata_file.exists():
                metadata_file.unlink()
            logger.info("Knowledge base is empty after sync; removed stale index files")
            return 0

        RAGHelpers.save_faiss_index_and_metadata(
            index,
            [{"chunk_id": chunk_id} for chunk_id in chunk_ids],
            faiss_file,
            metadata_file,
        )
        return len(chunk_ids)

    def _load_incremental_faiss_index(
        self,
        faiss_file: Path = None,
        metadata_file: Path = None,
    ):
        faiss_file = faiss_file or FAISS_INDEX_FILE
        metadata_file = metadata_file or METADATA_FILE
        if not faiss_file.exists() or not metadata_file.exists():
            return None

        try:
            index, metadata = RAGHelpers.load_faiss_index_and_metadata(faiss_file, metadata_file)
        except Exception as exc:
            logger.warning(f"Failed to load existing FAISS snapshot for incremental update: {exc}")
            return None

        if not isinstance(metadata, dict) or not metadata.get("uses_vector_ids"):
            logger.info("Existing FAISS snapshot does not use vector ids; falling back to full rebuild.")
            return None
        if not hasattr(index, "add_with_ids") or not hasattr(index, "remove_ids"):
            logger.info("Current FAISS index type does not support id-based incremental updates.")
            return None
        return index

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
        vector_ids = [int(item["chunk_id"]) for item in all_metadata]
        RAGHelpers.add_vectors_to_faiss_index(index, vec_array, vector_ids=vector_ids)
        return self._save_faiss_snapshot(index, conn, faiss_file, metadata_file)

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

        current_source_paths, path_lookup, current_fingerprints = self._discover_source_files_with_fingerprints(
            source_dir
        )
        current_paths = set(current_source_paths)
        discovered_documents = len(current_source_paths)
        empty_documents = 0

        conn = sqlite3.connect(DB_PATH)
        temp_faiss_file = FAISS_INDEX_FILE.with_suffix(f"{FAISS_INDEX_FILE.suffix}.tmp")
        temp_metadata_file = METADATA_FILE.with_suffix(f"{METADATA_FILE.suffix}.tmp")
        final_total = 0
        total_added = 0
        new_paths = set()
        refreshed_paths = set()
        unchanged_paths = set()
        stale_paths = set()
        try:
            RAGHelpers.ensure_schema(conn)
            snapshot_requires_refresh = RAGHelpers.is_snapshot_dirty(conn)
            snapshot_chunk_count = None
            if FAISS_INDEX_FILE.exists() and METADATA_FILE.exists():
                try:
                    _, snapshot_metadata = RAGHelpers.load_faiss_index_and_metadata(
                        FAISS_INDEX_FILE,
                        METADATA_FILE,
                    )
                    if not isinstance(snapshot_metadata, dict) or not snapshot_metadata.get("uses_vector_ids"):
                        snapshot_requires_refresh = True
                    else:
                        snapshot_chunk_ids = snapshot_metadata.get("chunk_ids", [])
                        if isinstance(snapshot_chunk_ids, list):
                            snapshot_chunk_count = len(snapshot_chunk_ids)
                        else:
                            snapshot_requires_refresh = True
                except Exception:
                    snapshot_requires_refresh = True
            db_chunk_count = int(conn.execute("SELECT COUNT(*) FROM metadata").fetchone()[0])
            if snapshot_chunk_count is not None and snapshot_chunk_count != db_chunk_count:
                snapshot_requires_refresh = True

            existing_source_records = RAGHelpers.get_source_file_records(conn)
            existing_source_paths = {
                path
                for path in existing_source_records.keys()
                if self._path_belongs_to_source_dir(path, source_dir)
            }
            stale_paths = {
                path
                for path in existing_source_paths
                if path not in current_paths
            }
            new_paths = current_paths - existing_source_paths
            refreshed_paths = {
                path
                for path in (current_paths & existing_source_paths)
                if existing_source_records.get(path, {}).get("fingerprint") != current_fingerprints.get(path)
            }
            unchanged_paths = current_paths - new_paths - refreshed_paths
            paths_to_replace = sorted(new_paths | refreshed_paths)
            paths_to_delete = sorted(stale_paths | refreshed_paths)

            snapshot_changed = bool(
                paths_to_replace
                or stale_paths
                or snapshot_requires_refresh
                or not FAISS_INDEX_FILE.exists()
                or not METADATA_FILE.exists()
            )
            force_full_snapshot_rebuild = snapshot_requires_refresh
            working_index = (
                self._load_incremental_faiss_index()
                if snapshot_changed and not force_full_snapshot_rebuild
                else None
            )
            can_apply_incremental_snapshot = working_index is not None

            if paths_to_delete:
                chunk_ids_to_delete = RAGHelpers.get_chunk_ids_by_paths(conn, paths_to_delete)
                if can_apply_incremental_snapshot and chunk_ids_to_delete:
                    try:
                        RAGHelpers.remove_ids_from_faiss_index(working_index, chunk_ids_to_delete)
                    except Exception as exc:
                        logger.warning(
                            f"Incremental FAISS deletion failed; falling back to full rebuild: {exc}"
                        )
                        can_apply_incremental_snapshot = False
                        working_index = None
                conn.execute("BEGIN IMMEDIATE")
                RAGHelpers.delete_metadata_by_paths(conn, paths_to_delete, commit=False)
                RAGHelpers.delete_source_file_records_by_paths(conn, paths_to_delete, commit=False)
                RAGHelpers.set_snapshot_dirty(conn, True, commit=False)
                conn.commit()
                logger.info(f"Removed stale metadata for {len(paths_to_delete)} source files")

            next_chunk_id = RAGHelpers.get_max_chunk_id(conn) + 1
            for file_index, source_path in enumerate(paths_to_replace, start=1):
                logger.info(f"Syncing document {file_index}/{len(paths_to_replace)}: {source_path}")
                file_path = path_lookup[source_path]
                texts = self._load_document_texts(file_path)
                chunks = self._split_source_texts(texts)
                if not chunks:
                    logger.info(f"Skipped empty document after splitting: {source_path}")
                    empty_documents += 1
                    chunk_hashes = []
                    embeddings = []
                else:
                    chunk_hashes = [
                        hashlib.sha256(f"{source_path}::{chunk}".encode("utf-8")).hexdigest()
                        for chunk in chunks
                    ]
                    embeddings = self._encode_texts_with_cache(chunks, chunk_hashes)

                records: List[Tuple[int, str, int, str, str]] = []
                record_ids: List[int] = []
                record_vectors: List[List[float]] = []
                for chunk_index, (chunk_text, chunk_hash, embedding) in enumerate(
                    zip(chunks, chunk_hashes, embeddings)
                ):
                    if not embedding:
                        continue
                    chunk_id = next_chunk_id
                    records.append(
                        (
                            chunk_id,
                            source_path,
                            chunk_index,
                            chunk_text,
                            chunk_hash,
                        )
                    )
                    record_ids.append(chunk_id)
                    record_vectors.append(embedding)
                    next_chunk_id += 1

                conn.execute("BEGIN IMMEDIATE")
                if records:
                    RAGHelpers.bulk_insert_metadata(conn, records, commit=False)
                    total_added += len(records)
                RAGHelpers.upsert_source_file_records(
                    conn,
                    [(source_path, current_fingerprints[source_path])],
                    commit=False,
                )
                RAGHelpers.set_snapshot_dirty(conn, True, commit=False)
                conn.commit()

                if can_apply_incremental_snapshot and record_vectors:
                    try:
                        RAGHelpers.add_vectors_to_faiss_index(
                            working_index,
                            np.array(record_vectors, dtype=np.float32),
                            vector_ids=record_ids,
                        )
                    except Exception as exc:
                        logger.warning(
                            f"Incremental FAISS addition failed; falling back to full rebuild: {exc}"
                        )
                        can_apply_incremental_snapshot = False
                        working_index = None

            if snapshot_changed:
                if can_apply_incremental_snapshot and working_index is not None:
                    final_total = self._save_faiss_snapshot(
                        working_index,
                        conn,
                        temp_faiss_file,
                        temp_metadata_file,
                    )
                else:
                    final_total = self._rebuild_faiss_from_db(
                        conn,
                        faiss_file=temp_faiss_file,
                        metadata_file=temp_metadata_file,
                    )
            else:
                final_total = conn.execute("SELECT COUNT(*) FROM metadata").fetchone()[0]

            with RAGHelpers.SNAPSHOT_FILE_LOCK:
                if snapshot_changed and temp_faiss_file.exists() and temp_metadata_file.exists():
                    os.replace(temp_faiss_file, FAISS_INDEX_FILE)
                    os.replace(temp_metadata_file, METADATA_FILE)
                elif snapshot_changed:
                    with suppress(FileNotFoundError):
                        FAISS_INDEX_FILE.unlink()
                    with suppress(FileNotFoundError):
                        METADATA_FILE.unlink()
            if snapshot_changed:
                conn.execute("BEGIN IMMEDIATE")
                RAGHelpers.set_snapshot_dirty(conn, False, commit=False)
                conn.commit()
        except Exception:
            with suppress(sqlite3.Error):
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
            "unchanged_documents": len(unchanged_paths),
            "removed_documents": len(stale_paths),
            "empty_documents": empty_documents,
            "written_chunks": total_added,
            "total_chunks": final_total,
            "duration_seconds": round(elapsed, 2),
            "snapshot_status": (
                "cleared"
                if final_total == 0
                else "updated"
                if snapshot_changed
                else "unchanged"
            ),
        }
