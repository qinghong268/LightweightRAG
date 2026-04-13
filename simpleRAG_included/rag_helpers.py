import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import faiss
import numpy as np
import requests

from .config_imports import logger
from .rag_exceptions import ModelConnectionError


class RAGHelpers:
    SNAPSHOT_FILE_LOCK = threading.RLock()

    @staticmethod
    def _ollama_headers() -> dict:
        return {"Content-Type": "application/json"}

    @staticmethod
    def _chat_completion(
        ollama_host: str,
        messages: List[dict],
        model_name: str,
        temperature: float,
    ) -> str:
        url = f"{ollama_host}/api/chat"
        data = {
            "model": model_name,
            "messages": messages,
            "options": {"temperature": temperature},
            "stream": False,
        }
        try:
            response = requests.post(
                url,
                headers=RAGHelpers._ollama_headers(),
                json=data,
                timeout=60,
            )
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "")
        except requests.exceptions.RequestException as exc:
            logger.error(f"Chat request failed: {exc}")
            raise ModelConnectionError(f"Unable to reach Ollama service: {exc}")

    @staticmethod
    def _chat_completion_stream(
        ollama_host: str,
        messages: List[dict],
        model_name: str,
        temperature: float,
    ) -> Iterable[str]:
        url = f"{ollama_host}/api/chat"
        data = {
            "model": model_name,
            "messages": messages,
            "options": {"temperature": temperature},
            "stream": True,
        }
        try:
            with requests.post(
                url,
                headers=RAGHelpers._ollama_headers(),
                json=data,
                stream=True,
                timeout=60,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line.decode("utf-8"))
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("done"):
                        break
                    yield chunk["message"]["content"]
        except requests.exceptions.RequestException as exc:
            logger.error(f"Streaming chat request failed: {exc}")
            raise ModelConnectionError(f"Unable to reach Ollama service: {exc}")

    @staticmethod
    def load_embedding_cache(cache_file_path: Path) -> Dict[str, List[float]]:
        if cache_file_path.exists():
            try:
                with open(cache_file_path, "r", encoding="utf-8") as file:
                    cache_data = json.load(file)
                logger.debug(
                    f"Loaded {len(cache_data)} embedding cache entries from {cache_file_path}"
                )
                return cache_data
            except (json.JSONDecodeError, FileNotFoundError) as exc:
                logger.warning(f"Embedding cache {cache_file_path} is invalid: {exc}")
        return {}

    @staticmethod
    def save_embedding_cache(cache: Dict[str, List[float]], cache_file_path: Path) -> None:
        with open(cache_file_path, "w", encoding="utf-8") as file:
            json.dump(cache, file, ensure_ascii=False, indent=2)
        logger.info(f"Saved embedding cache to {cache_file_path}")

    @staticmethod
    def ensure_schema(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                chunk_id INTEGER PRIMARY KEY,
                path TEXT,
                chunk_index INTEGER,
                content TEXT,
                chunk_hash TEXT
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS source_files (
                path TEXT PRIMARY KEY,
                fingerprint TEXT,
                updated_at TEXT
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS build_state (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            """
        )
        cols = [row[1] for row in conn.execute("PRAGMA table_info(metadata);").fetchall()]
        if "chunk_hash" not in cols:
            conn.execute("ALTER TABLE metadata ADD COLUMN chunk_hash TEXT;")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_path ON metadata(path);")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_chunk_hash_unique ON metadata(chunk_hash);")
        conn.commit()

    @staticmethod
    def get_build_state(conn: sqlite3.Connection, key: str, default: str = "") -> str:
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM build_state WHERE key = ?", (key,))
        row = cursor.fetchone()
        if not row:
            return default
        return str(row[0]) if row[0] is not None else default

    @staticmethod
    def set_build_state(
        conn: sqlite3.Connection,
        key: str,
        value: str,
        commit: bool = True,
    ) -> None:
        conn.execute(
            """
            INSERT INTO build_state (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            (key, str(value)),
        )
        if commit:
            conn.commit()

    @staticmethod
    def is_snapshot_dirty(conn: sqlite3.Connection) -> bool:
        raw_value = RAGHelpers.get_build_state(conn, "snapshot_dirty", default="0").strip().lower()
        return raw_value in {"1", "true", "yes", "y"}

    @staticmethod
    def set_snapshot_dirty(
        conn: sqlite3.Connection,
        dirty: bool,
        commit: bool = True,
    ) -> None:
        RAGHelpers.set_build_state(conn, "snapshot_dirty", "1" if dirty else "0", commit=commit)

    @staticmethod
    def bulk_insert_metadata(
        conn: sqlite3.Connection,
        records: List[Tuple],
        commit: bool = True,
    ) -> None:
        conn.executemany(
            "INSERT OR REPLACE INTO metadata (chunk_id, path, chunk_index, content, chunk_hash) VALUES (?, ?, ?, ?, ?)",
            records,
        )
        if commit:
            conn.commit()

    @staticmethod
    def delete_metadata_by_paths(
        conn: sqlite3.Connection,
        paths: List[str],
        commit: bool = True,
    ) -> None:
        if not paths:
            return
        placeholders = ",".join("?" for _ in paths)
        conn.execute(f"DELETE FROM metadata WHERE path IN ({placeholders})", tuple(paths))
        if commit:
            conn.commit()

    @staticmethod
    def get_chunk_ids_by_paths(
        conn: sqlite3.Connection,
        paths: List[str],
    ) -> List[int]:
        if not paths:
            return []
        placeholders = ",".join("?" for _ in paths)
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT chunk_id FROM metadata WHERE path IN ({placeholders}) ORDER BY chunk_id",
            tuple(paths),
        )
        return [int(row[0]) for row in cursor.fetchall()]

    @staticmethod
    def get_source_file_records(conn: sqlite3.Connection) -> Dict[str, Dict[str, Any]]:
        cursor = conn.cursor()
        cursor.execute("SELECT path, fingerprint, updated_at FROM source_files")
        return {
            row[0]: {
                "path": row[0],
                "fingerprint": row[1],
                "updated_at": row[2],
            }
            for row in cursor.fetchall()
        }

    @staticmethod
    def upsert_source_file_records(
        conn: sqlite3.Connection,
        records: List[Tuple[str, str]],
        commit: bool = True,
    ) -> None:
        if not records:
            return
        updated_at = datetime.now(timezone.utc).isoformat()
        conn.executemany(
            """
            INSERT INTO source_files (path, fingerprint, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                fingerprint=excluded.fingerprint,
                updated_at=excluded.updated_at
            """,
            [(path, fingerprint, updated_at) for path, fingerprint in records],
        )
        if commit:
            conn.commit()

    @staticmethod
    def delete_source_file_records_by_paths(
        conn: sqlite3.Connection,
        paths: List[str],
        commit: bool = True,
    ) -> None:
        if not paths:
            return
        placeholders = ",".join("?" for _ in paths)
        conn.execute(f"DELETE FROM source_files WHERE path IN ({placeholders})", tuple(paths))
        if commit:
            conn.commit()

    @staticmethod
    def get_all_metadata(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT chunk_id, path, chunk_index, content, chunk_hash FROM metadata ORDER BY chunk_id"
        )
        rows = cursor.fetchall()
        return [
            {
                "chunk_id": row[0],
                "path": row[1],
                "chunk_index": row[2],
                "content": row[3],
                "chunk_hash": row[4],
            }
            for row in rows
        ]

    @staticmethod
    def get_all_chunk_ids(conn: sqlite3.Connection) -> List[int]:
        cursor = conn.cursor()
        cursor.execute("SELECT chunk_id FROM metadata ORDER BY chunk_id")
        return [int(row[0]) for row in cursor.fetchall()]

    @staticmethod
    def get_metadata_by_chunk_ids(
        conn: sqlite3.Connection,
        chunk_ids: List[int],
    ) -> Dict[int, Dict[str, Any]]:
        if not chunk_ids:
            return {}

        placeholders = ",".join("?" for _ in chunk_ids)
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT chunk_id, path, chunk_index, content FROM metadata WHERE chunk_id IN ({placeholders})",
            tuple(chunk_ids),
        )
        return {
            row[0]: {"path": row[1], "chunk_index": row[2], "content": row[3]}
            for row in cursor.fetchall()
        }

    @staticmethod
    def get_max_chunk_id(conn: sqlite3.Connection) -> int:
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(chunk_id) FROM metadata")
        value = cursor.fetchone()[0]
        return int(value) if value is not None else -1

    @staticmethod
    def create_initial_faiss_index(dimension: int) -> faiss.Index:
        base_index = faiss.IndexFlatIP(dimension)
        id_map_cls = getattr(faiss, "IndexIDMap2", None) or getattr(faiss, "IndexIDMap", None)
        if id_map_cls is None:
            return base_index
        return id_map_cls(base_index)

    @staticmethod
    def add_vectors_to_faiss_index(
        index: faiss.Index,
        vectors: np.ndarray,
        vector_ids: List[int] = None,
    ) -> None:
        if vectors.size > 0:
            faiss.normalize_L2(vectors)
        if vector_ids is not None and hasattr(index, "add_with_ids"):
            ids_array = np.asarray(vector_ids, dtype=np.int64)
            index.add_with_ids(vectors, ids_array)
            return
        index.add(vectors)

    @staticmethod
    def remove_ids_from_faiss_index(index: faiss.Index, vector_ids: List[int]) -> int:
        if not vector_ids or not hasattr(index, "remove_ids"):
            return 0

        ids_array = np.asarray(vector_ids, dtype=np.int64)
        try:
            removed = index.remove_ids(ids_array)
        except TypeError:
            selector_cls = getattr(faiss, "IDSelectorBatch", None)
            swig_ptr = getattr(faiss, "swig_ptr", None)
            if selector_cls is None or swig_ptr is None:
                raise
            selector = selector_cls(ids_array.size, swig_ptr(ids_array))
            removed = index.remove_ids(selector)
        return int(removed)

    @staticmethod
    def _normalize_snapshot_metadata(metadata: Any) -> Dict[str, Any]:
        if isinstance(metadata, dict):
            chunk_ids = metadata.get("chunk_ids", [])
            uses_vector_ids = bool(metadata.get("uses_vector_ids"))
        else:
            chunk_ids = []
            if isinstance(metadata, list):
                for item in metadata:
                    if isinstance(item, dict) and item.get("chunk_id") is not None:
                        chunk_ids.append(int(item["chunk_id"]))
                    elif isinstance(item, int):
                        chunk_ids.append(int(item))
            uses_vector_ids = False

        return {
            "version": 2,
            "uses_vector_ids": uses_vector_ids,
            "chunk_ids": [int(chunk_id) for chunk_id in chunk_ids],
        }

    @staticmethod
    def save_faiss_index_and_metadata(
        index: faiss.Index,
        metadata_list: List[dict],
        faiss_file: Path,
        metadata_file: Path,
    ) -> None:
        faiss.write_index(index, str(faiss_file))
        simplified_metadata = {
            "version": 2,
            "uses_vector_ids": bool(hasattr(index, "add_with_ids")),
            "chunk_ids": [int(item["chunk_id"]) for item in metadata_list],
        }
        with open(metadata_file, "w", encoding="utf-8") as file:
            json.dump(simplified_metadata, file, ensure_ascii=False, indent=2)
        logger.info("Saved FAISS index and metadata map")

    @staticmethod
    def load_faiss_index_and_metadata(
        faiss_file: Path,
        metadata_file: Path,
    ) -> Tuple[faiss.Index, Dict[str, Any]]:
        with RAGHelpers.SNAPSHOT_FILE_LOCK:
            index = faiss.read_index(str(faiss_file))
            with open(metadata_file, "r", encoding="utf-8") as file:
                metadata = RAGHelpers._normalize_snapshot_metadata(json.load(file))
        logger.debug(
            f"Loaded FAISS index and metadata map: dimension={index.d}, total={index.ntotal}"
        )
        return index, metadata
