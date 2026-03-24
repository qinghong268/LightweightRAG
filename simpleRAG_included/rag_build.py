# rag_build.py
import asyncio
import hashlib
import time
import sqlite3
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any, Tuple, Iterator

from document_loader import batch_load_documents
from text_splitter import SmartTextSplitter
from .rag_helpers import RAGHelpers
from .rag_exceptions import BuildError
from .config_imports import (
    logger, BATCH_SIZE_DOCS, CHUNK_SIZE_DEFAULT, CHUNK_OVERLAP_DEFAULT,
    LOCAL_EMBEDDING_MODEL_PATH, SEMANTIC_SPLITTER_MODEL_NAME, 
    SEMANTIC_SPLITTER_THRESHOLD, SEMANTIC_SPLITTER_SEPARATORS,
    DB_PATH, FAISS_INDEX_FILE, METADATA_FILE
)

class RAGBuilder:
    def __init__(self, cache: Dict[str, List[float]], embedding_model_instance=None):
        self.cache = cache
        self.embedding_model = embedding_model_instance  # 使用传入的模型
        self.splitter = SmartTextSplitter(
            model_path=LOCAL_EMBEDDING_MODEL_PATH,
            model_name=SEMANTIC_SPLITTER_MODEL_NAME,
            threshold=SEMANTIC_SPLITTER_THRESHOLD,
            base_splitter_params={
                "chunk_size": CHUNK_SIZE_DEFAULT,
                "chunk_overlap": CHUNK_OVERLAP_DEFAULT,
                "separators": SEMANTIC_SPLITTER_SEPARATORS
            },
            load_timeout=60
        )

    def _load_document_batches(self, source_dir: Path, batch_size: int) -> Iterator[List[Any]]:
        all_docs = batch_load_documents(str(source_dir))
        for i in range(0, len(all_docs), batch_size):
            yield all_docs[i:i+batch_size]

    async def build_knowledge_base_async(self, source_dir: Path, chunk_size: int = None, overlap: int = None) -> None:
        start_time = time.time()
        logger.info(f"[构建] 开始处理：{source_dir}")

        if not source_dir.exists():
            raise BuildError(f"源目录不存在：{source_dir}")

        if chunk_size:
            self.splitter.base_splitter._chunk_size = chunk_size
        if overlap:
            self.splitter.base_splitter._chunk_overlap = overlap

        conn = sqlite3.connect(DB_PATH)
        RAGHelpers.ensure_schema(conn)

        # 增量模式：尽可能复用已有索引与元数据
        if FAISS_INDEX_FILE.exists() and METADATA_FILE.exists():
            try:
                initial_index, all_metadata = RAGHelpers.load_faiss_index_and_metadata(FAISS_INDEX_FILE, METADATA_FILE)
                logger.info(f"[构建] 进入增量更新模式，现有向量数：{initial_index.ntotal}")
            except Exception as e:
                logger.warning(f"[构建] 读取已有索引失败，将重新创建索引。原因: {e}")
                initial_index = None
                all_metadata = []
        else:
            initial_index = None
            all_metadata = []

        global_chunk_id = RAGHelpers.get_max_chunk_id(conn) + 1
        existing_hashes = RAGHelpers.get_existing_chunk_hashes(conn)
        existing_path_content_keys = RAGHelpers.get_existing_path_content_keys(conn)
        total_added = 0
        total_skipped = 0

        batches = list(self._load_document_batches(source_dir, BATCH_SIZE_DOCS))
        total = len(batches)

        for idx, batch_docs in enumerate(batches):
            logger.info(f"[进度] {idx+1}/{total} 批次处理中...")
            
            texts = [doc.page_content for doc in batch_docs]
            sources = [doc.metadata.get('source', 'unknown') for doc in batch_docs]
            
            all_chunks = []
            doc_chunk_counts = []
            
            for text in texts:
                if not text.strip():
                    doc_chunk_counts.append(0)
                    continue
                chunks = self.splitter._perform_semantic_split(text) if self.splitter.use_semantic_splitting and self.splitter.model else self.splitter.base_splitter.split_text(text)
                all_chunks.extend(chunks)
                doc_chunk_counts.append(len(chunks))

            if not all_chunks:
                continue

            # 使用主模型向量化
            model_to_use = self.embedding_model if self.embedding_model else self.splitter.model
            if model_to_use is None:
                raise Exception("未找到可用的嵌入模型进行向量化")
            
            with torch.no_grad():
                embeddings_array = model_to_use.encode(
                    all_chunks,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
            embeddings = embeddings_array.tolist()

            batch_valid_embs = []
            batch_valid_meta = []
            
            chunk_ptr = 0
            for i, count in enumerate(doc_chunk_counts):
                source_path = sources[i]
                for j in range(count):
                    if chunk_ptr >= len(all_chunks):
                        break
                    
                    chunk_text = all_chunks[chunk_ptr]
                    emb = embeddings[chunk_ptr]
                    
                    # 去重粒度按“来源文档 + 文本块”，避免不同文档同内容被误去重
                    h = hashlib.sha256(f"{source_path}::{chunk_text}".encode()).hexdigest()
                    if h in self.cache:
                        emb = self.cache[h]
                    else:
                        self.cache[h] = emb
                    
                    if emb and len(emb) > 0:
                        dedupe_key = (source_path, chunk_text)
                        if h in existing_hashes or dedupe_key in existing_path_content_keys:
                            total_skipped += 1
                            chunk_ptr += 1
                            continue
                        batch_valid_embs.append(emb)
                        batch_valid_meta.append({
                            "chunk_id": global_chunk_id,
                            "path": source_path,
                            "chunk_index": j,
                            "content": chunk_text,
                            "chunk_hash": h
                        })
                        existing_hashes.add(h)
                        existing_path_content_keys.add(dedupe_key)
                        global_chunk_id += 1
                    
                    chunk_ptr += 1

            if not batch_valid_embs:
                continue

            vec_array = np.array(batch_valid_embs, dtype=np.float32)
            if initial_index is None:
                dim = vec_array.shape[1]
                logger.info(f"创建新索引，维度：{dim}")
                initial_index = RAGHelpers.create_initial_faiss_index(dim)
            
            if initial_index.d != vec_array.shape[1]:
                raise BuildError(
                    f"索引维度不匹配：已有索引维度={initial_index.d}，当前向量维度={vec_array.shape[1]}。"
                )

            RAGHelpers.add_vectors_to_faiss_index(initial_index, vec_array)
            all_metadata.extend(batch_valid_meta)
            total_added += len(batch_valid_meta)

            records = [
                (m['chunk_id'], m['path'], m['chunk_index'], m['content'], m['chunk_hash'])
                for m in batch_valid_meta
            ]
            RAGHelpers.bulk_insert_metadata(conn, records)
            
            logger.info(f"批次 {idx+1} 完成：新增 {len(batch_valid_meta)} 个块。")

        conn.close()
        
        if initial_index:
            RAGHelpers.save_faiss_index_and_metadata(initial_index, all_metadata, FAISS_INDEX_FILE, METADATA_FILE)
        
        elapsed = time.time() - start_time
        logger.info(
            f"[完成] 总耗时 {elapsed:.2f}s, 索引总块数 {len(all_metadata)}, 本次新增 {total_added}, 本次跳过重复 {total_skipped}"
        )