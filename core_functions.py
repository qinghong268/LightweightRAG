"""
SimpleRAG 模块
将 RAG 的核心流程（构建、检索、压缩、生成）封装成一个类。
"""

import asyncio
import aiohttp
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Iterable, List, Dict, Any, Tuple, Iterator, TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any as AnyType

import faiss
import numpy as np
import requests

import document_loader
import text_splitter
import prompts
from config import (
    # Ollama
    OLLAMA_HOST,
    CHAT_MODEL,
    COMPRESSOR_MODEL,
    EMBEDDING_MODEL,
    # Files
    CACHE_FILE,
    DB_PATH,
    FAISS_INDEX_FILE,
    METADATA_FILE,
    # Build
    BATCH_SIZE_DOCS,
    CHUNK_SIZE_DEFAULT,
    CHUNK_OVERLAP_DEFAULT,
    LOCAL_EMBEDDING_MODEL_PATH,
    SEMANTIC_SPLITTER_MODEL_NAME,
    SEMANTIC_SPLITTER_THRESHOLD,
    SEMANTIC_SPLITTER_SEPARATORS,
    OLLAMA_EMBED_BATCH_SIZE_INTERNAL,
    # Query
    DEFAULT_TOP_K,
    DEFAULT_TOP_K_COMPRESSED,
    DEFAULT_THRESHOLD,
    OLLAMA_CHAT_TEMPERATURE_DEFAULT,
    OLLAMA_COMPRESSOR_TEMPERATURE_DEFAULT,
)


class SimpleRAG:
    def __init__(self):
        self.cache = self.load_embedding_cache()

    # --- Internal helpers ---
    def _ollama_headers(self) -> dict:
        return {"Content-Type": "application/json"}

    def _fetch_embedding(self, text: str) -> List[float]:
        """使用本地 Ollama 的嵌入模型获取向量"""
        url = f"{OLLAMA_HOST}/api/embeddings"
        data = {"model": EMBEDDING_MODEL, "prompt": text}
        response = requests.post(url, headers=self._ollama_headers(), json=data)
        response.raise_for_status()
        return response.json()["embedding"]

    async def _async_fetch_embedding(self, session: aiohttp.ClientSession, text: str) -> List[float]:
        """异步获取嵌入向量"""
        url = f"{OLLAMA_HOST}/api/embeddings"
        data = {"model": EMBEDDING_MODEL, "prompt": text}
        async with session.post(url, headers=self._ollama_headers(), json=data) as response:
            response.raise_for_status()
            result = await response.json()
            return result["embedding"]

    def _chat_completion(self, messages: List[dict], model_name: str, temperature: float) -> str:
        """非流式聊天完成"""
        url = f"{OLLAMA_HOST}/api/chat"
        data = {
            "model": model_name,
            "messages": messages,
            "options": {"temperature": temperature},
            "stream": False
        }
        response = requests.post(url, headers=self._ollama_headers(), json=data)
        response.raise_for_status()
        return response.json().get('message', {}).get('content', '')

    def _chat_completion_stream(self, messages: List[dict], model_name: str, temperature: float) -> Iterable[str]:
        """流式聊天完成"""
        url = f"{OLLAMA_HOST}/api/chat"
        data = {
            "model": model_name,
            "messages": messages,
            "options": {"temperature": temperature},
            "stream": True
        }
        response = requests.post(url, headers=self._ollama_headers(), json=data, stream=True)
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line.decode('utf-8'))
                    if chunk.get('done'):
                        break
                    yield chunk['message']['content']
                except json.JSONDecodeError:
                    continue

    @staticmethod
    def load_embedding_cache():
        """加载嵌入缓存"""
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                print(f"缓存文件 {CACHE_FILE} 损坏或不存在，将创建新的缓存。")
        return {}

    def save_embedding_cache(self):
        """保存嵌入缓存"""
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    # --- SQLite helpers ---
    @staticmethod
    def ensure_schema(conn: sqlite3.Connection) -> None:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                chunk_id INTEGER PRIMARY KEY, path TEXT, chunk_index INTEGER, content TEXT
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_path ON metadata(path);")
        conn.commit()

    @staticmethod
    def bulk_insert_metadata(conn: sqlite3.Connection, records: List[Tuple]):
        conn.executemany("INSERT OR REPLACE INTO metadata (chunk_id, path, chunk_index, content) VALUES (?, ?, ?, ?)", records)
        conn.commit()

    @staticmethod
    def get_metadata_by_chunk_id(conn: sqlite3.Connection, chunk_id: int) -> Dict[str, Any]:
        cursor = conn.cursor()
        cursor.execute("SELECT path, chunk_index, content FROM metadata WHERE chunk_id = ?", (chunk_id,))
        result = cursor.fetchone()
        if result:
            return {"path": result[0], "chunk_index": result[1], "content": result[2]}
        return {}

    # --- FAISS helpers ---
    @staticmethod
    def create_initial_faiss_index(dimension: int):
        return faiss.IndexFlatIP(dimension)

    @staticmethod
    def add_vectors_to_faiss_index(index: faiss.Index, vectors: np.ndarray):
        # 在添加到索引前，对所有文档向量进行 L2 归一化
        if vectors.size > 0: # 检查向量数组是否为空
            faiss.normalize_L2(vectors)
        index.add(vectors)

    @staticmethod
    def save_faiss_index_and_metadata(index: faiss.Index, metadata_list: List[dict]):
        faiss.write_index(index, FAISS_INDEX_FILE)
        simplified_metadata = [{"chunk_id": m["chunk_id"]} for m in metadata_list]
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(simplified_metadata, f, ensure_ascii=False, indent=2)
        print(f"[构建] FAISS 索引和元数据文件已保存。")

    @staticmethod
    def load_faiss_index_and_metadata():
        index = faiss.read_index(FAISS_INDEX_FILE)
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        return index, metadata

    def search_similar_with_faiss(self, query_vec: List[float], top_k: int, score_threshold: float) -> List[Tuple[float, int]]:
        try:
            index, metadata_map = self.load_faiss_index_and_metadata()
        except Exception as e:
            print(f"加载 FAISS 索引失败: {e}")
            return []

        query_array = np.array([query_vec], dtype=np.float32)
        faiss.normalize_L2(query_array)
        scores, indices = index.search(query_array, top_k)

        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            score = scores[0][i]
            # FAISS IndexFlatIP 在向量归一化后，其得分就是余弦相似度
            if score >= score_threshold:
                chunk_id = metadata_map[idx]["chunk_id"]
                results.append((float(score), chunk_id))
        return results

        # --- Build Phase ---
    async def build_knowledge_base_async(self, source_dir: Path, chunk_size: int = CHUNK_SIZE_DEFAULT, overlap: int = CHUNK_OVERLAP_DEFAULT):
        print(f"[构建] 开始处理来自 '{source_dir}' 的文档...")

        conn = sqlite3.connect(DB_PATH)
        self.ensure_schema(conn)

        initial_index = None
        global_chunk_id_counter = 0
        all_metadata_for_faiss = []

        async with aiohttp.ClientSession() as session:
            for batch_idx, batch_docs in enumerate(self._load_document_batches(source_dir, BATCH_SIZE_DOCS)):
                print(f"\n--- 开始处理第 {batch_idx + 1} 个批次 ---")
                
                # --- 将整个批次的处理逻辑（从await到存储）都包裹在try中 ---
                try:
                    batch_vectors, batch_metadata = await self._process_document_batch_async(
                        batch_docs, chunk_size, overlap, global_chunk_id_counter, session
                    )

                    # 检查是否返回了空数组，避免后续操作出错
                    if batch_vectors.size == 0 or len(batch_metadata) == 0:
                        print("本批次无有效向量或元数据，跳过存储。")
                        continue

                    if initial_index is None:
                        # 确保 batch_vectors 是有效的，再获取其维度
                        dimension = batch_vectors.shape[1]
                        initial_index = self.create_initial_faiss_index(dimension)

                    self.add_vectors_to_faiss_index(initial_index, batch_vectors)
                    all_metadata_for_faiss.extend(batch_metadata)
                    
                    records = [(m["chunk_id"], m["path"], m["chunk_index"], m["content"]) for m in batch_metadata]
                    self.bulk_insert_metadata(conn, records)

                    # 更新全局计数器
                    if batch_metadata:
                        global_chunk_id_counter = max(global_chunk_id_counter, batch_metadata[-1]['chunk_id'] + 1)

                    del batch_vectors, batch_metadata, records
                    print(f"--- 第 {batch_idx + 1} 个批次处理完成并已清理内存 ---\n")

                except Exception as e:
                    print(f"[错误] 处理第 {batch_idx + 1} 个批次时失败: {e}")
                    print("跳过此批次，继续处理下一个批次...")
                    continue # 跳过本次循环，处理下一个批次

        conn.close()
        self.save_faiss_index_and_metadata(initial_index, all_metadata_for_faiss)
        self.save_embedding_cache()
        print(f"\n[构建] 知识库构建完成。共索引了 {len(all_metadata_for_faiss)} 个文本块。")

    def _load_document_batches(self, source_dir: Path, batch_size: int) -> Iterator[List[object]]:
        all_docs = document_loader.batch_load_documents(str(source_dir))
        for i in range(0, len(all_docs), batch_size):
            yield all_docs[i:i+batch_size]

    async def _process_document_batch_async(
        self,
        batch_docs: List[Any],
        chunk_size: int,
        overlap: int,
        chunk_id_start: int,
        session: aiohttp.ClientSession
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        处理一个文档批次，返回该批次的向量数组和元数据列表。
        """
        print(f"[构建] 开始处理包含 {len(batch_docs)} 个文档的新批次...")
        
        # --- 重要修复：在函数入口处提前初始化所有关键变量 ---
        batch_metadata = []
        batch_embeddings_array = np.array([]).reshape(0, 0)
        final_metadata = []
        batch_embeddings_list = []

        splitter = text_splitter.SmartTextSplitter(
            model_path=LOCAL_EMBEDDING_MODEL_PATH,
            model_name=SEMANTIC_SPLITTER_MODEL_NAME,
            threshold=SEMANTIC_SPLITTER_THRESHOLD,
            base_splitter_params={"chunk_size": chunk_size, "chunk_overlap": overlap, "separators": SEMANTIC_SPLITTER_SEPARATORS},
            load_timeout=15
        )

        batch_texts = []
        current_chunk_id = chunk_id_start

        for doc_obj in batch_docs:
            content = doc_obj.page_content
            source_path = doc_obj.metadata.get("source", "unknown_source")
            chunks = splitter.split_text(content)
            for idx, chunk in enumerate(chunks):
                batch_texts.append(chunk)
                batch_metadata.append({"chunk_id": current_chunk_id, "path": source_path, "chunk_index": idx, "content": chunk})
                current_chunk_id += 1

        if not batch_texts:
            print("[构建] 当前批次没有产生任何文本块。")
            return batch_embeddings_array, batch_metadata # 返回初始值

        texts_to_embed, indices_to_embed, text_hashes_for_ordering = [], [], []
        for i, text in enumerate(batch_texts):
            text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
            text_hashes_for_ordering.append(text_hash)
            if text_hash in self.cache:
                print(f"[缓存命中] 块 #{chunk_id_start + i} 已在缓存中。")
            else:
                print(f"[待计算] 块 #{chunk_id_start + i} 需要向量化。")
                texts_to_embed.append(text)
                indices_to_embed.append(i)

        if texts_to_embed:
            print(f"[构建] 本批次发现 {len(texts_to_embed)} 个新块需要向量化...")
            for i in range(0, len(texts_to_embed), OLLAMA_EMBED_BATCH_SIZE_INTERNAL):
                sub_batch_texts = texts_to_embed[i:i+OLLAMA_EMBED_BATCH_SIZE_INTERNAL]
                sub_batch_indices = indices_to_embed[i:i+OLLAMA_EMBED_BATCH_SIZE_INTERNAL]

                sub_batch_embeddings = await asyncio.gather(*[self._async_fetch_embedding(session, text) for text in sub_batch_texts], return_exceptions=True)

                for j, (orig_idx, emb_result) in enumerate(zip(sub_batch_indices, sub_batch_embeddings)):
                    if isinstance(emb_result, Exception):
                        print(f"[错误] 向量化块 #{chunk_id_start + orig_idx} 时失败: {emb_result}")
                        continue
                    original_text = batch_texts[orig_idx]
                    text_hash = hashlib.sha256(original_text.encode('utf-8')).hexdigest()
                    self.cache[text_hash] = emb_result

        # --- 从缓存中提取向量，并处理缺失项 ---
        # batch_embeddings_list, final_metadata 已在此前初始化
        
        for md, hash_val in zip(batch_metadata, text_hashes_for_ordering):
            if hash_val in self.cache:
                batch_embeddings_list.append(self.cache[hash_val])
                final_metadata.append(md)
            else:
                print(f"[警告] 缓存不一致：找不到文本块 '{md['content'][:30]}...' (ID: {md['chunk_id']}) 的向量。已跳过。")

        # 如果经过筛选后，没有可用的向量，则返回空
        if not batch_embeddings_list:
            print("[构建] 本批次经过筛选后没有可用的向量。")
            return np.array([]).reshape(0, 0), []

        batch_embeddings_array = np.array(batch_embeddings_list, dtype=np.float32)
        print(f"[构建] 本批次处理完成，生成了 {len(final_metadata)} 个向量。")
        
        return batch_embeddings_array, final_metadata

    # --- Query Phase ---
    def retrieve_contexts(self, question: str, top_k: int = DEFAULT_TOP_K, score_threshold: float = DEFAULT_THRESHOLD) -> List[Dict[str, Any]]:
        query_vec = self._fetch_embedding(question)
        raw_results = self.search_similar_with_faiss(query_vec, top_k=top_k, score_threshold=score_threshold)
        
        conn = sqlite3.connect(DB_PATH)
        formatted_results = []
        for score, chunk_id in raw_results:
            metadata = self.get_metadata_by_chunk_id(conn, chunk_id)
            if metadata:
                formatted_results.append({"score": score, "chunk_id": chunk_id, **metadata})
        conn.close()
        return formatted_results

    def compress_contexts(self, retrieved_results: List[Dict[str, Any]], compressor_model: str = COMPRESSOR_MODEL, temperature: float = OLLAMA_COMPRESSOR_TEMPERATURE_DEFAULT) -> str:
        if not retrieved_results:
            return ""
        
        compression_messages = prompts.get_compress_prompt_template(retrieved_results)
        compressed_context = self._chat_completion(compression_messages, model_name=compressor_model, temperature=temperature)
        return compressed_context

    def prepare_final_prompt(self, question: str, contexts: List[Dict[str, Any]], compressed_context: str) -> List[dict]:
        if not compressed_context.strip():
            context_text = "\n\n".join(f"[source={item['path']}#chunk{item['chunk_index']}] {item['content']}" for item in contexts)
        else:
            context_text = compressed_context
        return prompts.get_rag_prompt_template(context_text, question)

    def answer_question(self, question: str, top_k_retrieve: int = DEFAULT_TOP_K, top_k_compressed: int = DEFAULT_TOP_K_COMPRESSED, score_threshold: float = DEFAULT_THRESHOLD) -> str:
        retrieved_results = self.retrieve_contexts(question, top_k=top_k_retrieve, score_threshold=score_threshold)
        if not retrieved_results:
            return "检索未命中相关片段，请检查知识库或降低阈值。"

        compressed_context = self.compress_contexts(retrieved_results[:top_k_compressed])
        messages = self.prepare_final_prompt(question, retrieved_results[:top_k_compressed], compressed_context)
        return self._chat_completion(messages, model_name=CHAT_MODEL, temperature=OLLAMA_CHAT_TEMPERATURE_DEFAULT)

            # --- Query Phase (Streaming) ---
    def answer_question_stream(self, question: str, top_k_retrieve: int = 5, top_k_compressed: int = 3, score_threshold: float = 0.3):
        """
        流式回答问题，包含详细的中间步骤。
        """
        import numpy as np
        import faiss
        import json

        # 1. 加载知识库
        yield f"\n--- 开始查询 ---\n"
        index, metadata_list = self.load_faiss_index_and_metadata()
        if index is None or len(metadata_list) == 0:
            yield "知识库为空或未正确加载。\n"
            return

        # 2. 向量化问题
        yield "--- 步骤 1: 向量化问题 ---\n"
        question_embedding = self._fetch_embedding(question)
        question_vector = np.array([question_embedding], dtype='float32')
        # 归一化查询向量，确保与索引中的向量计算的是余弦相似度
        faiss.normalize_L2(question_vector)
        yield f"问题向量化完成，维度: {question_vector.shape[1]}\n"

        # 3. 搜索相似块
        yield f"\n--- 步骤 2: 在知识库中搜索 Top-{top_k_retrieve} 个相似片段 ---\n"
        # FAISS 搜索，返回的距离即为余弦相似度分数（因为向量都已归一化）
        k = min(len(metadata_list), top_k_retrieve)
        similarities, indices = index.search(question_vector, k)

        # 4. 根据阈值过滤并排序
        results_with_scores = []
        conn = sqlite3.connect(DB_PATH)
        for i in range(len(indices[0])):
            idx = indices[0][i]
            # 从 FAISS 得到的 `similarities` 就是最终的余弦相似度分数
            similarity_score = similarities[0][i]

            if idx < len(metadata_list) and similarity_score >= score_threshold:
                chunk_id = metadata_list[idx]["chunk_id"]
                # 从数据库获取完整的元数据
                full_metadata = self.get_metadata_by_chunk_id(conn, chunk_id)
                if full_metadata:
                    results_with_scores.append((similarity_score, full_metadata['path'], full_metadata['chunk_index'], full_metadata['content']))
        
        conn.close()
        # 按相似度分数降序排列
        results_with_scores.sort(key=lambda x: x[0], reverse=True)
        relevant_results = results_with_scores

        # --- 打印 Stage 1 结果 ---
        yield f"=== 检索命中 (Stage 1) ===\n"
        for i, (similarity, path, chunk_index, content) in enumerate(relevant_results):
            yield f"[Top{i+1}] 相似度：{similarity:.4f}\n来源：{path}（段落 #{chunk_index}）\n{content[:100]}...\n----------------------------------------\n"

        if not relevant_results:
            yield "未找到相关的文档片段来回答您的问题。\n"
            return

        # 5. 压缩阶段 (Context Compression) - 使用类中现有的方法
        # 将检索结果转换为类中方法可接受的格式
        formatted_retrieved_results = [
            {"score": score, "chunk_id": i, "path": path, "chunk_index": chunk_idx, "content": content}
            for i, (score, path, chunk_idx, content) in enumerate(relevant_results)
        ]
        
        yield "\n--- 开始压缩上下文 ---\n"
        yield f"输入片段数: {len(formatted_retrieved_results)}\n\n"
        
        # 调用现有的压缩方法
        compressed_content = self.compress_contexts(formatted_retrieved_results[:top_k_compressed])
        yield f"调用压缩模型: {COMPRESSOR_MODEL}\n"
        yield f"压缩后的内容:\n{compressed_content}\n--- 压缩完成 ---\n\n"

        # 6. 准备最终提示并生成答案
        final_messages = self.prepare_final_prompt(question, formatted_retrieved_results[:top_k_compressed], compressed_content)
        yield f"=== 调用回答生成模型 ===\n"
        yield f"调用模型: {CHAT_MODEL}\n"
        # 将消息列表转换为字符串以便显示
        message_str = "\n".join([f"{msg['role']}: {msg['content'][:100]}..." for msg in final_messages])
        yield f"Prompt:\n{message_str}\n"

        # 流式生成最终答案
        yield "\n=== 最终回答 ===\n"
        for token in self._chat_completion_stream(final_messages, model_name=CHAT_MODEL, temperature=OLLAMA_CHAT_TEMPERATURE_DEFAULT):
            yield token