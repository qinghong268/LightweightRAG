"""
基于本地 Ollama 的 RAG 示例：
1. 读取本地 txt/pdf/docx 文件，切片并调用本地嵌入模型存入 sqlite。
2. 接收用户问题，生成嵌入，在本地 FAISS 向量库里检索相似片段。
3. 使用专门的模型对片段进行优化（压缩、融合、摘要）。
4. 将优化后的上下文连同问题一并提交给本地 Ollama 大模型，得到回答。

准备工作：
1. 安装依赖：pip install langchain-openai requests
2. 启动 Ollama 服务：ollama serve，拉取所需模型：ollama pull bge-m3:latest && ollama pull deepseek-r1:8b && ollama pull qwen:7b
"""

from __future__ import annotations
import json
import math
import os
import sqlite3
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Dict, Any
import faiss  # 新增：用于高效的向量搜索
import numpy as np  # 新增：用于数值计算

import requests
import document_loader
import text_splitter
import prompts

# --- Ollama 配置 ---
OLLAMA_HOST = "http://localhost:11434"
CHAT_MODEL = "deepseek-r1:8b"  # 用于最终回答生成的本地模型
COMPRESSOR_MODEL = "qwen:7b"  # 用于上下文压缩和优化的本地模型
EMBEDDING_MODEL = "bge-m3:latest" # 用于知识库文档索引的本地模型

DB_PATH = Path("knowledge_base.db") # 知识库数据库文件
DOC_DIR = Path("docs") # 知识库文档目录
DEFAULT_TOP_K = 5 # 检索 Top-K 个片段，为压缩提供更多素材
DEFAULT_TOP_K_COMPRESSED = 3 # 压缩后最终使用的片段数
DEFAULT_THRESHOLD = 0.3 # 默认相似度阈值

# --- FAISS 配置 ---
FAISS_INDEX_FILE = "faiss_index.bin"  # 存储 FAISS 索引的文件
METADATA_FILE = "metadata.json"       # 存储向量对应的元数据 (ID, path, chunk_index, content)


# ----------------------------- HTTP helpers for Ollama ----------------------------- #
def _ollama_headers() -> dict:
    return {"Content-Type": "application/json"}

def fetch_embedding(text: str) -> List[float]:
    """使用本地 Ollama 的 bge-m3:latest 模型获取嵌入向量"""
    url = f"{OLLAMA_HOST}/api/embeddings"
    data = {
        "model": EMBEDDING_MODEL,
        "prompt": text
    }
    response = requests.post(url, headers=_ollama_headers(), json=data)
    response.raise_for_status()
    result = response.json()
    return result["embedding"]

def _ollama_chat_completion(messages: List[dict], model_name: str = CHAT_MODEL, temperature: float = 0.7, stream: bool = False) -> str | Iterable[str]:
    """通过 Ollama API 进行聊天对话"""
    url = f"{OLLAMA_HOST}/api/chat"
    data = {
        "model": model_name,
        "messages": messages,
        "options": {"temperature": temperature},
        "stream": stream
    }
    
    if stream:
        # 流式模式
        response = requests.post(url, headers=_ollama_headers(), json=data, stream=True)
        response.raise_for_status()
        def generate():
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode('utf-8'))
                        if chunk.get('done'): # Ollama 流式响应的结束标志
                            break
                        if 'message' in chunk and 'content' in chunk['message']:
                            yield chunk['message']['content']
                    except json.JSONDecodeError:
                        continue
        return generate() # 返回生成器
        
    else:
        # 非流式模式：一次性获取完整响应
        # 注意：对于非流式请求，我们不能使用 stream=True，否则 response.json() 可能会失败
        response = requests.post(url, headers=_ollama_headers(), json=data, stream=False)
        response.raise_for_status()
        full_response_data = response.json()
        # Ollama /api/chat 的非流式响应结构是 {"model": "...", "created_at": "...", "message": {...}, "done": true, ...}
        return full_response_data.get('message', {}).get('content', '') # 返回字符串

def chat_completion(messages: List[dict], model_name: str, temperature: float = 0.7) -> str: # <--- 新增 model_name 参数
    """使用 Ollama 获取非流式回复"""
    return _ollama_chat_completion(messages, model_name=model_name, temperature=temperature, stream=False)

def chat_completion_stream(messages: List[dict], model_name: str, temperature: float = 0.7) -> Iterable[str]: # <--- 新增 model_name 参数
    """使用 Ollama 获取流式回复"""
    return _ollama_chat_completion(messages, model_name=model_name, temperature=temperature, stream=True)


# ----------------------------- SQLite helpers ----------------------------- #
def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
        """
    )
    conn.commit()


def upsert_chunk(
    conn: sqlite3.Connection,
    *,
    path: str,
    chunk_index: int,
    content: str,
    embedding: Sequence[float],
) -> None:
    conn.execute(
        """
        INSERT INTO documents (path, chunk_index, content, embedding)
        VALUES (?, ?, ?, ?)
        """,
        (path, chunk_index, content, json.dumps(list(embedding))),
    )


def load_all_chunks(conn: sqlite3.Connection) -> List[Tuple[int, str, int, str, List[float]]]:
    rows = conn.execute("SELECT id, path, chunk_index, content, embedding FROM documents").fetchall()
    return [
        (row[0], row[1], row[2], row[3], json.loads(row[4]))
        for row in rows
    ]


# ----------------------------- FAISS helpers ----------------------------- #
def save_faiss_index_and_metadata(vectors: np.ndarray, metadata_list: List[dict]):
    """将向量索引和元数据保存到磁盘"""
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)  # 使用内积 (Inner Product)，对于归一化向量，等价于余弦相似度
    
    # 注意：BGE 模型的向量通常需要进行 L2 归一化以使其等效于余弦相似度
    faiss.normalize_L2(vectors)
    index.add(vectors)
    
    faiss.write_index(index, FAISS_INDEX_FILE)
    
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)

def load_faiss_index_and_metadata():
    """从磁盘加载 FAISS 索引和元数据"""
    index = faiss.read_index(FAISS_INDEX_FILE)
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    return index, metadata

def search_similar_with_faiss(query_vec: List[float], top_k: int = 3, score_threshold: float = 0.3):
    """
    使用 FAISS 进行向量相似度搜索
    """
    try:
        index, metadata = load_faiss_index_and_metadata()
    except Exception as e:
        print(f"加载 FAISS 索引失败: {e}。可能需要先构建知识库。")
        return []

    query_array = np.array([query_vec], dtype=np.float32)
    faiss.normalize_L2(query_array)  # 查询向量也需要归一化
    
    scores, indices = index.search(query_array, top_k)
    
    # 注意：FAISS 的 IndexFlatIP 返回的是内积分数，对于归一化向量，这等于余弦相似度
    results = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        score = scores[0][i]
        # 过滤掉低于阈值的结果
        if score >= score_threshold:
            meta = metadata[idx]
            results.append((float(score), meta["content"], meta["path"], meta["chunk_index"]))
    
    return results


# ----------------------------- Text utilities ----------------------------- #
def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if len(vec_a) != len(vec_b):
        raise ValueError("向量维度不一致")
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ----------------------------- RAG pipeline ----------------------------- #
def build_knowledge_base(source_dir: Path, chunk_size: int = 400, overlap: int = 50) -> None:
    # 使用 document_loader 模块批量加载所有支持格式的文档
    print(f"正在从 '{source_dir}' 加载所有支持的文档...")
    all_docs = document_loader.batch_load_documents(str(source_dir))
    
    if not all_docs:
        raise RuntimeError(f"在目录 {source_dir} 下未找到或加载任何支持的文档")

    # --- 初始化语义感知切块器 ---
    print("正在初始化语义感知切块器...")
    splitter = text_splitter.SmartTextSplitter(
        model_name="BAAI/bge-large-zh-v1.5", # 嵌入模型
        threshold=0.8,                       # 语义相似度阈值
        base_splitter_params={
            "chunk_size": chunk_size,       # 初始块大小
            "chunk_overlap": overlap,       # 初始重叠长度
            "separators": ["\n\n", "\n", "。", "？", "！", "；", "?", "!", ";", "...", " ", ""],
        }
    )
    # ----------------------------------

    conn = sqlite3.connect(DB_PATH)
    ensure_schema(conn)

    total_chunks = 0
    all_vectors_for_faiss = []  # 收集所有向量用于 FAISS
    all_metadata_for_faiss = []  # 收集所有元数据用于 FAISS
    # 遍历所有由 document_loader 返回的 Document 对象
    for doc_obj in all_docs:
        content = doc_obj.page_content
        # 从 Document 对象的元数据中获取原始文件路径
        source_path = doc_obj.metadata.get("source", "unknown_source")
        
        # --- 使用 SmartTextSplitter 进行语义感知切块 ---
        chunks = splitter.split_text(content)
        # ----------------------------------------------------

        for idx, chunk in enumerate(chunks):
            embedding = fetch_embedding(chunk)
            vector_for_faiss = np.array(embedding, dtype=np.float32)  # 为 FAISS 准备
            # 将块索引、内容和元数据存入数据库
            upsert_chunk(
                conn, 
                path=source_path,      # 来源文件
                chunk_index=idx,       # 在该文档中的块序号
                content=chunk,         # 切分后的文本块
                embedding=embedding    # 对应的向量
            )
            conn.commit()
            # 为 FAISS 索引准备数据
            all_vectors_for_faiss.append(vector_for_faiss)
            all_metadata_for_faiss.append({
                "id": total_chunks,  # 这里的 ID 可以是 SQLite 中的 ID 或一个顺序 ID
                "path": source_path,
                "chunk_index": idx,
                "content": chunk
            })
            total_chunks += 1
            print(f"[索引] 来自 '{Path(source_path).name}' 的块 #{idx} 已写入")

    conn.close()
    # 构建并保存 FAISS 索引
    if all_vectors_for_faiss:
        vectors_array = np.stack(all_vectors_for_faiss)
        save_faiss_index_and_metadata(vectors_array, all_metadata_for_faiss)
    print(f"知识库构建完成。共索引了 {total_chunks} 个文本块，并更新了 FAISS 索引。")

def retrieve_contexts(
    question: str,
    top_k: int = 5, # <--- 增加 Top-K，为压缩提供更多信息
    score_threshold: float = 0.3,
) -> List[Tuple[float, str, str, int]]:
    # 直接使用 FAISS 进行搜索
    query_vec = fetch_embedding(question)
    raw_results = search_similar_with_faiss(query_vec, top_k=top_k, score_threshold=score_threshold)
    
    # 将结果转换为字典格式，便于传递给压缩模型
    formatted_results = [
        {"score": score, "content": content, "path": path, "chunk_index": chunk_idx}
        for score, content, path, chunk_idx in raw_results
    ]
    
    return formatted_results

def compress_contexts(
    retrieved_results: List[Dict[str, Any]],
    compressor_model: str = COMPRESSOR_MODEL,
    temperature: float = 0.3, # 压缩任务通常需要更确定性的输出
) -> str:
    """
    Stage 2: 使用专门的模型对检索到的上下文进行压缩和优化。
    """
    if not retrieved_results:
        return ""

    print("\n--- 开始压缩上下文 ---")
    print(f"输入片段数: {len(retrieved_results)}")
    
    # 1. 生成用于压缩的 Prompt
    compression_messages = prompts.get_compress_prompt_template(retrieved_results)
    
    print(f"\n调用压缩模型: {compressor_model}")
    print(json.dumps(compression_messages, ensure_ascii=False, indent=2))

    # 2. 调用压缩模型
    compressed_context = chat_completion(compression_messages, model_name=compressor_model, temperature=temperature)
    
    print(f"\n压缩后的内容:\n{compressed_context}")
    print("--- 压缩完成 ---\n")
    
    return compressed_context


def _prepare_prompt(
    question: str,
    top_k_retrieve: int = 5, # <--- 检索更多片段
    top_k_compressed: int = 3, # <--- 最终使用压缩后的片段数
    score_threshold: float = 0.3,
) -> Tuple[List[dict], str]:
    # Stage 1: Retrieve
    retrieved_results = retrieve_contexts(question, top_k=top_k_retrieve, score_threshold=score_threshold)
    if not retrieved_results:
        return [], "检索未命中相关片段，请检查知识库或降低阈值。"

    # 打印检索到的原始内容
    print("\n=== 检索命中 (Stage 1) ===")
    for idx, item in enumerate(retrieved_results, start=1):
        print(f"[Top{idx}] 相似度：{item['score']:.4f}")
        print(f"来源：{item['path']}（段落 #{item['chunk_index']}）")
        print(item['content'])
        print("-" * 40)

    # Stage 2: Compress
    compressed_context = compress_contexts(retrieved_results[:top_k_compressed]) # 只压缩前 top_k_compressed 个

    # 如果压缩失败或返回为空，则回退到原始上下文
    if not compressed_context.strip():
        print("警告: 压缩模型未返回有效内容，将回退到原始检索片段。")
        context_text = "\n\n".join(
            f"[source={item['path']}#chunk{item['chunk_index']}] {item['content']}"
            for item in retrieved_results[:top_k_compressed]
        )
    else:
        context_text = compressed_context

    # Stage 3: Generate (using the final optimized context)
    messages = prompts.get_rag_prompt_template(context_text, question)
    return messages, ""


def answer_question(question: str, top_k_retrieve: int = 5, top_k_compressed: int = 3, score_threshold: float = 0.3) -> str:
    messages, error = _prepare_prompt(question, top_k_retrieve=top_k_retrieve, top_k_compressed=top_k_compressed, score_threshold=score_threshold)
    if not messages:
        return error
    print("\n=== 调用回答生成模型 ===")
    print(f"调用模型: {CHAT_MODEL}")
    print(json.dumps(messages, ensure_ascii=False, indent=2))
    return chat_completion(messages, model_name=CHAT_MODEL)


def answer_question_stream(
    question: str,
    top_k_retrieve: int = 5,
    top_k_compressed: int = 3,
    score_threshold: float = 0.3,
    temperature: float = 0.7,
) -> Iterable[str]:
    messages, error = _prepare_prompt(question, top_k_retrieve=top_k_retrieve, top_k_compressed=top_k_compressed, score_threshold=score_threshold)
    if not messages:
        yield error
        return
    print("\n=== 调用回答生成模型 ===")
    print(f"调用模型: {CHAT_MODEL}")
    print(json.dumps(messages, ensure_ascii=False, indent=2))
    yield from chat_completion_stream(messages, model_name=CHAT_MODEL, temperature=temperature)


# ----------------------------- CLI entry ----------------------------- #
def interactive_cli() -> None:
    print("=== 硅基流动 RAG 命令行 ===")
    print("1. 知识库初始化（索引 doc 目录）")
    print("2. 开始对话（RAG 问答）")
    print("0. 退出")

    while True:
        choice = input("\n请选择操作 [1/2/0]：").strip()
        if choice == "1":
            source = input(f"文档目录（默认 {DOC_DIR}）：").strip()
            source_dir = Path(source) if source else DOC_DIR
            chunk_size = input("切片长度（默认 400）：").strip()
            overlap = input("切片重叠长度（默认 50）：").strip()
            build_knowledge_base(
                source_dir,
                chunk_size=int(chunk_size) if chunk_size else 400,
                overlap=int(overlap) if overlap else 50,
            )
        elif choice == "2":
            question = input("请输入你的问题：").strip()
            if not question:
                print("问题不能为空。")
                continue
            top_k_ret = input(f"检索片段数量（默认 {DEFAULT_TOP_K}）：").strip()
            top_k_comp = input(f"压缩后片段数量（默认 {DEFAULT_TOP_K_COMPRESSED}）：").strip()
            threshold = input(f"相似度阈值（默认 {DEFAULT_THRESHOLD}）：").strip()
            print("\n===== 回答开始 =====")
            for chunk in answer_question_stream(
                question,
                top_k_retrieve=int(top_k_ret) if top_k_ret else DEFAULT_TOP_K,
                top_k_compressed=int(top_k_comp) if top_k_comp else DEFAULT_TOP_K_COMPRESSED,
                score_threshold=float(threshold) if threshold else DEFAULT_THRESHOLD,
            ):
                print(chunk, end="", flush=True)
            print("\n===== 回答结束 =====")
        elif choice == "0":
            print("Bye~")
            break
        else:
            print("无效选项，请重新选择。")


if __name__ == "__main__":
    interactive_cli()