"""
基于硅基流动线上资源的最小 RAG 示例 (LangChain 版)：
1. 读取本地 txt 文件，切片并调用嵌入接口存入 sqlite。
2. 接收用户问题，生成嵌入，在本地向量库里检索相似片段。
3. 将片段连同问题一并提交给硅基流动大模型，得到回答。

准备工作：
1. 安装依赖：pip install langchain-openai requests
2. 修改 API_KEY 常量。
"""

from __future__ import annotations
import json
import math
import os
import sqlite3
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

# 导入 LangChain 组件
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import requests

API_BASE = "https://api.siliconflow.cn/v1"
CHAT_MODEL = "qwen3.5-flash" # 对话大模型
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5" # 知识库文档索引模型
API_KEY = os.environ["API_KEY"]

DB_PATH = Path("knowledge_base.db") # 知识库数据库文件
DOC_DIR = Path("doc") # 知识库文档目录
DEFAULT_TOP_K = 3 # 默认召回片段数量
DEFAULT_THRESHOLD = 0.3 # 默认相似度阈值


# ----------------------------- LangChain clients initialization ----------------------------- #
# 初始化 LangChain 客户端
def _get_embedding_client():
    if not API_KEY or API_KEY == "your_api_key":
        raise RuntimeError("请先配置有效的 API_KEY")
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=API_KEY,
        openai_api_base=API_BASE,
        # LangChain Embedding 默认可能检查上下文长度，硅基流动 API 不需要，禁用它
        check_embedding_ctx_length=False 
    )

def _get_chat_client(stream: bool = False, temperature: float = 0.7):
    if not API_KEY or API_KEY == "your_api_key":
        raise RuntimeError("请先配置有效的 API_KEY")
    return ChatOpenAI(
        model=CHAT_MODEL,
        openai_api_key=API_KEY,
        openai_api_base=API_BASE,
        temperature=temperature,
        streaming=stream
    )


# ----------------------------- HTTP helpers ----------------------------- #
def _headers() -> dict:
    if not API_KEY or API_KEY == "your_api_key":
        raise RuntimeError("请先配置有效的 API_KEY（可使用环境变量或修改常量）")
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }


def fetch_embedding(text: str) -> List[float]:
    """使用 LangChain 获取嵌入向量"""
    client = _get_embedding_client()
    # LangChain 的 embed_query 方法返回 List[float]
    return client.embed_query(text)


def chat_completion(messages: List[dict], temperature: float = 0.7) -> str:
    """使用 LangChain 获取非流式回复"""
    client = _get_chat_client(stream=False, temperature=temperature)
    
    # 将原始的字典消息列表转换为 LangChain 消息对象
    lc_messages = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
        # 可能还有其他角色，这里只处理常见的三种
        
    response = client.invoke(lc_messages)
    return response.content


def chat_completion_stream(messages: List[dict], temperature: float = 0.7) -> Iterable[str]:
    """使用 LangChain 获取流式回复"""
    client = _get_chat_client(stream=True, temperature=temperature)
    
    # 将原始的字典消息列表转换为 LangChain 消息对象
    lc_messages = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))

    # 使用 LangChain 的 stream 方法
    for chunk in client.stream(lc_messages):
        # chunk 是一个 MessageChunk 对象，其 content 属性包含文本块
        if chunk.content:
            yield chunk.content


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


# ----------------------------- Text utilities ----------------------------- #
def iter_txt_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.txt")):
        if path.is_file():
            yield path


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
        if start < 0:
            break
    return chunks


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
    txt_files = list(iter_txt_files(source_dir))
    if not txt_files:
        raise RuntimeError(f"目录 {source_dir} 下未找到 txt 文件")

    conn = sqlite3.connect(DB_PATH)
    ensure_schema(conn)

    for file_path in txt_files:
        text = file_path.read_text(encoding="utf-8")
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for idx, chunk in enumerate(chunks):
            embedding = fetch_embedding(chunk)
            upsert_chunk(conn, path=str(file_path), chunk_index=idx, content=chunk, embedding=embedding)
            conn.commit()
            print(f"[索引] {file_path.name} chunk#{idx} 已写入")

    conn.close()
    print("知识库构建完成。")


def retrieve_contexts(
    question: str,
    top_k: int = 3,
    score_threshold: float = 0.3,
) -> List[Tuple[float, str, str, int]]:
    conn = sqlite3.connect(DB_PATH)
    ensure_schema(conn)

    chunks = load_all_chunks(conn)
    conn.close()
    if not chunks:
        raise RuntimeError("知识库为空，请先执行 build 命令")

    query_vec = fetch_embedding(question)
    scored = [
        (cosine_similarity(query_vec, embedding), content, path, chunk_idx)
        for _, path, chunk_idx, content, embedding in chunks
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    filtered = [item for item in scored if item[0] >= score_threshold]
    if not filtered:
        return []
    return filtered[:top_k]


def _prepare_prompt(
    question: str,
    top_k: int = 3,
    score_threshold: float = 0.3,
) -> Tuple[List[dict], str]:
    contexts = retrieve_contexts(question, top_k=top_k, score_threshold=score_threshold)
    if not contexts:
        return [], "检索未命中相关片段，请检查知识库或降低阈值。"

    print("\n=== 检索命中 ===")
    for idx, (score, content, path, chunk_idx) in enumerate(contexts, start=1):
        print(f"[Top{idx}] 相似度：{score:.4f}")
        print(f"来源：{path}（段落 #{chunk_idx}）")
        print(content)
        print("-" * 40)

    context_text = "\n\n".join(
        f"[source={path}#chunk{chunk_idx}] {content}"
        for _, content, path, chunk_idx in contexts
    )

    messages = [
        {
            "role": "system",
            "content": (
                "你是一位知识库问答助手。回答必须严格基于提供的片段，"
                "并在涉及事实的句子末尾使用 [source=文件路径#chunk编号] 进行引用。"
                "若片段不足以回答，请明确说明。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"已知文档片段：\n{context_text}\n\n"
                f"问题：{question}\n"
                "请基于片段回答，确保在相关句子后附上对应的 source 引用。"
            ),
        },
    ]
    return messages, ""


def answer_question(question: str, top_k: int = 3, score_threshold: float = 0.3) -> str:
    messages, error = _prepare_prompt(question, top_k=top_k, score_threshold=score_threshold)
    if not messages:
        return error
    print("\n=== 调用大模型 ===")
    print(json.dumps(messages, ensure_ascii=False, indent=2))
    return chat_completion(messages)


def answer_question_stream(
    question: str,
    top_k: int = 3,
    score_threshold: float = 0.3,
    temperature: float = 0.7,
) -> Iterable[str]:
    messages, error = _prepare_prompt(question, top_k=top_k, score_threshold=score_threshold)
    if not messages:
        yield error
        return
    print("\n=== 调用大模型 ===")
    print(json.dumps(messages, ensure_ascii=False, indent=2))
    yield from chat_completion_stream(messages, temperature=temperature)


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
            top_k = input(f"召回片段数量（默认 {DEFAULT_TOP_K}）：").strip()
            threshold = input(f"相似度阈值（默认 {DEFAULT_THRESHOLD}）：").strip()
            print("\n===== 回答开始 =====")
            for chunk in answer_question_stream(
                question,
                top_k=int(top_k) if top_k else DEFAULT_TOP_K,
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