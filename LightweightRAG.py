from pathlib import Path
import asyncio

import simpleRAG_content
from config import (
    CHUNK_OVERLAP_DEFAULT,
    CHUNK_SIZE_DEFAULT,
    DEFAULT_THRESHOLD,
    DEFAULT_TOP_K,
    DEFAULT_TOP_K_COMPRESSED,
    DOC_DIR,
)


def get_rag_instance(current_instance):
    if current_instance is None:
        return simpleRAG_content.SimpleRAG()
    return current_instance


def interactive_cli():
    rag_instance = None

    print("操作简介")
    print("1. 初始化知识库（索引 docs 目录）")
    print("2. 开始对话（RAG 问答）")
    print("0. 退出")

    while True:
        choice = input("\n请选择操作 [1/2/0]: ").strip()
        if choice == "1":
            source = input(f"文档目录（默认 {DOC_DIR}）: ").strip()
            source_dir = Path(source) if source else DOC_DIR
            chunk_size_input = input(f"切片长度（默认 {CHUNK_SIZE_DEFAULT}）: ").strip()
            overlap_input = input(f"切片重叠长度（默认 {CHUNK_OVERLAP_DEFAULT}）: ").strip()
            chunk_size = int(chunk_size_input) if chunk_size_input else CHUNK_SIZE_DEFAULT
            overlap = int(overlap_input) if overlap_input else CHUNK_OVERLAP_DEFAULT

            rag_instance = get_rag_instance(rag_instance)
            asyncio.run(
                rag_instance.build_knowledge_base_async(
                    source_dir,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
            )

        elif choice == "2":
            question = input("请输入你的问题: ").strip()
            if not question:
                print("问题不能为空。")
                continue

            top_k_ret_input = input(f"检索片段数量（默认 {DEFAULT_TOP_K}）: ").strip()
            top_k_comp_input = input(
                f"压缩后片段数量（默认 {DEFAULT_TOP_K_COMPRESSED}）: "
            ).strip()
            threshold_input = input(f"相似度阈值（默认 {DEFAULT_THRESHOLD}）: ").strip()
            top_k_ret = int(top_k_ret_input) if top_k_ret_input else DEFAULT_TOP_K
            top_k_comp = int(top_k_comp_input) if top_k_comp_input else DEFAULT_TOP_K_COMPRESSED
            score_threshold = float(threshold_input) if threshold_input else DEFAULT_THRESHOLD

            rag_instance = get_rag_instance(rag_instance)
            print("\n回答开始")
            buffered_output = []
            for chunk in rag_instance.answer_question_stream(
                question,
                top_k_retrieve=top_k_ret,
                top_k_compressed=top_k_comp,
                score_threshold=score_threshold,
            ):
                if chunk == simpleRAG_content.SimpleRAG.ANSWER_REPLACE_MARKER:
                    buffered_output = []
                    continue
                if chunk.startswith(simpleRAG_content.SimpleRAG.DEBUG_LOG_PREFIX):
                    continue
                buffered_output.append(chunk)
            print("".join(buffered_output), end="", flush=True)
            print("\n回答结束")

        elif choice == "0":
            print("结束")
            break
        else:
            print("无效选项，请重新选择。")


if __name__ == "__main__":
    interactive_cli()
