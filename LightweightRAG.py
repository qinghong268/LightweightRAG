# LightweightRAG.py
"""
RAG 应用主入口(CLI)
负责处理用户输入和输出，调用SimpleRAG模块执行具体逻辑。
"""

from pathlib import Path
import asyncio
import simpleRAG_content
from config import DOC_DIR, CHUNK_SIZE_DEFAULT, CHUNK_OVERLAP_DEFAULT, DEFAULT_TOP_K, DEFAULT_TOP_K_COMPRESSED, DEFAULT_THRESHOLD

def interactive_cli():
    rag_instance = simpleRAG_content.SimpleRAG() # 创建RAG实例

    print("操作简介")
    print("1.知识库初始化（索引docs目录）")
    print("2.开始对话（RAG问答）")
    print("0.退出")

    while True:
        choice = input("\n请选择操作[1/2/0]：").strip()
        if choice == "1":
            source = input(f"文档目录（默认 {DOC_DIR}）：").strip()
            source_dir = Path(source) if source else DOC_DIR
            chunk_size_input = input(f"切片长度（默认{CHUNK_SIZE_DEFAULT}）：").strip()
            overlap_input = input(f"切片重叠长度（默认{CHUNK_OVERLAP_DEFAULT}）：").strip()
            chunk_size = int(chunk_size_input) if chunk_size_input else CHUNK_SIZE_DEFAULT
            overlap = int(overlap_input) if overlap_input else CHUNK_OVERLAP_DEFAULT
            
            # 调用SimpleRAG实例的构建方法
            asyncio.run(rag_instance.build_knowledge_base_async(source_dir, chunk_size=chunk_size, overlap=overlap))
            
        elif choice == "2":
            question = input("请输入你的问题：").strip()
            if not question:
                print("问题不能为空。")
                continue
            
            top_k_ret_input = input(f"检索片段数量（默认{DEFAULT_TOP_K}）：").strip()
            top_k_comp_input = input(f"压缩后片段数量（默认{DEFAULT_TOP_K_COMPRESSED}）：").strip()
            threshold_input = input(f"相似度阈值（默认{DEFAULT_THRESHOLD}）：").strip()
            top_k_ret = int(top_k_ret_input) if top_k_ret_input else DEFAULT_TOP_K
            top_k_comp = int(top_k_comp_input) if top_k_comp_input else DEFAULT_TOP_K_COMPRESSED
            score_threshold = float(threshold_input) if threshold_input else DEFAULT_THRESHOLD
            
            print("\n回答开始")
            # 调用SimpleRAG实例的流式回答方法
            for chunk in rag_instance.answer_question_stream(
                question,
                top_k_retrieve=top_k_ret,
                top_k_compressed=top_k_comp,
                score_threshold=score_threshold,
            ):
                print(chunk, end="", flush=True)
            print("\n回答结束")
            
        elif choice == "0":
            print("结束")
            break
        else:
            print("无效选项，请重新选择。")

if __name__ == "__main__":
    interactive_cli()