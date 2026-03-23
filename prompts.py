# prompts.py
from typing import List, Dict, Any

# 提示模板

SYSTEM_PROMPT = """
你是一位知识库问答助手。回答必须严格基于提供的片段，
并在涉及事实的句子末尾使用[source=文件路径#chunk编号]进行引用。
若片段不足以回答，请明确说明。
""".strip()

def get_rag_prompt_template(context_text: str, question: str) -> List[dict]:
    """
    生成RAG问答所需的prompt消息列表。
    
    Args:
        context_text:检索到的上下文文本。
        question:用户的问题。

    Returns:
        包含system和user消息的字典列表。
    """
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f"已知文档片段：\n{context_text}\n\n"
                f"问题：{question}\n"
                "请基于片段回答，确保在相关句子后附上对应的source引用。"
            ),
        },
    ]
    return messages

# 用于压缩的提示

COMPRESS_SYSTEM_PROMPT = """
你是一个专业的信息摘要和整合专家。你的任务是分析提供的多个文本片段，
去除冗余信息，将相关主题的内容合并，并生成1-3个高质量、简洁、连贯的摘要。
在每个摘要的末尾，必须标注出其来源的文件名和chunk编号，格式为[source=文件名#chunk编号]。
""".strip()

def get_compress_prompt_template(retrieved_results: List[Dict[str, Any]]) -> List[dict]:
    """
    生成用于压缩上下文的prompt消息列表。

    Args:
        retrieved_results: 从向量库检索到的结果列表，每个元素包含 'score', 'content', 'path', 'chunk_index'。

    Returns:
        包含system和user消息的字典列表。
    """
    # 构建输入文本，包含所有检索到的片段及其来源
    input_text = ""
    for item in retrieved_results:
        input_text += f"[Source: {item['path']}#chunk{item['chunk_index']}] {item['content']}\n\n"

    messages = [
        {
            "role": "system",
            "content": COMPRESS_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f"请分析以下文本片段，并按要求生成摘要：\n\n{input_text}"
            ),
        },
    ]
    return messages