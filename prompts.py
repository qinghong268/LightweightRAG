from typing import Any, Dict, List


SYSTEM_PROMPT = """
你是一个基于检索增强生成（RAG）的中文问答助手。
请始终以检索到的知识片段作为事实依据。
当你陈述由知识片段支持的事实时，必须在句子末尾附上 [source=PATH#chunkN] 格式的引用。
直接回答用户问题，不要写“根据检索到的知识片段……”“以下是基于这些信息的整理和回答……”这类前言。
不要输出 "---" 这类分隔线，不要为了排版加入大量空行或空格。
如果检索片段不足以支持回答，请明确说明，不要猜测。
对话历史只用于理解代词、上下文指代和追问关系，除非相同信息也被检索片段支持，否则不要把对话历史当作事实来源。
如果用户询问的是对话本身，例如“上一个问题是什么”“上一条回答是什么”，可以直接基于对话历史回答，此时引用可以省略。
无论检索到的资料是中文还是英文，最终回答都统一使用中文表达。
如果证据来自英文资料，可以将其内容准确转述为中文，但必须保留原始引用，不要因为资料是英文就忽略它。
""".strip()


QUERY_REWRITE_SYSTEM_PROMPT = """
你负责把用户的最新问题改写成适合检索的独立查询。
只使用最近对话来补全指代、省略和上下文关系。
只返回最终改写后的查询，不要添加解释、项目符号、标签或额外文字。
改写后的查询统一使用中文表达。
如果问题中涉及常见英文术语、产品名、框架名、接口名、模型名或缩写，在有助于检索时可以保留这些关键英文词，但不要把整条查询改写成英文。
不要因为资料可能是英文而刻意回避英文关键词。
""".strip()


COMPRESS_SYSTEM_PROMPT = """
你是一个上下文压缩助手。
请把多个召回片段中的重叠证据整合为简洁摘要，只保留有助于回答问题的信息。
每一句摘要都必须保留 [source=PATH#chunkN] 格式的引用。
无论原始资料是中文还是英文，压缩后的表述统一使用中文。
如果某条证据来自英文资料，请将其准确转述为中文，同时保留对应引用。
不要因为资料是英文就忽略其中的重要信息，也不要把中英内容混杂堆叠在一句里。
优先输出适合直接喂给中文回答模型使用的、表达统一的中文证据摘要。
""".strip()


CONVERSATION_SUMMARY_SYSTEM_PROMPT = """
你负责为后续检索支持总结较早的对话历史。
只保留对后续追问有帮助的稳定事实、实体、目标和已解析的指代关系。
不要输出思维链。
不要添加来源引用。
总结保持简短、客观，并统一使用中文。
""".strip()


CONVERSATION_META_SYSTEM_PROMPT = """
你负责回答关于当前对话本身的问题。
只能使用提供的对话历史，不要虚构对话中不存在的细节。
如果所需信息在历史中不存在，请明确说明。
回答统一使用中文。
""".strip()


def get_query_rewrite_prompt_template(question: str, history_text: str = "") -> List[dict]:
    history_section = history_text.strip() if history_text and history_text.strip() else "None"
    return [
        {
            "role": "system",
            "content": QUERY_REWRITE_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f"Recent conversation:\n{history_section}\n\n"
                f"Latest user question:\n{question}\n\n"
                "Rewrite the latest user question into a standalone retrieval query."
            ),
        },
    ]


def get_rag_prompt_template(
    context_text: str,
    question: str,
    history_text: str = "",
) -> List[dict]:
    history_section = history_text.strip() if history_text and history_text.strip() else "None"
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f"Recent conversation:\n{history_section}\n\n"
                f"Retrieved knowledge snippets:\n{context_text}\n\n"
                f"Current user question:\n{question}\n\n"
                "Answer the current user question using the retrieved snippets and include citations."
            ),
        },
    ]


def get_compress_prompt_template(retrieved_results: List[Dict[str, Any]]) -> List[dict]:
    input_text = ""
    for item in retrieved_results:
        input_text += f"[Source: {item['path']}#chunk{item['chunk_index']}] {item['content']}\n\n"

    return [
        {
            "role": "system",
            "content": COMPRESS_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                "Compress the following retrieved snippets into a concise, citation-preserving summary:\n\n"
                f"{input_text}"
            ),
        },
    ]


def get_conversation_summary_prompt_template(history_text: str) -> List[dict]:
    return [
        {
            "role": "system",
            "content": CONVERSATION_SUMMARY_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                "Summarize the following earlier conversation for future retrieval and follow-up understanding:\n\n"
                f"{history_text}"
            ),
        },
    ]


def get_conversation_meta_prompt_template(question: str, history_text: str) -> List[dict]:
    return [
        {
            "role": "system",
            "content": CONVERSATION_META_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f"Conversation history:\n{history_text}\n\n"
                f"Current question about the conversation:\n{question}\n\n"
                "Answer using only the conversation history."
            ),
        },
    ]
