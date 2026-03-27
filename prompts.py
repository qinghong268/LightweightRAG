from typing import Any, Dict, List


SYSTEM_PROMPT = """
You are a retrieval-augmented question answering assistant.
Use the retrieved knowledge snippets as the primary source of truth.
When you state facts supported by the snippets, append citations in the format [source=PATH#chunkN].
If the retrieved snippets are insufficient, say so clearly instead of guessing.
Conversation history is only for understanding references such as pronouns or follow-up questions.
Do not treat conversation history as a factual source unless the same information is supported by the retrieved snippets.
""".strip()


QUERY_REWRITE_SYSTEM_PROMPT = """
You rewrite a user's latest question into a standalone retrieval query.
Use the recent conversation only to resolve references and ellipsis.
Return only the final rewritten question.
Do not add explanations, bullets, labels, or extra text.
""".strip()


COMPRESS_SYSTEM_PROMPT = """
You are a context compression assistant.
Merge overlapping evidence from multiple retrieved snippets into a concise summary.
Keep only information that helps answer the question.
Every summary sentence must preserve source citations in the format [source=PATH#chunkN].
""".strip()


CONVERSATION_SUMMARY_SYSTEM_PROMPT = """
You summarize an earlier conversation for retrieval support.
Keep only the durable facts, entities, goals, and resolved references that help future follow-up questions.
Do not include chain-of-thought.
Do not include source citations.
Keep the summary short and factual.
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
