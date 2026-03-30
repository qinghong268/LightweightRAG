import asyncio
import contextlib
import html
import io
import json
import logging
import os
import re
import sqlite3
import sys
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("NO_COLOR", "1")

logging.getLogger().setLevel(logging.WARNING)
for noisy_logger_name in (
    "httpx",
    "httpcore",
    "urllib3",
    "gradio",
    "gradio.analytics",
    "gradio.networking",
    "faiss",
    "faiss.loader",
):
    logging.getLogger(noisy_logger_name).setLevel(logging.WARNING)

try:
    from requests.exceptions import RequestsDependencyWarning

    warnings.filterwarnings("ignore", category=RequestsDependencyWarning)
    warnings.simplefilter("ignore", RequestsDependencyWarning)
    warnings.filterwarnings(
        "ignore",
        message=r".*doesn't match a supported version.*",
        module=r"requests\..*",
    )
except Exception:
    pass

import gradio as gr

try:
    import simpleRAG_content
    from config import (
        CACHE_FILE,
        CHAT_MODEL,
        CHUNK_OVERLAP_DEFAULT,
        CHUNK_SIZE_DEFAULT,
        COMPRESSOR_MODEL,
        CONVERSATION_STATE_FILE,
        DB_PATH,
        DEFAULT_THRESHOLD,
        DEFAULT_TOP_K,
        DEFAULT_TOP_K_COMPRESSED,
        DOC_DIR,
        EMBEDDING_MODEL,
        FAISS_INDEX_FILE,
        METADATA_FILE,
        RERANK_MODEL,
    )
    from doc_converter import preprocess_doc_files_for_build
    from simpleRAG_included.conversation_store import ConversationStore
    from simpleRAG_included.rag_helpers import RAGHelpers
except ImportError as exc:
    print(f"Import failed: {exc}")
    sys.exit(1)


def clean_log_text(text):
    if not text:
        return ""
    text = re.sub(r"\n\s*\n", "\n", text)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n".join(lines)


def is_process_log(text):
    stripped = text.strip()
    prefixes = [
        "Start retrieval",
        "开始检索流程",
        "Step 0:",
        "步骤 0：",
        "Step 1:",
        "步骤 1：",
        "Step 2:",
        "步骤 2：",
        "Step 3:",
        "步骤 3：",
        "Original question:",
        "原始问题：",
        "Rewritten query:",
        "改写后的查询：",
        "Vectorized query dimension:",
        "查询向量维度：",
        "Reranker status:",
        "重排器状态：",
        "Compression model:",
        "压缩模型：",
        "Compressed context:",
        "压缩后的上下文：",
        "Compression complete",
        "上下文压缩完成",
        "Answer generation model:",
        "回答生成模型：",
        "Prompt:",
        "提示词：",
        "Final answer",
        "最终回答",
        "Diagnostics:",
        "诊断信息：",
        "[Notice]",
        "[提示]",
        "Cache hit:",
        "缓存命中：",
        "Threshold fallback kept only",
        "阈值回退后仅保留",
        "No relevant knowledge snippets were found.",
        "No relevant knowledge snippets were retrieved.",
        "未检索到相关知识片段。",
        "未检索到相关知识片段，请检查知识库或适当降低分数阈值。",
        "Knowledge base is empty or failed to load.",
        "知识库为空或加载失败。",
        "知识库加载失败：",
        "Compressing retrieved context",
        "开始压缩召回上下文",
        "Compressed context failed citation validation;",
        "压缩结果未通过引用校验",
        "Input snippet count:",
        "输入片段数量：",
    ]
    if any(stripped.startswith(prefix) for prefix in prefixes):
        return True
    if re.match(r"^\[(Top\d+|Rank\d+|Initial Rank\d+|候选\d+|重排\d+|初排\d+)\]", stripped):
        return True
    return False


def parse_debug_event(text):
    prefix = simpleRAG_content.SimpleRAG.DEBUG_LOG_PREFIX
    if not isinstance(text, str) or not text.startswith(prefix):
        return None
    raw_payload = text[len(prefix) :]
    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def format_debug_content(content):
    if isinstance(content, str):
        return content.strip()
    return json.dumps(content, ensure_ascii=False, indent=2)


def localize_reranker_status(status):
    status_map = {
        "pending": "待处理",
        "ready": "就绪",
        "not_loaded": "未加载",
        "unavailable": "不可用",
        "skipped_conversation_only": "仅对话历史",
        "unknown": "未知",
    }
    return status_map.get(str(status or "").strip(), str(status or "未知"))


def escape_html(text):
    return html.escape(text).replace("\n", "<br>")


def generate_logs_html(log_list):
    if not log_list:
        return "<div style='color: #888; font-style: italic;'>暂无历史记录。</div>"

    html_content = "<div class='log-container'>"
    for item in log_list:
        safe_label = escape_html(item["label"])
        safe_details = escape_html(item["details"])
        html_content += f"""
        <details style="border: 1px solid #e5e7eb; border-radius: 6px; margin-bottom: 8px; background: #f9fafb;">
            <summary style="padding: 10px; cursor: pointer; font-weight: 600; color: #374151; list-style: none; display: flex; justify-content: space-between; align-items: center;">
                <span>{safe_label}</span>
                <span style="font-size: 0.8em; color: #9ca3af;">查看</span>
            </summary>
            <div style="padding: 10px; border-top: 1px solid #e5e7eb; font-family: monospace; font-size: 0.85em; color: #4b5563; white-space: pre-wrap; background: #fff;">
{safe_details}
            </div>
        </details>
        """
    html_content += "</div>"
    return html_content


def build_results_panel_html(results, title, score_key):
    if not results:
        return (
            f"<div style='padding: 10px; border: 1px dashed #d1d5db; border-radius: 8px; color: #6b7280;'>"
            f"{title}：暂无结果。</div>"
        )

    cards = [f"<div style='font-weight: 700; margin-bottom: 10px;'>{html.escape(title)}</div>"]
    score_label = {
        "score": "相似度",
        "rerank_score": "重排分数",
    }.get(score_key, score_key)
    for index_id, item in enumerate(results, start=1):
        path = html.escape(str(item.get("path", "未知来源")))
        chunk_index = html.escape(str(item.get("chunk_index", "未知分块")))
        score = item.get(score_key)
        if isinstance(score, float):
            score_text = f"{score_label}={score:.4f}"
        else:
            score_text = f"{score_label}=不可用"
        score_text = html.escape(score_text)
        snippet = html.escape(str(item.get("content", "")))
        cards.append(
            f"""
            <details style="border: 1px solid #e5e7eb; border-radius: 8px; margin-bottom: 8px; background: #ffffff;">
                <summary style="padding: 10px; cursor: pointer; font-weight: 600;">
                    #{index_id} {path}#chunk{chunk_index} | {score_text}
                </summary>
                <div style="padding: 10px; border-top: 1px solid #e5e7eb; white-space: pre-wrap; font-family: monospace; font-size: 0.85em;">
{snippet}
                </div>
            </details>
            """
        )
    return "".join(cards)


def generate_retrieval_html(retrieved_results, reranked_results):
    return (
        "<div style='display: grid; grid-template-columns: 1fr; gap: 12px;'>"
        f"{build_results_panel_html(retrieved_results, '召回结果', 'score')}"
        f"{build_results_panel_html(reranked_results, '重排结果', 'rerank_score')}"
        "</div>"
    )


def generate_citation_preview_html(answer_text, source_results):
    source_results = source_results or []
    matches = re.findall(r"\[source=([^\]#]+)#chunk(\d+)\]", answer_text or "", flags=re.IGNORECASE)
    if not matches:
        return "<div style='color: #888; font-style: italic;'>最新回答里还没有引用。</div>"

    lookup = {
        (str(item.get("path", "")).strip(), str(item.get("chunk_index", "")).strip()): item
        for item in source_results
    }
    unique_matches = []
    seen = set()
    for path, chunk_index in matches:
        key = (path.strip(), chunk_index.strip())
        if key not in seen:
            seen.add(key)
            unique_matches.append(key)

    nav_links = []
    cards = []
    for index_id, (path, chunk_index) in enumerate(unique_matches, start=1):
        anchor = f"citation-{index_id}"
        label = html.escape(f"{Path(path).name}#chunk{chunk_index}")
        nav_links.append(f"<a href='#{anchor}' style='margin-right: 10px;'>{label}</a>")
        item = lookup.get((path, chunk_index))
        snippet = html.escape(
            str(item.get("content", "当前召回结果中没有可预览的内容。"))
            if item
            else "当前召回结果中没有可预览的内容。"
        )
        full_path = html.escape(path)
        cards.append(
            f"""
            <div id="{anchor}" style="border: 1px solid #e5e7eb; border-radius: 8px; margin-top: 10px; background: #fff;">
                <div style="padding: 10px; font-weight: 700; border-bottom: 1px solid #e5e7eb;">
                    {label}
                </div>
                <div style="padding: 10px; color: #6b7280; font-size: 0.9em;">{full_path}</div>
                <div style="padding: 10px; white-space: pre-wrap; font-family: monospace; font-size: 0.85em;">
{snippet}
                </div>
            </div>
            """
        )

    return (
        "<div>"
        "<div style='font-weight: 700; margin-bottom: 8px;'>引用导航</div>"
        f"<div style='margin-bottom: 8px;'>{''.join(nav_links)}</div>"
        f"{''.join(cards)}"
        "</div>"
    )


def truncate_text(text, limit=120):
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def create_workflow_state(question="", top_k_ret=None, top_k_comp=None, threshold=None, multi_turn_enabled=True):
    return {
        "question": question,
        "top_k_ret": top_k_ret,
        "top_k_comp": top_k_comp,
        "threshold": threshold,
        "multi_turn_enabled": bool(multi_turn_enabled),
        "cache_status": "pending",
        "cache_entries": 0,
        "history_messages": 0,
        "summary_used": False,
        "rewrite_mode": "pending",
        "rewritten_query": "",
        "retrieved_count": 0,
        "reranked_count": 0,
        "reranker_status": "pending",
        "compression_mode": "pending",
        "citation_retry": False,
        "final_status": "idle",
        "status_detail": "等待发起请求。",
    }


def generate_system_workflow_html(workflow_state=None):
    state = workflow_state or create_workflow_state()
    status_map = {
        "idle": ("空闲", "#6b7280", "#f3f4f6"),
        "running": ("运行中", "#9a3412", "#ffedd5"),
        "streaming": ("生成中", "#1d4ed8", "#dbeafe"),
        "completed": ("已完成", "#166534", "#dcfce7"),
        "no_results": ("未命中", "#92400e", "#fef3c7"),
        "kb_unavailable": ("知识库不可用", "#991b1b", "#fee2e2"),
        "error": ("错误", "#991b1b", "#fee2e2"),
    }
    status_label, status_text_color, status_bg = status_map.get(
        state.get("final_status", "idle"),
        status_map["idle"],
    )
    cache_label = {
        "pending": "待处理",
        "hit": "命中",
        "miss": "未命中",
        "disabled": "已禁用",
    }.get(state.get("cache_status", "pending"), "待处理")
    compression_label = {
        "pending": "待处理",
        "compressed": "压缩证据已就绪",
        "raw_fallback": "回退到原始证据",
    }.get(state.get("compression_mode", "pending"), "待处理")
    rewrite_label = {
        "pending": "待处理",
        "used": "已启用上下文改写",
        "skipped": "已跳过改写",
        "conversation_only": "直接使用会话历史回答",
    }.get(state.get("rewrite_mode", "pending"), "待处理")
    conversation_mode = "开启" if state.get("multi_turn_enabled") else "关闭"
    summary_label = "已启用" if state.get("summary_used") else "未使用"
    citation_label = "已触发" if state.get("citation_retry") else "无需触发"
    rewritten_query = truncate_text(state.get("rewritten_query", "") or state.get("question", ""), 140)

    cards = [
        (
            "对话理解",
            [
                f"多轮上下文：{conversation_mode}",
                f"纳入历史消息：{state.get('history_messages', 0)}",
                f"摘要记忆：{summary_label}",
                f"改写策略：{rewrite_label}",
                f"解析后的查询：{rewritten_query or '待处理'}",
            ],
        ),
        (
            "证据流水线",
            [
                f"响应缓存：{cache_label}",
                f"缓存条目数：{state.get('cache_entries', 0)}",
                f"召回片段数：{state.get('retrieved_count', 0)}",
                f"重排证据块数：{state.get('reranked_count', 0)}",
                f"重排器状态：{localize_reranker_status(state.get('reranker_status', 'pending'))}",
            ],
        ),
        (
            "回答生成",
            [
                f"压缩路径：{compression_label}",
                f"引用补救：{citation_label}",
                f"当前状态：{status_label}",
                f"运行说明：{truncate_text(state.get('status_detail', ''), 120)}",
            ],
        ),
        (
            "执行参数",
            [
                f"问题：{truncate_text(state.get('question', ''), 110) or '当前没有问题'}",
                f"召回数量：{state.get('top_k_ret', '-')}",
                f"重排数量：{state.get('top_k_comp', '-')}",
                f"分数阈值：{state.get('threshold', '-')}",
            ],
        ),
    ]

    cards_html = []
    for title, lines in cards:
        cards_html.append(
            f"""
            <div style="padding: 14px; border: 1px solid #e5e7eb; border-radius: 12px; background: #ffffff;">
                <div style="font-weight: 700; margin-bottom: 8px; color: #111827;">{html.escape(title)}</div>
                <div style="color: #4b5563; line-height: 1.55;">
                    {'<br>'.join(html.escape(line) for line in lines)}
                </div>
            </div>
            """
        )

    return f"""
    <div style="padding: 14px; border: 1px solid #dbe4f0; border-radius: 14px; background: linear-gradient(135deg, #f8fbff 0%, #ffffff 100%);">
        <div style="display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; flex-wrap: wrap;">
            <div>
                <div style="font-size: 1.08em; font-weight: 800; color: #111827;">系统流程</div>
                <div style="margin-top: 4px; color: #6b7280;">
                    用简洁视图展示系统如何理解问题、组织证据并生成最终回答。
                </div>
            </div>
            <div style="padding: 6px 12px; border-radius: 999px; background: {status_bg}; color: {status_text_color}; font-weight: 700;">
                {html.escape(status_label)}
            </div>
        </div>
        <div style="margin-top: 10px; color: #4b5563;">{html.escape(state.get("status_detail", "等待发起请求。"))}</div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-top: 14px;">
            {''.join(cards_html)}
        </div>
    </div>
    """


def generate_build_report_html(report=None):
    if not report:
        return "<div style='color: #888; font-style: italic;'>暂无构建报告。</div>"

    snapshot_label = "快照已更新" if report.get("snapshot_status") == "updated" else "快照已清空"
    return f"""
    <div style="padding: 14px; border: 1px solid #e5e7eb; border-radius: 12px; background: #ffffff;">
        <div style="font-size: 1.05em; font-weight: 800; color: #111827;">增量构建报告</div>
        <div style="margin-top: 6px; color: #6b7280;">
            这次知识库同步会对比当前文档目录与已有索引，并刷新当前使用的快照。
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin-top: 14px;">
            <div style="padding: 12px; border: 1px solid #e5e7eb; border-radius: 10px; background: #f9fafb;">
                <div style="font-weight: 700;">范围</div>
                <div style="margin-top: 6px;">源目录：{html.escape(str(report.get('source_dir', 'N/A')))}</div>
                <div>发现文档数：{report.get('discovered_documents', 0)}</div>
                <div>有效文档数：{report.get('active_documents', 0)}</div>
            </div>
            <div style="padding: 12px; border: 1px solid #e5e7eb; border-radius: 10px; background: #f9fafb;">
                <div style="font-weight: 700;">文档同步</div>
                <div style="margin-top: 6px;">新增文档：{report.get('new_documents', 0)}</div>
                <div>刷新文档：{report.get('refreshed_documents', 0)}</div>
                <div>移除文档：{report.get('removed_documents', 0)}</div>
                <div>跳过空文档：{report.get('empty_documents', 0)}</div>
            </div>
            <div style="padding: 12px; border: 1px solid #e5e7eb; border-radius: 10px; background: #f9fafb;">
                <div style="font-weight: 700;">分块同步</div>
                <div style="margin-top: 6px;">新写入分块：{report.get('written_chunks', 0)}</div>
                <div>索引总分块数：{report.get('total_chunks', 0)}</div>
                <div>快照状态：{html.escape(snapshot_label)}</div>
                <div>构建耗时：{report.get('duration_seconds', 0)} 秒</div>
            </div>
        </div>
    </div>
    """


def apply_debug_event_to_workflow_state(workflow_state, event, content):
    if event == "cache_status" and isinstance(content, dict):
        workflow_state["cache_status"] = "hit" if content.get("hit") else "miss"
        workflow_state["cache_entries"] = int(content.get("entries", workflow_state.get("cache_entries", 0)))
        if content.get("hit"):
            workflow_state["status_detail"] = "命中了响应缓存，已直接复用已有结果。"
    elif event == "history_mode" and isinstance(content, dict):
        workflow_state["summary_used"] = bool(content.get("summary_used"))
        workflow_state["history_messages"] = int(content.get("history_messages", 0))
    elif event == "rewrite_mode":
        workflow_state["rewrite_mode"] = str(content or "pending")
    elif event == "rewritten_query":
        workflow_state["rewritten_query"] = str(content or "")
    elif event == "retrieved_results" and isinstance(content, list):
        workflow_state["retrieved_count"] = len(content)
        if workflow_state.get("rewrite_mode") == "conversation_only":
            return
        if workflow_state["cache_status"] != "hit":
            workflow_state["status_detail"] = f"已召回 {len(content)} 条候选证据片段。"
    elif event == "reranked_results" and isinstance(content, list):
        workflow_state["reranked_count"] = len(content)
        if workflow_state.get("rewrite_mode") == "conversation_only":
            return
        if workflow_state["cache_status"] != "hit":
            workflow_state["status_detail"] = f"已将证据收敛为 {len(content)} 条高置信片段。"


def apply_process_log_to_workflow_state(workflow_state, text):
    stripped = str(text or "").strip()
    if not stripped:
        return
    if stripped in {"Start retrieval", "开始检索流程"}:
        workflow_state["final_status"] = "running"
        workflow_state["status_detail"] = "开始执行检索与推理流程。"
    elif stripped.startswith(("Step 0: answer from conversation history", "步骤 0：直接根据对话历史回答")):
        workflow_state["status_detail"] = "当前问题被识别为对话历史问题，直接基于会话历史回答。"
    elif stripped.startswith(("Step 0:", "步骤 0：")):
        workflow_state["status_detail"] = "正在理解请求并解析对话中的指代。"
    elif stripped.startswith(("Step 1: vectorize", "步骤 1：向量化")):
        workflow_state["status_detail"] = "正在将解析后的查询转换为向量。"
    elif stripped.startswith(("Step 1: reuse cached response package", "步骤 1：复用缓存的响应包")):
        workflow_state["final_status"] = "completed"
        workflow_state["status_detail"] = "相同请求命中缓存，已复用完整响应包。"
    elif stripped.startswith(("Step 2:", "步骤 2：")):
        workflow_state["status_detail"] = "正在知识库中搜索相关证据。"
    elif stripped.startswith(("Step 3:", "步骤 3：")):
        workflow_state["status_detail"] = "正在重排并筛选证据集合。"
    elif stripped.startswith(("Reranker status:", "重排器状态：")):
        workflow_state["reranker_status"] = re.split(r"[:：]", stripped, maxsplit=1)[1].strip()
    elif stripped.startswith(("Compression model:", "压缩模型：")):
        workflow_state["compression_mode"] = "compressed"
        workflow_state["status_detail"] = "正在压缩重叠证据，生成更精炼的上下文。"
    elif stripped.startswith(("Compressed context failed citation validation;", "压缩结果未通过引用校验")):
        workflow_state["compression_mode"] = "raw_fallback"
        workflow_state["status_detail"] = "压缩结果丢失了引用准确性，已回退到原始证据。"
    elif stripped in {"Final answer", "最终回答"}:
        workflow_state["final_status"] = "streaming"
        workflow_state["status_detail"] = "正在生成带引用依据的最终回答。"
    elif stripped.startswith(("[Notice] Answer citations were incomplete.", "[提示] 回答中的引用不完整")):
        workflow_state["citation_retry"] = True
        workflow_state["status_detail"] = "回答中的引用不完整，已基于原始证据重试。"
    elif stripped.startswith(("No relevant knowledge snippets were found.", "No relevant knowledge snippets were retrieved.", "未检索到相关知识片段。", "未检索到相关知识片段，请检查知识库或适当降低分数阈值。")):
        workflow_state["final_status"] = "no_results"
        workflow_state["status_detail"] = "没有证据通过当前问题的召回阈值。"
    elif stripped.startswith(("Knowledge base is empty or failed to load.", "Knowledge base failed to load:", "知识库为空或加载失败。", "知识库加载失败：")):
        workflow_state["final_status"] = "kb_unavailable"
        workflow_state["status_detail"] = "知识库快照不可用，或加载失败。"
    elif stripped.startswith(("Diagnostics:", "诊断信息：")):
        workflow_state["final_status"] = "completed"
        workflow_state["status_detail"] = "端到端流程已成功完成。"
        reranker_match = re.search(r"(?:reranker|重排器)=([^,，]+)", stripped)
        retrieval_match = re.search(r"(?:retrieval_fallback|召回回退)=([^,，\s]+)", stripped)
        citation_match = re.search(r"(?:citation_retry|引用重试)=([^,，\s]+)", stripped)
        if reranker_match:
            workflow_state["reranker_status"] = reranker_match.group(1).strip()
        if citation_match:
            workflow_state["citation_retry"] = citation_match.group(1).strip().lower() in {"true", "是"}
        if retrieval_match and retrieval_match.group(1).strip().lower() in {"true", "是"}:
            workflow_state["status_detail"] = "流程已完成，并触发了保守的召回回退策略。"
    elif stripped.startswith(("[Error]", "Error:", "[错误]")):
        workflow_state["final_status"] = "error"
        workflow_state["status_detail"] = stripped


def _format_timestamp(path_obj):
    if not path_obj.exists():
        return "暂无"
    return datetime.fromtimestamp(path_obj.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def generate_knowledge_base_status_html():
    vector_count = "0"
    metadata_count = "0"
    document_count = "0"
    db_rows = "0"
    status_message = "未找到知识库文件。"

    if FAISS_INDEX_FILE.exists() and METADATA_FILE.exists():
        try:
            index, metadata = RAGHelpers.load_faiss_index_and_metadata(FAISS_INDEX_FILE, METADATA_FILE)
            vector_count = str(index.ntotal if index is not None else 0)
            metadata_count = str(len(metadata))
            status_message = "知识库快照可用。"
        except Exception as exc:
            status_message = f"知识库快照加载失败：{html.escape(str(exc))}"

    if DB_PATH.exists():
        try:
            conn = sqlite3.connect(DB_PATH)
            try:
                db_rows = str(conn.execute("SELECT COUNT(*) FROM metadata").fetchone()[0])
                document_count = str(conn.execute("SELECT COUNT(DISTINCT path) FROM metadata").fetchone()[0])
            finally:
                conn.close()
        except Exception as exc:
            status_message += f" SQLite 读取警告：{html.escape(str(exc))}"

    cache_entries = "0"
    if CACHE_FILE.exists():
        try:
            cache_data = RAGHelpers.load_embedding_cache(CACHE_FILE)
            cache_entries = str(max(0, len(cache_data) - (1 if "__model_name__" in cache_data else 0)))
        except Exception:
            cache_entries = "未知"

    query_cache_entries = "0"
    summary_cache_entries = "0"
    runtime_rag = globals().get("rag_instance")
    if runtime_rag is not None:
        try:
            runtime_cache_metrics = runtime_rag.get_runtime_cache_metrics()
            query_cache_entries = str(runtime_cache_metrics.get("query_cache_entries", 0))
            summary_cache_entries = str(runtime_cache_metrics.get("summary_cache_entries", 0))
        except Exception:
            query_cache_entries = "未知"
            summary_cache_entries = "未知"

    return f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px;">
        <div style="padding: 12px; border: 1px solid #e5e7eb; border-radius: 8px; background: #fff;">
            <div style="font-weight: 700; margin-bottom: 6px;">知识库状态</div>
            <div>{html.escape(status_message)}</div>
            <div style="color: #6b7280; margin-top: 6px;">最近索引更新时间：{html.escape(_format_timestamp(FAISS_INDEX_FILE))}</div>
        </div>
        <div style="padding: 12px; border: 1px solid #e5e7eb; border-radius: 8px; background: #fff;">
            <div style="font-weight: 700; margin-bottom: 6px;">数量统计</div>
            <div>向量数：{vector_count}</div>
            <div>元数据条目：{metadata_count}</div>
            <div>数据库行数：{db_rows}</div>
            <div>不同文档数：{document_count}</div>
            <div>嵌入缓存条目：{cache_entries}</div>
            <div>查询缓存条目：{query_cache_entries}</div>
            <div>摘要缓存条目：{summary_cache_entries}</div>
        </div>
        <div style="padding: 12px; border: 1px solid #e5e7eb; border-radius: 8px; background: #fff;">
            <div style="font-weight: 700; margin-bottom: 6px;">模型信息</div>
            <div>对话模型：{html.escape(CHAT_MODEL)}</div>
            <div>压缩模型：{html.escape(COMPRESSOR_MODEL)}</div>
            <div>嵌入模型：{html.escape(EMBEDDING_MODEL)}</div>
            <div>重排模型：{html.escape(RERANK_MODEL)}</div>
        </div>
    </div>
    """


def apply_parameter_preset(preset_name):
    presets = {
        "均衡（默认）": (DEFAULT_TOP_K, DEFAULT_TOP_K_COMPRESSED, DEFAULT_THRESHOLD),
        "快速": (3, 2, 0.45),
        "深入": (8, 5, 0.2),
    }
    return presets.get(
        preset_name,
        (DEFAULT_TOP_K, DEFAULT_TOP_K_COMPRESSED, DEFAULT_THRESHOLD),
    )


conversation_store = ConversationStore(CONVERSATION_STATE_FILE)
DEBUG_EVENT_LABELS = {
    "cache_status": "缓存状态",
    "history_mode": "历史模式",
    "history_summary": "历史摘要",
    "retrieval_history": "检索历史",
    "rewrite_mode": "改写模式",
    "rewritten_query": "改写结果",
    "retrieved_results": "召回结果",
    "reranked_results": "重排结果",
}


def _get_conversation_state():
    return conversation_store.get_state()


def _get_conversation_messages():
    return _get_conversation_state().get("messages", [])


def _is_request_session_current(session_id):
    current_session_id = str(_get_conversation_state().get("active_session_id", "")).strip()
    return bool(session_id) and current_session_id == session_id


def _should_persist_answer(answer_text):
    return bool(str(answer_text or "").strip()) and not simpleRAG_content.is_non_persistent_assistant_message(answer_text)


def _serialize_history_editor(messages):
    return json.dumps(messages or [], ensure_ascii=False, indent=2)


def generate_conversation_state_html(state=None):
    state = state or _get_conversation_state()
    session_id = str(state.get("active_session_id", ""))
    updated_at = state.get("updated_at") or "暂无"
    messages = state.get("messages", [])
    return f"""
    <div style="padding: 12px; border: 1px solid #e5e7eb; border-radius: 10px; background: #ffffff;">
        <div style="font-weight: 700; margin-bottom: 6px;">当前会话状态</div>
        <div>会话 ID：{html.escape(session_id)}</div>
        <div>消息条数：{len(messages)}</div>
        <div>最近保存时间：{html.escape(str(updated_at))}</div>
    </div>
    """


def generate_conversation_history_manager_html(messages):
    if not messages:
        return "<div style='color: #888; font-style: italic;'>当前没有已保存的历史对话。</div>"

    cards = []
    for index_id, item in enumerate(messages, start=1):
        role = "用户" if item.get("role") == "user" else "助手"
        content = html.escape(str(item.get("content", "")))
        cards.append(
            f"""
            <details style="border: 1px solid #e5e7eb; border-radius: 8px; margin-bottom: 8px; background: #ffffff;">
                <summary style="padding: 10px; cursor: pointer; font-weight: 600;">
                    #{index_id} {role}
                </summary>
                <div style="padding: 10px; border-top: 1px solid #e5e7eb; white-space: pre-wrap; font-family: monospace; font-size: 0.85em;">
{content}
                </div>
            </details>
            """
        )
    return "".join(cards)


def load_conversation_manager_views():
    state = _get_conversation_state()
    messages = state.get("messages", [])
    return (
        messages,
        _serialize_history_editor(messages),
        generate_conversation_history_manager_html(messages),
        generate_conversation_state_html(state),
        "已加载当前持久化会话历史。",
    )


def build_conversation_manager_views(messages, status_text):
    state = _get_conversation_state()
    preview_state = {
        **state,
        "messages": messages,
    }
    return (
        _serialize_history_editor(messages),
        generate_conversation_history_manager_html(messages),
        generate_conversation_state_html(preview_state),
        status_text,
    )


def _validate_history_messages(parsed):
    if not isinstance(parsed, list):
        return None, "历史编辑内容必须是 JSON 数组。"

    validated = []
    for index, item in enumerate(parsed, start=1):
        if not isinstance(item, dict):
            return None, f"第 {index} 条消息必须是 JSON 对象，格式应为 {{\"role\": \"user\", \"content\": \"...\"}}。"

        if "role" not in item:
            return None, f"第 {index} 条消息缺少 role 字段。"
        if "content" not in item:
            return None, f"第 {index} 条消息缺少 content 字段。"

        role = item.get("role")
        content = item.get("content")

        if not isinstance(role, str):
            return None, f"第 {index} 条消息的 role 必须是字符串。"
        role = role.strip()
        if role not in {"user", "assistant"}:
            return None, f"第 {index} 条消息的 role 只能是 user 或 assistant。"

        if not isinstance(content, str):
            return None, f"第 {index} 条消息的 content 必须是字符串。"
        content = content.strip()
        if not content:
            return None, f"第 {index} 条消息的 content 不能为空。"

        validated.append({"role": role, "content": content})

    return validated, None


def save_history_edits(history_editor_text):
    state = _get_conversation_state()
    messages = state.get("messages", [])

    try:
        parsed = json.loads(history_editor_text or "[]")
    except json.JSONDecodeError as exc:
        return (
            messages,
            history_editor_text,
            generate_conversation_history_manager_html(messages),
            generate_conversation_state_html(state),
            f"保存编辑失败：JSON 解析错误（第 {exc.lineno} 行，第 {exc.colno} 列）：{exc.msg}",
        )

    validated_messages, validation_error = _validate_history_messages(parsed)
    if validation_error:
        return (
            messages,
            history_editor_text,
            generate_conversation_history_manager_html(messages),
            generate_conversation_state_html(state),
            f"保存编辑失败：{validation_error}",
        )

    saved_state = conversation_store.set_messages(validated_messages)
    saved_messages = saved_state.get("messages", [])
    return (
        saved_messages,
        _serialize_history_editor(saved_messages),
        generate_conversation_history_manager_html(saved_messages),
        generate_conversation_state_html(saved_state),
        "历史修改已保存，后续对话会基于新的历史内容继续。",
    )


def reset_conversation_session():
    state = conversation_store.reset_session()
    messages = state.get("messages", [])
    return (
        messages,
        [],
        _serialize_history_editor(messages),
        generate_conversation_history_manager_html(messages),
        generate_conversation_state_html(state),
        generate_system_workflow_html(),
        "",
        "",
        "<div style='color: #888; font-style: italic;'>暂无历史记录。</div>",
        generate_retrieval_html([], []),
        generate_citation_preview_html("", []),
        "已清空当前会话，并创建新的会话 ID。",
    )


def request_clear_conversation_confirmation():
    return gr.update(visible=False), gr.update(visible=True)


def cancel_clear_conversation_confirmation():
    return gr.update(visible=True), gr.update(visible=False)


def confirm_reset_conversation_session():
    return (
        *reset_conversation_session(),
        gr.update(visible=True),
        gr.update(visible=False),
    )


class OutputLogger:
    def write_log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        return f"[{timestamp}] {message}\n"


logger = OutputLogger()
rag_instance = None
rag_instance_lock = threading.Lock()


def get_rag_instance():
    global rag_instance
    if rag_instance is not None:
        return rag_instance

    with rag_instance_lock:
        if rag_instance is None:
            rag_instance = simpleRAG_content.SimpleRAG()
    return rag_instance


def get_last_build_report():
    runtime_rag = globals().get("rag_instance")
    if runtime_rag is None:
        return {}
    try:
        return runtime_rag.get_last_build_report()
    except Exception:
        return {}


def refresh_knowledge_base_panels():
    return generate_knowledge_base_status_html(), generate_build_report_html(get_last_build_report())


def _append_doc_preprocess_logs(logs, report):
    if not report:
        return logs, False

    has_warning = False
    detected_doc_files = int(report.get("detected_doc_files", 0) or 0)
    converted_doc_files = int(report.get("converted_doc_files", 0) or 0)
    archived_doc_files = int(report.get("archived_doc_files", 0) or 0)
    skipped_existing_docx = report.get("skipped_existing_docx", []) or []
    failed_doc_files = report.get("failed_doc_files", []) or []
    archive_failed_doc_files = report.get("archive_failed_doc_files", []) or []
    errors = report.get("errors", []) or []

    if detected_doc_files == 0:
        logs += logger.write_log("未发现需要预处理的 .doc 文件。")
        return logs, False

    logs += logger.write_log(f"发现 {detected_doc_files} 个 .doc 文件，已在构建前尝试预处理。")
    if converted_doc_files:
        logs += logger.write_log(
            f"已成功转换 {converted_doc_files} 个 .doc 文件，其中 {archived_doc_files} 个原始 .doc 已移入 backup 目录。"
        )

    if skipped_existing_docx:
        has_warning = True
        logs += logger.write_log(
            f"有 {len(skipped_existing_docx)} 个 .doc 因同名 .docx 已存在而跳过自动转换，原 .doc 保留在原位。"
        )
        for item in skipped_existing_docx[:5]:
            logs += logger.write_log(f"跳过转换：{item}")
        if len(skipped_existing_docx) > 5:
            logs += logger.write_log(f"其余跳过项还有 {len(skipped_existing_docx) - 5} 个。")

    if failed_doc_files:
        has_warning = True
        logs += logger.write_log(
            f"有 {len(failed_doc_files)} 个 .doc 转换失败，本次构建将继续，但这些 .doc 不会入库。"
        )
        for item in failed_doc_files[:5]:
            logs += logger.write_log(f"转换失败：{item}")
        if len(failed_doc_files) > 5:
            logs += logger.write_log(f"其余失败项还有 {len(failed_doc_files) - 5} 个。")
        logs += logger.write_log("如需处理失败的 .doc，可单独运行 doc_converter.py。")

    if archive_failed_doc_files:
        has_warning = True
        logs += logger.write_log(
            f"有 {len(archive_failed_doc_files)} 个 .doc 已转成 .docx，但原始 .doc 移入 backup 失败。"
        )
        for item in archive_failed_doc_files[:5]:
            logs += logger.write_log(f"归档失败：{item}")
        if len(archive_failed_doc_files) > 5:
            logs += logger.write_log(f"其余归档失败项还有 {len(archive_failed_doc_files) - 5} 个。")

    if errors:
        has_warning = True
        for item in errors:
            logs += logger.write_log(f".doc 预处理提示：{item}")

    return logs, has_warning


def _build_busy_control_updates(mode=None):
    is_chat_busy = mode == "chat"
    is_build_busy = mode == "build"
    is_busy = is_chat_busy or is_build_busy

    msg_placeholder = "输入你的问题..."
    if is_chat_busy:
        msg_placeholder = "回答生成中，请稍候..."
    elif is_build_busy:
        msg_placeholder = "知识库构建中，请稍候..."

    submit_label = "生成中..." if is_chat_busy else "发送"
    build_label = "构建中..." if is_build_busy else "构建索引"

    return (
        gr.update(interactive=not is_busy, placeholder=msg_placeholder),
        gr.update(value=submit_label, interactive=not is_busy),
        gr.update(interactive=not is_busy),
        gr.update(interactive=not is_busy),
        gr.update(interactive=not is_busy),
        gr.update(interactive=not is_busy),
        gr.update(interactive=not is_busy),
        gr.update(interactive=not is_busy),
        gr.update(value=build_label, interactive=not is_busy),
        gr.update(interactive=not is_busy),
    )


def lock_controls_for_chat():
    return _build_busy_control_updates("chat")


def lock_controls_for_build():
    return _build_busy_control_updates("build")


def unlock_controls():
    return _build_busy_control_updates()


def _coerce_optional_int(value):
    if value in (None, ""):
        return None
    return int(value)


def _run_async_in_thread(coro):
    result = {"value": None, "error": None}

    def _runner():
        try:
            result["value"] = asyncio.run(coro)
        except Exception as exc:
            result["error"] = exc

    worker = threading.Thread(target=_runner, daemon=False)
    worker.start()
    worker.join()

    if result["error"] is not None:
        raise result["error"]
    return result["value"]


def build_knowledge_base_task(source_dir_str, chunk_size, overlap, progress=gr.Progress()):
    source_dir = Path(source_dir_str) if source_dir_str else DOC_DIR
    logs = logger.write_log(f"正在扫描源目录：{source_dir}")

    if not source_dir.exists() or not source_dir.is_dir():
        logs += logger.write_log(f"目录不存在或不是文件夹：{source_dir}")
        return (
            "失败",
            clean_log_text(logs),
            generate_knowledge_base_status_html(),
            generate_build_report_html(get_last_build_report()),
        )

    try:
        progress(0, desc="准备构建中...")
        logs += logger.write_log("正在初始化 RAG 引擎...")
        rag = get_rag_instance()
        logs += logger.write_log("RAG 引擎已就绪。")
        progress(0.05, desc="RAG 引擎初始化完成")

        async def run_build():
            return await rag.build_knowledge_base_async(
                source_dir,
                chunk_size=_coerce_optional_int(chunk_size),
                overlap=_coerce_optional_int(overlap),
            )

        captured_stdout = io.StringIO()
        captured_stderr = io.StringIO()
        with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
            build_report = _run_async_in_thread(run_build())

        runtime_output = (captured_stdout.getvalue() + captured_stderr.getvalue()).strip()
        if runtime_output:
            logs += runtime_output
            if not logs.endswith("\n"):
                logs += "\n"

        try:
            index, _ = RAGHelpers.load_faiss_index_and_metadata(FAISS_INDEX_FILE, METADATA_FILE)
            count = index.ntotal if index is not None else "未知"
            logs += logger.write_log(f"构建完成。当前向量总数：{count}")
            logs += logger.write_log("索引文件已保存到本地。")
        except Exception:
            logs += logger.write_log("构建已完成。")

        return (
            "成功",
            clean_log_text(logs),
            generate_knowledge_base_status_html(),
            generate_build_report_html(build_report),
        )
    except Exception as exc:
        logs += logger.write_log(f"构建失败：{exc}")
        return (
            "失败",
            clean_log_text(logs),
            generate_knowledge_base_status_html(),
            generate_build_report_html(get_last_build_report()),
        )

def build_knowledge_base_task_with_doc_preprocess(
    source_dir_str,
    chunk_size,
    overlap,
    progress=gr.Progress(),
):
    source_dir = Path(source_dir_str) if source_dir_str else DOC_DIR
    logs = logger.write_log(f"正在扫描源目录：{source_dir}")

    if not source_dir.exists() or not source_dir.is_dir():
        logs += logger.write_log(f"目录不存在或不是文件夹：{source_dir}")
        return (
            "失败",
            clean_log_text(logs),
            generate_knowledge_base_status_html(),
            generate_build_report_html(get_last_build_report()),
        )

    try:
        progress(0, desc="准备构建中...")
        logs += logger.write_log("正在检查并预处理 .doc 文件...")
        preprocess_report = preprocess_doc_files_for_build(
            source_dir,
            skip_backup=True,
            allow_install=False,
        )
        logs, has_preprocess_warning = _append_doc_preprocess_logs(logs, preprocess_report)
        progress(0.08, desc="文档预处理完成")

        logs += logger.write_log("正在初始化 RAG 引擎...")
        rag = get_rag_instance()
        logs += logger.write_log("RAG 引擎已就绪。")
        progress(0.15, desc="RAG 引擎初始化完成")

        async def run_build():
            return await rag.build_knowledge_base_async(
                source_dir,
                chunk_size=_coerce_optional_int(chunk_size),
                overlap=_coerce_optional_int(overlap),
            )

        captured_stdout = io.StringIO()
        captured_stderr = io.StringIO()
        with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
            build_report = _run_async_in_thread(run_build())

        runtime_output = (captured_stdout.getvalue() + captured_stderr.getvalue()).strip()
        if runtime_output:
            logs += runtime_output
            if not logs.endswith("\n"):
                logs += "\n"

        try:
            index, _ = RAGHelpers.load_faiss_index_and_metadata(FAISS_INDEX_FILE, METADATA_FILE)
            count = index.ntotal if index is not None else "未知"
            logs += logger.write_log(f"构建完成，当前向量总数：{count}")
            logs += logger.write_log("索引文件已保存到本地。")
        except Exception:
            logs += logger.write_log("构建已完成。")

        return (
            "成功（含预处理提示）" if has_preprocess_warning else "成功",
            clean_log_text(logs),
            generate_knowledge_base_status_html(),
            generate_build_report_html(build_report),
        )
    except Exception as exc:
        logs += logger.write_log(f"构建失败：{exc}")
        return (
            "失败",
            clean_log_text(logs),
            generate_knowledge_base_status_html(),
            generate_build_report_html(get_last_build_report()),
        )


def answer_question_task(question, history, top_k_ret, top_k_comp, threshold, multi_turn_enabled, log_history_state):
    persisted_history = _get_conversation_messages()
    request_session_id = str(_get_conversation_state().get("active_session_id", "")).strip()
    if not isinstance(history, list):
        history = persisted_history
    if not isinstance(log_history_state, list):
        log_history_state = []

    latest_retrieved_results = []
    latest_reranked_results = []
    workflow_state = create_workflow_state(
        question=question,
        top_k_ret=top_k_ret,
        top_k_comp=top_k_comp,
        threshold=threshold,
        multi_turn_enabled=multi_turn_enabled,
    )
    current_q_logs = ""
    debug_lines = []
    conversation_status_text = "当前会话历史会在回答完成后自动保存。"

    def emit(history_value, log_state_value, answer_text, manager_messages=None):
        if not _is_request_session_current(request_session_id):
            return None
        manager_source = history_value if manager_messages is None else manager_messages
        (
            _history_editor_value,
            history_manager_value,
            session_state_value,
            history_status_value,
        ) = build_conversation_manager_views(manager_source, conversation_status_text)
        return (
            "",
            history_value,
            log_state_value,
            generate_system_workflow_html(workflow_state),
            clean_log_text(current_q_logs),
            "\n\n".join(debug_lines),
            generate_logs_html(log_state_value),
            generate_retrieval_html(latest_retrieved_results, latest_reranked_results),
            generate_citation_preview_html(answer_text, latest_reranked_results or latest_retrieved_results),
            history_manager_value,
            session_state_value,
            history_status_value,
        )

    if not question:
        current_q_logs = logger.write_log("错误：问题不能为空。")
        workflow_state["final_status"] = "error"
        workflow_state["status_detail"] = "请求为空，流程未启动。"
        payload = emit(history, log_history_state, "")
        if payload is not None:
            yield payload
        return

    current_q_logs = logger.write_log(f"正在处理新问题：{question}")
    current_q_logs += logger.write_log(
        f"参数：召回数={top_k_ret}，压缩数={top_k_comp}，阈值={threshold}"
    )
    current_q_logs += logger.write_log(f"多轮上下文开关：{bool(multi_turn_enabled)}")

    workflow_state["final_status"] = "running"
    workflow_state["status_detail"] = "已收到请求，正在准备检索流程。"
    new_history = persisted_history + [{"role": "user", "content": question}]
    new_history.append({"role": "assistant", "content": "正在检索并推理..."})
    payload = emit(new_history, log_history_state, "")
    if payload is not None:
        yield payload
    else:
        return

    full_answer = ""

    try:
        k_ret = int(top_k_ret) if top_k_ret is not None else DEFAULT_TOP_K
        k_comp = int(top_k_comp) if top_k_comp is not None else DEFAULT_TOP_K_COMPRESSED
        thresh = float(threshold) if threshold is not None else DEFAULT_THRESHOLD
        workflow_state["top_k_ret"] = k_ret
        workflow_state["top_k_comp"] = k_comp
        workflow_state["threshold"] = thresh

        current_q_logs += logger.write_log("正在初始化 RAG 引擎...")
        workflow_state["status_detail"] = "正在初始化 RAG 运行时。"
        payload = emit(new_history, log_history_state, full_answer)
        if payload is not None:
            yield payload
        else:
            return

        rag = get_rag_instance()
        current_q_logs += logger.write_log("RAG 引擎已就绪。")
        workflow_state["status_detail"] = "RAG 运行时初始化完成，开始检索证据。"
        payload = emit(new_history, log_history_state, full_answer)
        if payload is not None:
            yield payload
        else:
            return

        effective_history = persisted_history if multi_turn_enabled else []
        if not multi_turn_enabled:
            workflow_state["cache_status"] = "disabled"

        for chunk in rag.answer_question_stream(
            question,
            top_k_retrieve=k_ret,
            top_k_compressed=k_comp,
            score_threshold=thresh,
            history=effective_history,
        ):
            debug_event = parse_debug_event(chunk)
            if debug_event:
                event = debug_event.get("event", "unknown")
                content = debug_event.get("content", "")
                if event == "retrieved_results" and isinstance(content, list):
                    latest_retrieved_results = content
                elif event == "reranked_results" and isinstance(content, list):
                    latest_reranked_results = content
                else:
                    debug_label = DEBUG_EVENT_LABELS.get(event, event)
                    debug_lines.append(f"[{debug_label}]\n{format_debug_content(content)}")
                apply_debug_event_to_workflow_state(workflow_state, event, content)
                payload = emit(new_history, log_history_state, full_answer)
                if payload is not None:
                    yield payload
                else:
                    return
                continue

            stripped_chunk = chunk.strip()
            if is_process_log(chunk):
                apply_process_log_to_workflow_state(workflow_state, stripped_chunk)
                if stripped_chunk in {
                    "No relevant knowledge snippets were found.",
                    "No relevant knowledge snippets were retrieved.",
                    "Knowledge base is empty or failed to load.",
                    "未检索到相关知识片段。",
                    "未检索到相关知识片段，请检查知识库或适当降低分数阈值。",
                    "知识库为空或加载失败。",
                } or stripped_chunk.startswith(("Knowledge base failed to load:", "知识库加载失败：")):
                    full_answer = stripped_chunk
                    new_history[-1]["content"] = full_answer
                elif stripped_chunk.startswith(("[Error]", "[错误]")):
                    full_answer = stripped_chunk
                    new_history[-1]["content"] = full_answer
                current_q_logs += chunk + "\n"
                payload = emit(new_history, log_history_state, full_answer)
                if payload is not None:
                    yield payload
                else:
                    return
            else:
                if chunk == simpleRAG_content.SimpleRAG.ANSWER_REPLACE_MARKER:
                    full_answer = ""
                    new_history[-1]["content"] = ""
                    workflow_state["citation_retry"] = True
                    workflow_state["status_detail"] = "正在基于原始引用证据刷新回答。"
                    payload = emit(new_history, log_history_state, full_answer)
                    if payload is not None:
                        yield payload
                    else:
                        return
                    continue
                full_answer += chunk
                new_history[-1]["content"] = full_answer
                workflow_state["final_status"] = "streaming"
                workflow_state["status_detail"] = "正在向界面流式输出有依据的回答。"
                payload = emit(new_history, log_history_state, full_answer)
                if payload is not None:
                    yield payload
                else:
                    return

        if workflow_state["final_status"] == "running":
            workflow_state["final_status"] = "completed"
            workflow_state["status_detail"] = "流程已成功完成。"

        current_q_logs += logger.write_log("回答生成完成。")
        if not _is_request_session_current(request_session_id):
            return

        if _should_persist_answer(full_answer):
            final_history = persisted_history + [
                {"role": "user", "content": question},
                {"role": "assistant", "content": full_answer},
            ]
            conversation_store.set_messages(final_history)
            manager_history = final_history
            display_history = final_history
            conversation_status_text = "当前会话历史已自动保存。"
        else:
            manager_history = persisted_history
            display_history = new_history
            conversation_status_text = "本次结果仅作为运行提示显示，未写入历史。"
        log_entry_label = f"{question[:30]}{'...' if len(question) > 30 else ''} ({time.strftime('%H:%M:%S')})"
        new_log_history = [{"label": log_entry_label, "details": clean_log_text(current_q_logs)}] + log_history_state
        if len(new_log_history) > 20:
            new_log_history = new_log_history[:20]

        payload = emit(display_history, new_log_history, full_answer, manager_messages=manager_history)
        if payload is not None:
            yield payload
    except Exception as exc:
        error_msg = str(exc)
        current_q_logs += logger.write_log(f"错误：{error_msg}")
        workflow_state["final_status"] = "error"
        workflow_state["status_detail"] = f"请求失败：{error_msg}"
        new_history[-1]["content"] = f"{full_answer}\n\n[系统错误]：{error_msg}"
        conversation_status_text = "本次回答失败，历史未自动写入持久化会话。"
        log_entry_label = f"{question[:30]}...（错误）({time.strftime('%H:%M:%S')})"
        new_log_history = [{"label": log_entry_label, "details": clean_log_text(current_q_logs)}] + log_history_state
        payload = emit(new_history, new_log_history, full_answer, manager_messages=persisted_history)
        if payload is not None:
            yield payload


initial_conversation_state = _get_conversation_state()
initial_conversation_messages = initial_conversation_state.get("messages", [])


with gr.Blocks(title="大模型/RAG/AI开发知识体系轻量问答系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 大模型/RAG/AI开发知识体系轻量问答系统")

    with gr.Tabs():
        with gr.TabItem("对话问答"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="对话记录",
                        height=500,
                        show_copy_button=True,
                        type="messages",
                        value=initial_conversation_messages,
                    )
                    with gr.Row():
                        msg_input = gr.Textbox(
                            label="问题",
                            placeholder="输入你的问题...",
                            scale=4,
                            container=False,
                        )
                        submit_btn = gr.Button("发送", variant="primary", scale=1)
                    clear_btn = gr.Button("清空对话", size="sm")
                    with gr.Row(visible=False) as clear_confirm_row:
                        gr.Markdown("确认清空当前全部对话历史吗？此操作不可撤销。")
                        clear_confirm_btn = gr.Button("确认清空", variant="stop")
                        clear_cancel_btn = gr.Button("取消")

                with gr.Column(scale=1):
                    gr.Markdown("### 检索设置")
                    preset_dropdown = gr.Dropdown(
                        choices=["均衡（默认）", "快速", "深入"],
                        value="均衡（默认）",
                        label="参数预设",
                    )
                    top_k_ret_slider = gr.Slider(1, 50, value=DEFAULT_TOP_K, step=1, label="召回数量")
                    top_k_comp_slider = gr.Slider(1, 20, value=DEFAULT_TOP_K_COMPRESSED, step=1, label="重排数量")
                    threshold_slider = gr.Slider(0.0, 1.0, value=DEFAULT_THRESHOLD, step=0.05, label="分数阈值")
                    multi_turn_toggle = gr.Checkbox(label="启用多轮上下文", value=True)

            gr.Markdown("### 处理历史")
            log_history_state = gr.State([])
            workflow_html = gr.HTML(
                value=generate_system_workflow_html(),
                label="系统流程",
            )
            log_html_display = gr.HTML(
                value="<div style='color: #888; font-style: italic;'>暂无历史记录。</div>",
                label="历史记录列表",
            )
            current_process_log = gr.Textbox(
                label="当前处理日志",
                lines=5,
                max_lines=8,
                interactive=False,
                visible=True,
            )
            conversation_debug_log = gr.Textbox(
                label="多轮调试信息",
                lines=8,
                max_lines=14,
                interactive=False,
                visible=True,
            )
            retrieval_results_html = gr.HTML(
                value=generate_retrieval_html([], []),
                label="检索结果",
            )
            citation_preview_html = gr.HTML(
                value=generate_citation_preview_html("", []),
                label="引用预览",
            )

        with gr.TabItem("对话历史管理"):
            gr.Markdown("### 历史管理")
            session_state_html = gr.HTML(
                value=generate_conversation_state_html(initial_conversation_state),
                label="会话状态",
            )
            history_manager_html = gr.HTML(
                value=generate_conversation_history_manager_html(initial_conversation_messages),
                label="历史预览",
            )
            history_editor = gr.Code(
                label="历史编辑（JSON）",
                value=_serialize_history_editor(initial_conversation_messages),
                language="json",
                lines=24,
                interactive=True,
            )
            history_manager_status = gr.Textbox(
                label="历史管理状态",
                value="已加载当前持久化会话历史。",
                interactive=False,
            )
            with gr.Row():
                load_history_btn = gr.Button("重新加载历史")
                save_history_btn = gr.Button("保存历史修改", variant="primary")

        with gr.TabItem("知识库构建"):
            gr.Markdown("### 构建 / 重建索引")
            knowledge_base_status_html = gr.HTML(
                value=generate_knowledge_base_status_html(),
                label="知识库状态",
            )
            build_report_html = gr.HTML(
                value=generate_build_report_html(),
                label="构建报告",
            )
            with gr.Row():
                with gr.Column():
                    dir_input = gr.Textbox(label="文档目录", value=str(DOC_DIR))
                    chunk_size_input = gr.Number(label="分块大小", value=CHUNK_SIZE_DEFAULT, precision=0)
                    overlap_input = gr.Number(label="分块重叠", value=CHUNK_OVERLAP_DEFAULT, precision=0)
                    build_btn = gr.Button("构建索引", variant="primary")
                    refresh_status_btn = gr.Button("刷新知识库状态")
                with gr.Column():
                    build_status = gr.Textbox(label="构建状态", interactive=False)
                    build_log = gr.Textbox(label="构建日志", lines=15, interactive=False)

    def process_and_update(question, history, k1, k2, th, multi_turn_enabled, log_state):
        yield from answer_question_task(question, history, k1, k2, th, multi_turn_enabled, log_state)

    busy_control_outputs = [
        msg_input,
        submit_btn,
        clear_btn,
        clear_confirm_btn,
        clear_cancel_btn,
        load_history_btn,
        save_history_btn,
        history_editor,
        build_btn,
        refresh_status_btn,
    ]

    preset_dropdown.change(
        fn=apply_parameter_preset,
        inputs=[preset_dropdown],
        outputs=[top_k_ret_slider, top_k_comp_slider, threshold_slider],
    )

    msg_input.submit(
        fn=lock_controls_for_chat,
        inputs=None,
        outputs=busy_control_outputs,
        queue=False,
    ).then(
        fn=process_and_update,
        inputs=[msg_input, chatbot, top_k_ret_slider, top_k_comp_slider, threshold_slider, multi_turn_toggle, log_history_state],
        outputs=[
            msg_input,
            chatbot,
            log_history_state,
            workflow_html,
            current_process_log,
            conversation_debug_log,
            log_html_display,
            retrieval_results_html,
            citation_preview_html,
            history_manager_html,
            session_state_html,
            history_manager_status,
        ],
    ).then(
        fn=unlock_controls,
        inputs=None,
        outputs=busy_control_outputs,
        queue=False,
    )

    submit_btn.click(
        fn=lock_controls_for_chat,
        inputs=None,
        outputs=busy_control_outputs,
        queue=False,
    ).then(
        fn=process_and_update,
        inputs=[msg_input, chatbot, top_k_ret_slider, top_k_comp_slider, threshold_slider, multi_turn_toggle, log_history_state],
        outputs=[
            msg_input,
            chatbot,
            log_history_state,
            workflow_html,
            current_process_log,
            conversation_debug_log,
            log_html_display,
            retrieval_results_html,
            citation_preview_html,
            history_manager_html,
            session_state_html,
            history_manager_status,
        ],
    ).then(
        fn=unlock_controls,
        inputs=None,
        outputs=busy_control_outputs,
        queue=False,
    )

    clear_btn.click(
        fn=request_clear_conversation_confirmation,
        inputs=None,
        outputs=[clear_btn, clear_confirm_row],
        queue=False,
    )

    clear_cancel_btn.click(
        fn=cancel_clear_conversation_confirmation,
        inputs=None,
        outputs=[clear_btn, clear_confirm_row],
        queue=False,
    )

    clear_confirm_btn.click(
        fn=confirm_reset_conversation_session,
        inputs=None,
        outputs=[
            chatbot,
            log_history_state,
            history_editor,
            history_manager_html,
            session_state_html,
            workflow_html,
            current_process_log,
            conversation_debug_log,
            log_html_display,
            retrieval_results_html,
            citation_preview_html,
            history_manager_status,
            clear_btn,
            clear_confirm_row,
        ],
        queue=False,
    )

    load_history_btn.click(
        fn=load_conversation_manager_views,
        inputs=None,
        outputs=[chatbot, history_editor, history_manager_html, session_state_html, history_manager_status],
        queue=False,
    )

    save_history_btn.click(
        fn=save_history_edits,
        inputs=[history_editor],
        outputs=[chatbot, history_editor, history_manager_html, session_state_html, history_manager_status],
    )

    build_btn.click(
        fn=lock_controls_for_build,
        inputs=None,
        outputs=busy_control_outputs,
        queue=False,
    ).then(
        fn=build_knowledge_base_task_with_doc_preprocess,
        inputs=[dir_input, chunk_size_input, overlap_input],
        outputs=[build_status, build_log, knowledge_base_status_html, build_report_html],
    ).then(
        fn=unlock_controls,
        inputs=None,
        outputs=busy_control_outputs,
        queue=False,
    )

    refresh_status_btn.click(
        fn=refresh_knowledge_base_panels,
        inputs=None,
        outputs=[knowledge_base_status_html, build_report_html],
        queue=False,
    )


if False and __name__ == "__main__":
    print("系统启动中，请稍候...")
    try:
        demo.launch(server_name="0.0.0.0", server_port=7860, quiet=True)
    except TypeError:
        demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    print("系统启动中，请稍候...")
    local_url = "http://127.0.0.1:7860"
    try:
        launch_result = demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            quiet=True,
            prevent_thread_lock=True,
        )
    except TypeError:
        try:
            launch_result = demo.launch(
                server_name="0.0.0.0",
                server_port=7860,
                prevent_thread_lock=True,
            )
        except TypeError:
            launch_result = demo.launch(server_name="0.0.0.0", server_port=7860)

    if isinstance(launch_result, tuple) and len(launch_result) >= 2 and launch_result[1]:
        local_url = str(launch_result[1]).rstrip("/")
    elif getattr(demo, "local_url", None):
        local_url = str(demo.local_url).rstrip("/")

    print(f"启动成功，Web 界面地址：{local_url}")

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("程序已停止。")
