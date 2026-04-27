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
import uuid
import warnings
from datetime import datetime
from pathlib import Path

from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    render_template,
    request,
    send_from_directory,
    stream_with_context,
)

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("NO_COLOR", "1")

logging.getLogger().setLevel(logging.WARNING)
for noisy_logger_name in (
    "httpx",
    "httpcore",
    "urllib3",
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

try:
    import simpleRAG_content
    from config import (
        CACHE_FILE,
        CHAT_MODEL,
        CHUNK_OVERLAP_DEFAULT,
        CHUNK_SIZE_DEFAULT,
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
        "Retrieval query:",
        "Retrieval context prepared:",
        "用于生成的上下文长度：",
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


def _to_project_relative_path(path_text):
    raw = str(path_text or "").strip()
    if not raw:
        return ""
    try:
        project_root = Path(__file__).resolve().parent
        path_obj = Path(raw).resolve()
        rel_path = path_obj.relative_to(project_root)
        return str(rel_path).replace("\\", "/")
    except Exception:
        return raw.replace("\\", "/")


def _extract_citation_keys(answer_text):
    matches = re.findall(r"source=([^\],#]+)#chunk(\d+)", str(answer_text or ""), flags=re.IGNORECASE)
    keys = set()
    for path, chunk_index in matches:
        rel_path = _to_project_relative_path(path).lower()
        keys.add((rel_path, str(chunk_index).strip()))
    return keys


def _compute_online_eval_snapshot(question, answer_text, retrieved_results, reranked_results, elapsed_ms):
    retrieved_results = retrieved_results or []
    reranked_results = reranked_results or []
    answer_text = str(answer_text or "")

    retrieved_count = len(retrieved_results)
    reranked_count = len(reranked_results)
    answer_length = len(answer_text.strip())

    citation_keys = _extract_citation_keys(answer_text)
    source_keys = set()
    source_pool = reranked_results if reranked_results else retrieved_results
    for item in source_pool:
        rel_path = _to_project_relative_path(item.get("path", "")).lower()
        chunk_index = str(item.get("chunk_index", "")).strip()
        if rel_path and chunk_index:
            source_keys.add((rel_path, chunk_index))

    matched_citations = len(citation_keys.intersection(source_keys))
    citation_total = len(citation_keys)
    citation_coverage = (matched_citations / citation_total) if citation_total > 0 else 0.0
    compression_ratio = (reranked_count / float(retrieved_count)) if retrieved_count > 0 else 0.0





    hit_rate_at_k = 1.0 if matched_citations > 0 else 0.0
    recall_at_k = min(1.0, (matched_citations / float(retrieved_count))) if retrieved_count > 0 else 0.0
    precision_at_k = min(1.0, (matched_citations / float(reranked_count))) if reranked_count > 0 else 0.0

    latency_s = float(elapsed_ms or 0.0) / 1000.0
    latency_score = 1.0 / (1.0 + (latency_s / 30.0))
    overall_score = (
        0.30 * hit_rate_at_k
        + 0.25 * recall_at_k
        + 0.15 * precision_at_k
        + 0.15 * citation_coverage
        + 0.15 * latency_score
    ) * 100.0

    return {
        "question": str(question or "").strip(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "overall_score": round(overall_score, 1),
        "latency_seconds": round(latency_s, 2),
        "retrieved_count": retrieved_count,
        "reranked_count": reranked_count,
        "hit_rate_at_k": round(hit_rate_at_k, 4),
        "recall_at_k": round(recall_at_k, 4),
        "precision_at_k": round(precision_at_k, 4),
        "citation_coverage": round(citation_coverage, 4),
        "compression_ratio": round(compression_ratio, 4),
        "answer_length": answer_length,
    }

def generate_online_eval_html(snapshot):
    if not snapshot:
        return "<div style='color: #888; font-style: italic;'>暂无评估结果，请先发起一次问答。</div>"

    question_preview = html.escape(truncate_text(snapshot.get("question", ""), 80))
    return f"""
    <div style="border: 1px solid #e5e7eb; border-radius: 10px; padding: 12px; background: #fff;">
        <div style="font-size:12px;color:#6b7280;margin-bottom:8px;">{html.escape(snapshot.get("timestamp", ""))}</div>
        <div style="font-size:12px;color:#6b7280;margin-bottom:10px;">\u672c\u8f6e\u95ee\u9898\uff1a{question_preview}</div>
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:8px;">
            <div><strong>综合得分</strong><br>{snapshot.get("overall_score", 0):.1f}</div>
            <div><strong>端到端时延(秒)</strong><br>{snapshot.get("latency_seconds", 0):.2f}</div>
            <div><strong>检索候选数</strong><br>{int(snapshot.get("retrieved_count", 0))}</div>
            <div><strong>重排保留数</strong><br>{int(snapshot.get("reranked_count", 0))}</div>
            <div><strong>召回率@K</strong><br>{snapshot.get("recall_at_k", 0):.4f}</div>
            <div><strong>精确率@K</strong><br>{snapshot.get("precision_at_k", 0):.4f}</div>
            <div><strong>命中率@K</strong><br>{snapshot.get("hit_rate_at_k", 0):.4f}</div>
            <div><strong>引用覆盖率</strong><br>{snapshot.get("citation_coverage", 0):.4f}</div>
            <div><strong>重排压缩率</strong><br>{snapshot.get("compression_ratio", 0):.4f}</div>
            <div><strong>回答长度</strong><br>{int(snapshot.get("answer_length", 0))}</div>
        </div>
    </div>
    """

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
            <div>嵌入模型：{html.escape(EMBEDDING_MODEL)}</div>
            <div>重排模型：{html.escape(RERANK_MODEL)}</div>
        </div>
    </div>
    """


conversation_store = ConversationStore(CONVERSATION_STATE_FILE)
ANSWER_REPLACE_MARKER = getattr(simpleRAG_content.SimpleRAG, "ANSWER_REPLACE_MARKER", None)

def _get_conversation_state():
    return conversation_store.get_state()


def _get_conversation_messages():
    return _get_conversation_state().get("messages", [])


def _set_active_chat_request_id(request_id):
    global active_chat_request_id
    with active_chat_request_lock:
        active_chat_request_id = str(request_id or "").strip()


def _is_request_id_current(request_id):
    with active_chat_request_lock:
        current_request_id = active_chat_request_id
    return bool(request_id) and current_request_id == str(request_id).strip()


def _is_request_session_current(session_id):
    current_session_id = str(_get_conversation_state().get("active_session_id", "")).strip()
    return bool(session_id) and current_session_id == session_id


def _is_request_current(session_id, request_id):
    return _is_request_session_current(session_id) and _is_request_id_current(request_id)


def _should_persist_answer(answer_text):
    return bool(str(answer_text or "").strip()) and not simpleRAG_content.is_non_persistent_assistant_message(answer_text)


def reset_conversation_session():
    return conversation_store.reset_session()


class OutputLogger:
    def write_log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        return f"[{timestamp}] {message}\n"


logger = OutputLogger()
rag_instance = None
rag_instance_lock = threading.Lock()
active_chat_request_id = ""
active_chat_request_lock = threading.Lock()


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


def build_knowledge_base_task_with_doc_preprocess(
    source_dir_str,
    chunk_size,
    overlap,
    progress=None,
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
        if callable(progress):
            progress(0, desc="准备构建中...")
        logs += logger.write_log("正在检查并预处理 .doc 文件...")
        preprocess_report = preprocess_doc_files_for_build(
            source_dir,
            skip_backup=True,
            allow_install=False,
        )
        logs, has_preprocess_warning = _append_doc_preprocess_logs(logs, preprocess_report)
        if callable(progress):
            progress(0.08, desc="文档预处理完成")

        logs += logger.write_log("正在初始化 RAG 引擎...")
        rag = get_rag_instance()
        logs += logger.write_log("RAG 引擎已就绪。")
        if callable(progress):
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


def answer_question_task(question, history, top_k_ret, top_k_comp, threshold, multi_turn_enabled, log_history_state, request_id=None):
    persisted_history = _get_conversation_messages()
    request_session_id = str(_get_conversation_state().get("active_session_id", "")).strip()
    request_id = str(request_id or uuid.uuid4()).strip()
    _set_active_chat_request_id(request_id)
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
    request_started_at = time.perf_counter()
    final_online_eval_snapshot = None

    def emit(history_value, log_state_value, answer_text):
        if not _is_request_current(request_session_id, request_id):
            return None
        return (
            "",
            history_value,
            log_state_value,
            generate_system_workflow_html(workflow_state),
            clean_log_text(current_q_logs),
            generate_logs_html(log_state_value),
            generate_retrieval_html(latest_retrieved_results, latest_reranked_results),
            generate_online_eval_html(final_online_eval_snapshot),
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
                if ANSWER_REPLACE_MARKER is not None and chunk == ANSWER_REPLACE_MARKER:
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
        if not _is_request_current(request_session_id, request_id):
            return

        if _should_persist_answer(full_answer):
            final_history = persisted_history + [
                {"role": "user", "content": question},
                {"role": "assistant", "content": full_answer},
            ]
            conversation_store.set_messages(final_history)
            display_history = final_history
        else:
            display_history = new_history
        log_entry_label = f"{question[:30]}{'...' if len(question) > 30 else ''} ({time.strftime('%H:%M:%S')})"
        new_log_history = [{"label": log_entry_label, "details": clean_log_text(current_q_logs)}] + log_history_state
        if len(new_log_history) > 20:
            new_log_history = new_log_history[:20]

        elapsed_ms = (time.perf_counter() - request_started_at) * 1000.0
        final_online_eval_snapshot = _compute_online_eval_snapshot(
            question,
            full_answer,
            latest_retrieved_results,
            latest_reranked_results,
            elapsed_ms,
        )
        payload = emit(display_history, new_log_history, full_answer)
        if payload is not None:
            yield payload
    except Exception as exc:
        error_msg = str(exc)
        current_q_logs += logger.write_log(f"错误：{error_msg}")
        workflow_state["final_status"] = "error"
        workflow_state["status_detail"] = f"请求失败：{error_msg}"
        new_history[-1]["content"] = f"{full_answer}\n\n[系统错误]：{error_msg}"
        log_entry_label = f"{question[:30]}...（错误）({time.strftime('%H:%M:%S')})"
        new_log_history = [{"label": log_entry_label, "details": clean_log_text(current_q_logs)}] + log_history_state
        elapsed_ms = (time.perf_counter() - request_started_at) * 1000.0
        final_online_eval_snapshot = _compute_online_eval_snapshot(
            question,
            full_answer,
            latest_retrieved_results,
            latest_reranked_results,
            elapsed_ms,
        )
        payload = emit(new_history, new_log_history, full_answer)
        if payload is not None:
            yield payload

ROOT_DIR = Path(__file__).resolve().parent
ASSET_FILES = {"lightweightrag.css", "lightweightrag.js"}

PRESET_OPTIONS = {
    "均衡（默认）": {
        "top_k_ret": DEFAULT_TOP_K,
        "top_k_comp": DEFAULT_TOP_K_COMPRESSED,
        "threshold": DEFAULT_THRESHOLD,
    },
    "快速": {
        "top_k_ret": 3,
        "top_k_comp": 2,
        "threshold": 0.45,
    },
    "深入": {
        "top_k_ret": 8,
        "top_k_comp": 5,
        "threshold": 0.2,
    },
}


def _build_initial_page_state():
    return {
        "page_title": "RAG/检索增强技术问答系统",
        "preset_default": "均衡（默认）",
        "top_k_ret_default": DEFAULT_TOP_K,
        "top_k_comp_default": DEFAULT_TOP_K_COMPRESSED,
        "threshold_default": DEFAULT_THRESHOLD,
        "multi_turn_default": True,
        "chatbot": _get_conversation_messages(),
        "log_history_state": [],
        "workflow_html": generate_system_workflow_html(),
        "current_process_log": "",
        "log_html_display": generate_logs_html([]),
        "retrieval_results_html": generate_retrieval_html([], []),
        "online_eval_html": generate_online_eval_html(None),
        "knowledge_base_status_html": generate_knowledge_base_status_html(),
        "build_report_html": generate_build_report_html(get_last_build_report()),
        "doc_dir": str(DOC_DIR),
        "chunk_size_default": CHUNK_SIZE_DEFAULT,
        "chunk_overlap_default": CHUNK_OVERLAP_DEFAULT,
        "build_status": "",
        "build_log": "",
    }


def _tuple_to_chat_payload(payload_tuple):
    if not isinstance(payload_tuple, (list, tuple)) or len(payload_tuple) < 8:
        raise ValueError("Unexpected chat payload format.")

    return {
        "chatbot": payload_tuple[1],
        "log_history_state": payload_tuple[2],
        "workflow_html": payload_tuple[3],
        "current_process_log": payload_tuple[4],
        "log_html_display": payload_tuple[5],
        "retrieval_results_html": payload_tuple[6],
        "online_eval_html": payload_tuple[7],
    }


def _safe_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value, default=True):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    return default


def _chat_stream_generator(payload):
    question = payload.get("question", "")
    history = payload.get("history", _get_conversation_messages())
    log_history_state = payload.get("log_history_state", [])

    top_k_ret = _safe_int(payload.get("top_k_ret"), DEFAULT_TOP_K)
    top_k_comp = _safe_int(payload.get("top_k_comp"), DEFAULT_TOP_K_COMPRESSED)
    threshold = _safe_float(payload.get("threshold"), DEFAULT_THRESHOLD)
    multi_turn_enabled = _safe_bool(payload.get("multi_turn_enabled"), True)
    request_id = str(payload.get("request_id", "")).strip() or str(uuid.uuid4())

    try:
        for update in answer_question_task(
            question,
            history,
            top_k_ret,
            top_k_comp,
            threshold,
            multi_turn_enabled,
            log_history_state,
            request_id=request_id,
        ):
            if update is None:
                continue
            yield json.dumps(
                {"type": "update", "data": _tuple_to_chat_payload(update)},
                ensure_ascii=False,
            ) + "\n"
    except Exception as exc:
        yield json.dumps({"type": "error", "error": str(exc)}, ensure_ascii=False) + "\n"


app = Flask(__name__, template_folder=str(ROOT_DIR))


@app.get("/")
def index_page():
    return render_template(
        "index.html",
        initial_state=_build_initial_page_state(),
        presets=PRESET_OPTIONS,
    )


@app.get("/assets/<path:filename>")
def serve_asset(filename):
    if filename not in ASSET_FILES:
        abort(404)
    return send_from_directory(ROOT_DIR, filename)


@app.post("/api/chat/stream")
def chat_stream_api():
    payload = request.get_json(silent=True) or {}
    return Response(
        stream_with_context(_chat_stream_generator(payload)),
        mimetype="application/x-ndjson; charset=utf-8",
        headers={"Cache-Control": "no-cache, no-transform"},
    )


@app.post("/api/conversation/clear")
def clear_conversation_api():
    reset_conversation_session()
    return jsonify(
        {
            "chatbot": _get_conversation_messages(),
            "log_history_state": [],
            "workflow_html": generate_system_workflow_html(),
            "current_process_log": "",
            "log_html_display": generate_logs_html([]),
            "retrieval_results_html": generate_retrieval_html([], []),
            "online_eval_html": generate_online_eval_html(None),
        }
    )


@app.get("/api/knowledge-base/panels")
def knowledge_base_panels_api():
    knowledge_base_status_html, build_report_html = refresh_knowledge_base_panels()
    return jsonify(
        {
            "knowledge_base_status_html": knowledge_base_status_html,
            "build_report_html": build_report_html,
        }
    )


@app.post("/api/knowledge-base/build")
def knowledge_base_build_api():
    payload = request.get_json(silent=True) or {}
    source_dir = payload.get("source_dir", str(DOC_DIR))
    chunk_size = payload.get("chunk_size", CHUNK_SIZE_DEFAULT)
    overlap = payload.get("overlap", CHUNK_OVERLAP_DEFAULT)

    build_status, build_log, knowledge_base_status_html, build_report_html = build_knowledge_base_task_with_doc_preprocess(
        source_dir,
        chunk_size,
        overlap,
    )

    return jsonify(
        {
            "build_status": build_status,
            "build_log": build_log,
            "knowledge_base_status_html": knowledge_base_status_html,
            "build_report_html": build_report_html,
        }
    )


if __name__ == "__main__":
    host = "0.0.0.0"
    port = 7860
    local_url = f"http://127.0.0.1:{port}"
    print("系统启动中，请稍候...")
    print(f"启动成功，Web 界面地址：{local_url}")
    app.run(host=host, port=port, debug=False, use_reloader=False)
