import asyncio
import contextlib
import html
import io
import json
import re
import sqlite3
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import gradio as gr

try:
    import simpleRAG_content
    from config import (
        CACHE_FILE,
        CHAT_MODEL,
        CHUNK_OVERLAP_DEFAULT,
        CHUNK_SIZE_DEFAULT,
        COMPRESSOR_MODEL,
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
        "Step 0:",
        "Step 1:",
        "Step 2:",
        "Step 3:",
        "Original question:",
        "Rewritten query:",
        "Vectorized query dimension:",
        "Reranker status:",
        "Compression model:",
        "Compressed context:",
        "Compression complete",
        "Answer generation model:",
        "Prompt:",
        "Final answer",
        "Diagnostics:",
        "[Notice]",
        "Threshold fallback kept only",
        "No relevant knowledge snippets were found.",
        "Knowledge base is empty or failed to load.",
        "Compressed context failed citation validation;",
        "Input snippet count:",
    ]
    if any(stripped.startswith(prefix) for prefix in prefixes):
        return True
    if re.match(r"^\[(Top\d+|Rank\d+|Initial Rank\d+)\]", stripped):
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


def escape_html(text):
    return html.escape(text).replace("\n", "<br>")


def generate_logs_html(log_list):
    if not log_list:
        return "<div style='color: #888; font-style: italic;'>No history yet.</div>"

    html_content = "<div class='log-container'>"
    for item in log_list:
        safe_label = escape_html(item["label"])
        safe_details = escape_html(item["details"])
        html_content += f"""
        <details style="border: 1px solid #e5e7eb; border-radius: 6px; margin-bottom: 8px; background: #f9fafb;">
            <summary style="padding: 10px; cursor: pointer; font-weight: 600; color: #374151; list-style: none; display: flex; justify-content: space-between; align-items: center;">
                <span>{safe_label}</span>
                <span style="font-size: 0.8em; color: #9ca3af;">view</span>
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
            f"{title}: no results yet.</div>"
        )

    cards = [f"<div style='font-weight: 700; margin-bottom: 10px;'>{html.escape(title)}</div>"]
    for index_id, item in enumerate(results, start=1):
        path = html.escape(str(item.get("path", "unknown")))
        chunk_index = html.escape(str(item.get("chunk_index", "unknown")))
        score = item.get(score_key)
        if isinstance(score, float):
            score_text = f"{score_key}={score:.4f}"
        else:
            score_text = f"{score_key}=n/a"
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
        f"{build_results_panel_html(retrieved_results, 'Retrieved Results', 'score')}"
        f"{build_results_panel_html(reranked_results, 'Reranked Results', 'rerank_score')}"
        "</div>"
    )


def generate_citation_preview_html(answer_text, source_results):
    source_results = source_results or []
    matches = re.findall(r"\[source=([^\]#]+)#chunk(\d+)\]", answer_text or "", flags=re.IGNORECASE)
    if not matches:
        return "<div style='color: #888; font-style: italic;'>No citations in the latest answer yet.</div>"

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
        snippet = html.escape(str(item.get("content", "Preview unavailable in current retrieval results.")) if item else "Preview unavailable in current retrieval results.")
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
        "<div style='font-weight: 700; margin-bottom: 8px;'>Citation Navigation</div>"
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
        "status_detail": "Awaiting a request.",
    }


def generate_system_workflow_html(workflow_state=None):
    state = workflow_state or create_workflow_state()
    status_map = {
        "idle": ("Idle", "#6b7280", "#f3f4f6"),
        "running": ("Running", "#9a3412", "#ffedd5"),
        "streaming": ("Generating", "#1d4ed8", "#dbeafe"),
        "completed": ("Completed", "#166534", "#dcfce7"),
        "no_results": ("No Match", "#92400e", "#fef3c7"),
        "kb_unavailable": ("Knowledge Base Unavailable", "#991b1b", "#fee2e2"),
        "error": ("Error", "#991b1b", "#fee2e2"),
    }
    status_label, status_text_color, status_bg = status_map.get(
        state.get("final_status", "idle"),
        status_map["idle"],
    )
    cache_label = {
        "pending": "Pending",
        "hit": "Hit",
        "miss": "Miss",
        "disabled": "Disabled",
    }.get(state.get("cache_status", "pending"), "Pending")
    compression_label = {
        "pending": "Pending",
        "compressed": "Compressed evidence ready",
        "raw_fallback": "Raw evidence fallback",
    }.get(state.get("compression_mode", "pending"), "Pending")
    rewrite_label = {
        "pending": "Pending",
        "used": "Context-aware rewrite applied",
        "skipped": "Rewrite skipped",
    }.get(state.get("rewrite_mode", "pending"), "Pending")
    conversation_mode = "Enabled" if state.get("multi_turn_enabled") else "Disabled"
    summary_label = "Enabled" if state.get("summary_used") else "Not used"
    citation_label = "Triggered" if state.get("citation_retry") else "Not needed"
    rewritten_query = truncate_text(state.get("rewritten_query", "") or state.get("question", ""), 140)

    cards = [
        (
            "Conversation Grounding",
            [
                f"Multi-turn context: {conversation_mode}",
                f"History messages considered: {state.get('history_messages', 0)}",
                f"Summary memory: {summary_label}",
                f"Rewrite strategy: {rewrite_label}",
                f"Resolved query: {rewritten_query or 'Pending'}",
            ],
        ),
        (
            "Evidence Pipeline",
            [
                f"Response cache: {cache_label}",
                f"Cache entries: {state.get('cache_entries', 0)}",
                f"Retrieved snippets: {state.get('retrieved_count', 0)}",
                f"Reranked evidence blocks: {state.get('reranked_count', 0)}",
                f"Reranker status: {state.get('reranker_status', 'pending')}",
            ],
        ),
        (
            "Answer Synthesis",
            [
                f"Compression path: {compression_label}",
                f"Citation recovery: {citation_label}",
                f"Current status: {status_label}",
                f"Operational note: {truncate_text(state.get('status_detail', ''), 120)}",
            ],
        ),
        (
            "Execution Profile",
            [
                f"Question: {truncate_text(state.get('question', ''), 110) or 'No active question'}",
                f"Retrieve count: {state.get('top_k_ret', '-')}",
                f"Rerank count: {state.get('top_k_comp', '-')}",
                f"Score threshold: {state.get('threshold', '-')}",
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
                <div style="font-size: 1.08em; font-weight: 800; color: #111827;">System Workflow</div>
                <div style="margin-top: 4px; color: #6b7280;">
                    A compact view of how the engine interprets the question, grounds evidence, and prepares the final answer.
                </div>
            </div>
            <div style="padding: 6px 12px; border-radius: 999px; background: {status_bg}; color: {status_text_color}; font-weight: 700;">
                {html.escape(status_label)}
            </div>
        </div>
        <div style="margin-top: 10px; color: #4b5563;">{html.escape(state.get("status_detail", "Awaiting a request."))}</div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-top: 14px;">
            {''.join(cards_html)}
        </div>
    </div>
    """


def generate_build_report_html(report=None):
    if not report:
        return "<div style='color: #888; font-style: italic;'>No build report yet.</div>"

    snapshot_label = "Snapshot updated" if report.get("snapshot_status") == "updated" else "Snapshot cleared"
    return f"""
    <div style="padding: 14px; border: 1px solid #e5e7eb; border-radius: 12px; background: #ffffff;">
        <div style="font-size: 1.05em; font-weight: 800; color: #111827;">Incremental Build Report</div>
        <div style="margin-top: 6px; color: #6b7280;">
            The knowledge base sync compared the current document folder against the persisted index and refreshed the live snapshot.
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin-top: 14px;">
            <div style="padding: 12px; border: 1px solid #e5e7eb; border-radius: 10px; background: #f9fafb;">
                <div style="font-weight: 700;">Scope</div>
                <div style="margin-top: 6px;">Source directory: {html.escape(str(report.get('source_dir', 'N/A')))}</div>
                <div>Discovered documents: {report.get('discovered_documents', 0)}</div>
                <div>Active documents: {report.get('active_documents', 0)}</div>
            </div>
            <div style="padding: 12px; border: 1px solid #e5e7eb; border-radius: 10px; background: #f9fafb;">
                <div style="font-weight: 700;">Document Sync</div>
                <div style="margin-top: 6px;">New documents: {report.get('new_documents', 0)}</div>
                <div>Refreshed documents: {report.get('refreshed_documents', 0)}</div>
                <div>Removed documents: {report.get('removed_documents', 0)}</div>
                <div>Skipped empty documents: {report.get('empty_documents', 0)}</div>
            </div>
            <div style="padding: 12px; border: 1px solid #e5e7eb; border-radius: 10px; background: #f9fafb;">
                <div style="font-weight: 700;">Chunk Sync</div>
                <div style="margin-top: 6px;">Newly written chunks: {report.get('written_chunks', 0)}</div>
                <div>Total indexed chunks: {report.get('total_chunks', 0)}</div>
                <div>Snapshot status: {html.escape(snapshot_label)}</div>
                <div>Build duration: {report.get('duration_seconds', 0)}s</div>
            </div>
        </div>
    </div>
    """


def apply_debug_event_to_workflow_state(workflow_state, event, content):
    if event == "cache_status" and isinstance(content, dict):
        workflow_state["cache_status"] = "hit" if content.get("hit") else "miss"
        workflow_state["cache_entries"] = int(content.get("entries", workflow_state.get("cache_entries", 0)))
        if content.get("hit"):
            workflow_state["status_detail"] = "A matching request was served from the response cache."
    elif event == "history_mode" and isinstance(content, dict):
        workflow_state["summary_used"] = bool(content.get("summary_used"))
        workflow_state["history_messages"] = int(content.get("history_messages", 0))
    elif event == "rewrite_mode":
        workflow_state["rewrite_mode"] = str(content or "pending")
    elif event == "rewritten_query":
        workflow_state["rewritten_query"] = str(content or "")
    elif event == "retrieved_results" and isinstance(content, list):
        workflow_state["retrieved_count"] = len(content)
        if workflow_state["cache_status"] != "hit":
            workflow_state["status_detail"] = f"Retrieved {len(content)} candidate evidence snippets."
    elif event == "reranked_results" and isinstance(content, list):
        workflow_state["reranked_count"] = len(content)
        if workflow_state["cache_status"] != "hit":
            workflow_state["status_detail"] = f"Condensed the evidence set to {len(content)} high-confidence snippets."


def apply_process_log_to_workflow_state(workflow_state, text):
    stripped = str(text or "").strip()
    if not stripped:
        return
    if stripped == "Start retrieval":
        workflow_state["final_status"] = "running"
        workflow_state["status_detail"] = "Launching the retrieval and reasoning workflow."
    elif stripped.startswith("Step 0:"):
        workflow_state["status_detail"] = "Interpreting the request and resolving conversational references."
    elif stripped.startswith("Step 1: vectorize"):
        workflow_state["status_detail"] = "Converting the resolved query into vector space."
    elif stripped.startswith("Step 1: reuse cached response package"):
        workflow_state["final_status"] = "completed"
        workflow_state["status_detail"] = "Reused a cached response package for an identical request."
    elif stripped.startswith("Step 2:"):
        workflow_state["status_detail"] = "Searching the knowledge base for relevant evidence."
    elif stripped.startswith("Step 3:"):
        workflow_state["status_detail"] = "Reranking and refining the evidence set."
    elif stripped.startswith("Reranker status:"):
        workflow_state["reranker_status"] = stripped.split(":", 1)[1].strip()
    elif stripped.startswith("Compression model:"):
        workflow_state["compression_mode"] = "compressed"
        workflow_state["status_detail"] = "Compressing overlapping evidence into a concise context pack."
    elif stripped.startswith("Compressed context failed citation validation;"):
        workflow_state["compression_mode"] = "raw_fallback"
        workflow_state["status_detail"] = "Switched to raw evidence because the compressed version lost citation fidelity."
    elif stripped == "Final answer":
        workflow_state["final_status"] = "streaming"
        workflow_state["status_detail"] = "Synthesizing the final answer with grounded citations."
    elif stripped.startswith("[Notice] Answer citations were incomplete."):
        workflow_state["citation_retry"] = True
        workflow_state["status_detail"] = "Answer citations were retried against raw evidence."
    elif stripped.startswith("No relevant knowledge snippets were found."):
        workflow_state["final_status"] = "no_results"
        workflow_state["status_detail"] = "No evidence passed the retrieval threshold for this question."
    elif stripped.startswith("Knowledge base is empty or failed to load.") or stripped.startswith("Knowledge base failed to load:"):
        workflow_state["final_status"] = "kb_unavailable"
        workflow_state["status_detail"] = "The knowledge base snapshot is unavailable or could not be loaded."
    elif stripped.startswith("Diagnostics:"):
        workflow_state["final_status"] = "completed"
        workflow_state["status_detail"] = "The end-to-end workflow completed successfully."
        reranker_match = re.search(r"reranker=([^,]+)", stripped)
        retrieval_match = re.search(r"retrieval_fallback=([^,]+)", stripped)
        citation_match = re.search(r"citation_retry=([^\s]+)", stripped)
        if reranker_match:
            workflow_state["reranker_status"] = reranker_match.group(1).strip()
        if citation_match:
            workflow_state["citation_retry"] = citation_match.group(1).strip().lower() == "true"
        if retrieval_match and retrieval_match.group(1).strip().lower() == "true":
            workflow_state["status_detail"] = "The workflow completed successfully after a conservative retrieval fallback."
    elif stripped.startswith("[Error]") or stripped.startswith("Error:"):
        workflow_state["final_status"] = "error"
        workflow_state["status_detail"] = stripped


def _format_timestamp(path_obj):
    if not path_obj.exists():
        return "N/A"
    return datetime.fromtimestamp(path_obj.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def generate_knowledge_base_status_html():
    vector_count = "0"
    metadata_count = "0"
    document_count = "0"
    db_rows = "0"
    status_message = "Knowledge base files not found."

    if FAISS_INDEX_FILE.exists() and METADATA_FILE.exists():
        try:
            index, metadata = RAGHelpers.load_faiss_index_and_metadata(FAISS_INDEX_FILE, METADATA_FILE)
            vector_count = str(index.ntotal if index is not None else 0)
            metadata_count = str(len(metadata))
            status_message = "Knowledge base snapshot is available."
        except Exception as exc:
            status_message = f"Knowledge base snapshot failed to load: {html.escape(str(exc))}"

    if DB_PATH.exists():
        try:
            conn = sqlite3.connect(DB_PATH)
            try:
                db_rows = str(conn.execute("SELECT COUNT(*) FROM metadata").fetchone()[0])
                document_count = str(conn.execute("SELECT COUNT(DISTINCT path) FROM metadata").fetchone()[0])
            finally:
                conn.close()
        except Exception as exc:
            status_message += f" SQLite read warning: {html.escape(str(exc))}"

    cache_entries = "0"
    if CACHE_FILE.exists():
        try:
            cache_data = RAGHelpers.load_embedding_cache(CACHE_FILE)
            cache_entries = str(max(0, len(cache_data) - (1 if "__model_name__" in cache_data else 0)))
        except Exception:
            cache_entries = "unknown"

    query_cache_entries = "0"
    summary_cache_entries = "0"
    runtime_rag = globals().get("rag_instance")
    if runtime_rag is not None:
        try:
            runtime_cache_metrics = runtime_rag.get_runtime_cache_metrics()
            query_cache_entries = str(runtime_cache_metrics.get("query_cache_entries", 0))
            summary_cache_entries = str(runtime_cache_metrics.get("summary_cache_entries", 0))
        except Exception:
            query_cache_entries = "unknown"
            summary_cache_entries = "unknown"

    return f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px;">
        <div style="padding: 12px; border: 1px solid #e5e7eb; border-radius: 8px; background: #fff;">
            <div style="font-weight: 700; margin-bottom: 6px;">Knowledge Base Status</div>
            <div>{html.escape(status_message)}</div>
            <div style="color: #6b7280; margin-top: 6px;">Last index update: {html.escape(_format_timestamp(FAISS_INDEX_FILE))}</div>
        </div>
        <div style="padding: 12px; border: 1px solid #e5e7eb; border-radius: 8px; background: #fff;">
            <div style="font-weight: 700; margin-bottom: 6px;">Counts</div>
            <div>Vectors: {vector_count}</div>
            <div>Metadata entries: {metadata_count}</div>
            <div>DB rows: {db_rows}</div>
            <div>Distinct documents: {document_count}</div>
            <div>Embedding cache entries: {cache_entries}</div>
            <div>Query cache entries: {query_cache_entries}</div>
            <div>Summary cache entries: {summary_cache_entries}</div>
        </div>
        <div style="padding: 12px; border: 1px solid #e5e7eb; border-radius: 8px; background: #fff;">
            <div style="font-weight: 700; margin-bottom: 6px;">Models</div>
            <div>Chat: {html.escape(CHAT_MODEL)}</div>
            <div>Compressor: {html.escape(COMPRESSOR_MODEL)}</div>
            <div>Embedding: {html.escape(EMBEDDING_MODEL)}</div>
            <div>Reranker: {html.escape(RERANK_MODEL)}</div>
        </div>
    </div>
    """


def apply_parameter_preset(preset_name):
    presets = {
        "Balanced (Default)": (DEFAULT_TOP_K, DEFAULT_TOP_K_COMPRESSED, DEFAULT_THRESHOLD),
        "Fast": (3, 2, 0.45),
        "Deep": (8, 5, 0.2),
    }
    return presets.get(
        preset_name,
        (DEFAULT_TOP_K, DEFAULT_TOP_K_COMPRESSED, DEFAULT_THRESHOLD),
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
    logs = logger.write_log(f"Scanning source directory: {source_dir}")

    if not source_dir.exists() or not source_dir.is_dir():
        logs += logger.write_log(f"Directory does not exist or is not a folder: {source_dir}")
        return (
            "Failed",
            clean_log_text(logs),
            generate_knowledge_base_status_html(),
            generate_build_report_html(get_last_build_report()),
        )

    try:
        progress(0, desc="Preparing build...")
        logs += logger.write_log("Initializing RAG engine...")
        rag = get_rag_instance()
        logs += logger.write_log("RAG engine ready.")
        progress(0.05, desc="RAG engine initialized")

        async def run_build():
            return await rag.build_knowledge_base_async(
                source_dir,
                chunk_size=int(chunk_size),
                overlap=int(overlap),
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
            count = index.ntotal if index is not None else "unknown"
            logs += logger.write_log(f"Build complete. Total vectors: {count}")
            logs += logger.write_log("Index files were saved locally.")
        except Exception:
            logs += logger.write_log("Build finished.")

        return (
            "Success",
            clean_log_text(logs),
            generate_knowledge_base_status_html(),
            generate_build_report_html(build_report),
        )
    except Exception as exc:
        logs += logger.write_log(f"Build failed: {exc}")
        return (
            "Failed",
            clean_log_text(logs),
            generate_knowledge_base_status_html(),
            generate_build_report_html(get_last_build_report()),
        )


def answer_question_task(question, history, top_k_ret, top_k_comp, threshold, multi_turn_enabled, log_history_state):
    if not isinstance(history, list):
        history = []
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

    def emit(history_value, log_state_value, answer_text):
        return (
            history_value,
            log_state_value,
            generate_system_workflow_html(workflow_state),
            clean_log_text(current_q_logs),
            "\n\n".join(debug_lines),
            generate_logs_html(log_state_value),
            generate_retrieval_html(latest_retrieved_results, latest_reranked_results),
            generate_citation_preview_html(answer_text, latest_reranked_results or latest_retrieved_results),
        )

    if not question:
        current_q_logs = logger.write_log("Error: question cannot be empty.")
        workflow_state["final_status"] = "error"
        workflow_state["status_detail"] = "The request was empty, so the workflow did not start."
        yield emit(history, log_history_state, "")
        return

    current_q_logs = logger.write_log(f"Processing new question: {question}")
    current_q_logs += logger.write_log(
        f"Parameters -> retrieve_k={top_k_ret}, compressed_k={top_k_comp}, threshold={threshold}"
    )
    current_q_logs += logger.write_log(f"Multi-turn context enabled: {bool(multi_turn_enabled)}")

    workflow_state["final_status"] = "running"
    workflow_state["status_detail"] = "Request received. Preparing the retrieval workflow."
    new_history = history + [{"role": "user", "content": question}]
    new_history.append({"role": "assistant", "content": "Searching and reasoning..."})
    yield emit(new_history, log_history_state, "")

    full_answer = ""

    try:
        k_ret = int(top_k_ret) if top_k_ret is not None else DEFAULT_TOP_K
        k_comp = int(top_k_comp) if top_k_comp is not None else DEFAULT_TOP_K_COMPRESSED
        thresh = float(threshold) if threshold is not None else DEFAULT_THRESHOLD
        workflow_state["top_k_ret"] = k_ret
        workflow_state["top_k_comp"] = k_comp
        workflow_state["threshold"] = thresh

        current_q_logs += logger.write_log("Initializing RAG engine...")
        workflow_state["status_detail"] = "Initializing the RAG runtime."
        yield emit(new_history, log_history_state, full_answer)

        rag = get_rag_instance()
        current_q_logs += logger.write_log("RAG engine ready.")
        workflow_state["status_detail"] = "RAG runtime initialized. Starting evidence retrieval."
        yield emit(new_history, log_history_state, full_answer)

        effective_history = history if multi_turn_enabled else []
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
                    debug_lines.append(f"[{event}]\n{format_debug_content(content)}")
                apply_debug_event_to_workflow_state(workflow_state, event, content)
                yield emit(new_history, log_history_state, full_answer)
                continue

            stripped_chunk = chunk.strip()
            if is_process_log(chunk):
                apply_process_log_to_workflow_state(workflow_state, stripped_chunk)
                if stripped_chunk in {
                    "No relevant knowledge snippets were found.",
                    "Knowledge base is empty or failed to load.",
                } or stripped_chunk.startswith("Knowledge base failed to load:"):
                    full_answer = stripped_chunk
                    new_history[-1]["content"] = full_answer
                elif stripped_chunk.startswith("[Error]"):
                    full_answer = stripped_chunk
                    new_history[-1]["content"] = full_answer
                current_q_logs += chunk + "\n"
                yield emit(new_history, log_history_state, full_answer)
            else:
                if chunk == simpleRAG_content.SimpleRAG.ANSWER_REPLACE_MARKER:
                    full_answer = ""
                    new_history[-1]["content"] = ""
                    workflow_state["citation_retry"] = True
                    workflow_state["status_detail"] = "Refreshing the answer against raw cited evidence."
                    yield emit(new_history, log_history_state, full_answer)
                    continue
                full_answer += chunk
                new_history[-1]["content"] = full_answer
                workflow_state["final_status"] = "streaming"
                workflow_state["status_detail"] = "Streaming the grounded answer to the interface."
                yield emit(new_history, log_history_state, full_answer)

        if workflow_state["final_status"] == "running":
            workflow_state["final_status"] = "completed"
            workflow_state["status_detail"] = "The workflow completed successfully."

        current_q_logs += logger.write_log("Answer generation completed.")
        log_entry_label = f"{question[:30]}{'...' if len(question) > 30 else ''} ({time.strftime('%H:%M:%S')})"
        new_log_history = [{"label": log_entry_label, "details": clean_log_text(current_q_logs)}] + log_history_state
        if len(new_log_history) > 20:
            new_log_history = new_log_history[:20]

        yield emit(new_history, new_log_history, full_answer)
    except Exception as exc:
        error_msg = str(exc)
        current_q_logs += logger.write_log(f"Error: {error_msg}")
        workflow_state["final_status"] = "error"
        workflow_state["status_detail"] = f"Request failed: {error_msg}"
        new_history[-1]["content"] = f"{full_answer}\n\n[System Error]: {error_msg}"
        log_entry_label = f"{question[:30]}... (Error) ({time.strftime('%H:%M:%S')})"
        new_log_history = [{"label": log_entry_label, "details": clean_log_text(current_q_logs)}] + log_history_state
        yield emit(new_history, new_log_history, full_answer)


with gr.Blocks(title="Lightweight RAG", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Lightweight RAG")

    with gr.Tabs():
        with gr.TabItem("Chat"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=500,
                        show_copy_button=True,
                        type="messages",
                        value=[],
                    )
                    with gr.Row():
                        msg_input = gr.Textbox(
                            label="Question",
                            placeholder="Ask a question...",
                            scale=4,
                            container=False,
                        )
                        submit_btn = gr.Button("Send", variant="primary", scale=1)
                    clear_btn = gr.Button("Clear conversation", size="sm")

                with gr.Column(scale=1):
                    gr.Markdown("### Retrieval Settings")
                    preset_dropdown = gr.Dropdown(
                        choices=["Balanced (Default)", "Fast", "Deep"],
                        value="Balanced (Default)",
                        label="Parameter preset",
                    )
                    top_k_ret_slider = gr.Slider(1, 50, value=DEFAULT_TOP_K, step=1, label="Retrieve count")
                    top_k_comp_slider = gr.Slider(1, 20, value=DEFAULT_TOP_K_COMPRESSED, step=1, label="Rerank count")
                    threshold_slider = gr.Slider(0.0, 1.0, value=DEFAULT_THRESHOLD, step=0.05, label="Score threshold")
                    multi_turn_toggle = gr.Checkbox(label="Enable multi-turn context", value=True)

            gr.Markdown("### Processing History")
            log_history_state = gr.State([])
            log_html_display = gr.HTML(
                value="<div style='color: #888; font-style: italic;'>No history yet.</div>",
                label="History list",
            )
            current_process_log = gr.Textbox(
                label="Current process log",
                lines=5,
                max_lines=8,
                interactive=False,
                visible=True,
            )
            conversation_debug_log = gr.Textbox(
                label="Multi-turn debug",
                lines=8,
                max_lines=14,
                interactive=False,
                visible=True,
            )
            retrieval_results_html = gr.HTML(
                value=generate_retrieval_html([], []),
                label="Retrieval results",
            )
            citation_preview_html = gr.HTML(
                value=generate_citation_preview_html("", []),
                label="Citation preview",
            )

        with gr.TabItem("Build Knowledge Base"):
            gr.Markdown("### Build / Rebuild Index")
            knowledge_base_status_html = gr.HTML(
                value=generate_knowledge_base_status_html(),
                label="Knowledge base status",
            )
            with gr.Row():
                with gr.Column():
                    dir_input = gr.Textbox(label="Document directory", value=str(DOC_DIR))
                    chunk_size_input = gr.Number(label="Chunk size", value=CHUNK_SIZE_DEFAULT, precision=0)
                    overlap_input = gr.Number(label="Chunk overlap", value=CHUNK_OVERLAP_DEFAULT, precision=0)
                    build_btn = gr.Button("Build index", variant="primary")
                    refresh_status_btn = gr.Button("Refresh knowledge base status")
                with gr.Column():
                    build_status = gr.Textbox(label="Build status", interactive=False)
                    build_log = gr.Textbox(label="Build logs", lines=15, interactive=False)

    def process_and_update(question, history, k1, k2, th, multi_turn_enabled, log_state):
        yield from answer_question_task(question, history, k1, k2, th, multi_turn_enabled, log_state)

    preset_dropdown.change(
        fn=apply_parameter_preset,
        inputs=[preset_dropdown],
        outputs=[top_k_ret_slider, top_k_comp_slider, threshold_slider],
    )

    msg_input.submit(
        fn=process_and_update,
        inputs=[msg_input, chatbot, top_k_ret_slider, top_k_comp_slider, threshold_slider, multi_turn_toggle, log_history_state],
        outputs=[chatbot, log_history_state, current_process_log, conversation_debug_log, log_html_display, retrieval_results_html, citation_preview_html],
    )

    submit_btn.click(
        fn=process_and_update,
        inputs=[msg_input, chatbot, top_k_ret_slider, top_k_comp_slider, threshold_slider, multi_turn_toggle, log_history_state],
        outputs=[chatbot, log_history_state, current_process_log, conversation_debug_log, log_html_display, retrieval_results_html, citation_preview_html],
    )

    clear_btn.click(
        lambda: (
            [],
            [],
            "",
            "",
            "<div style='color: #888; font-style: italic;'>No history yet.</div>",
            generate_retrieval_html([], []),
            generate_citation_preview_html("", []),
        ),
        None,
        [chatbot, log_history_state, current_process_log, conversation_debug_log, log_html_display, retrieval_results_html, citation_preview_html],
        queue=False,
    )

    build_btn.click(
        fn=build_knowledge_base_task,
        inputs=[dir_input, chunk_size_input, overlap_input],
        outputs=[build_status, build_log, knowledge_base_status_html],
    )

    refresh_status_btn.click(
        fn=generate_knowledge_base_status_html,
        inputs=None,
        outputs=[knowledge_base_status_html],
        queue=False,
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
