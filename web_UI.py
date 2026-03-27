import asyncio
import contextlib
import html
import io
import json
import re
import sys
import threading
import time
from pathlib import Path

import gradio as gr

try:
    import simpleRAG_content
    from config import (
        CHUNK_OVERLAP_DEFAULT,
        CHUNK_SIZE_DEFAULT,
        DEFAULT_THRESHOLD,
        DEFAULT_TOP_K,
        DEFAULT_TOP_K_COMPRESSED,
        DOC_DIR,
        FAISS_INDEX_FILE,
        METADATA_FILE,
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

    if not source_dir.exists():
        return "Failed", logs + logger.write_log(f"Directory does not exist: {source_dir}")

    try:
        progress(0, desc="Preparing build...")
        logs += logger.write_log("Initializing RAG engine...")
        rag = get_rag_instance()
        logs += logger.write_log("RAG engine ready.")
        progress(0.05, desc="RAG engine initialized")

        async def run_build():
            await rag.build_knowledge_base_async(
                source_dir,
                chunk_size=int(chunk_size),
                overlap=int(overlap),
            )

        captured_stdout = io.StringIO()
        captured_stderr = io.StringIO()
        with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
            _run_async_in_thread(run_build())

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

        return "Success", clean_log_text(logs)
    except Exception as exc:
        logs += logger.write_log(f"Build failed: {exc}")
        return "Failed", clean_log_text(logs)


def answer_question_task(question, history, top_k_ret, top_k_comp, threshold, log_history_state):
    if not isinstance(history, list):
        history = []
    if not isinstance(log_history_state, list):
        log_history_state = []

    if not question:
        yield (
            history,
            log_history_state,
            logger.write_log("Error: question cannot be empty."),
            "",
            generate_logs_html(log_history_state),
        )
        return

    current_q_logs = logger.write_log(f"Processing new question: {question}")
    current_q_logs += logger.write_log(
        f"Parameters -> retrieve_k={top_k_ret}, compressed_k={top_k_comp}, threshold={threshold}"
    )

    debug_lines = []
    new_history = history + [{"role": "user", "content": question}]
    new_history.append({"role": "assistant", "content": "Searching and reasoning..."})
    yield (
        new_history,
        log_history_state,
        clean_log_text(current_q_logs),
        "",
        generate_logs_html(log_history_state),
    )

    full_answer = ""

    try:
        k_ret = int(top_k_ret) if top_k_ret is not None else DEFAULT_TOP_K
        k_comp = int(top_k_comp) if top_k_comp is not None else DEFAULT_TOP_K_COMPRESSED
        thresh = float(threshold) if threshold is not None else DEFAULT_THRESHOLD
        current_q_logs += logger.write_log("Initializing RAG engine...")
        yield (
            new_history,
            log_history_state,
            clean_log_text(current_q_logs),
            "\n\n".join(debug_lines),
            generate_logs_html(log_history_state),
        )
        rag = get_rag_instance()
        current_q_logs += logger.write_log("RAG engine ready.")
        yield (
            new_history,
            log_history_state,
            clean_log_text(current_q_logs),
            "\n\n".join(debug_lines),
            generate_logs_html(log_history_state),
        )

        for chunk in rag.answer_question_stream(
            question,
            top_k_retrieve=k_ret,
            top_k_compressed=k_comp,
            score_threshold=thresh,
            history=history,
        ):
            debug_event = parse_debug_event(chunk)
            if debug_event:
                event = debug_event.get("event", "unknown")
                content = str(debug_event.get("content", "")).strip()
                debug_lines.append(f"[{event}]\n{content}")
                yield (
                    new_history,
                    log_history_state,
                    clean_log_text(current_q_logs),
                    "\n\n".join(debug_lines),
                    generate_logs_html(log_history_state),
                )
                continue

            if is_process_log(chunk):
                current_q_logs += chunk + "\n"
                yield (
                    new_history,
                    log_history_state,
                    clean_log_text(current_q_logs),
                    "\n\n".join(debug_lines),
                    generate_logs_html(log_history_state),
                )
            else:
                if chunk == simpleRAG_content.SimpleRAG.ANSWER_REPLACE_MARKER:
                    full_answer = ""
                    new_history[-1]["content"] = ""
                    yield (
                        new_history,
                        log_history_state,
                        clean_log_text(current_q_logs),
                        "\n\n".join(debug_lines),
                        generate_logs_html(log_history_state),
                    )
                    continue
                full_answer += chunk
                new_history[-1]["content"] = full_answer
                yield (
                    new_history,
                    log_history_state,
                    clean_log_text(current_q_logs),
                    "\n\n".join(debug_lines),
                    generate_logs_html(log_history_state),
                )

        current_q_logs += logger.write_log("Answer generation completed.")
        log_entry_label = f"{question[:30]}{'...' if len(question) > 30 else ''} ({time.strftime('%H:%M:%S')})"
        new_log_history = [{"label": log_entry_label, "details": clean_log_text(current_q_logs)}] + log_history_state
        if len(new_log_history) > 20:
            new_log_history = new_log_history[:20]

        yield (
            new_history,
            new_log_history,
            clean_log_text(current_q_logs),
            "\n\n".join(debug_lines),
            generate_logs_html(new_log_history),
        )
    except Exception as exc:
        error_msg = str(exc)
        current_q_logs += logger.write_log(f"Error: {error_msg}")
        new_history[-1]["content"] = f"{full_answer}\n\n[System Error]: {error_msg}"
        log_entry_label = f"{question[:30]}... (Error) ({time.strftime('%H:%M:%S')})"
        new_log_history = [{"label": log_entry_label, "details": clean_log_text(current_q_logs)}] + log_history_state
        yield (
            new_history,
            new_log_history,
            clean_log_text(current_q_logs),
            "\n\n".join(debug_lines),
            generate_logs_html(new_log_history),
        )


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
                    top_k_ret_slider = gr.Slider(1, 50, value=DEFAULT_TOP_K, step=1, label="Retrieve count")
                    top_k_comp_slider = gr.Slider(1, 20, value=DEFAULT_TOP_K_COMPRESSED, step=1, label="Rerank count")
                    threshold_slider = gr.Slider(0.0, 1.0, value=DEFAULT_THRESHOLD, step=0.05, label="Score threshold")

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

        with gr.TabItem("Build Knowledge Base"):
            gr.Markdown("### Build / Rebuild Index")
            with gr.Row():
                with gr.Column():
                    dir_input = gr.Textbox(label="Document directory", value=str(DOC_DIR))
                    chunk_size_input = gr.Number(label="Chunk size", value=CHUNK_SIZE_DEFAULT, precision=0)
                    overlap_input = gr.Number(label="Chunk overlap", value=CHUNK_OVERLAP_DEFAULT, precision=0)
                    build_btn = gr.Button("Build index", variant="primary")
                with gr.Column():
                    build_status = gr.Textbox(label="Build status", interactive=False)
                    build_log = gr.Textbox(label="Build logs", lines=15, interactive=False)

    def process_and_update(question, history, k1, k2, th, log_state):
        yield from answer_question_task(question, history, k1, k2, th, log_state)

    msg_input.submit(
        fn=process_and_update,
        inputs=[msg_input, chatbot, top_k_ret_slider, top_k_comp_slider, threshold_slider, log_history_state],
        outputs=[chatbot, log_history_state, current_process_log, conversation_debug_log, log_html_display],
    )

    submit_btn.click(
        fn=process_and_update,
        inputs=[msg_input, chatbot, top_k_ret_slider, top_k_comp_slider, threshold_slider, log_history_state],
        outputs=[chatbot, log_history_state, current_process_log, conversation_debug_log, log_html_display],
    )

    clear_btn.click(
        lambda: ([], [], "", "", "<div style='color: #888; font-style: italic;'>No history yet.</div>"),
        None,
        [chatbot, log_history_state, current_process_log, conversation_debug_log, log_html_display],
        queue=False,
    )

    build_btn.click(
        fn=build_knowledge_base_task,
        inputs=[dir_input, chunk_size_input, overlap_input],
        outputs=[build_status, build_log],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
