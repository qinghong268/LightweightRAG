import gradio as gr
import asyncio
import time
import threading
from pathlib import Path
import sys
import os
import re
import html

# 导入配置和核心模块
try:
    import simpleRAG_content
    from config import DOC_DIR, CHUNK_SIZE_DEFAULT, CHUNK_OVERLAP_DEFAULT, DEFAULT_TOP_K, DEFAULT_TOP_K_COMPRESSED, DEFAULT_THRESHOLD
except ImportError as e:
    print(f"导入失败：{e}")
    sys.exit(1)

# 工具函数

def clean_log_text(text):
    """清洗日志：去除多余空行"""
    if not text:
        return ""
    text = re.sub(r'\n\s*\n', '\n', text)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return '\n'.join(lines)

def is_process_log(text):
    """判断是否为过程日志"""
    keywords = [
        "开始查询", "步骤", "改写后问题", "向量化", "相似片段", "Top", "相似度:", 
        "来源:", "重排", "重排分数", "Rank", "压缩上下文", "输入片段数", 
        "调用压缩模型", "摘要", "调用回答生成模型", "Prompt:", "system:", "user:",
        "压缩完成", "初始检索命中"
    ]
    if any(k in text for k in keywords):
        return True
    if re.match(r'^\[(Top|Rank)', text.strip()):
        return True
    if ("来源:" in text and (".txt" in text or ".pdf" in text or ".docx" in text)):
        return True
    return False

def escape_html(text):
    """转义HTML特殊字符，防止注入"""
    return html.escape(text).replace('\n', '<br>')

def generate_logs_html(log_list):
    """
    将日志列表转换为HTML字符串，使用<details>实现折叠效果
    """
    if not log_list:
        return "<div style='color: #888; font-style: italic;'>暂无历史处理记录。</div>"
    
    html_content = "<div class='log-container'>"
    
    for i, item in enumerate(log_list):
        label = item["label"]
        details = item["details"]
        
        # 将换行符转换为<br>,并转义内容
        safe_details = escape_html(details)
        safe_label = escape_html(label)
        
        # 使用原生HTML details/summary实现折叠
        # open属性控制默认是否展开，这里设为false
        html_content += f"""
        <details style="border: 1px solid #e5e7eb; border-radius: 6px; margin-bottom: 8px; background: #f9fafb;">
            <summary style="padding: 10px; cursor: pointer; font-weight: 600; color: #374151; list-style: none; display: flex; justify-content: space-between; align-items: center;">
                <span> {safe_label}</span>
                <span style="font-size: 0.8em; color: #9ca3af;">▼</span>
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
rag_instance = simpleRAG_content.SimpleRAG()

def _run_async_in_thread(coro):
    """
    在线程中运行协程，避免在已运行事件循环中直接 run_until_complete。
    """
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

# 核心逻辑

def build_knowledge_base_task(source_dir_str, chunk_size, overlap, progress=gr.Progress()):
    source_dir = Path(source_dir_str) if source_dir_str else DOC_DIR
    logs = logger.write_log(f"开始扫描目录：{source_dir}")
    
    if not source_dir.exists():
        return "失败", logs + logger.write_log(f" 目录不存在：{source_dir}")

    try:
        progress(0, desc="初始化引擎...")
        
        async def run_build():
            await rag_instance.build_knowledge_base_async(
                source_dir, 
                chunk_size=int(chunk_size), 
                overlap=int(overlap)
            )
        
        _run_async_in_thread(run_build())
        
        try:
            count = len(rag_instance.index) if hasattr(rag_instance, 'index') else "未知"
            logs += logger.write_log(f" 索引构建成功！总向量数：{count}")
            logs += logger.write_log(f" 索引文件已保存至本地。")
        except:
            logs += logger.write_log(" 构建流程结束。")
            
        return "成功", clean_log_text(logs)
    except Exception as e:
        logs += logger.write_log(f" 构建失败：{str(e)}")
        return "失败", clean_log_text(logs)

def answer_question_task(question, history, top_k_ret, top_k_comp, threshold, log_history_state):
    """
    问答任务
    """
    if not isinstance(history, list):
        history = []
    if not isinstance(log_history_state, list):
        log_history_state = []

    if not question:
        # 返回时也要保持log_history_state不变
        yield history, log_history_state, logger.write_log("错误：问题不能为空。"), generate_logs_html(log_history_state)
        return

    current_q_logs = logger.write_log(f" 开始处理新问题：{question}")
    current_q_logs += logger.write_log(f" 参数 -> K:{top_k_ret}, Compress:{top_k_comp}, Thresh:{threshold}")
    
    new_history = history + [{"role": "user", "content": question}]
    new_history.append({"role": "assistant", "content": " 正在深度检索与分析中..."})
    
    # 初始 yield
    yield new_history, log_history_state, clean_log_text(current_q_logs), generate_logs_html(log_history_state)

    full_answer = ""
    
    try:
        k_ret = int(top_k_ret) if top_k_ret is not None else DEFAULT_TOP_K
        k_comp = int(top_k_comp) if top_k_comp is not None else DEFAULT_TOP_K_COMPRESSED
        thresh = float(threshold) if threshold is not None else DEFAULT_THRESHOLD

        for chunk in rag_instance.answer_question_stream(
            question,
            top_k_retrieve=k_ret,
            top_k_compressed=k_comp,
            score_threshold=thresh,
        ):
            if is_process_log(chunk):
                current_q_logs += chunk + "\n"
                # 实时刷新日志文本和HTML列表（虽然列表内容还没变，但保持同步）
                yield new_history, log_history_state, clean_log_text(current_q_logs), generate_logs_html(log_history_state)
            else:
                full_answer += chunk
                new_history[-1]["content"] = full_answer
                yield new_history, log_history_state, clean_log_text(current_q_logs), generate_logs_html(log_history_state)

        # 完成：将当前日志加入历史列表
        current_q_logs += logger.write_log(" 回答生成完毕。")
        log_entry_label = f"{question[:30]}{'...' if len(question)>30 else ''} ({time.strftime('%H:%M:%S')})"
        
        new_log_history = [{
            "label": log_entry_label,
            "details": clean_log_text(current_q_logs)
        }] + log_history_state
        
        if len(new_log_history) > 20:
            new_log_history = new_log_history[:20]

        # 返回更新后的state和新生成的HTML
        yield new_history, new_log_history, clean_log_text(current_q_logs), generate_logs_html(new_log_history)

    except Exception as e:
        error_msg = str(e)
        current_q_logs += logger.write_log(f" 发生错误：{error_msg}")
        new_history[-1]["content"] = f"{full_answer}\n\n[系统错误]:{error_msg}"
        
        log_entry_label = f"{question[:30]}... (Error) ({time.strftime('%H:%M:%S')})"
        new_log_history = [{
            "label": log_entry_label,
            "details": clean_log_text(current_q_logs)
        }] + log_history_state
        
        yield new_history, new_log_history, clean_log_text(current_q_logs), generate_logs_html(new_log_history)

# UI构建

with gr.Blocks(title="Lightweight RAG", theme=gr.themes.Soft()) as demo:
    gr.Markdown("#  Lightweight智能问答系统")
    
    with gr.Tabs():
        with gr.TabItem(" 智能问答"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label=" 对话记录",
                        height=500,
                        show_copy_button=True,
                        type="messages",
                        value=[] 
                    )
                    with gr.Row():
                        msg_input = gr.Textbox(label="输入问题", placeholder="请输入你的问题...", scale=4, container=False)
                        submit_btn = gr.Button("发送", variant="primary", scale=1)
                    clear_btn = gr.Button(" 清空对话", size="sm")

                with gr.Column(scale=1):
                    gr.Markdown(" 检索参数")
                    top_k_ret_slider = gr.Slider(1, 50, value=DEFAULT_TOP_K, step=1, label="检索数量")
                    top_k_comp_slider = gr.Slider(1, 20, value=DEFAULT_TOP_K_COMPRESSED, step=1, label="重排数量")
                    threshold_slider = gr.Slider(0.0, 1.0, value=DEFAULT_THRESHOLD, step=0.05, label="相似度阈值")
            
            gr.Markdown(" 历史处理过程(点击展开)")
            
            # 状态存储
            log_history_state = gr.State([])
            
            # 【关键修改】使用 gr.HTML 来渲染动态列表，而不是 gr.Column
            log_html_display = gr.HTML(
                value="<div style='color: #888; font-style: italic;'>暂无历史处理记录。</div>",
                label="历史记录列表"
            )
            
            # 当前实时日志（可选，保留用于调试或实时查看）
            current_process_log = gr.Textbox(
                label=" 当前实时运行流",
                lines=4,
                max_lines=6,
                interactive=False,
                visible=True
            )

        with gr.TabItem(" 知识库构建"):
            gr.Markdown(" 初始化索引")
            with gr.Row():
                with gr.Column():
                    dir_input = gr.Textbox(label="文档目录路径", value=str(DOC_DIR))
                    chunk_size_input = gr.Number(label="切片长度", value=CHUNK_SIZE_DEFAULT, precision=0)
                    overlap_input = gr.Number(label="重叠长度", value=CHUNK_OVERLAP_DEFAULT, precision=0)
                    build_btn = gr.Button(" 开始构建索引", variant="primary")
                with gr.Column():
                    build_status = gr.Textbox(label="构建状态", interactive=False)
                    build_log = gr.Textbox(label="构建详细日志", lines=15, interactive=False)

    # 事件绑定
    
    def process_and_update(question, history, k1, k2, th, log_state):
        yield from answer_question_task(question, history, k1, k2, th, log_state)

    # 绑定提交事件
    # outputs现在包含：chatbot, state, textbox, html
    submit_event = msg_input.submit(
        fn=process_and_update,
        inputs=[msg_input, chatbot, top_k_ret_slider, top_k_comp_slider, threshold_slider, log_history_state],
        outputs=[chatbot, log_history_state, current_process_log, log_html_display]
    )
    
    submit_btn.click(
        fn=process_and_update,
        inputs=[msg_input, chatbot, top_k_ret_slider, top_k_comp_slider, threshold_slider, log_history_state],
        outputs=[chatbot, log_history_state, current_process_log, log_html_display]
    )

    # 清空按钮重置HTML
    clear_btn.click(
        lambda: ([], [], "", "<div style='color: #888; font-style: italic;'>暂无历史处理记录。</div>"), 
        None, 
        [chatbot, log_history_state, current_process_log, log_html_display], 
        queue=False
    )

    build_btn.click(
        fn=build_knowledge_base_task,
        inputs=[dir_input, chunk_size_input, overlap_input],
        outputs=[build_status, build_log]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)