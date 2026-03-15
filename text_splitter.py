# text_splitter_improved.py
import os
import threading
import queue
import torch
from sentence_transformers import SentenceTransformer, util

def _load_model_in_thread(model_identifier, device, result_queue):
    """
    在独立线程中加载模型的辅助函数
    """
    try:
        if os.path.exists(model_identifier):
            model = SentenceTransformer(model_identifier, device=device)
        else:
            model = SentenceTransformer(model_identifier, device=device)
        result_queue.put(('success', model))
    except Exception as e:
        result_queue.put(('error', e))

class SmartTextSplitter:
    """
    智能语义文本分割器
    结合递归字符分割和语义相似度合并
    """
    def __init__(self, model_path=None, model_name=None, threshold=0.75, base_splitter_params=None, device=None, load_timeout=30):
        """
        初始化语义感知文本分割器
        
        参数:
            model_path: 本地模型路径（优先级最高）
            model_name: 模型名称，如果提供了model_path则忽略此参数
            threshold: 语义相似度阈值，高于此值则合并文本块
            base_splitter_params: 基础递归分割器的参数
            device: 运行设备，如 'cuda' 或 'cpu'
            load_timeout: 模型加载超时时间（秒）
        """
        self.threshold = threshold
        self.load_timeout = load_timeout
        
        # 初始化基础分割器（用于初次分句/分段）
        if base_splitter_params is None:
            base_splitter_params = {
                "chunk_size": 256,
                "chunk_overlap": 50,
                "separators": ["\n\n", "\n", "。", "？", "！", "；", "?", "!", ";", "..."],
                "length_function": len,
                "is_separator_regex": False
            }
        
        # 注意：这里需要在方法内部导入，避免循环导入问题
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        self.base_splitter = RecursiveCharacterTextSplitter(**base_splitter_params)
        
        # 设备选择
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # --- 改进部分：加入模型加载失败的回退机制和跨平台超时 ---
        self.use_semantic_splitting = True  # 默认启用语义分割
        self.model = None # 预先定义model属性

        # 选择要加载的模型标识符 (优先级: model_path > model_name > default_model)
        model_to_load = model_path if model_path is not None and os.path.exists(model_path) else model_name
        if model_to_load is None:
            model_to_load = 'all-MiniLM-L6-v2'

        try:
            if model_path is not None and os.path.exists(model_path):
                print(f"[信息] 从本地路径加载模型: {model_path}")
            else:
                print(f"[信息] 加载模型: {model_to_load}")

            # 创建一个队列来在线程间传递结果
            result_queue = queue.Queue()

            # 创建加载模型的线程
            load_thread = threading.Thread(
                target=_load_model_in_thread,
                args=(model_to_load, device, result_queue)
            )
            load_thread.daemon = True  # 设置为守护线程
            load_thread.start()

            # 设置一个定时器来强制中断加载
            timer = threading.Timer(self.load_timeout, lambda t: t.join() or t._stop() if t.is_alive() else None, [load_thread])
            timer.start()

            # 等待加载完成或超时
            load_thread.join(timeout=self.load_timeout)
            timer.cancel()  # 成功加载后，取消定时器

            if load_thread.is_alive():
                # 线程仍在运行，说明超时了
                print(f"[警告] 模型加载超时 ({self.load_timeout} 秒)。")
                print("[信息] 将回退到基于字符长度的基础切分。")
                self.use_semantic_splitting = False
            else:
                # 线程已结束，检查结果
                try:
                    status, value = result_queue.get_nowait()
                    if status == 'success':
                        self.model = value
                        print(f"[信息] 语义模型加载成功，设备: {device}, 阈值: {threshold}")
                    else:
                        raise value  # 抛出加载时发生的异常
                except queue.Empty:
                    # 理论上不应该发生，但作为安全网
                    print("[警告] 模型加载线程异常退出，未返回结果。")
                    print("[信息] 将回退到基于字符长度的基础切分。")
                    self.use_semantic_splitting = False
                    
        except Exception as e:
            print(f"[警告] 语义模型加载失败: {e}")
            print("[信息] 将回退到基于字符长度的基础切分。")
            self.use_semantic_splitting = False

    def split_text(self, text: str):
        """主分割方法"""
        if not text or not text.strip():
            return []

        # 根据模型加载情况选择不同的分割策略
        if self.use_semantic_splitting and self.model is not None:
            return self._perform_semantic_split(text)
        else:
            # 回退到基础分割器
            print("[信息] 执行基础字符分割...")
            return self.base_splitter.split_text(text)

    def _perform_semantic_split(self, text: str):
        """执行完整的语义分割流程"""
        # 第一步：用基础分割器得到初始块
        initial_chunks = self.base_splitter.split_text(text)
        if len(initial_chunks) <= 1:
            return initial_chunks
        
        print(f"[信息] 初始分割为 {len(initial_chunks)} 个块")
        
        # 第二步：计算所有初始块的嵌入向量
        with torch.no_grad():
            embeddings = self.model.encode(
                initial_chunks,
                convert_to_tensor=True,
                show_progress_bar=False,
                normalize_embeddings=True
            )
        
        # 第三步：基于语义相似度合并块
        merged_chunks = []
        current_chunk = initial_chunks[0]
        current_embedding = embeddings[0]
        
        for i in range(1, len(initial_chunks)):
            next_chunk = initial_chunks[i]
            next_embedding = embeddings[i]
            
            # 计算当前块与下一块的语义相似度
            similarity = util.cos_sim(current_embedding, next_embedding).item()
            
            if similarity >= self.threshold:
                # 语义连贯，合并
                current_chunk += " " + next_chunk
                # 注意：这里为了性能，使用了近似的向量更新，更精确的做法是重新编码合并后的文本块，但这会增加计算量
                current_embedding = next_embedding  
            else:
                # 语义不连贯，保存当前块，开始新的块
                merged_chunks.append(current_chunk.strip())
                current_chunk = next_chunk
                current_embedding = next_embedding
        
        # 添加最后一个块
        if current_chunk:
            merged_chunks.append(current_chunk.strip())
        
        print(f"[信息] 语义感知合并后为 {len(merged_chunks)} 个块")
        return merged_chunks