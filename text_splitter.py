# text_splitter.py
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
        self.threshold = threshold
        self.load_timeout = load_timeout
        
        if base_splitter_params is None:
            base_splitter_params = {
                "chunk_size": 256,
                "chunk_overlap": 50,
                "separators": ["\n\n", "\n", "。", "？", "！", "；", "?", "!", ";", "..."],
                "length_function": len,
                "is_separator_regex": False
            }
        
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        self.base_splitter = RecursiveCharacterTextSplitter(**base_splitter_params)
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.use_semantic_splitting = True
        self.model = None

        model_to_load = model_path if model_path is not None and os.path.exists(model_path) else model_name
        if model_to_load is None:
            model_to_load = 'all-MiniLM-L6-v2'

        try:
            result_queue = queue.Queue()
            load_thread = threading.Thread(
                target=_load_model_in_thread,
                args=(model_to_load, device, result_queue)
            )
            load_thread.daemon = True
            load_thread.start()

            load_thread.join(timeout=self.load_timeout)

            if load_thread.is_alive():
                print(f"[警告]模型加载超时 ({self.load_timeout} 秒)。")
                print("[信息]将回退到基于字符长度的基础切分。")
                self.use_semantic_splitting = False
            else:
                status, value = result_queue.get_nowait()
                if status == 'success':
                    self.model = value
                    print(f"[信息]语义模型加载成功，设备:{device}, 阈值:{threshold}")
                else:
                    raise value
        except Exception as e:
            print(f"[警告]语义模型加载失败: {e}")
            print("[信息]将回退到基于字符长度的基础切分。")
            self.use_semantic_splitting = False

    def split_text(self, text: str):
        if not text or not text.strip():
            return []

        if self.use_semantic_splitting and self.model is not None:
            return self._perform_semantic_split(text)
        else:
            print("[信息]执行基础字符分割...")
            return self.base_splitter.split_text(text)

    def _perform_semantic_split(self, text: str):
        initial_chunks = self.base_splitter.split_text(text)
        if len(initial_chunks) <= 1:
            return initial_chunks
        max_merged_chars = int(self.base_splitter._chunk_size * 1.3)
        
        with torch.no_grad():
            embeddings = self.model.encode(
                initial_chunks,
                convert_to_tensor=True,
                show_progress_bar=False,
                normalize_embeddings=True
            )
        
        merged_chunks = []
        current_chunk = initial_chunks[0]
        current_embedding = embeddings[0]
        
        for i in range(1, len(initial_chunks)):
            next_chunk = initial_chunks[i]
            next_embedding = embeddings[i]
            similarity = util.cos_sim(current_embedding, next_embedding).item()
            can_merge_by_length = (len(current_chunk) + 1 + len(next_chunk)) <= max_merged_chars
            
            if similarity >= self.threshold and can_merge_by_length:
                current_chunk += " " + next_chunk
                current_embedding = self.model.encode(current_chunk, convert_to_tensor=True, normalize_embeddings=True)
            else:
                merged_chunks.append(current_chunk.strip())
                current_chunk = next_chunk
                current_embedding = next_embedding
        
        if current_chunk:
            merged_chunks.append(current_chunk.strip())
        
        return merged_chunks