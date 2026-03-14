# text_splitter.py
import re
from typing import List
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

class SmartTextSplitter:
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-zh-v1.5",
        threshold: float = 0.8,
        base_splitter_params: dict = None,
        device: str = "cpu",
    ):
        """
        初始化语义感知切块器。

        Args:
            model_name: 用于计算语义相似度的SentenceTransformer模型名称。
            threshold: 语义相似度阈值，高于此值则合并。
            base_splitter_params: 传递给基础字符切分器的参数。
            device: 计算设备 ('cpu' 或 'cuda')。
        """
        self.threshold = threshold
        self.device = device
        self.base_params = base_splitter_params or {
            "chunk_size": 400,
            "chunk_overlap": 50,
            "separators": ["\n\n", "\n", "。", "？", "！", "；", "?", "!", ";", "...", " ", ""],
        }

        # 尝试加载语义模型
        try:
            print(f"[信息] 尝试加载模型: {model_name}")
            self.model = SentenceTransformer(model_name, device=device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            print("[信息] 语义模型加载成功，将使用语义感知切分。")
            self.use_semantic_splitting = True
        except Exception as e:
            print(f"[警告] 语义模型加载失败: {e}")
            print("[信息] 将回退到基于字符长度的简单切分。")
            self.use_semantic_splitting = False
            # 初始化基础切分器参数
            self.chunk_size = self.base_params.get("chunk_size", 400)
            self.chunk_overlap = self.base_params.get("chunk_overlap", 50)
            self.separators = self.base_params.get("separators", ["\n\n", "\n", "。", "？", "！", "；", "?", "!", ";", "...", " ", ""])

    def split_text(self, text: str) -> List[str]:
        """
        根据配置切分文本。
        """
        if self.use_semantic_splitting:
            return self._semantic_split(text)
        else:
            return self._simple_split(text)

    def _semantic_split(self, text: str) -> List[str]:
        """
        使用语义相似度进行切分（简化版模拟）。
        这里为了演示，我们直接回退到简单切分，因为完整的语义合并逻辑比较复杂。
        实际应用中，这里会进行句子嵌入、相似度计算和合并。
        """
        print("[信息] 由于完整语义逻辑较复杂，当前回退到简单切分以保证运行。")
        return self._simple_split(text)

    def _simple_split(self, text: str) -> List[str]:
        """
        基于字符长度和分隔符的简单切分。
        """
        chunks = []
        current_pos = 0
        text_len = len(text)

        while current_pos < text_len:
            chunk_end = min(current_pos + self.chunk_size, text_len)
            
            # 寻找最佳分割点
            best_separator_pos = -1
            for separator in self.separators:
                # 从 chunk_end 往前找最近的分隔符
                pos = text.rfind(separator, current_pos, chunk_end)
                if pos != -1:
                    if best_separator_pos == -1 or pos > best_separator_pos:
                        best_separator_pos = pos
            
            # 如果找到了合适的分隔符，则在此处分割
            if best_separator_pos != -1 and best_separator_pos > current_pos:
                chunk_text = text[current_pos : best_separator_pos + len(self.separators[0])] # 这里简化处理，保留第一个分隔符
                chunks.append(chunk_text)
                current_pos = best_separator_pos + len(self.separators[0])
            else:
                # 如果在范围内找不到分隔符，则强制按长度切分
                chunk_text = text[current_pos:chunk_end]
                chunks.append(chunk_text)
                current_pos = chunk_end
            
            # 处理重叠
            if self.chunk_overlap > 0 and current_pos < text_len:
                current_pos = max(current_pos - self.chunk_overlap, best_separator_pos + 1 if best_separator_pos != -1 else current_pos - self.chunk_overlap)

        # 清理可能产生的空块或纯空白块
        final_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        return final_chunks

# --- Legacy function for backward compatibility ---
def legacy_recursive_split(text: str, max_length: int = 400, overlap: int = 50) -> List[str]:
    """
    旧版递归切分函数，作为备用。
    """
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(para) > max_length:
            # 如果段落本身就超长，则按字符切分
            sub_chunks = [para[i:i+max_length] for i in range(0, len(para), max_length)]
            for sub_chunk in sub_chunks:
                if len(sub_chunk.strip()) > 0:
                    chunks.append(sub_chunk.strip())
        else:
            if len(current_chunk) + len(para) <= max_length:
                current_chunk += "\n" + para
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = para
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # 再次按 max_length 和 overlap 精细调整
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_length:
            final_chunks.append(chunk)
        else:
            # 对过长的chunk再次切分
            for i in range(0, len(chunk), max_length - overlap):
                c = chunk[i:i + max_length]
                if c.strip(): # 只添加非空块
                    final_chunks.append(c)
    return final_chunks

if __name__ == "__main__":
    # 测试代码
    test_text = "这是第一段。它包含了一些信息。这是第二段。\n\n这是新的一节。内容和前面不同。这里还有一个句子。"
    splitter = SmartTextSplitter(chunk_size=30, chunk_overlap=5)
    result = splitter.split_text(test_text)
    for i, chunk in enumerate(result):
        print(f"Chunk {i}: {repr(chunk)}")