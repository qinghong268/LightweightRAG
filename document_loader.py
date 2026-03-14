# document_loader.py
import os
from pathlib import Path
from typing import List, Union
from langchain_core.documents import Document

# --- Import loaders with error handling ---
try:
    from langchain_community.document_loaders import PyPDFLoader
    PDF_AVAILABLE = True
except ImportError:
    print("警告: 未安装 PyPDF2 或其他 PDF 库。无法加载 PDF 文件。")
    PDF_AVAILABLE = False

try:
    from langchain_community.document_loaders import Docx2txtLoader
    DOCX_AVAILABLE = True
except ImportError:
    print("警告: 未安装 python-docx。无法加载 Word (.docx) 文件。")
    DOCX_AVAILABLE = False

try:
    from langchain_community.document_loaders import TextLoader
    TEXT_AVAILABLE = True
except ImportError:
    print("警告: 未安装 langchain-community 中的 TextLoader。")
    TEXT_AVAILABLE = False


def batch_load_documents(folder_path: str) -> List[Document]:
    """
    批量加载指定文件夹中的所有支持格式的文档。
    """
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        print(f"错误: 文件夹 '{folder_path}' 不存在或不是一个目录。")
        return []

    all_docs = []
    supported_extensions = {'.txt'}
    if PDF_AVAILABLE:
        supported_extensions.add('.pdf')
    if DOCX_AVAILABLE:
        supported_extensions.add('.docx')

    print("="*50)
    for file_path in folder.glob('*'):
        if file_path.suffix.lower() in supported_extensions:
            print(f"处理文件: {file_path.name}")
            try:
                docs = load_single_document(str(file_path))
                if docs:
                    all_docs.extend(docs)
                    print(f"{file_path.suffix.upper()[1:]}加载成功: {file_path.name}")
                else:
                    print(f"警告: {file_path.name} 加载后无内容或被过滤。")
            except Exception as e:
                print(f"错误: 加载 {file_path.name} 时发生异常: {e}")
        else:
            print(f"跳过不支持的文件: {file_path.name}")
    print("="*50)

    print(f"批量处理完成！\n总文件数: {len([p for p in folder.glob('*') if p.suffix.lower() in supported_extensions])}")
    print(f"成功文档块数: {len(all_docs)}")
    return all_docs


def load_single_document(file_path: str) -> List[Document]:
    """
    根据文件扩展名加载单个文档。
    """
    loader = None
    if file_path.lower().endswith(".pdf") and PDF_AVAILABLE:
        loader = PyPDFLoader(file_path)
    elif file_path.lower().endswith(".docx") and DOCX_AVAILABLE:
        loader = Docx2txtLoader(file_path)
    elif file_path.lower().endswith(".txt") and TEXT_AVAILABLE:
        # TextLoader 需要 encoding 参数来正确处理中文
        loader = TextLoader(file_path, encoding='utf-8')
    else:
        # 文件类型不受支持或缺少相应库
        return []

    if loader:
        # load() 方法返回一个 Document 对象的列表
        docs = loader.load()
        # 为每个 Document 对象添加来源元数据
        for doc in docs:
            doc.metadata['source'] = file_path
        return docs
    return []


if __name__ == "__main__":
    # 测试代码
    docs = batch_load_documents("./docs")
    for i, doc in enumerate(docs):
        print(f"Doc {i+1} from {doc.metadata['source']}: {doc.page_content[:50]}...")