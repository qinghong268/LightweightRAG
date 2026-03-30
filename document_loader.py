import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document

try:
    from langchain_community.document_loaders import PyPDFLoader

    PDF_AVAILABLE = True
except ImportError:
    print("Warning: PDF loader dependency is missing. PDF files will be skipped.")
    PDF_AVAILABLE = False

try:
    from langchain_community.document_loaders import Docx2txtLoader

    DOCX_AVAILABLE = True
except ImportError:
    print("Warning: DOCX loader dependency is missing. DOCX files will be skipped.")
    DOCX_AVAILABLE = False


TEXT_FALLBACK_ENCODINGS = (
    "utf-8-sig",
    "utf-8",
    "gb18030",
    "gbk",
    "gb2312",
)
MAX_UNSUPPORTED_LOG_EXAMPLES = 10
SKIPPED_DIRECTORY_NAMES = {"backup"}


def _iter_files(folder: Path) -> List[Path]:
    discovered_files: List[Path] = []
    for current_root, dir_names, file_names in os.walk(folder):
        dir_names[:] = [
            name for name in dir_names if name.lower() not in SKIPPED_DIRECTORY_NAMES
        ]
        root_path = Path(current_root)
        for file_name in file_names:
            discovered_files.append(root_path / file_name)

    return sorted(
        (path for path in discovered_files if path.is_file()),
        key=lambda path: str(path.relative_to(folder)).lower(),
    )


def _load_text_document(file_path: Path) -> List[Document]:
    last_error = None
    for encoding in TEXT_FALLBACK_ENCODINGS:
        try:
            content = file_path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except OSError as exc:
            last_error = exc
            break
        return [
            Document(
                page_content=content,
                metadata={"source": str(file_path), "encoding": encoding},
            )
        ]

    raise ValueError(
        f"Unable to decode text file {file_path} with fallback encodings: {last_error}"
    )


def batch_load_documents(folder_path: str) -> List[Document]:
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        print(f"Error: folder '{folder_path}' does not exist or is not a directory.")
        return []

    all_docs: List[Document] = []
    supported_extensions = {".txt"}
    if PDF_AVAILABLE:
        supported_extensions.add(".pdf")
    if DOCX_AVAILABLE:
        supported_extensions.add(".docx")

    discovered_files = _iter_files(folder)
    supported_files = [
        file_path
        for file_path in discovered_files
        if file_path.suffix.lower() in supported_extensions
    ]
    unsupported_examples = []
    unsupported_count = 0

    print("=" * 50)
    for file_path in discovered_files:
        relative_path = file_path.relative_to(folder)
        if file_path.suffix.lower() not in supported_extensions:
            unsupported_count += 1
            if len(unsupported_examples) < MAX_UNSUPPORTED_LOG_EXAMPLES:
                unsupported_examples.append(str(relative_path))
            continue

        print(f"Processing file: {relative_path}")
        try:
            docs = load_single_document(str(file_path))
            if docs:
                all_docs.extend(docs)
                print(
                    f"Loaded {file_path.suffix.upper()[1:]} file successfully: {relative_path}"
                )
            else:
                print(f"Warning: file had no usable content after loading: {relative_path}")
        except Exception as exc:
            print(f"Error: failed to load {relative_path}: {exc}")
    print("=" * 50)

    print("Batch processing completed:")
    print(f"Supported files found: {len(supported_files)}")
    if unsupported_count:
        print(f"Unsupported files skipped: {unsupported_count}")
        for example in unsupported_examples:
            print(f"Unsupported example: {example}")
        remaining = unsupported_count - len(unsupported_examples)
        if remaining > 0:
            print(f"... and {remaining} more unsupported files.")
    print(f"Document chunks loaded: {len(all_docs)}")
    return all_docs


def load_single_document(file_path: str) -> List[Document]:
    path = Path(file_path)
    loader = None
    if file_path.lower().endswith(".pdf") and PDF_AVAILABLE:
        loader = PyPDFLoader(file_path)
    elif file_path.lower().endswith(".docx") and DOCX_AVAILABLE:
        loader = Docx2txtLoader(file_path)
    elif file_path.lower().endswith(".txt"):
        return _load_text_document(path)
    else:
        return []

    docs = loader.load() if loader else []
    for doc in docs:
        doc.metadata["source"] = str(path)
    return docs


if __name__ == "__main__":
    docs = batch_load_documents("./docs")
    for i, doc in enumerate(docs):
        print(f"Doc{i + 1} from {doc.metadata['source']}: {doc.page_content[:50]}...")
