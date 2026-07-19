# LightweightRAG

## 1. 项目简介

本仓库为毕业设计 **《基于 LLM 信息检索增强的智能问答系统设计与实现》**（作者：王锦鸿）的配套工程实现 **LightweightRAG**。

系统采用「离线构建知识库 + 在线检索问答」架构：本地文档经分块与向量化写入 FAISS / SQLite，用户提问时通过 **bge-m3** 召回、**bge-reranker-v2-m3** 重排序，再调用 **Ollama** 上的 **deepseek-r1:8b** 流式生成答案，并通过 Flask Web 界面展示检索、重排与工作流过程。

## 2. 功能列表

- **知识库构建**：支持 PDF、TXT、Markdown、DOCX 等格式；可选 `.doc` 预处理；中文编码自动探测
- **增量构建**：文档指纹比对，跳过未变更文件，复用 embedding 缓存
- **向量召回**：FAISS + bge-m3，可配置 Top-K 与相似度阈值
- **重排序**：FlagEmbedding bge-reranker-v2-m3，保留 Top-3 上下文
- **Ollama 对话**：流式输出，支持多轮上下文与清空会话
- **Web 可观察性**：检索结果、重排结果、工作流步骤、证据面板、在线评测 HTML
- **离线评测**：`rag_evaluator.py` 与 SciFact 公开检索评测脚本 `experiments/public_retrieval_eval.py`

## 3. 环境要求

| 项目 | 说明 |
|------|------|
| Python | 3.10+（建议 64 位） |
| Ollama | 本地安装并运行，默认 `http://localhost:11434` |
| 磁盘空间 | 模型约 2–4 GB；语料与索引视个人文档规模而定 |
| 操作系统 | Windows 为主（含 `pywin32`、`.doc` 转换）；Linux/macOS 可运行核心 RAG，`.doc` 需另行处理 |

## 4. 安装依赖

在项目根目录执行：

```powershell
pip install -r requirements.txt
```

SciFact 评测额外需要：

```powershell
pip install ir_datasets
```

## 5. 安装 Ollama 并拉取对话模型

1. 从 [https://ollama.com](https://ollama.com) 安装 Ollama
2. 拉取论文默认对话模型：

```powershell
ollama pull deepseek-r1:8b
```

3. 确认服务可用：

```powershell
curl http://localhost:11434/api/tags
```

响应 JSON 中应包含 `deepseek-r1:8b`。

## 6. 下载 embedding 与 reranker 模型

模型**不随仓库分发**，请自行下载到下列目录（路径相对项目根目录）：

| 模型 | HuggingFace 仓库 | 本地目录 |
|------|------------------|----------|
| bge-m3 | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | `models/bge-m3/` |
| bge-reranker-v2-m3 | [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | `models/bge-reranker-v2-m3/` |

示例（需已安装 `huggingface_hub` 或 `git lfs`）：

```powershell
# 使用 huggingface-cli（pip install huggingface_hub）
huggingface-cli download BAAI/bge-m3 --local-dir models/bge-m3
huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir models/bge-reranker-v2-m3
```

`config.py` 中 `LOCAL_EMBEDDING_MODEL_PATH` 与 reranker 路径均指向上述目录。

## 7. 准备语料（docs/）

将待索引文档放入 **`docs/`** 目录（可含子目录）。**完整 RAG 语料不包含在 GitHub 仓库中**；克隆后需自行拷贝 PDF、TXT、DOCX 等文件。

设计文档与实现计划保留在 `docs/superpowers/`，会随仓库跟踪。

## 8. 启动应用

```powershell
python LightweightRAG.py
```

启动成功后终端会打印本地地址，默认：

**http://127.0.0.1:7860**

在浏览器中打开即可使用 Web 界面。

## 9. 在 Web 中构建知识库

1. 打开上述地址
2. 在界面中选择或确认语料目录（默认为 `docs/`）
3. 点击 **「构建知识库」**
4. 等待构建完成，查看分块数量、构建日志与状态面板

构建产物（FAISS 索引、SQLite、元数据等）写入项目根目录，已被 `.gitignore` 排除，请勿提交。

## 10. SciFact 公开检索评测

在项目根目录运行：

```powershell
python experiments/public_retrieval_eval.py
```

结果写入 `experiments/results/`（JSON + Markdown）。首次运行会下载 SciFact 数据集，耗时与网络有关；若缺少 `ir_datasets`，请先执行第 4 节中的额外安装命令。

## 11. 不要提交的文件

以下内容仅保留在本地，**请勿 `git add` 或推送到 GitHub**：

| 类型 | 路径 / 模式 |
|------|-------------|
| 模型权重 | `models/` |
| 向量索引 | `faiss_index.bin`、`*.bin` |
| 知识库 | `knowledge_base.db` |
| 缓存与元数据 | `embedding_cache.json`、`metadata.json` |
| 会话状态 | `conversation_state.json` |
| RAG 语料 | `docs/download/`、`docs/haystack/`、`docs/huggingface_learn/`、`docs/langchain/`、`docs/llamaindex/`、`docs/qdrant/`、`docs/transformers/` 及 `docs/` 下大量 PDF/TXT |
| 虚拟环境 | `.venv/`、`venv/` |
| 论文私有材料 | `graduation_thesis/`、`LightweightRAG_旁置材料/` 等（见 `.gitignore`） |

**可以提交**：Python 源码、`index.html` / CSS / JS、`requirements.txt`、`.gitignore`、本 README、`experiments/public_retrieval_eval.py`、`eval_dataset_template.json`、`docs/superpowers/**`。

## 12. 目录结构（简表）

```
LightweightRAG/
├── LightweightRAG.py          # Flask 主程序
├── simpleRAG_content.py       # RAG 编排入口
├── config.py                  # 路径与检索默认参数
├── document_loader.py         # 多格式文档加载
├── doc_converter.py           # .doc 预处理
├── text_splitter.py           # 分块与语义合并
├── prompts.py                 # 提示词模板
├── rag_evaluator.py           # 离线评测
├── index.html                 # Web 前端
├── lightweightrag.css
├── lightweightrag.js
├── requirements.txt
├── simpleRAG_included/        # 构建、查询、会话等核心模块
├── experiments/
│   └── public_retrieval_eval.py
├── docs/                      # 语料目录（内容需自备；superpowers/ 为设计文档）
├── models/                    # 本地模型（需自备，已 gitignore）
├── faiss_index.bin            # 运行时生成（已 gitignore）
├── knowledge_base.db          # 运行时生成（已 gitignore）
└── embedding_cache.json       # 运行时生成（已 gitignore）
```

## 默认参数（与论文一致）

- 召回 Top-K：5；相似度阈值：0.3；重排保留：3
- 分块：220 字 / 重叠 40；语义合并阈值：0.85
- 对话模型：`deepseek-r1:8b`；Embedding：`bge-m3`；Reranker：`bge-reranker-v2-m3`
