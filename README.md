# 大模型/RAG/AI开发知识体系轻量问答系统

## 第一部分：项目简介与正确运行方式

这是一个本地化运行的轻量 RAG 问答项目，主要面向中文问答场景，同时允许知识库中混合中文和英文资料。项目使用本地嵌入模型建立知识库索引，使用 Ollama 提供的大模型完成查询改写、上下文压缩和最终回答生成，并提供基于 Gradio 的 Web 界面。

项目当前的核心能力包括：

- 构建本地知识库，支持 `.txt`、`.pdf`、`.docx`
- 在知识库构建前自动递归预处理 `.doc` 文件
- 支持多轮对话和历史持久化
- 支持引用校验、压缩失败回退、回答引用不完整重试
- 提供知识库构建页、对话页、历史管理页

### 1. 运行环境

建议环境：

- Windows
- Python 3.10 及以上
- 已安装并可正常运行 Ollama
- 如果需要自动处理 `.doc` 文件，机器上还需要安装 Microsoft Word

### 2. 安装依赖

在项目根目录执行：

```powershell
pip install -r requirements.txt
```

说明：

- `FlagEmbedding` 已加入依赖，用于重排器功能
- `pywin32` 用于 `.doc` 转 `.docx`
- `gradio` 用于 Web 界面

### 3. 准备 Ollama 模型

请确保 Ollama 已启动，并提前拉取项目默认使用的模型：

```powershell
ollama pull deepseek-r1:8b
ollama pull qwen:7b
```

默认配置见 [`config.py`](/d:/RAGprojects/OutdatedAttempt/LightweightRAG/config.py)：

- 问答模型：`deepseek-r1:8b`
- 压缩模型：`qwen:7b`

### 4. 准备本地向量模型

项目默认使用以下本地模型目录：

- `models/bge-m3`
- `models/bge-reranker-v2-m3`

你可以：

- 手动将模型放到 `models/` 目录
- 或运行 [`download_model.py`](/d:/RAGprojects/OutdatedAttempt/LightweightRAG/download_model.py) 下载到项目本地 `models/` 目录

执行方式：

```powershell
python download_model.py
```

### 5. 准备知识库文档

将你的知识库文件放入 [`docs/`](/d:/RAGprojects/OutdatedAttempt/LightweightRAG/docs) 目录，支持：

- `.txt`
- `.pdf`
- `.docx`
- `.doc`

关于 `.doc`：

- Web 构建知识库时，会先递归扫描 `.doc`
- 成功转换后会在原目录生成同名 `.docx`
- 原始 `.doc` 会移动到 `backup/相对路径/`
- `backup` 目录默认不会参与知识库构建
- 如果 `.doc` 转换失败，本次构建会继续，但失败文件不会入库

### 6. 启动 Web 界面

```powershell
python LightweightRAG.py
```

默认访问地址通常为：

```text
http://127.0.0.1:7860
```

程序启动后，控制台通常会显示类似下面的提示：

```text
系统启动中，请稍候...
启动成功，Web 界面地址：http://localhost:7860
```

推荐使用顺序：

1. 先进入“知识库构建”页面构建索引
2. 再进入“对话问答”页面提问
3. 如需查看或修改持久化对话历史，可进入“对话历史管理”页面

### 7. 单独运行 `.doc` 转换工具

如果你只想先批量处理旧版 Word 文档，可以单独运行：

```powershell
python doc_converter.py
```

或者指定目录：

```powershell
python doc_converter.py --input ./docs
```

当前独立运行模式下：

- 会递归扫描目录和子目录
- 会跳过 `backup` 目录
- 会在原位生成 `.docx`
- 会将原 `.doc` 移到 `backup/相对路径/`
- 会尝试申请管理员权限

## 第二部分：项目细节说明

### 1. 项目结构

主要文件说明：

- [`LightweightRAG.py`](/d:/RAGprojects/OutdatedAttempt/LightweightRAG/LightweightRAG.py)：Gradio Web 界面入口
- [`simpleRAG_content.py`](/d:/RAGprojects/OutdatedAttempt/LightweightRAG/simpleRAG_content.py)：问答主流程、历史处理、缓存、引用校验
- [`document_loader.py`](/d:/RAGprojects/OutdatedAttempt/LightweightRAG/document_loader.py)：知识库文档加载
- [`text_splitter.py`](/d:/RAGprojects/OutdatedAttempt/LightweightRAG/text_splitter.py)：文本切分与语义合并
- [`doc_converter.py`](/d:/RAGprojects/OutdatedAttempt/LightweightRAG/doc_converter.py)：`.doc` 转 `.docx` 工具和构建前预处理逻辑
- [`prompts.py`](/d:/RAGprojects/OutdatedAttempt/LightweightRAG/prompts.py)：提示词模板
- [`config.py`](/d:/RAGprojects/OutdatedAttempt/LightweightRAG/config.py)：项目配置

`simpleRAG_included/` 目录：

- [`rag_build.py`](/d:/RAGprojects/OutdatedAttempt/LightweightRAG/simpleRAG_included/rag_build.py)：知识库构建核心
- [`rag_query.py`](/d:/RAGprojects/OutdatedAttempt/LightweightRAG/simpleRAG_included/rag_query.py)：检索、重排、压缩
- [`rag_helpers.py`](/d:/RAGprojects/OutdatedAttempt/LightweightRAG/simpleRAG_included/rag_helpers.py)：数据库、FAISS、Ollama 请求帮助函数
- [`conversation_store.py`](/d:/RAGprojects/OutdatedAttempt/LightweightRAG/simpleRAG_included/conversation_store.py)：单会话持久化
- [`config_imports.py`](/d:/RAGprojects/OutdatedAttempt/LightweightRAG/simpleRAG_included/config_imports.py)：配置统一导入

### 2. 知识库构建流程

知识库构建的大致流程如下：

1. Web 侧先检查并预处理 `.doc`
2. 文档加载器递归扫描 `docs/`，并跳过 `backup/`
3. 加载支持的文档类型
4. 使用切分器将文档拆分为 chunk
5. 使用本地嵌入模型生成向量
6. 将 chunk 元数据写入 SQLite
7. 从数据库重建 FAISS 索引与元数据快照
8. 原子替换 `faiss_index.bin` 和 `metadata.json`

构建产物：

- [`knowledge_base.db`](/d:/RAGprojects/OutdatedAttempt/LightweightRAG/knowledge_base.db)
- [`faiss_index.bin`](/d:/RAGprojects/OutdatedAttempt/LightweightRAG/faiss_index.bin)
- [`metadata.json`](/d:/RAGprojects/OutdatedAttempt/LightweightRAG/metadata.json)
- [`embedding_cache.json`](/d:/RAGprojects/OutdatedAttempt/LightweightRAG/embedding_cache.json)

### 3. 问答流程

问答主流程如下：

1. 读取当前持久化对话历史
2. 判断是否属于“关于对话本身的问题”
3. 如有必要，先结合历史改写查询
4. 对问题向量化
5. 从 FAISS 中召回候选片段
6. 用重排器对结果重排序
7. 调用压缩模型压缩上下文
8. 调用最终问答模型生成回答
9. 校验引用
10. 若引用不完整，则回退到原始上下文重试

### 4. 多轮对话与历史管理

当前是单窗口、单会话模式，持久化文件为：

- [`conversation_state.json`](/d:/RAGprojects/OutdatedAttempt/LightweightRAG/conversation_state.json)

当前实现特点：

- 会自动保存用户消息和合格的助手回答
- 系统性提示、错误提示、无知识库提示不会写入正式历史
- 支持通过 Web 页面直接查看和编辑历史 JSON
- 编辑后的历史会影响后续多轮对话
- “清空对话”带二次确认

### 5. 缓存与状态

项目包含两类主要缓存：

- 嵌入缓存：减少重复向量化
- 查询缓存：减少相同问题在相同知识库快照上的重复检索和生成

知识库快照状态变化后，查询缓存会自动失效。

### 6. 已知注意事项

以下内容不是阻塞问题，但使用时需要知道：

- `download_model.py` 现在默认下载到项目本地 `models/` 目录
- 如果未安装 Microsoft Word，`.doc` 自动转换能力无法使用
- 如果 `FlagEmbedding` 不可用，系统会降级为不做重排，但问答仍可继续
- 如果控制台出现 `RequestsDependencyWarning`，通常是当前 Python 环境里的 `requests` 相关依赖版本组合不够干净，不一定影响运行，但建议后续整理环境
- 项目根目录中存在较多运行产物文件，属于正常现象
