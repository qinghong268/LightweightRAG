# LightweightRAG Complete Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Acceleration:** Use `dispatching-parallel-agents` for independent domains inside a task. Tasks 1–3 touch different file sets and MAY run in parallel after this plan is approved. Task 4 (E2E verify) is sequential and depends on 1–3.

**Goal:** Make LightweightRAG fully implement thesis features, verify with local Ollama, and package a code-only GitHub demo with README.

**Architecture:** Keep existing offline-build + online-QA stack; restore missing runtime files; harden model paths via `BASE_DIR`; expand `.gitignore`; rewrite `README.md`; verify end-to-end.

**Tech Stack:** Python, Flask, FAISS, SentenceTransformers, FlagEmbedding, SQLite, Ollama (`deepseek-r1:8b`), bge-m3, bge-reranker-v2-m3

## Global Constraints

- Workspace root: `D:\RAGprojects\LightweightRAG`
- Do not restore `graduation_thesis/` or defense `outputs/` into the repo
- Do not commit `models/`, full RAG corpus, `faiss_index.bin`, `knowledge_base.db`, `embedding_cache.json`, `conversation_state.json`
- Keep thesis defaults: TopK=5, threshold=0.3, rerank keep=3, chunk 220/40, semantic threshold 0.85
- Spec: `docs/superpowers/specs/2026-07-19-lightweightrag-complete-implementation-design.md`
- Parallel policy: Tasks 1, 2, 3 are independent file domains → dispatch in one parallel batch; Task 4 after merge

---

### Task 1: Restore runtime files + fix `config.py`

**Files:**
- Create/Restore: `conversation_state.json`
- Restore: `测试问题.txt` from `D:\RAGprojects\LightweightRAG_旁置材料\测试问题.txt`
- Modify: `config.py`

**Interfaces:**
- Produces: `LOCAL_RERANK_MODEL_PATH: str` in `config.py` equal to `str(BASE_DIR / "models" / "bge-reranker-v2-m3")`
- Produces: valid UTF-8 docstring on `config.py`
- Consumes: existing `BASE_DIR`, `LOCAL_EMBEDDING_MODEL_PATH` pattern

- [x] **Step 1: Restore `测试问题.txt`**

```powershell
Copy-Item -LiteralPath "D:\RAGprojects\LightweightRAG_旁置材料\测试问题.txt" -Destination "D:\RAGprojects\LightweightRAG\测试问题.txt" -Force
```

Expected: file exists at project root.

- [x] **Step 2: Ensure `conversation_state.json` exists**

If archive copy exists, copy it; else write:

```json
{
  "active_session_id": "00000000-0000-0000-0000-000000000001",
  "updated_at": null,
  "messages": []
}
```

- [x] **Step 3: Rewrite `config.py` header + add rerank path**

Replace garbed docstring with:

```python
"""LightweightRAG configuration: paths, models, and retrieval defaults."""
```

Add after `LOCAL_EMBEDDING_MODEL_PATH`:

```python
LOCAL_RERANK_MODEL_PATH = str(BASE_DIR / "models" / "bge-reranker-v2-m3")
```

Keep all existing numeric defaults unchanged.

- [x] **Step 4: Export new symbol from `simpleRAG_included/config_imports.py`**

Add `LOCAL_RERANK_MODEL_PATH` to the import list from `config`.

- [x] **Step 5: Smoke import**

```powershell
python -c "import config; print(config.LOCAL_RERANK_MODEL_PATH); from pathlib import Path; print(Path(config.LOCAL_RERANK_MODEL_PATH).exists())"
```

Expected: path under `...\models\bge-reranker-v2-m3` and `True` if model dir present.

- [x] **Step 6: Commit**

```powershell
git add config.py simpleRAG_included/config_imports.py conversation_state.json "测试问题.txt"
git commit -m "fix: restore session assets and harden config model paths"
```

Note: `conversation_state.json` may be gitignored later in Task 3; still create locally. If already ignored, do not force-add.

---

### Task 2: Harden reranker absolute path in `rag_query.py`

**Files:**
- Modify: `simpleRAG_included/rag_query.py`
- Optionally modify: `simpleRAG_included/config_imports.py` (if Task 1 not yet merged, import `LOCAL_RERANK_MODEL_PATH`)

**Interfaces:**
- Consumes: `LOCAL_RERANK_MODEL_PATH` from config / config_imports
- Produces: `RAGQuerier._reranker_model_path` resolves to absolute local model directory (not `./models/...`)

- [x] **Step 1: Change path construction**

Replace:

```python
self._reranker_model_path = f"./models/{reranker_model_name}"
```

With logic:

```python
from pathlib import Path
try:
    from config import LOCAL_RERANK_MODEL_PATH, BASE_DIR
except ImportError:
    from .config_imports import LOCAL_RERANK_MODEL_PATH  # after Task 1 export
    from config import BASE_DIR

# Prefer explicit local path; fall back to BASE_DIR / models / name
candidate = Path(LOCAL_RERANK_MODEL_PATH)
if not candidate.exists():
    candidate = Path(BASE_DIR) / "models" / reranker_model_name
self._reranker_model_path = str(candidate)
```

Match existing import style in the file (keep relative imports consistent with the package).

- [x] **Step 2: Unit-style check without loading GPU weights if slow**

```powershell
python -c "from simpleRAG_included.rag_query import RAGQuerier; from config import OLLAMA_HOST, CHAT_MODEL, RERANK_MODEL, LOCAL_RERANK_MODEL_PATH; q=RAGQuerier(OLLAMA_HOST, CHAT_MODEL, RERANK_MODEL); print(q._reranker_model_path); print(q._reranker_model_path.replace('\\\\','/').endswith('models/bge-reranker-v2-m3') or 'bge-reranker-v2-m3' in q._reranker_model_path)"
```

Expected: path contains `models\bge-reranker-v2-m3` and is absolute.

- [x] **Step 3: Commit**

```powershell
git add simpleRAG_included/rag_query.py
git commit -m "fix: load reranker from BASE_DIR absolute path"
```

---

### Task 3: GitHub packaging — `.gitignore` + `README.md` + `requirements.txt`

**Files:**
- Modify: `.gitignore`
- Create: `README.md` (currently deleted in working tree)
- Modify: `requirements.txt`

**Interfaces:**
- Produces: ignore rules that keep `docs/superpowers/**` tracked while ignoring RAG corpus dirs and runtime artifacts
- Produces: Chinese README covering install → models → docs → build → run → eval
- Produces: `requirements.txt` includes Flask (currently missing) and other runtime imports as needed

- [x] **Step 1: Append project-specific ignores to `.gitignore`**

Add (do not remove existing thesis ignores):

```gitignore
# LightweightRAG runtime / large local assets
models/
*.bin
knowledge_base.db
embedding_cache.json
conversation_state.json
metadata.json
__pycache__/
*.pyc
.venv/
venv/

# RAG corpus (keep design/plans tracked)
/docs/download/
/docs/haystack/
/docs/huggingface_learn/
/docs/langchain/
/docs/llamaindex/
/docs/qdrant/
/docs/transformers/
/docs/*.pdf
/docs/*.txt

# Local archive / non-repo materials
/LightweightRAG_旁置材料/
```

Ensure `docs/superpowers/` is NOT ignored.

- [x] **Step 2: Fix `requirements.txt`**

Ensure at least:

```text
Flask>=3.0.0
FlagEmbedding==1.3.5
faiss-cpu==1.13.2
langchain-community==0.4.1
langchain-core==1.2.7
langchain-text-splitters==1.1.0
numpy==2.2.5
pywin32==311
requests==2.32.5
sentence-transformers==5.3.0
torch==2.10.0
pypdf>=4.0.0
python-docx>=1.0.0
```

(Adjust versions only if import smoke fails; prefer adding missing packages over changing pinned ones unnecessarily.)

- [x] **Step 3: Write Chinese `README.md`**

Must include sections:
1. 项目简介（论文题目 + LightweightRAG）
2. 功能列表（构建/增量/召回/重排/Ollama/Web 可观察/评测）
3. 环境要求（Python、Ollama、磁盘空间提示）
4. 安装依赖：`pip install -r requirements.txt`
5. 安装 Ollama 并拉取 `deepseek-r1:8b`
6. 下载模型到 `models/bge-m3` 与 `models/bge-reranker-v2-m3`
7. 将语料放入 `docs/`（说明不随仓库分发）
8. 启动：`python LightweightRAG.py`，浏览器访问提示的本地端口
9. Web 中构建知识库
10. 运行 SciFact：`python experiments/public_retrieval_eval.py`
11. **不要提交**的文件清单
12. 目录结构简表

- [x] **Step 4: Commit**

```powershell
git add .gitignore requirements.txt README.md
git commit -m "docs: add README and ignore local models/indexes for GitHub"
```

---

### Task 4: End-to-end verification + gap patches (sequential)

**Files:**
- Possibly modify any module that fails live verification
- Test: live Flask + Ollama + existing local `docs/` + models

**Interfaces:**
- Consumes: Tasks 1–3 outputs
- Produces: verification notes (what passed/failed) and minimal code patches for real gaps only

- [x] **Step 1: Ollama health**

```powershell
curl http://localhost:11434/api/tags
```

Expected: JSON including `deepseek-r1:8b` (or pull it).

- [x] **Step 2: Start app**

```powershell
python LightweightRAG.py
```

Expected: Flask listens (note printed host/port); open in browser.

- [x] **Step 3: Build or confirm KB**

Use Web「构建知识库」or existing index. Expected: panels show chunk counts; no crash.

- [x] **Step 4: QA smoke with `测试问题.txt`**

Ask 1–2 questions. Expected: streamed answer + retrieval/rerank panels + workflow updates.

- [x] **Step 5: Multi-turn + clear**

Ask a follow-up pronoun question; then clear conversation. Expected: history-aware answer; clear resets UI/state file messages.

- [ ] **Step 6: SciFact attempt**（进行中：缩小语料后台重跑，不阻塞其它交付）

```powershell
python experiments/public_retrieval_eval.py
```

Expected: writes under `experiments/results/` OR documented dependency failure with exact error.

- [x] **Step 7: Patch only real gaps found; re-smoke critical path; commit**

```powershell
git add -A
git status
git commit -m "fix: close live verification gaps for full thesis feature path"
```

(Only stage code/docs; do not force-add ignored models/indexes.)

- [x] **Step 8: Write handoff checklist for user**

In chat reply, list **GitHub 应上传** vs **不要上传** matching `.gitignore` and README.

---

## Parallel execution map

| Batch | Agents | Domains |
|-------|--------|---------|
| P1 | Agent-A Task1, Agent-B Task2, Agent-C Task3 | config/restore ‖ rerank path ‖ README/gitignore/reqs |
| P2 | Parent merges + resolves conflicts on `config_imports.py` if both A/B touched it | integrate |
| P3 | Parent or single agent Task4 | live E2E |

If Task1 and Task2 both edit `config_imports.py`, prefer: Task1 owns `config_imports.py`; Task2 only edits `rag_query.py` and imports `LOCAL_RERANK_MODEL_PATH` from `config` directly to avoid merge conflict.

## Spec coverage check

| Spec unit | Task |
|-----------|------|
| Unit A restore | Task 1 |
| Unit B config/path | Task 1 + 2 |
| Unit C feature completeness | Task 4 |
| Unit D GitHub packaging | Task 3 |
| Unit E verification | Task 4 |

## Self-review

- No TBD placeholders
- `LOCAL_RERANK_MODEL_PATH` naming consistent across Task 1–2
- Task 2 avoids editing `config_imports.py` when parallel with Task 1 (conflict avoidance)
- Flask missing from requirements captured in Task 3
