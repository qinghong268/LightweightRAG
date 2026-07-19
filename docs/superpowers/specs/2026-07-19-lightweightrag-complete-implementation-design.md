# LightweightRAG Complete Implementation Design

Date: 2026-07-19  
Project: `D:\RAGprojects\LightweightRAG`  
Thesis reference: 王锦鸿《基于 LLM 信息检索增强的智能问答系统设计与实现》终稿 docx  

## 1. Goal and non-goals

### Goal
Deliver a **complete, runnable** LightweightRAG that implements all thesis-described product functions, verified on the author’s machine with Ollama (`deepseek-r1:8b`), and prepared for a **public GitHub demo**.

Success means:
1. Flask app starts; browser QA shows retrieval, rerank, workflow, and streamed answers.
2. Knowledge-base build works (multi-format docs, optional `.doc` preprocess, incremental fingerprint path).
3. Local embedding (`models/bge-m3`) and reranker (`models/bge-reranker-v2-m3`) load via stable absolute paths.
4. Multi-turn conversation + clear session + `conversation_state.json` persistence work.
5. Offline eval path works: `rag_evaluator.py` + `experiments/public_retrieval_eval.py` (SciFact) when deps allow.
6. Repository ships **code + config + README**; excludes models, corpus, indexes, and caches.
7. Final handoff includes an explicit **GitHub upload / do-not-upload** checklist.

### Non-goals
- Rewriting the thesis manuscript or restoring `graduation_thesis/` / defense `outputs/` into the engineering repo.
- Cloud multi-tenancy, auth, hybrid sparse+dense retrieval, or other features not specified in the thesis.
- Uploading `models/`, full `docs/` corpus, `faiss_index.bin`, `embedding_cache.json`, or `knowledge_base.db` to GitHub.

## 2. Constraints (confirmed with user)

| Topic | Decision |
|-------|----------|
| Scope | Full thesis feature completeness (not defense-minimal) |
| Runtime verification | Author has Ollama + `deepseek-r1:8b`; agent must run build/QA and attempt SciFact |
| Archive restore | Restore runtime-needed files; keep thesis/PPT archives outside the repo |
| GitHub | Code + config + README only; user self-manages models and knowledge base via README |

## 3. Current state (gap scan)

### Present and aligned with thesis (code-level)
- Offline + online architecture: `LightweightRAG.py`, `simpleRAG_content.py`, `simpleRAG_included/rag_build.py`, `rag_query.py`, `rag_helpers.py`
- Document load / encoding / discovery: `document_loader.py`
- `.doc` preprocess: `doc_converter.py`
- Chunking defaults 220/40 + semantic merge threshold 0.85: `text_splitter.py`, `config.py`
- FAISS + SQLite + metadata + embedding cache integration
- Rerank hook `_rerank_results`, stream API `/api/chat/stream`, KB build/panels, conversation clear
- Workflow / evidence / online-eval HTML generation
- SciFact script: `experiments/public_retrieval_eval.py`
- Local models directories exist for bge-m3 and bge-reranker-v2-m3

### Gaps / defects to fix
1. `conversation_state.json` missing from workspace (moved to `D:\RAGprojects\LightweightRAG_旁置材料`); must restore or recreate empty state at `CONVERSATION_STATE_FILE`.
2. `config.py` header docstring encoding corruption.
3. Reranker path uses cwd-relative `./models/{name}` — fragile; must resolve from `BASE_DIR`.
4. `.gitignore` does not yet exclude models/indexes/caches/db — required before GitHub.
5. No project `README.md` for zero-to-run deployment.
6. `测试问题.txt` moved out — restore for local demo prompts (small file; optional for git via sample copy in README).
7. End-to-end runtime not yet verified in this session (Ollama/build/QA/SciFact).

Thesis-only materials remain in `LightweightRAG_旁置材料` and stay out of the engineering tree.

## 4. Target architecture (unchanged layering)

```
docs/ → load/split → SQLite + FAISS + metadata.json + embedding_cache.json
user Q → bge-m3 retrieve → threshold filter → bge-reranker-v2-m3
     → prompt context → Ollama deepseek-r1:8b stream
     → Flask APIs → Web workflow / evidence / online-eval panels
```

Orchestration remains: `LightweightRAG.py` → `SimpleRAG` → build/query/conversation modules.

### Default parameters (thesis)
- `DEFAULT_TOP_K = 5`
- `DEFAULT_THRESHOLD = 0.3`
- `DEFAULT_TOP_K_COMPRESSED = 3` (rerank keep)
- `MIN_RETRIEVE_KEEP = 2`
- `CHUNK_SIZE_DEFAULT = 220`, `CHUNK_OVERLAP_DEFAULT = 40`
- `SEMANTIC_SPLITTER_THRESHOLD = 0.85`
- `CHAT_MODEL = deepseek-r1:8b`, `OLLAMA_HOST = http://localhost:11434`

### Fallback behaviors (must preserve)
- Bad files skipped with build report notes
- Semantic splitter failure → basic splitter
- Reranker unavailable → vector Top-K
- Snapshot inconsistency → full rebuild
- Ollama down → explicit error in UI/API (no silent empty success)

## 5. Work plan (implementation units)

### Unit A — Workspace integrity
- Restore `conversation_state.json` (or recreate empty via `ConversationStore`)
- Restore `测试问题.txt` to project root for local demo questions
- Do **not** restore `graduation_thesis/`, `outputs/`, `THESIS_WORKSPACE.txt` into the repo root

### Unit B — Config and path hardening
- Fix `config.py` encoding / docstring
- Add explicit `LOCAL_RERANK_MODEL_PATH = BASE_DIR / "models" / "bge-reranker-v2-m3"` (or equivalent)
- Update `rag_query.py` (and any duplicate path logic) to use `BASE_DIR`-anchored paths
- Audit `requirements.txt` against imports actually used

### Unit C — Feature completeness pass
Run thesis checklist against live behavior; patch only real gaps:
1. Multi-format ingest + Chinese encodings + doc convert
2. Incremental fingerprint build + cache reuse
3. Retrieve → filter → rerank → generate
4. Multi-turn + clear session
5. Web observability panels
6. Offline evaluator + SciFact script

Strategy: **measure first, patch minimally** — do not rewrite working modules.

### Unit D — GitHub demo packaging
- Expand `.gitignore` for: `models/`, knowledge artifacts (`*.bin`, `knowledge_base.db`, `embedding_cache.json`, `conversation_state.json`), `__pycache__/`, local corpus policy for `docs/` (keep `docs/superpowers/` tracked; ignore RAG corpus content or use `docs/.gitkeep` + README instructions)
- Author Chinese `README.md`: env, deps, Ollama, model download layout, how to populate `docs/`, build KB, start app, run SciFact, what not to commit
- Provide final upload checklist in the implementation handoff message

### Unit E — Verification (required)
1. Dependency/model path check
2. Start Flask; open home page
3. Build KB (or confirm existing index consistent)
4. Ask questions from `测试问题.txt`: retrieval + rerank + stream + workflow visible
5. Multi-turn + clear conversation
6. Attempt SciFact eval; record metrics path or failure cause

## 6. GitHub artifact policy

### Upload (yes)
- Python sources: `LightweightRAG.py`, `simpleRAG_content.py`, `config.py`, `document_loader.py`, `doc_converter.py`, `text_splitter.py`, `prompts.py`, `rag_evaluator.py`, `simpleRAG_included/**`
- Web: `index.html`, `lightweightrag.css`, `lightweightrag.js`
- `requirements.txt`, `.gitignore`, `README.md`
- `experiments/public_retrieval_eval.py` (+ small result samples only if tiny and non-sensitive)
- `eval_dataset_template.json` if used by evaluator
- `docs/superpowers/**` (design/plans)

### Do not upload (no)
- `models/**`
- Full RAG corpus under `docs/` (except tracked superpowers docs / optional tiny sample)
- `faiss_index.bin`, `knowledge_base.db`, `metadata.json` (runtime-generated; rebuildable), `embedding_cache.json`
- `conversation_state.json`
- `graduation_thesis/`, defense `outputs/`, private thesis archives
- `__pycache__/`, virtualenvs, `.lock` temp model dirs

Note: Author’s local machine may keep corpus/models/indexes for running; git must not track them.

## 7. Risks

| Risk | Mitigation |
|------|------------|
| SciFact deps (`ir_datasets`) missing/slow | Script still ships; README documents install; verification records skip reason if blocked |
| First rerank latency ~8s | Expected per thesis; not treated as defect |
| `docs/` both corpus and design path | gitignore rules distinguish `docs/superpowers/**` vs corpus |
| Accidental large-file commit | `.gitignore` + README warning + upload checklist |

## 8. Approval record

- Approach: complete-feature repair (not defense-minimal, not full rewrite)
- Design §1 Goal/boundary: approved
- Design §2 Fix list/order: approved
- Design §3 Architecture/risks: approved
- Verification mode: local Ollama A
- GitHub mode: code+config+README only

## 9. Next step after user reviews this spec

Invoke `writing-plans` to produce `docs/superpowers/plans/2026-07-19-lightweightrag-complete-implementation.md`, then implement and verify.
