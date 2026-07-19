import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import faiss
import ir_datasets
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (  # noqa: E402
    CHAT_MODEL,
    DEFAULT_THRESHOLD,
    DEFAULT_TOP_K,
    DEFAULT_TOP_K_COMPRESSED,
    LOCAL_EMBEDDING_MODEL_PATH,
    OLLAMA_HOST,
    RERANK_MODEL,
)
from simpleRAG_included.rag_query import RAGQuerier  # noqa: E402


def _doc_text(doc, max_chars: int) -> str:
    title = str(getattr(doc, "title", "") or "").strip()
    text = str(getattr(doc, "text", "") or "").strip()
    content = f"{title}\n{text}".strip()
    if max_chars > 0 and len(content) > max_chars:
        return content[:max_chars]
    return content


def _dcg(relevances: Sequence[int]) -> float:
    return sum((rel / math.log2(index + 2)) for index, rel in enumerate(relevances))


def _metrics(ranking: Sequence[str], relevant: Dict[str, int], k: int) -> Dict[str, float]:
    top = list(ranking[:k])
    if not relevant:
        return {"hit": 0.0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0}

    hit = 1.0 if any(doc_id in relevant for doc_id in top) else 0.0
    recall = sum(1 for doc_id in top if doc_id in relevant) / float(len(relevant))

    mrr = 0.0
    for rank, doc_id in enumerate(top, start=1):
        if doc_id in relevant:
            mrr = 1.0 / rank
            break

    gains = [int(relevant.get(doc_id, 0)) for doc_id in top]
    ideal = sorted((int(value) for value in relevant.values()), reverse=True)[:k]
    ndcg = (_dcg(gains) / _dcg(ideal)) if ideal and _dcg(ideal) > 0 else 0.0
    return {"hit": hit, "recall": recall, "mrr": mrr, "ndcg": ndcg}


def _mean_metric(rows: Iterable[Dict[str, float]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return round(sum(values) / len(values), 4) if values else 0.0


def _load_qrels(dataset) -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str, int]] = {}
    for item in dataset.qrels_iter():
        if int(item.relevance) <= 0:
            continue
        qrels.setdefault(str(item.query_id), {})[str(item.doc_id)] = int(item.relevance)
    return qrels


def _select_queries(dataset, qrels: Dict[str, Dict[str, int]], max_queries: int) -> List[Tuple[str, str]]:
    selected: List[Tuple[str, str]] = []
    for query in dataset.queries_iter():
        query_id = str(query.query_id)
        if query_id not in qrels:
            continue
        selected.append((query_id, str(query.text)))
        if len(selected) >= max_queries:
            break
    return selected


def _load_corpus(
    dataset,
    selected_queries: List[Tuple[str, str]],
    qrels: Dict[str, Dict[str, int]],
    max_corpus_docs: int,
    max_doc_chars: int,
):
    required_doc_ids = {
        doc_id
        for query_id, _ in selected_queries
        for doc_id in qrels.get(query_id, {})
    }

    docs: Dict[str, str] = {}
    for doc in dataset.docs_iter():
        doc_id = str(doc.doc_id)
        if len(docs) < max_corpus_docs or doc_id in required_doc_ids:
            docs[doc_id] = _doc_text(doc, max_doc_chars)
        if len(docs) >= max_corpus_docs and required_doc_ids.issubset(docs.keys()):
            break

    missing = sorted(required_doc_ids - docs.keys())
    if missing:
        raise RuntimeError(f"Missing relevant documents in sampled corpus: {missing[:10]}")
    return docs


def run(args) -> Dict[str, object]:
    dataset = ir_datasets.load(args.dataset)
    qrels = _load_qrels(dataset)
    selected_queries = _select_queries(dataset, qrels, args.max_queries)
    if not selected_queries:
        raise RuntimeError("No queries with relevance labels were selected.")

    docs = _load_corpus(dataset, selected_queries, qrels, args.max_corpus_docs, args.max_doc_chars)
    doc_ids = list(docs.keys())
    doc_texts = [docs[doc_id] for doc_id in doc_ids]

    model_load_start = time.perf_counter()
    embedding_model = SentenceTransformer(LOCAL_EMBEDDING_MODEL_PATH)
    model_load_seconds = time.perf_counter() - model_load_start

    encode_start = time.perf_counter()
    doc_embeddings = embedding_model.encode(
        doc_texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")
    corpus_encode_seconds = time.perf_counter() - encode_start

    index = faiss.IndexFlatIP(doc_embeddings.shape[1])
    index.add(doc_embeddings)

    querier = RAGQuerier(OLLAMA_HOST, CHAT_MODEL, RERANK_MODEL)

    vector_rows: List[Dict[str, float]] = []
    rerank_rows: List[Dict[str, float]] = []
    examples = []
    retrieve_times = []
    rerank_times = []

    for query_id, query_text in selected_queries:
        query_start = time.perf_counter()
        query_vec = embedding_model.encode(query_text, normalize_embeddings=True).astype("float32")
        scores, indices = index.search(np.asarray([query_vec], dtype="float32"), args.candidate_k)
        retrieve_times.append(time.perf_counter() - query_start)

        vector_ranking = [doc_ids[int(idx)] for idx in indices[0] if int(idx) >= 0]
        vector_rows.append(_metrics(vector_ranking, qrels[query_id], args.metric_k))

        candidates = [
            {
                "path": doc_id,
                "chunk_index": 0,
                "content": docs[doc_id],
                "score": float(scores[0][rank]),
            }
            for rank, doc_id in enumerate(vector_ranking)
        ]

        rerank_start = time.perf_counter()
        reranked = querier._rerank_results(query_text, candidates, top_k=args.metric_k)
        rerank_times.append(time.perf_counter() - rerank_start)
        rerank_ranking = [str(item["path"]) for item in reranked]
        rerank_rows.append(_metrics(rerank_ranking, qrels[query_id], args.metric_k))

        if len(examples) < 5:
            examples.append(
                {
                    "query_id": query_id,
                    "query": query_text,
                    "relevant_doc_ids": list(qrels[query_id].keys()),
                    "vector_top": vector_ranking[: args.metric_k],
                    "rerank_top": rerank_ranking[: args.metric_k],
                }
            )

    def summarize(rows: List[Dict[str, float]], avg_time: float) -> Dict[str, float]:
        return {
            f"Hit@{args.metric_k}": _mean_metric(rows, "hit"),
            f"Recall@{args.metric_k}": _mean_metric(rows, "recall"),
            f"MRR@{args.metric_k}": _mean_metric(rows, "mrr"),
            f"nDCG@{args.metric_k}": _mean_metric(rows, "ndcg"),
            "AvgTimeSeconds": round(avg_time, 4),
        }

    vector_summary = summarize(vector_rows, sum(retrieve_times) / len(retrieve_times))
    rerank_summary = summarize(
        rerank_rows,
        (sum(retrieve_times) + sum(rerank_times)) / len(rerank_times),
    )

    return {
        "dataset": args.dataset,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "embedding_model_path": LOCAL_EMBEDDING_MODEL_PATH,
            "rerank_model": RERANK_MODEL,
            "default_top_k": DEFAULT_TOP_K,
            "default_top_k_compressed": DEFAULT_TOP_K_COMPRESSED,
            "default_threshold": DEFAULT_THRESHOLD,
            "max_queries": args.max_queries,
            "max_corpus_docs": args.max_corpus_docs,
            "max_doc_chars": args.max_doc_chars,
            "candidate_k": args.candidate_k,
            "metric_k": args.metric_k,
        },
        "runtime": {
            "model_load_seconds": round(model_load_seconds, 3),
            "corpus_encode_seconds": round(corpus_encode_seconds, 3),
            "reranker_status": querier.get_reranker_status(),
        },
        "sample": {
            "query_count": len(selected_queries),
            "corpus_doc_count": len(docs),
        },
        "results": {
            "vector": vector_summary,
            "vector_plus_rerank": rerank_summary,
        },
        "examples": examples,
    }


def write_outputs(result: Dict[str, object], output_dir: Path) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"public_retrieval_eval_{stamp}.json"
    md_path = output_dir / f"public_retrieval_eval_{stamp}.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    metric_names = list(result["results"]["vector"].keys())
    lines = [
        "# Public Retrieval Evaluation",
        "",
        f"- Dataset: `{result['dataset']}`",
        f"- Queries: {result['sample']['query_count']}",
        f"- Corpus documents: {result['sample']['corpus_doc_count']}",
        f"- Reranker status: `{result['runtime']['reranker_status']}`",
        "",
        "| Method | " + " | ".join(metric_names) + " |",
        "|---|" + "|".join(["---:"] * len(metric_names)) + "|",
    ]
    for label, row in [
        ("基础向量检索", result["results"]["vector"]),
        ("向量检索+重排序", result["results"]["vector_plus_rerank"]),
    ]:
        lines.append("| " + label + " | " + " | ".join(str(row[name]) for name in metric_names) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate vector retrieval and reranking on a public IR dataset.")
    parser.add_argument("--dataset", default="beir/scifact/test")
    parser.add_argument("--max-queries", type=int, default=20)
    parser.add_argument("--max-corpus-docs", type=int, default=2000)
    parser.add_argument("--max-doc-chars", type=int, default=1200)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--metric-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", default=str(ROOT_DIR / "experiments" / "results"))
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    output = run(cli_args)
    json_file, md_file = write_outputs(output, Path(cli_args.output_dir))
    print(f"JSON: {json_file}")
    print(f"Markdown: {md_file}")
