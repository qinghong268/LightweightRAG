import json
import math
import re
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


RELEVANT_CHUNK_KEYS = (
    "relevant_chunks",
    "ground_truth_chunks",
    "positive_chunks",
    "relevant_passages",
    "relevant_contexts",
    "relevant_chunk_keys",
    "相关分块",
    "相关片段",
    "相关上下文",
)

ANSWER_KEYS = (
    "ground_truth_answer",
    "expected_answer",
    "reference_answer",
    "标准答案",
    "参考答案",
)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _tokenize_text(text: str) -> List[str]:
    text = str(text or "").lower()

    return re.findall(r"[\u3400-\u9fff]|[a-z0-9]+", text)


def _normalize_answer_text(text: str) -> str:
    tokens = _tokenize_text(text)
    return " ".join(tokens).strip()


def _answer_exact_match(prediction: str, ground_truth: str) -> float:
    return 1.0 if _normalize_answer_text(prediction) == _normalize_answer_text(ground_truth) else 0.0


def _answer_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = _tokenize_text(prediction)
    gold_tokens = _tokenize_text(ground_truth)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    gold_counts: Dict[str, int] = {}
    for token in gold_tokens:
        gold_counts[token] = gold_counts.get(token, 0) + 1

    overlap = 0
    for token in pred_tokens:
        count = gold_counts.get(token, 0)
        if count > 0:
            overlap += 1
            gold_counts[token] = count - 1

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return (2.0 * precision * recall) / (precision + recall)


def _normalize_path(path: Any) -> str:
    return str(path or "").strip().replace("\\", "/").lower()


def _paths_equivalent(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    if a.endswith("/" + b) or b.endswith("/" + a):
        return True
    return False


def _parse_chunk_key(text: Any) -> Optional[Dict[str, Any]]:
    raw = str(text or "").strip()
    if not raw:
        return None
    match = re.match(r"^(.*)#chunk(\d+)$", raw, flags=re.IGNORECASE)
    if match:
        return {
            "path": _normalize_path(match.group(1)),
            "chunk_index": int(match.group(2)),
            "relevance": 1.0,
        }
    return {
        "path": _normalize_path(raw),
        "chunk_index": None,
        "relevance": 1.0,
    }


def _parse_relevant_chunks(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    parsed: List[Dict[str, Any]] = []
    for key in RELEVANT_CHUNK_KEYS:
        value = sample.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, str):
                chunk = _parse_chunk_key(item)
                if chunk:
                    parsed.append(chunk)
                continue

            if not isinstance(item, dict):
                continue

            path_value = item.get("path")
            if path_value is None:
                path_value = item.get("source")
            if path_value is None:
                path_value = item.get("file")
            if path_value is None:
                path_value = item.get("doc")
            if path_value is None:
                path_value = item.get("路径")
            if path_value is None:
                path_value = item.get("文件")
            if path_value is None:
                path_value = item.get("文档")

            chunk_index = item.get("chunk_index")
            if chunk_index is None and item.get("chunk") is not None:
                chunk_index = item.get("chunk")
            if chunk_index is None and item.get("分块索引") is not None:
                chunk_index = item.get("分块索引")
            if chunk_index is None and item.get("分块") is not None:
                chunk_index = item.get("分块")

            parsed_chunk_index = None
            if chunk_index is not None:
                try:
                    parsed_chunk_index = int(chunk_index)
                except (TypeError, ValueError):
                    parsed_chunk_index = None

            if path_value is None:
                key_text = item.get("key") or item.get("chunk_key")
                chunk = _parse_chunk_key(key_text)
                if chunk:
                    chunk["relevance"] = _safe_float(
                        item.get("relevance", item.get("score", item.get("label", item.get("相关性", 1.0)))),
                        1.0,
                    )
                    parsed.append(chunk)
                continue

            parsed.append(
                {
                    "path": _normalize_path(path_value),
                    "chunk_index": parsed_chunk_index,
                    "relevance": _safe_float(
                        item.get("relevance", item.get("score", item.get("label", item.get("相关性", 1.0)))),
                        1.0,
                    ),
                }
            )

    dedup: Dict[str, Dict[str, Any]] = {}
    for chunk in parsed:
        chunk_path = _normalize_path(chunk.get("path"))
        chunk_index = chunk.get("chunk_index")
        chunk_key = f"{chunk_path}#chunk{chunk_index}" if chunk_index is not None else f"{chunk_path}#doc"
        existing = dedup.get(chunk_key)
        if existing is None or float(chunk.get("relevance", 0.0)) > float(existing.get("relevance", 0.0)):
            dedup[chunk_key] = {
                "path": chunk_path,
                "chunk_index": chunk_index,
                "relevance": max(0.0, float(chunk.get("relevance", 1.0))),
            }
    return list(dedup.values())


def _match_relevance(result_item: Dict[str, Any], relevant_chunks: List[Dict[str, Any]]) -> float:
    result_path = _normalize_path(result_item.get("path"))
    result_chunk_index = result_item.get("chunk_index")
    try:
        result_chunk_index = int(result_chunk_index)
    except Exception:
        result_chunk_index = None

    for chunk in relevant_chunks:
        rel_path = chunk.get("path", "")
        rel_chunk_index = chunk.get("chunk_index")
        if rel_chunk_index is not None and result_chunk_index != int(rel_chunk_index):
            continue
        if _paths_equivalent(result_path, rel_path):
            return max(0.0, float(chunk.get("relevance", 1.0)))
    return 0.0


def _compute_ranking_metrics(
    results: List[Dict[str, Any]],
    relevant_chunks: List[Dict[str, Any]],
    k: int,
) -> Dict[str, float]:
    k = max(1, int(k))
    top_results = list(results or [])[:k]
    total_relevant = len(relevant_chunks)

    gains: List[float] = []
    hit_count = 0
    first_relevant_rank: Optional[int] = None

    for rank, item in enumerate(top_results, start=1):
        relevance = _match_relevance(item, relevant_chunks)
        gains.append(relevance)
        if relevance > 0:
            hit_count += 1
            if first_relevant_rank is None:
                first_relevant_rank = rank

    recall = (hit_count / total_relevant) if total_relevant > 0 else 0.0
    precision = hit_count / float(k)
    hit_rate = 1.0 if hit_count > 0 else 0.0
    mrr = (1.0 / float(first_relevant_rank)) if first_relevant_rank else 0.0

    dcg = 0.0
    for rank, gain in enumerate(gains, start=1):
        dcg += (2.0**gain - 1.0) / math.log2(rank + 1.0)

    ideal_gains = sorted(
        [max(0.0, float(item.get("relevance", 1.0))) for item in relevant_chunks],
        reverse=True,
    )[:k]
    idcg = 0.0
    for rank, gain in enumerate(ideal_gains, start=1):
        idcg += (2.0**gain - 1.0) / math.log2(rank + 1.0)
    ndcg = (dcg / idcg) if idcg > 0 else 0.0

    return {
        "hit_rate_at_k": hit_rate,
        "recall_at_k": recall,
        "precision_at_k": precision,
        "mrr_at_k": mrr,
        "ndcg_at_k": ndcg,

        "context_recall_at_k": recall,
        "context_precision_at_k": precision,
        "retrieved_relevant_count": float(hit_count),
        "relevant_count": float(total_relevant),
    }


def _average(values: Iterable[float]) -> float:
    numeric = [float(v) for v in values]
    return statistics.fmean(numeric) if numeric else 0.0


def _p95(values: List[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(float(v) for v in values)
    index = max(0, math.ceil(0.95 * len(sorted_values)) - 1)
    return sorted_values[index]


def _round_metrics(payload: Any, digits: int = 6) -> Any:
    if isinstance(payload, dict):
        return {key: _round_metrics(value, digits=digits) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_round_metrics(value, digits=digits) for value in payload]
    if isinstance(payload, float):
        return round(payload, digits)
    return payload


def _load_eval_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    suffix = dataset_path.suffix.lower()
    text = dataset_path.read_text(encoding="utf-8")

    if suffix == ".jsonl":
        records: List[Dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(payload)
        return records

    payload = json.loads(text)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("samples"), list):
            return [item for item in payload["samples"] if isinstance(item, dict)]
        if isinstance(payload.get("data"), list):
            return [item for item in payload["data"] if isinstance(item, dict)]
        if isinstance(payload.get("样本"), list):
            return [item for item in payload["样本"] if isinstance(item, dict)]
        if isinstance(payload.get("数据"), list):
            return [item for item in payload["数据"] if isinstance(item, dict)]
        if payload.get("question"):
            return [payload]
        if payload.get("问题"):
            return [payload]
    raise ValueError("Unsupported evaluation dataset format. Expect JSON/JSONL with sample objects.")


def build_evaluation_dataset_template() -> Dict[str, Any]:
    return {
        "description": "RAG evaluation dataset template",
        "supported_format": "json | jsonl",
        "sample": {
            "id": "q1",
            "question": "什么是RAG？",
            "ground_truth_answer": "RAG 是检索增强生成，通过先检索知识再生成答案。",
            "relevant_chunks": [
                {"path": "docs/rag_intro.txt", "chunk_index": 0, "relevance": 2},
                {"path": "docs/rag_intro.txt", "chunk_index": 1, "relevance": 1},
            ],
            "top_k_retrieve": 5,
            "top_k_compressed": 3,
            "score_threshold": 0.3,
            "history": [],
        },
        "metric_notes": {
            "retrieval": ["hit_rate_at_k", "recall_at_k", "precision_at_k", "mrr_at_k", "ndcg_at_k"],
            "answer": ["exact_match", "f1"],
        },
    }


def run_rag_evaluation(
    rag: Any,
    dataset_path: Path,
    default_top_k_retrieve: int,
    default_top_k_compressed: int,
    default_score_threshold: float,
) -> Dict[str, Any]:
    samples = _load_eval_dataset(dataset_path)
    started_at = time.perf_counter()

    retrieval_pre_scores: Dict[str, List[float]] = {
        "hit_rate_at_k": [],
        "recall_at_k": [],
        "precision_at_k": [],
        "mrr_at_k": [],
        "ndcg_at_k": [],
        "context_recall_at_k": [],
        "context_precision_at_k": [],
    }
    retrieval_post_scores: Dict[str, List[float]] = {
        "hit_rate_at_k": [],
        "recall_at_k": [],
        "precision_at_k": [],
        "mrr_at_k": [],
        "ndcg_at_k": [],
        "context_recall_at_k": [],
        "context_precision_at_k": [],
    }
    answer_em_scores: List[float] = []
    answer_f1_scores: List[float] = []
    latency_ms_list: List[float] = []

    sample_results: List[Dict[str, Any]] = []
    error_count = 0
    retrieval_labeled_count = 0
    answer_labeled_count = 0

    for index, sample in enumerate(samples, start=1):
        sample_id = sample.get("id", sample.get("编号", index))
        question = str(sample.get("question", sample.get("问题", "")) or "").strip()
        if not question:
            error_count += 1
            sample_results.append(
                {
                    "id": sample_id,
                    "error": "缺少问题字段",
                }
            )
            continue

        top_k_retrieve = _safe_int(
            sample.get(
                "top_k_retrieve",
                sample.get("top_k_ret", sample.get("检索前K", sample.get("召回前K", default_top_k_retrieve))),
            ),
            default_top_k_retrieve,
        )
        top_k_compressed = _safe_int(
            sample.get("top_k_compressed", sample.get("top_k_comp", sample.get("重排前K", default_top_k_compressed))),
            default_top_k_compressed,
        )
        score_threshold = _safe_float(sample.get("score_threshold", sample.get("阈值", default_score_threshold)), default_score_threshold)
        history = sample.get("history", sample.get("历史"))
        if not isinstance(history, list):
            history = []

        relevant_chunks = _parse_relevant_chunks(sample)
        ground_truth_answer = ""
        for key in ANSWER_KEYS:
            if sample.get(key):
                ground_truth_answer = str(sample.get(key) or "")
                break

        answer_text = ""
        retrieved_results: List[Dict[str, Any]] = []
        reranked_results: List[Dict[str, Any]] = []
        diagnostics: Dict[str, Any] = {}
        elapsed_ms = 0.0

        try:
            start = time.perf_counter()
            answer_text = str(
                rag.answer_question(
                    question,
                    top_k_retrieve=top_k_retrieve,
                    top_k_compressed=top_k_compressed,
                    score_threshold=score_threshold,
                    history=history,
                )
                or ""
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latency_ms_list.append(elapsed_ms)

            cache_key = rag._build_query_cache_key(
                question,
                history,
                top_k_retrieve,
                top_k_compressed,
                score_threshold,
            )
            cached = rag._get_cached_query_entry(cache_key) or {}
            if isinstance(cached, dict):
                retrieved_results = list(cached.get("retrieved_results", []) or [])
                reranked_results = list(cached.get("reranked_results", []) or [])
                diagnostics = dict(cached.get("diagnostics", {}) or {})
        except Exception as exc:
            error_count += 1
            sample_results.append(
                {
                    "id": sample_id,
                    "question": question,
                    "error": str(exc),
                }
            )
            continue

        sample_payload: Dict[str, Any] = {
            "id": sample_id,
            "question": question,
            "latency_ms": elapsed_ms,
            "top_k_retrieve": top_k_retrieve,
            "top_k_compressed": top_k_compressed,
            "score_threshold": score_threshold,
            "retrieved_count": len(retrieved_results),
            "reranked_count": len(reranked_results),
            "diagnostics": diagnostics,
        }

        if relevant_chunks:
            retrieval_labeled_count += 1
            pre_metrics = _compute_ranking_metrics(retrieved_results, relevant_chunks, top_k_retrieve)
            post_source = reranked_results if reranked_results else retrieved_results
            post_metrics = _compute_ranking_metrics(post_source, relevant_chunks, top_k_compressed)
            sample_payload["retrieval_pre_rerank"] = pre_metrics
            sample_payload["retrieval_post_rerank"] = post_metrics
            for key in retrieval_pre_scores:
                retrieval_pre_scores[key].append(pre_metrics[key])
                retrieval_post_scores[key].append(post_metrics[key])

        if ground_truth_answer:
            answer_labeled_count += 1
            exact_match = _answer_exact_match(answer_text, ground_truth_answer)
            f1_score = _answer_f1(answer_text, ground_truth_answer)
            answer_em_scores.append(exact_match)
            answer_f1_scores.append(f1_score)
            sample_payload["answer_metrics"] = {
                "exact_match": exact_match,
                "f1": f1_score,
            }

        sample_results.append(sample_payload)

    total_time_ms = (time.perf_counter() - started_at) * 1000.0
    summary = {
        "dataset_path": str(dataset_path.resolve(strict=False)),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "total_samples": len(samples),
        "successful_samples": len(samples) - error_count,
        "error_samples": error_count,
        "retrieval_labeled_samples": retrieval_labeled_count,
        "answer_labeled_samples": answer_labeled_count,
        "total_eval_time_ms": total_time_ms,
        "avg_latency_ms": _average(latency_ms_list),
        "p95_latency_ms": _p95(latency_ms_list),
    }

    retrieval_metrics = {
        "pre_rerank": {key: _average(values) for key, values in retrieval_pre_scores.items()},
        "post_rerank": {key: _average(values) for key, values in retrieval_post_scores.items()},
    }
    answer_metrics = {
        "exact_match": _average(answer_em_scores),
        "f1": _average(answer_f1_scores),
    }

    return _round_metrics(
        {
            "summary": summary,
            "retrieval_metrics": retrieval_metrics,
            "answer_metrics": answer_metrics,
            "samples": sample_results,
            "metric_definitions": {
                "retrieval": {
                    "hit_rate_at_k": "前K结果中是否至少出现1条相关分块。",
                    "recall_at_k": "前K结果命中的相关分块数 / 全部相关分块数。",
                    "precision_at_k": "前K结果命中的相关分块数 / K。",
                    "mrr_at_k": "首个相关分块排名倒数的平均值。",
                    "ndcg_at_k": "考虑相关性等级与排序位置的归一化折损增益。",
                },
                "answer": {
                    "exact_match": "回答与标准答案归一化后完全一致的比例。",
                    "f1": "基于中英文混合分词的词级重叠 F1 分数。",
                },
            },
        }
    )
