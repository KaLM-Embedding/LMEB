from __future__ import annotations
from collections import defaultdict
from typing import Mapping, Sequence
from mteb.types import RelevantDocumentsType


def recall_cap(
    qrels: RelevantDocumentsType,
    results: Mapping[str, Mapping[str, float]],
    k_values: Sequence[int],
    skip_first_result: bool = False,
) -> dict[str, list[float | None]]:
    """
    R_cap@k = (# relevant in top-k) / min(#total_relevant, k)

    - Returns None if the query has no relevant documents (denominator = 0).
    - If skip_first_result=True, the first ranked result is removed before applying the top-k cutoff
    (useful for special evaluation settings in some datasets).
    """
    capped_recall: dict[str, list[float | None]] = defaultdict(list)
    k_values = list(k_values)
    k_max = max(k_values)

    for query_id, doc_scores in results.items():
        ranked = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)

        if skip_first_result:
            ranked = ranked[1:]

        top_hits = ranked[:k_max]

        num_relevant_total = sum(1 for doc_id, rel in qrels[query_id].items() if rel > 0)

        for k in k_values:
            hits = 0
            for doc_id, _ in top_hits[:k]:
                if qrels[query_id].get(doc_id, 0) > 0:
                    hits += 1

            denom = min(num_relevant_total, k)
            if denom == 0:
                capped_recall[f"R_cap_at_{k}"].append(None)
            else:
                capped_recall[f"R_cap_at_{k}"].append(hits / denom)

    return capped_recall


def evaluate_recall_cap(
    qrels: RelevantDocumentsType,
    results: dict[str, dict[str, float]],
    k_values: Sequence[int],
    ignore_identical_ids: bool = False,
    skip_first_result: bool = False,
) -> tuple[dict[str, list[float | None]], dict[str, float | None]]:
    """
    Compute recall_cap separately, following a similar parameter style to `Retriever.evaluate`.

    Returns:
    - recall_cap_lists: {"R_cap_at_10": [..per query..], ...}  (per-query values)
    - recall_cap_avg:   {"R_cap_at_10": 0.12345, ...}         (macro average, ignoring None; if all are None, value is None)
    """
    if ignore_identical_ids:
        results = {qid: dict(rels) for qid, rels in results.items()}
        for qid, rels in results.items():
            if qid in rels:
                rels.pop(qid, None)

    recall_cap_lists = recall_cap(qrels, results, k_values, skip_first_result=skip_first_result)

    recall_cap_avg: dict[str, float | None] = {}
    for metric_name, vals in recall_cap_lists.items():
        valid = [v for v in vals if v is not None]
        if not valid:
            recall_cap_avg[metric_name] = None
        else:
            recall_cap_avg[metric_name] = round(sum(valid) / len(valid), 5)

    return recall_cap_lists, recall_cap_avg