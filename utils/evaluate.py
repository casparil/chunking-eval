from typing import Any, Dict, List, Tuple

import pytrec_eval

from config import K_VALUES


def _compute_results(qrels: dict, results: dict, k_values: List[int]) -> Tuple[
    Dict[str, List[float]],
    Dict[str, List[float]],
    Dict[str, List[float]],
    Dict[str, List[float]],
]:
    """
    Computes a set of retrieval metrics at different k-values for the passed results dictionary, which contains query
    ids, document ids and the retrieval score. The computed metrics are returned as dictionaries.

    :param qrels: The dictionary containing information on relevant documents for each query.
    :param results: The retrieved documents per query along with their score.
    :param k_values: The k-values for which retrieval metrics are computed.
    :return: The calculated retrieval metrics.
    """
    all_ndcgs, all_aps, all_recalls, all_precisions = {}, {}, {}, {}

    for k in k_values:
        all_ndcgs[f"NDCG@{k}"] = []
        all_aps[f"MAP@{k}"] = []
        all_recalls[f"Recall@{k}"] = []
        all_precisions[f"P@{k}"] = []

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {map_string, ndcg_string, recall_string, precision_string}
    )
    scores = evaluator.evaluate(results)

    for query_id in scores.keys():
        for k in k_values:
            all_ndcgs[f"NDCG@{k}"].append(scores[query_id]["ndcg_cut_" + str(k)])
            all_aps[f"MAP@{k}"].append(scores[query_id]["map_cut_" + str(k)])
            all_recalls[f"Recall@{k}"].append(scores[query_id]["recall_" + str(k)])
            all_precisions[f"P@{k}"].append(scores[query_id]["P_" + str(k)])

    ndcg, _map, recall, precision = (
        all_ndcgs.copy(),
        all_aps.copy(),
        all_recalls.copy(),
        all_precisions.copy(),
    )

    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(sum(ndcg[f"NDCG@{k}"]) / len(scores), 5) if len(scores) > 0 else 0
        _map[f"MAP@{k}"] = round(sum(_map[f"MAP@{k}"]) / len(scores), 5) if len(scores) > 0 else 0
        recall[f"Recall@{k}"] = round(sum(recall[f"Recall@{k}"]) / len(scores), 5) if len(scores) > 0 else 0
        precision[f"P@{k}"] = round(sum(precision[f"P@{k}"]) / len(scores), 5) if len(scores) > 0 else 0

    return ndcg, _map, recall, precision


def _get_rankings(results: dict, qrels_rel: dict, qrels: dict, top_k: int) -> Dict[Any, Dict[str, Dict[str, int]]]:
    """
    Stores the position of all relevant, random and distractor documents retrieved in the top-k results for a certain
    query in a dictionary.

    :param results: Dictionary containing the results of the top-k retrieval.
    :param qrels_rel: The qrels containing only relevant documents.
    :param qrels: The qrels that also contain information on distractors.
    :return: The constructed dictionary.
    """
    rankings = {}
    for qid in results:
        rankings[qid] = {
            "relevant": {},
            "distractor": {},
            "random": {},
        }
        scores = {}

        for cid in results[qid]:
            scores[cid] = results[qid][cid]

        scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))

        for idx, cid in enumerate(scores):
            if idx >= top_k:
                break
            if cid in qrels_rel[qid] and qrels_rel[qid][cid] > 0:
                rankings[qid]["relevant"][cid] = idx + 1
            elif cid in qrels[qid] and qrels[qid][cid] == "distractor":
                rankings[qid]["distractor"][cid] = idx + 1
            else:
                rankings[qid]["random"][cid] = idx + 1

    return rankings


def evaluate_results(results: dict, qrels: dict, qrels_relevant_only: dict):
    """
    Calculates different retrieval metrics for a set of K values for the given retrieved documents. Additionally, the
    position at which each relevant, distractor and irrelevant document was retrieved is stored for each query.

    :param results: The top-k retrieved documents per query along with their score.
    :param qrels: The dictionary containing information on whether a document is relevant or distractor.
    :param qrels_relevant_only: The regular qrels containing the scores for relevant documents.
    :return: The calculated retrieval metrics and rankings.
    """
    ndcg, _map, recall, precision = _compute_results(qrels_relevant_only.copy(), results, K_VALUES)
    scores = {
        **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
        **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
        **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
        **{
            f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()
        }
    }

    rankings = _get_rankings(results.copy(), qrels_relevant_only.copy(), qrels.copy(), max(K_VALUES))
    scores["rankings"] = rankings
    return scores
