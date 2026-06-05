import itertools
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import click
import numpy as np
import pandas as pd
import pytrec_eval
from statsmodels.stats.multitest import multipletests

from config import RESULTS_PATH, K_VALUES
from utils.dataset_loading import load_data

METHODS = ["token", "sentence", "enriched_title", "late", "semantic", "enriched_summary", "contextual", "summary"]
MODELS = ["gemma", "qwen", "snowflake"]
DATASETS = ["kilt", "core"]


def _get_per_query_metrics(rankings: dict, qrels: dict) -> Dict[str, Dict[str, float]]:
    """
    Computes NDCG@10 and Recall@100 per query from the given rankings.

    :param rankings: A dictionary containing ranking information on top retrieved documents per query.
    :param qrels: The relevance judgements.
    :return: The computed query-wise scores.
    """
    results = defaultdict(dict)
    k = max(K_VALUES) + 1

    for qid in rankings:
        for doc_id in rankings[qid]["relevant"]:
            results[qid][doc_id] = k - rankings[qid]["relevant"][doc_id]
        for doc_id in rankings[qid]["distractor"]:
            results[qid][doc_id] = k - rankings[qid]["distractor"][doc_id]
        for doc_id in rankings[qid]["random"]:
            results[qid][doc_id] = k - rankings[qid]["random"][doc_id]

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, ["ndcg_cut_10", "recall_100"]
    )
    return evaluator.evaluate(results)


def _collect_data() -> pd.DataFrame:
    """
    Computes query-wise retrieval scores per method, model and dataset and returns the results in a DataFrame.

    :return: The calculated query scores and relevant metadata as a pandas DataFrame.
    """
    results = []

    for dataset in DATASETS:
        _, _, _, qrels = load_data(dataset, True)
        for model in MODELS:
            for method in METHODS:
                data_path = os.path.join(RESULTS_PATH, model, dataset, method)

                if not os.path.isdir(data_path):
                    click.echo(f"Could not find any results at path {data_path}.")
                    continue

                for file_name in os.listdir(data_path):
                    if file_name.endswith(".json"):
                        corpus_size = file_name.split(".")[0]
                        if corpus_size.isnumeric():
                            corpus_size = int(corpus_size)
                            data = json.load(open(os.path.join(data_path, file_name), "r"))
                            rankings = data["rankings"]
                            scores = _get_per_query_metrics(rankings, qrels)

                            for qid in scores.keys():
                                results.append([
                                        model, dataset, method, corpus_size, qid, scores[qid]["ndcg_cut_10"],
                                        scores[qid]["recall_100"]
                                    ])

    return pd.DataFrame(results, columns=["model", "dataset", "method", "size", "qid", "ndcg@10", "recall@100"])


def _paired_approx_randomization(
        x: pd.Series, y: pd.Series, n_iter: int = 10000, seed: int = 42
) -> Tuple[float, float]:
    """
    Performs an approximate randomization test between query-wise retrieval scores of two methods. The calculated
    p-value and the observed mean difference are returned.

    :param x: The retrieval scores of method a.
    :param y: The retrieval scores of method b.
    :param n_iter: The number of iterations
    :param seed: The seed to use for sign permutations.
    :return:
    """
    rng = np.random.default_rng(seed)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    diffs = x - y
    observed = np.mean(diffs)

    # Random sign flips are equivalent to paired swaps
    signs = rng.choice([-1, 1], size=(n_iter, len(diffs)))
    randomized_diffs = signs * diffs
    null_stats = randomized_diffs.mean(axis=1)

    # Add 1 for conservative finite-sample p-value
    p_value = (np.sum(np.abs(null_stats) >= abs(observed)) + 1) / (n_iter + 1)

    return observed, p_value


def _methods_pairwise(
        df: pd.DataFrame, methods: List[str], n_iter: int = 10000, seed: int = 42
) -> List[Dict[str, str | float]]:
    """
    Copmutes significance scores between all unique method combinations for the query-wise retrieval scores. Results are
    returned as a list of dictionaries, with each entry containing the necessary metadata and p-values per pairwise
    comparison.

    :param df: The DataFrame containing the query scores.
    :param methods: The list of methods to compare.
    :param n_iter: The number of iterations to use in the randomization test.
    :param seed: The seed to use for the randomization test.
    :return: A list of results per method combination.
    """
    results = []

    for method_a, method_b in itertools.combinations(methods, 2):
        a = df[df["method"] == method_a][["qid", "ndcg@10", "recall@100"]]
        b = df[df["method"] == method_b][["qid", "ndcg@10", "recall@100"]]

        diff_ndcg, p_ndcg = _paired_approx_randomization(a["ndcg@10"], b["ndcg@10"], n_iter, seed)
        diff_recall, p_recall = _paired_approx_randomization(a["recall@100"], b["recall@100"], n_iter, seed)

        results.append({
            "method_a": method_a,
            "method_b": method_b,
            "n_queries": len(a),
            "ndcg_mean_a": a["ndcg@10"].mean(),
            "ndcg_mean_b": b["ndcg@10"].mean(),
            "recall_mean_a": a["recall@100"].mean(),
            "recall_mean_b": b["recall@100"].mean(),
            "ndcg_mean_diff": diff_ndcg,
            "recall_mean_diff": diff_recall,
            "p_ndcg_raw": p_ndcg,
            "p_recall_raw": p_recall,
        })

    return results


@click.command()
@click.argument("n_iter", type=int, default=10000)
@click.argument("seed", type=int, default=42)
def statistics(n_iter: int, seed: int) -> None:
    """
    Computes pairwise significance scores between chunking methods via randomization testing for each model, dataset
    and corpus size.

    :param n_iter: The number of iterations to use in the randomization test.
    :param seed: The seed to use for the randomization test.
    """
    if not os.path.isfile(os.path.join(RESULTS_PATH, "all.parquet")):
        df = _collect_data()
        df.to_parquet(os.path.join(RESULTS_PATH, "all.parquet"))
    else:
        df = pd.read_parquet(os.path.join(RESULTS_PATH, "all.parquet"))

    outputs = None

    for model in MODELS:
        for dataset in DATASETS:
            corpus_sizes = df[df["dataset"] == dataset]["size"].unique()
            for corpus_size in corpus_sizes:
                sub_df = df[(df["model"] == model) & (df["dataset"] == dataset) & (df["size"] == corpus_size)]
                methods = METHODS.copy()
                if corpus_size == max(corpus_sizes):
                    methods.remove("contextual")
                results = _methods_pairwise(sub_df, methods, n_iter, seed)
                p_ndcg = [res["p_ndcg_raw"] for res in results]
                p_recall = [res["p_recall_raw"] for res in results]
                rejected_ndcg, p_adj_ndcg, _, _  = multipletests(p_ndcg, alpha=0.05, method="holm")
                rejected_recall, p_adj_recall, _, _ = multipletests(p_recall, alpha=0.05, method="holm")
                assert len(rejected_ndcg) == len(rejected_recall) == len(results)

                for res, rej_rec, rej_ndcg, p_ndcg, p_rec in zip(
                        results, rejected_recall, rejected_ndcg, p_adj_ndcg, p_adj_recall
                ):
                    res["rejected_ndcg"] = rej_ndcg
                    res["p_adj_ndcg"] = p_ndcg
                    res["rejected_recall"] = rej_rec
                    res["p_adj_recall"] = p_rec
                    res["model"] = model
                    res["dataset"] = dataset
                    res["corpus_size"] = corpus_size

                click.echo(
                    f"Evaluated significance for model {model} on dataset {dataset} at corpus size {corpus_size}."
                )
                results_df = pd.DataFrame(results)

                if outputs is None:
                    outputs = results_df
                else:
                    outputs = pd.concat([outputs, results_df], ignore_index=True)

    outputs.to_parquet(os.path.join(RESULTS_PATH, "tests.parquet"))


if __name__ == "__main__":
    statistics()
