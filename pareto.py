import json
import os
from typing import Tuple, List

import click
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from config import RESULTS_PATH, PLOTS_PATH


METHODS = ["token", "sentence", "enriched_title", "late", "semantic", "enriched_summary", "contextual", "summary"]
LABELS = ["Token", "Sentence", "Enriched (T.)", "Late", "Semantic", "Enriched (S.)", "Contextual", "Summary"]
CMAP = plt.get_cmap("tab10")
COLORS = CMAP(np.linspace(0, 1, 10))[:len(METHODS)]
CHUNK_TIME_TOKEN, CHUNK_TIME_ENRICHED = 0, 0


def _collect_data(result_path: str) -> Tuple[float, float, float, float, float, float]:
    metrics_file = os.path.join(result_path, "10000.json")
    stats_file = os.path.join(result_path, "stats_0.json")
    stats_inference = os.path.join(result_path, "inference.json")
    recall, memory, chunk_time = 0, 0, 0
    ndcg, latency, memory_inference = 0, 0, 0
    num_stats = 1

    if os.path.isfile(metrics_file) and os.path.isfile(stats_file):
        num_stats = 0
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
            recall = round(metrics["recall_at_100"] * 100, 2)
            ndcg = round(metrics["ndcg_at_10"] * 100, 2)
        with open(stats_file, "r") as f:
            stats = json.load(f)
            memory = stats["peak_memory"]
        with open(stats_inference, "r") as f:
            stats = json.load(f)
            memory_inference = sum(stats["index_sizes"])
            latency += stats["retrieval_time"]

        for file_name in os.listdir(result_path):
            if "stats" in file_name and file_name.endswith(".json"):
                num_stats += 1
                with open(os.path.join(result_path, file_name), "r") as f:
                    stats = json.load(f)
                    chunk_time += stats["chunk_time"]
                    latency += stats["retrieval_time"]

    latency = latency / num_stats
    chunk_time = chunk_time / num_stats
    return recall, ndcg, memory / 1_000**2, chunk_time, memory_inference / 1_000**2, 55 / latency


def _pareto_front(x, y):
    points = np.column_stack((x, y))
    pareto = np.ones(len(points), dtype=bool)

    for i, p in enumerate(points):
        if pareto[i]:
            pareto[pareto] = np.any(points[pareto] > p, axis=1)
            pareto[i] = True

    return np.where(pareto)[0]


def _plot_pareto(docs_per_sec: List[float], recalls: List[float], ram_usage: List[float], sorted_idx: List[int],
                 model: str, inference: bool = False):
    file_name = "pareto_inf.pdf" if inference else "pareto_train.pdf"
    y_label = "NDCG@10 in %" if inference else "Recall@100 in %"
    x_label = "Processed #Queries / Second" if inference else "Processed #Docs / Second"
    save_path = os.path.join(PLOTS_PATH, model, file_name)
    ram_usage = np.array(ram_usage)
    if inference:
        ram_norm = (ram_usage - ram_usage.min()) / (ram_usage.max() - ram_usage.min())
        min_size, max_size = 200, 500
        sizes = min_size + ram_norm * (max_size - min_size)
    else:
        sizes = ram_usage
    patches = []
    plt.figure(figsize=(8, 6))

    plt.scatter(
        docs_per_sec,
        recalls,
        s=sizes,
        color=COLORS[:len(docs_per_sec)],
        alpha=0.7
    )

    # Pareto front line
    plt.plot(
        [docs_per_sec[idx] for idx in sorted_idx],
        [recalls[idx] for idx in sorted_idx],
        linestyle="--",
        linewidth=2
    )

    # Create manual legend entries for each method
    for color, label in zip(COLORS[:len(docs_per_sec)], LABELS[:len(docs_per_sec)]):
        patches.append(Line2D([0], [0], color=color, label=label, marker="o", markersize=12))

    plt.legend(handles=patches, title="Chunking Method", fontsize=12, title_fontsize=14)
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=300)
    plt.close()


@click.command()
@click.argument("model", type=str)
def pareto(model: str):
    result_path = os.path.join(RESULTS_PATH, model)
    recalls, ram_usage, docs_per_sec = [], [], []
    ndcgs, ram_inference, queries_per_sec = [], [], []
    chunk_time_token, chunk_time_enriched = 0, 0

    for method in METHODS:
        if not os.path.exists(os.path.join(result_path, method)):
            continue

        recall, ndcg, memory, chunk_time, memory_inference, latency = _collect_data(os.path.join(result_path, method))

        if method == "token":
            chunk_time_token = chunk_time
        elif method == "enriched_title":
            chunk_time_enriched = chunk_time

        if method == "summary":
            docs_per_sec.append(10_000 / (1_200_000 / 100))
        elif method == "enriched_summary":
            docs_per_sec.append(10_000 / (chunk_time_enriched + 1_200_000 / 100))
        elif method == "contextual":
            docs_per_sec.append(10_000 / (chunk_time_token + 4_092_862 / 100))
        else:
            docs_per_sec.append(10_000 / chunk_time)

        recalls.append(recall)
        ram_usage.append(memory)

        if method != "summary":
            ndcgs.append(ndcg)
            ram_inference.append(memory_inference)
            queries_per_sec.append(latency)

    pareto_idx = _pareto_front(docs_per_sec, recalls)
    sorted_idx = pareto_idx[np.argsort(np.array(docs_per_sec)[pareto_idx])].tolist()
    _plot_pareto(docs_per_sec, recalls, ram_usage, sorted_idx, model)
    pareto_idx = _pareto_front(queries_per_sec, ndcgs)
    sorted_idx = pareto_idx[np.argsort(np.array(queries_per_sec)[pareto_idx])].tolist()
    _plot_pareto(queries_per_sec, ndcgs, ram_inference, sorted_idx, model, True)


if __name__ == "__main__":
    pareto()
