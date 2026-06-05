import json
import os
from typing import Tuple, List

import click
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from config import RESULTS_PATH, PLOTS_PATH


METHODS = ["token", "sentence", "enriched_title", "late", "semantic", "enriched_summary", "contextual", "summary"]
LABELS = ["Token", "Sentence", "Enriched (Title)", "Late", "Semantic", "Enriched (Summary)", "Contextual", "Summary"]
CMAP = plt.get_cmap("tab10")
COLORS = CMAP(np.linspace(0, 1, 10))[:len(METHODS)]
CHUNK_TIME_TOKEN, CHUNK_TIME_ENRICHED = 0, 0


def _collect_data(result_path: str) -> Tuple[float, float, float]:
    """
    Collects the chunk time, retrieval time and maximum RAM usage across all runs and computes the average memory usage
    in GB as well as the average document and query throughput.

    :param result_path: The path to the folder containing the JSON files with runtime metrics.
    :return: Average memory usage, document and query throughput.
    """
    stats_file = os.path.join(result_path, "stats_0.json")
    chunk_times, latencies, memories = [], [], []

    if os.path.isfile(stats_file):
        for file_name in os.listdir(result_path):
            if file_name == "stats_all.json" or "inference" in file_name:
                continue
            if "stats" in file_name and file_name.endswith(".json"):
                with open(os.path.join(result_path, file_name), "r") as f:
                    stats = json.load(f)
                    chunk_times.append(stats["chunk_time"])
                    latencies.append(stats["retrieval_time"])
                    memories.append(stats["max_rss"]) # Size in KB

    click.echo(
        f"Chunk time min-max: {min(chunk_times)}-{max(chunk_times)}, "
        f"query time min-max: {min(latencies)}-{max(latencies)}"
    )
    return np.mean(memories) / 1_000**2, np.mean(chunk_times), 2837 / np.mean(latencies)


def _plot_pareto(
        methods: List[str],
        docs_per_sec: List[float],
        queries_per_sec: List[float],
        ram_usage: List[float],
        model: str
) -> None:
    """
    Creates a scatter plot that compares each method's average document throughput on the x-axis to its average query
    throughput on the y-axis. Ram usage is represented circles with a text label stating the required RAM in GB.
    Method labels are placed near their respective circles. The plot is stored as a PDF and PNG file.

    :param methods: The method names for the collected metrics.
    :param docs_per_sec: The document throughput per method.
    :param queries_per_sec: The query throughput per method.
    :param ram_usage: The RAM usage per method.
    :param model: The name of the model for which runtime metrics were measured.
    """
    save_path = os.path.join(PLOTS_PATH, model, f"pareto.pdf")
    ram_usage = np.array(ram_usage)
    ram_norm = (ram_usage - ram_usage.min()) / (ram_usage.max() - ram_usage.min())
    min_size, max_size = 500, 2000
    sizes = min_size + ram_norm * (max_size - min_size)
    plt.figure(figsize=(8, 6))

    summary_idx = methods.index("summary")
    summary_ram = ram_usage[summary_idx]
    docs_summary = docs_per_sec[summary_idx]
    queries_summary = queries_per_sec[summary_idx]
    size_summary = sizes[summary_idx]
    summary_label = LABELS[METHODS.index("summary")]
    plot_methods = methods.copy()
    colors = np.array([COLORS[METHODS.index(method)] for method in methods])
    summary_color = colors[summary_idx]

    docs_per_sec.pop(summary_idx)
    queries_per_sec.pop(summary_idx)
    plot_methods.pop(summary_idx)
    colors = np.delete(colors, summary_idx, axis=0)
    sizes = np.delete(sizes, summary_idx)
    ram_usage = np.delete(ram_usage, summary_idx)

    plt.scatter(
        docs_per_sec,
        queries_per_sec,
        s=sizes,
        color=colors,
        alpha=0.7
    )

    for x, y, ram in zip(docs_per_sec, queries_per_sec, ram_usage):
        plt.text(x, y, f"{ram:.1f}", ha="center", va="center", fontsize=9)

    method_label_offsets = {
        "token": (0, 15),
        "sentence": (0, 15),
        "enriched_title": (0, -15),
        "late": (0, -25),
        "semantic": (0, 18),
        "enriched_summary": (30, -25),
        "contextual": (0, 20),
    }
    for method, x, y, color in zip(plot_methods, docs_per_sec, queries_per_sec, colors):
        dx, dy = method_label_offsets.get(method, (0, 26))
        plt.annotate(
            LABELS[METHODS.index(method)],
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            va="bottom" if dy > 0 else "top" if dy < 0 else "center",
            fontsize=12,
            color=color,
            annotation_clip=False,
        )

    ax = plt.gca()

    plt.xlabel("Indexing (Processed #Docs / Second)", fontsize=16)
    plt.ylabel("Inference (Processed #Queries / Second)", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(-5, 85)

    axins = inset_axes(
        ax,
        width="20%",
        height="20%",
        loc="upper left",
        bbox_to_anchor=(0.1, 0.0, 1.0, 0.98),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    axins.scatter(
        [docs_summary],
        [queries_summary],
        s=[size_summary],
        color=[summary_color],
        alpha=0.7
    )
    axins.text(
        docs_summary,
        queries_summary,
        f"{summary_ram:.1f}",
        ha="center",
        va="center",
        fontsize=9,
    )
    axins.annotate(
        summary_label,
        xy=(docs_summary, queries_summary),
        xytext=(0, 15),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=12,
        color=summary_color,
        annotation_clip=False,
    )
    axins.tick_params(axis="x", labelsize=12)
    axins.tick_params(axis="y", labelsize=12)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=300)
    plt.close()


@click.command()
@click.argument("model", type=str, default="gemma")
@click.argument("dataset", type=str, default="kilt")
def pareto(model: str, dataset: str) -> None:
    """
    Creates a runtime metric plot that compares memory usage, document and query throughput of different methods.
    Requires calculated stats.jons files in the results folders of the respective methods.

    :param model: The model for which the plot should be generated.
    :param dataset: The dataset for which results were calculated.
    """
    result_path = os.path.join(RESULTS_PATH, model, dataset)
    ram_usage, docs_per_sec, queries_per_sec = [], [], []
    methods = []

    for method in METHODS:
        if not os.path.exists(os.path.join(result_path, method)):
            continue

        memory, chunk_time, latency = _collect_data(os.path.join(result_path, method))
        methods.append(method)
        docs_per_sec.append(10_000 / chunk_time)
        ram_usage.append(memory)
        queries_per_sec.append(latency)

    _plot_pareto(methods, docs_per_sec, queries_per_sec, ram_usage, model)


if __name__ == "__main__":
    pareto()
