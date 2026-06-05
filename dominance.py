import os

import click
import pandas as pd
from matplotlib import pyplot as plt

from config import RESULTS_PATH


METRICS = {
    "ndcg": {
        "diff_col": "ndcg_mean_diff",
        "rejected_col": "rejected_ndcg",
    },
    "recall": {
        "diff_col": "recall_mean_diff",
        "rejected_col": "rejected_recall",
    },
}

METHOD_LABELS = {
    "enriched_summary": "enriched (summary)",
    "enriched_title": "enriched (title)",
}


def _method_label(method: str) -> str:
    """
    Exchanges the name of the given method with the one specified in the dictionary, if present.

    :param method: The original name of the method.
    :return: Returns the replacement name, if present, or the original name of the method.
    """
    return METHOD_LABELS.get(method, method)


def _validate_columns(df: pd.DataFrame) -> None:
    """
    Checks that all columns required for calculating dominance scores are present in the DataFrame.

    :param df: The DataFrame to validate.
    """
    required = {"method_a", "method_b"}
    for metric in METRICS.values():
        required.add(metric["diff_col"])
        required.add(metric["rejected_col"])

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _metric_dominance(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    For each pair of chunking methods present in the given DataFrame, the times method a achieves significantly better
    scores than method b for a specific metric is calculated. For each pair, the number of significant wins, losses and
    ties (no significant difference) along with the total number of tests is stored in a dictionary per method. Finally,
    all dictionaries are returned as a DataFrame.

    :param df: The DataFrame containing information about chunking method performance.
    :param metric: The metric for which scores should be calculated.
    :return: The DataFrame containing the calculated information.
    """
    diff_col = METRICS[metric]["diff_col"]
    rejected_col = METRICS[metric]["rejected_col"]
    rows = []

    for (left, right), pair_df in df.groupby(["method_a", "method_b"], sort=True):
        total_tests = len(pair_df)

        left_wins = (pair_df[rejected_col] & (pair_df[diff_col] > 0)).sum()
        right_wins = (pair_df[rejected_col] & (pair_df[diff_col] < 0)).sum()
        ties_or_not_significant = total_tests - left_wins - right_wins

        rows.append({
            "method_a": left,
            "method_b": right,
            "metric": metric,
            "significant_wins": int(left_wins),
            "significant_losses": int(right_wins),
            "not_significant_or_tied": int(ties_or_not_significant),
            "total_tests": int(total_tests),
            "dominance_score": left_wins / total_tests if total_tests else 0.0,
        })
        rows.append({
            "method_a": right,
            "method_b": left,
            "metric": metric,
            "significant_wins": int(right_wins),
            "significant_losses": int(left_wins),
            "not_significant_or_tied": int(ties_or_not_significant),
            "total_tests": int(total_tests),
            "dominance_score": right_wins / total_tests if total_tests else 0.0,
        })

    return pd.DataFrame(rows)


def compute_dominance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the dominance score between each unique pairs of chunking methods. The dominance score between two methods
    is calculated as the fraction of times method a achieves significantly better scores than method b over all models
    and corpora per retrieval metric. The calculated information is returned as a DataFrame.

    :param df: The DataFrame containing information about chunking method performance.
    :return: The DataFrame containing the dominance scores per method pair.
    """
    _validate_columns(df)
    metric_scores = pd.concat(
        [_metric_dominance(df, metric) for metric in METRICS],
        ignore_index=True,
    )

    overall_scores = (
        metric_scores
        .groupby(["method_a", "method_b"], as_index=False)
        .agg({
            "significant_wins": "sum",
            "significant_losses": "sum",
            "not_significant_or_tied": "sum",
            "total_tests": "sum",
        })
    )
    overall_scores["metric"] = "overall"
    overall_scores["dominance_score"] = (
        overall_scores["significant_wins"] / overall_scores["total_tests"]
    )

    return pd.concat([metric_scores, overall_scores], ignore_index=True)[[
        "method_a",
        "method_b",
        "metric",
        "significant_wins",
        "significant_losses",
        "not_significant_or_tied",
        "total_tests",
        "dominance_score",
    ]].sort_values(
        ["metric", "dominance_score", "method_a", "method_b"],
        ascending=[True, False, True, True],
        ignore_index=True,
    )


def dominance_matrix(scores: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Given the DataFrame that contains calculated dominance scores for pairs of chunking methods and the name of the
    retrieval metric to consider, constructs a 2D matrix in which each row contains the list of dominance scores for a
    specific method. Each column in the resulting matrix corresponds to the dominated chunking method. Chunking methods
    are ordered by their overall dominance score, i.e. the chunking method that dominates others most often comes first.

    :param scores: The DataFrame containing the calculated dominance scores.
    :param metric: The metric for which the matrix should be constructed, i.e. NDCG@10 or Recall@100.
    :return: The constructed dominance matrix DataFrame.
    """
    metric_scores = scores[scores["metric"] == metric]
    if metric_scores.empty:
        available = ", ".join(sorted(scores["metric"].unique()))
        raise ValueError(f"Unknown metric '{metric}'. Available metrics: {available}")

    method_rankings = (
        metric_scores
        .groupby("method_a", as_index=False)["dominance_score"]
        .sum()
        .sort_values(["dominance_score", "method_a"], ascending=[False, True])
    )
    methods = method_rankings["method_a"].tolist()
    matrix = metric_scores.pivot(
        index="method_a",
        columns="method_b",
        values="dominance_score",
    ).reindex(index=methods, columns=methods)

    for method in methods:
        matrix.loc[method, method] = pd.NA

    return matrix


def save_heatmap(ndcg_matrix: pd.DataFrame, recall_matrix: pd.DataFrame, output_path: str) -> None:
    """
    Saves the calculated dominance matrix scores for NDCG and Recall in a combined heatmap.

    :param ndcg_matrix: The NDCG@10 dominance score matrix.
    :param recall_matrix: The Recall@100 dominance score matrix.
    :param output_path: The path to save the heatmap to.
    """
    fig_width = max(14, (len(ndcg_matrix.columns) + len(recall_matrix.columns)) * 0.95)
    fig_height = max(7, max(len(ndcg_matrix.index), len(recall_matrix.index)) * 1.05)
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), constrained_layout=True)
    image = None

    for ax, matrix in zip(axes, [ndcg_matrix, recall_matrix]):
        masked = matrix.astype(float).to_numpy()
        image = ax.imshow(masked, cmap="viridis", vmin=0.0, vmax=1.0)

        x_labels = [_method_label(method) for method in matrix.columns]
        y_labels = [_method_label(method) for method in matrix.index]
        ax.set_xticks(range(len(matrix.columns)), labels=x_labels, rotation=45, ha="right")
        ax.set_yticks(range(len(matrix.index)), labels=y_labels)
        ax.tick_params(axis="both", labelsize=14)

        for row_idx, row in enumerate(matrix.index):
            for col_idx, col in enumerate(matrix.columns):
                value = matrix.loc[row, col]
                if pd.isna(value):
                    ax.text(col_idx, row_idx, "-", ha="center", va="center", color="black", fontsize=12)
                    continue

                text_color = "white" if value < 0.45 else "black"
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=14,
                )

    for ax, label in zip(axes, ["a", "b"]):
        ax.text(
            -0.08, 1.03, label, transform=ax.transAxes, ha="left", va="top", fontsize=18, fontweight="bold"
        )


    cbar_ax = axes[-1].inset_axes([1.04, 0.0, 0.04, 1.0])
    colorbar = fig.colorbar(image, cax=cbar_ax)
    colorbar.ax.tick_params(labelsize=14)
    colorbar.set_label("Significant Win Rates", fontsize=18)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    fig.savefig(output_path.replace(".png", ".pdf"))
    plt.close(fig)


@click.command()
@click.argument("inputs", type=str, default="tests.parquet")
def dominance(inputs: str) -> None:
    """
    Calculates dominance scores, i.e. the fraction of times chunking method a performs significantly better than method
    b, for each pair of chunking methods. The scores are stored as CSV files and as a heatmap.

    :param inputs: Path to the input file that contains information about chunking method performance.
    """
    df = pd.read_parquet(inputs)
    scores = compute_dominance(df)
    scores.to_csv(os.path.join(RESULTS_PATH, f"dominance_scores.csv"), index=False)

    ndcg_matrix = dominance_matrix(scores, "ndcg")
    recall_matrix = dominance_matrix(scores, "recall")
    ndcg_matrix.to_csv(os.path.join(RESULTS_PATH, f"dominance_matrix_ndcg.csv"))
    recall_matrix.to_csv(os.path.join(RESULTS_PATH, f"dominance_matrix_recall.csv"))
    save_heatmap(ndcg_matrix, recall_matrix, os.path.join(RESULTS_PATH, f"dominance_scores.png"))

    click.echo(scores.to_string(index=False))
    click.echo(f"\nDominance matrix (ndcg):")
    click.echo(ndcg_matrix.to_string(float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    dominance()
