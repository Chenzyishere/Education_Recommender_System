import matplotlib.pyplot as plt
import numpy as np


def plot_paper_figure():
    """
    Draw a publication-style comparison figure.

    Notes:
    - Left axis: AUC bars
    - Right axis: logic score line
    - This script is standalone demo plotting utility.
    """
    # Example values (replace with your final table if needed).
    models = ["PureCF", "DKT", "SAKT", "KG-SAKT"]
    auc_scores = [0.5874, 0.8458, 0.8198, 0.8122]
    logic_scores = [42.5, 82.22, 77.49, 98.83]

    x = np.arange(len(models))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

    # AUC bars (left y-axis)
    bars = ax1.bar(
        x,
        auc_scores,
        width,
        color="#3498db",
        alpha=0.75,
        label="Best AUC",
        edgecolor="black",
        linewidth=1,
    )
    ax1.set_ylabel("Best AUC (Higher is better)", fontsize=12, fontweight="bold")
    ax1.set_ylim(0.4, 1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=11)

    # Value labels for bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Logic score line (right y-axis)
    ax2 = ax1.twinx()
    ax2.plot(
        x,
        logic_scores,
        color="#e74c3c",
        marker="D",
        markersize=8,
        linewidth=2.5,
        label="Logic Score (%)",
    )
    ax2.set_ylabel("Logic Score (%) (Higher is better)", color="#e74c3c", fontsize=12, fontweight="bold")
    ax2.set_ylim(30, 110)
    ax2.tick_params(axis="y", labelcolor="#e74c3c")

    # Value labels for line points
    for i, score in enumerate(logic_scores):
        ax2.text(i, score + 2.5, f"{score:.2f}%", color="#e74c3c", ha="center", fontsize=9, fontweight="bold")

    plt.title("Comparison of Models on Accuracy and Logical Consistency", fontsize=14, pad=18)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)

    # Merge legends from both axes.
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True)

    plt.tight_layout()
    plt.savefig("experimental_results_comparison.png")
    print("[Saved] experimental figure -> experimental_results_comparison.png")


if __name__ == "__main__":
    plot_paper_figure()
