import json
import os
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RENDERED_DIR = os.path.join(PROJECT_ROOT, "rendered")

KG_JSON = os.path.join(DATA_DIR, "kg_adj_list.json")
METRIC_CSV = os.path.join(DATA_DIR, "logic_metrics_comparison.csv")
REC_JSON = os.path.join(DATA_DIR, "recommendation_simulation.json")


def ensure_rendered_dir():
    os.makedirs(RENDERED_DIR, exist_ok=True)


def clear_old_charts():
    """
    Remove old generated chart files so rendered/ contains fresh outputs only.
    Also clear previous chart files in data/ to keep outputs centralized.
    """
    removed = 0
    for folder in [DATA_DIR, RENDERED_DIR]:
        if not os.path.exists(folder):
            continue
        for name in os.listdir(folder):
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".svg")):
                path = os.path.join(folder, name)
                try:
                    os.remove(path)
                    removed += 1
                except OSError:
                    pass
    print(f"[Render] cleared old chart files: {removed}")


def plot_kg_matrix():
    """1) Adjacency list -> knowledge matrix heatmap."""
    with open(KG_JSON, "r", encoding="utf-8") as f:
        kg_adj = json.load(f)

    indices = []
    for t, prereqs in kg_adj.items():
        indices.append(int(t))
        indices.extend(int(p) for p in prereqs)
    n_skills = max(indices) if indices else 1

    mat = np.zeros((n_skills + 1, n_skills + 1), dtype=np.float32)
    for t, prereqs in kg_adj.items():
        t_idx = int(t)
        for p in prereqs:
            p_idx = int(p)
            if 0 <= t_idx <= n_skills and 0 <= p_idx <= n_skills:
                mat[t_idx, p_idx] = 1.0

    plt.figure(figsize=(8, 7), dpi=200)
    plt.imshow(mat[1:, 1:], cmap="Blues", aspect="auto")
    plt.colorbar(label="Prerequisite Link (0/1)")
    plt.title("Knowledge Matrix from KG Adjacency")
    plt.xlabel("Prerequisite Skill Index")
    plt.ylabel("Target Skill Index")
    plt.tight_layout()
    out = os.path.join(RENDERED_DIR, "kg_adjacency_matrix.png")
    plt.savefig(out)
    plt.close()
    print(f"[Render] saved: {out}")


def plot_model_comparison():
    """2) Four-model comparison chart (AUC + RMSE)."""
    df = pd.read_csv(METRIC_CSV)
    order = ["Pure-CF", "DKT", "SAKT", "KG-SAKT"]
    df["Model"] = pd.Categorical(df["Model"], categories=order, ordered=True)
    df = df.sort_values("Model")

    x = np.arange(len(df))
    width = 0.36

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=220)
    auc_bars = ax1.bar(x - width / 2, df["AUC"], width, label="AUC", color="#4c78a8")
    ax1.set_ylabel("AUC (Higher is better)")
    ax1.set_ylim(0.45, 0.90)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["Model"].tolist())

    ax2 = ax1.twinx()
    rmse_bars = ax2.bar(x + width / 2, df["RMSE"], width, label="RMSE", color="#f58518")
    ax2.set_ylabel("RMSE (Lower is better)")
    ax2.set_ylim(0.35, 0.70)

    for b in auc_bars:
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005, f"{b.get_height():.3f}", ha="center", fontsize=8)
    for b in rmse_bars:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005, f"{b.get_height():.3f}", ha="center", fontsize=8)

    plt.title("Four-Model Comparison (AUC & RMSE)")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    plt.tight_layout()
    out = os.path.join(RENDERED_DIR, "four_models_comparison.png")
    plt.savefig(out)
    plt.close()
    print(f"[Render] saved: {out}")


def _top1_per_student(rec_payload: Dict[str, Dict]) -> List[dict]:
    rows = []
    for student, info in rec_payload.items():
        recs = info.get("recommendations", [])
        if not recs:
            continue
        top = sorted(recs, key=lambda x: float(x.get("score", 0.0)), reverse=True)[0]
        rows.append(
            {
                "student": student,
                "level": info.get("level", "N/A"),
                "acc": float(info.get("recent_accuracy", 0.0)),
                "top_skill": str(top.get("skill_name", f"Skill {top.get('skill_id', '?')}")),
                "top_score": float(top.get("score", 0.0)),
            }
        )
    return rows


def plot_student_recommendation_example():
    """3) Student recommendation example chart."""
    with open(REC_JSON, "r", encoding="utf-8") as f:
        rec_payload = json.load(f)
    rows = _top1_per_student(rec_payload)
    if not rows:
        print("[Render] no recommendation rows found.")
        return

    students = [r["student"] for r in rows]
    scores = [r["top_score"] for r in rows]
    accs = [r["acc"] for r in rows]
    labels = [f"{r['top_skill']}" for r in rows]

    x = np.arange(len(rows))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6), dpi=220)
    b1 = ax.bar(x - width / 2, scores, width, color="#54a24b", label="Top-1 Recommendation Score")
    b2 = ax.bar(x + width / 2, accs, width, color="#b279a2", label="Recent Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(students)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score / Accuracy")
    ax.set_title("Student Recommendation Example (Top-1)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for i, bar in enumerate(b1):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, labels[i], ha="center", va="bottom", fontsize=8, rotation=0)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{bar.get_height():.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    out = os.path.join(RENDERED_DIR, "student_recommendation_example.png")
    plt.savefig(out)
    plt.close()
    print(f"[Render] saved: {out}")


def main():
    ensure_rendered_dir()
    clear_old_charts()
    plot_kg_matrix()
    plot_model_comparison()
    plot_student_recommendation_example()


if __name__ == "__main__":
    main()
