import argparse
import json
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Prefer common CJK fonts when available to improve Chinese rendering.
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def project_root() -> str:
    # utils/case_study_viz.py -> project root
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def data_dir() -> str:
    return os.path.join(project_root(), "data")


def rendered_dir() -> str:
    return os.path.join(project_root(), "rendered")


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_students(payload) -> List[Dict]:
    """
    Normalize recommendation payload into a unified list:
    [{"name": ..., "level": ..., "recent_accuracy": ..., "recommendations": [...]}]
    """
    students: List[Dict] = []

    if isinstance(payload, dict):
        for name, info in payload.items():
            if not isinstance(info, dict):
                continue
            students.append(
                {
                    "name": str(name),
                    "level": info.get("level", "N/A"),
                    "recent_accuracy": float(info.get("recent_accuracy", 0.0)),
                    "recommendations": info.get("recommendations", []),
                }
            )
        return students

    if isinstance(payload, list):
        for i, info in enumerate(payload):
            if not isinstance(info, dict):
                continue
            students.append(
                {
                    "name": str(info.get("name", f"Student-{i+1}")),
                    "level": info.get("level", "N/A"),
                    "recent_accuracy": float(info.get("recent_accuracy", 0.0)),
                    "recommendations": info.get("recommendations", []),
                }
            )
    return students


def short_skill_name(text: str) -> str:
    """
    Keep concise skill labels for plotting.
    Example: "Ordering Fractions (分数排序)" -> "Ordering Fractions"
    """
    s = str(text or "").strip()
    if "(" in s:
        s = s.split("(", 1)[0].strip()
    return s if s else "Unknown Skill"


def load_skill_name_map(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    raw = load_json(path)
    return {str(k): str(v) for k, v in raw.items()}


def ensure_dir(path: str):
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)


def flatten_rows(students: List[Dict]) -> List[Dict]:
    rows = []
    for student in students:
        s_name = student["name"]
        level = student["level"]
        acc = student["recent_accuracy"]
        for rec in student.get("recommendations", []):
            rows.append(
                {
                    "student_name": s_name,
                    "level": level,
                    "recent_accuracy": acc,
                    "skill_id": int(rec.get("skill_id", -1)),
                    "skill_name": short_skill_name(rec.get("skill_name", "")),
                    "score": float(rec.get("score", 0.0)),
                    "mastery_prob": float(rec.get("mastery_prob", 0.0)),
                    "readiness": float(rec.get("readiness", 0.0)),
                }
            )
    return rows


def plot_score_bars(rows: List[Dict], out_path: str):
    """
    Figure 1: recommendation score bars for all student-skill pairs.
    """
    if not rows:
        print("[Warn] No rows available, skip score bar figure.")
        return

    # Sort by student then descending recommendation score.
    rows = sorted(rows, key=lambda x: (x["student_name"], -x["score"]))
    labels = [f'{r["student_name"]} | {r["skill_name"]}' for r in rows]
    scores = [r["score"] for r in rows]

    plt.figure(figsize=(14, max(5, 0.45 * len(rows))), dpi=180)
    bars = plt.barh(range(len(rows)), scores, color="#4c78a8")
    plt.yticks(range(len(rows)), labels, fontsize=8)
    plt.xlabel("Recommendation Score")
    plt.title("Case Study: Recommended Skills and Scores")
    plt.gca().invert_yaxis()
    plt.xlim(0.0, 1.05)

    for bar, score in zip(bars, scores):
        plt.text(
            x=min(1.02, bar.get_width() + 0.01),
            y=bar.get_y() + bar.get_height() / 2.0,
            s=f"{score:.3f}",
            va="center",
            fontsize=8,
        )

    plt.tight_layout()
    ensure_dir(out_path)
    plt.savefig(out_path)
    plt.close()
    print(f"[Saved] score bar figure -> {out_path}")


def plot_prereq_paths(
    students: List[Dict],
    kg_adj: Dict[str, List[int]],
    skill_name_map: Dict[str, str],
    out_path: str,
):
    """
    Figure 2: path-style prerequisite graph per student profile.
    Left column = prerequisite nodes, right column = recommended target skills.
    """
    if not students:
        print("[Warn] No students available, skip path figure.")
        return

    n = len(students)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), dpi=180, squeeze=False)

    for i, student in enumerate(students):
        ax = axes[0, i]
        recs = student.get("recommendations", [])
        if not recs:
            ax.set_title(f'{student["name"]}\n(no recommendations)')
            ax.axis("off")
            continue

        rec_nodes = []
        pre_nodes = {}
        y_step = 1.8

        # Place recommendation nodes on the right.
        for r_idx, rec in enumerate(recs):
            skill_id = int(rec.get("skill_id", -1))
            y = -r_idx * y_step
            label = short_skill_name(rec.get("skill_name", f"Skill {skill_id}"))
            rec_nodes.append((skill_id, label, y))

            prereqs = kg_adj.get(str(skill_id), [])
            for p_idx, p in enumerate(prereqs[:3]):
                if p not in pre_nodes:
                    # Spread prereq nodes around the target row for readability.
                    offset = (p_idx - 1) * 0.45
                    p_label = short_skill_name(skill_name_map.get(str(p), f"Skill {p}"))
                    pre_nodes[p] = {"label": p_label, "y": y + offset}

        # Draw edges prereq -> recommended skill.
        for skill_id, _, y_target in rec_nodes:
            for p in kg_adj.get(str(skill_id), [])[:3]:
                if p in pre_nodes:
                    ax.annotate(
                        "",
                        xy=(1.0, y_target),
                        xytext=(0.0, pre_nodes[p]["y"]),
                        arrowprops=dict(
                            arrowstyle="->",
                            lw=1.7,
                            color="#7f8c8d",
                            alpha=0.85,
                        ),
                    )

        # Draw prerequisite nodes.
        for p, meta in pre_nodes.items():
            ax.scatter(0.0, meta["y"], s=420, color="#d9d9d9", edgecolors="#7f8c8d", zorder=3)
            ax.text(0.0, meta["y"], meta["label"], ha="center", va="center", fontsize=8)

        # Draw recommendation nodes.
        for skill_id, label, y_target in rec_nodes:
            ax.scatter(1.0, y_target, s=540, color="#4c78a8", edgecolors="#2f5f8f", zorder=3)
            ax.text(1.0, y_target, label, ha="center", va="center", fontsize=8, color="white")

        level = student.get("level", "N/A")
        acc = float(student.get("recent_accuracy", 0.0))
        ax.set_title(f'{student["name"]}\nLevel={level} | RecentAcc={acc:.0%}', fontsize=10)
        ax.set_xlim(-0.55, 1.55)
        ax.set_ylim(-max(1.5, y_step * (len(recs) - 1) + 1.2), 1.2)
        ax.axis("off")

        # Column hints.
        ax.text(0.0, 1.0, "Prerequisites", ha="center", va="bottom", fontsize=9, color="#555")
        ax.text(1.0, 1.0, "Recommended Skills", ha="center", va="bottom", fontsize=9, color="#555")

    fig.suptitle("Case Study: KG-SAKT Recommendation Paths", fontsize=12, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    ensure_dir(out_path)
    plt.savefig(out_path)
    plt.close()
    print(f"[Saved] prerequisite path figure -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize recommendation simulation results from JSON."
    )
    default_input = os.path.join(data_dir(), "recommendation_simulation.json")
    default_kg = os.path.join(data_dir(), "kg_adj_list.json")
    default_skill_map = os.path.join(data_dir(), "skill_map.json")
    default_out_score = os.path.join(rendered_dir(), "case_study_scores.png")
    default_out_path = os.path.join(rendered_dir(), "case_study_paths.png")

    parser.add_argument("--input", default=default_input, help="recommendation_simulation.json path")
    parser.add_argument("--kg", default=default_kg, help="KG adjacency json path")
    parser.add_argument("--skill-map", default=default_skill_map, help="skill_map.json path")
    parser.add_argument("--out-score", default=default_out_score, help="output score bar figure path")
    parser.add_argument("--out-path", default=default_out_path, help="output path figure path")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input json not found: {args.input}")

    payload = load_json(args.input)
    students = normalize_students(payload)
    if not students:
        raise ValueError("No valid student recommendation records found in input json.")

    kg_adj = load_json(args.kg) if os.path.exists(args.kg) else {}
    skill_name_map = load_skill_name_map(args.skill_map)

    rows = flatten_rows(students)
    plot_score_bars(rows, args.out_score)
    plot_prereq_paths(students, kg_adj, skill_name_map, args.out_path)

    print(f"[Done] students visualized: {len(students)}")


if __name__ == "__main__":
    main()
