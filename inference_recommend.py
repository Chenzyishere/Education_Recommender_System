import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch

from models.kg_sakt import KGSAKTModel

# ==========================================
# 1) Environment and paths
# ==========================================
def resolve_device() -> torch.device:
    """
    Default to CPU to avoid CUDA probe stalls on some Windows setups.
    Set env `INFER_DEVICE=cuda` if you want to force GPU.
    """
    requested = os.environ.get("INFER_DEVICE", "cpu").strip().lower()
    if requested == "cuda":
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = resolve_device()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

KG_JSON_PATH = os.path.join(DATA_DIR, "kg_adj_list.json")
SKILL_MAP_JSON_PATH = os.path.join(DATA_DIR, "skill_map.json")
SKILL_MAP_CSV_PATH = os.path.join(DATA_DIR, "skill_map.csv")
MODEL_WEIGHTS = os.path.join(DATA_DIR, "kg_sakt_model.pth")
OUTPUT_JSON_PATH = os.path.join(DATA_DIR, "recommendation_simulation.json")

MAX_SEQ = 100
TOP_K = 3
MASTERY_HIGH = 0.85
READINESS_MIN = 0.60
ZPD_CENTER = 0.60
ZPD_HALF_WIDTH = 0.30


# ==========================================
# 2) Utilities
# ==========================================
def get_width(text: str) -> int:
    """Estimate monospace display width for mixed CJK/ASCII text."""
    return sum(2 if "\u4e00" <= ch <= "\u9fff" else 1 for ch in str(text))


def clip_text(text: str, max_width: int) -> str:
    """
    Clip text by display width and add ellipsis when needed.
    This avoids broken table alignment caused by very long skill names/reasons.
    """
    s = str(text)
    if get_width(s) <= max_width:
        return s

    ellipsis = "..."
    budget = max_width - len(ellipsis)
    if budget <= 0:
        return ellipsis[:max_width]

    out = []
    used = 0
    for ch in s:
        w = 2 if "\u4e00" <= ch <= "\u9fff" else 1
        if used + w > budget:
            break
        out.append(ch)
        used += w
    return "".join(out) + ellipsis


def format_cell(text: str, width: int) -> str:
    s = str(text)
    if get_width(s) > width:
        s = clip_text(s, width)
    return s + " " * max(0, width - get_width(s))


def wrap_text_by_width(text: str, width: int) -> List[str]:
    """
    Wrap text into multiple lines based on display width.
    Keeps all information instead of truncating.
    """
    s = str(text)
    if s == "":
        return [""]

    lines = []
    current = []
    used = 0
    for ch in s:
        ch_w = 2 if "\u4e00" <= ch <= "\u9fff" else 1
        if used + ch_w > width and current:
            lines.append("".join(current))
            current = [ch]
            used = ch_w
        else:
            current.append(ch)
            used += ch_w
    if current:
        lines.append("".join(current))
    return lines


def load_skill_map() -> Dict[str, str]:
    """Load skill mapping from JSON first, fallback to CSV."""
    if os.path.exists(SKILL_MAP_JSON_PATH):
        with open(SKILL_MAP_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {str(k): str(v) for k, v in data.items()}

    if os.path.exists(SKILL_MAP_CSV_PATH):
        skill_map = {}
        with open(SKILL_MAP_CSV_PATH, "r", encoding="utf-8") as f:
            next(f, None)
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    old_id, new_id = parts[0], parts[1]
                    skill_map[str(new_id)] = f"Skill {new_id} (old:{old_id})"
        return skill_map

    return {}


def load_kg_adj() -> Dict[str, List[int]]:
    if not os.path.exists(KG_JSON_PATH):
        return {}
    with open(KG_JSON_PATH, "r", encoding="utf-8") as f:
        kg_adj_raw = json.load(f)
    kg_adj = {}
    for k, v in kg_adj_raw.items():
        kg_adj[str(k)] = [int(x) for x in v]
    return kg_adj


def infer_n_skills(kg_adj: Dict[str, List[int]], skill_map: Dict[str, str]) -> int:
    candidates = []
    for k, v in kg_adj.items():
        candidates.append(int(k))
        candidates.extend(int(x) for x in v)
    candidates.extend(int(k) for k in skill_map.keys() if str(k).isdigit())
    return max(candidates) if candidates else 124


def zpd_score(prob: float) -> float:
    """Prefer skills in the zone of proximal development."""
    return max(0.0, 1.0 - abs(prob - ZPD_CENTER) / ZPD_HALF_WIDTH)


# ==========================================
# 3) Model loading and prediction backend
# ==========================================
def try_load_current_kgsakt(
    kg_adj: Dict[str, List[int]], n_skills: int
) -> Tuple[torch.nn.Module, str]:
    if not os.path.exists(MODEL_WEIGHTS):
        return None, "no_weight_file"

    state_dict = torch.load(MODEL_WEIGHTS, map_location=DEVICE)
    if not isinstance(state_dict, dict):
        return None, "invalid_weight_format"

    key_set = set(state_dict.keys())
    current_style = "exercise_embed.weight" in key_set and "query_embed.weight" in key_set
    if not current_style:
        return None, "legacy_weight_detected"

    if "query_embed.weight" in state_dict:
        inferred_n_skills = int(state_dict["query_embed.weight"].shape[0]) - 1
    elif "fc_full.bias" in state_dict:
        inferred_n_skills = int(state_dict["fc_full.bias"].shape[0]) - 1
    else:
        inferred_n_skills = n_skills

    model = KGSAKTModel(
        n_skills=inferred_n_skills,
        kg_adj=kg_adj,
        max_seq=MAX_SEQ,
        use_time_feature=False,
    ).to(DEVICE)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, "model_loaded"


def estimate_mastery_heuristic(
    history_skills: List[int], history_corrects: List[int], kg_adj: Dict[str, List[int]], n_skills: int
) -> np.ndarray:
    mastery = np.full(n_skills + 1, 0.35, dtype=np.float32)
    seen_count = np.zeros(n_skills + 1, dtype=np.float32)

    for s, c in zip(history_skills, history_corrects):
        if s <= 0 or s > n_skills:
            continue
        seen_count[s] += 1.0
        eta = 0.22 / np.sqrt(seen_count[s])
        target = 0.85 if c == 1 else 0.20
        mastery[s] = (1.0 - eta) * mastery[s] + eta * target

    for s in range(1, n_skills + 1):
        prereqs = kg_adj.get(str(s), [])
        if len(prereqs) == 0:
            continue
        vals = [mastery[p] for p in prereqs if 0 < p <= n_skills]
        if vals:
            mastery[s] = float(np.clip(0.8 * mastery[s] + 0.2 * float(np.mean(vals)), 0.0, 1.0))

    mastery[0] = 0.0
    return mastery


def predict_mastery_distribution(
    model: torch.nn.Module,
    history_skills: List[int],
    history_corrects: List[int],
    n_skills: int,
    kg_adj: Dict[str, List[int]],
) -> np.ndarray:
    if model is None:
        return estimate_mastery_heuristic(history_skills, history_corrects, kg_adj, n_skills)

    skills = np.array(history_skills[-MAX_SEQ:], dtype=np.int64)
    corrects = np.array(history_corrects[-MAX_SEQ:], dtype=np.int64)
    interactions = skills + corrects * n_skills

    pad_len = MAX_SEQ - len(skills)
    x = np.pad(interactions, (pad_len, 0), constant_values=0)
    q = np.pad(skills, (pad_len, 0), constant_values=0)

    x_t = torch.LongTensor(x).unsqueeze(0).to(DEVICE)
    q_t = torch.LongTensor(q).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(q_t, x_t)
        probs = torch.sigmoid(logits[:, -1, :]).squeeze(0).cpu().numpy()
    return probs


# ==========================================
# 4) Recommendation policy
# ==========================================
def prereq_readiness(skill_id: int, mastery: np.ndarray, kg_adj: Dict[str, List[int]]) -> Tuple[float, float]:
    prereqs = kg_adj.get(str(skill_id), [])
    if len(prereqs) == 0:
        return 1.0, 1.0

    vals = [float(mastery[p]) for p in prereqs if 0 < p < len(mastery)]
    if not vals:
        return 0.0, 0.0

    covered = [1.0 if v >= 0.5 else 0.0 for v in vals]
    return float(np.mean(covered)), float(np.mean(vals))


def generate_reason(skill_id: int, prob: float, readiness: float, prereq_mean: float, kg_adj: Dict[str, List[int]]) -> str:
    if str(skill_id) in kg_adj:
        level = "高" if readiness >= 0.8 else "中" if readiness >= 0.6 else "低"
        return f"前置覆盖{level}（均值{prereq_mean:.2f}），当前掌握{prob:.2f}，适合作为下一步学习。"
    if 0.45 <= prob <= 0.75:
        return "处于最近发展区，预计学习收益较高。"
    if prob > 0.75:
        return "掌握较高，可用于巩固与迁移练习。"
    return "基础尚弱，建议先补齐前置知识。"


def recommend_resources(
    mastery: np.ndarray,
    history_skills: List[int],
    kg_adj: Dict[str, List[int]],
    skill_map: Dict[str, str],
    top_k: int = TOP_K,
) -> List[Dict]:
    n_skills = len(mastery) - 1
    history_counter = {}
    for s in history_skills:
        history_counter[s] = history_counter.get(s, 0) + 1
    max_repeat = max(history_counter.values()) if history_counter else 1

    candidates = []
    for s in range(1, n_skills + 1):
        prob = float(mastery[s])
        if prob >= MASTERY_HIGH:
            continue

        readiness, prereq_mean = prereq_readiness(s, mastery, kg_adj)
        if readiness < READINESS_MIN:
            continue

        zpd = zpd_score(prob)
        novelty = 1.0 - history_counter.get(s, 0) / max_repeat
        score = 0.55 * zpd + 0.35 * readiness + 0.10 * novelty

        candidates.append(
            {
                "skill_id": s,
                "skill_name": skill_map.get(str(s), f"Skill {s}"),
                "mastery_prob": round(prob, 4),
                "readiness": round(readiness, 4),
                "score": round(float(score), 4),
                "reason": generate_reason(s, prob, readiness, prereq_mean, kg_adj),
            }
        )

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]


def student_level(corrects: List[int]) -> Tuple[str, float]:
    recent = corrects[-5:] if len(corrects) >= 5 else corrects
    avg = float(sum(recent) / max(1, len(recent)))
    if avg < 0.4:
        return "初级", avg
    if avg < 0.7:
        return "中级", avg
    return "高级", avg


# ==========================================
# 5) Simulation runner
# ==========================================
def simulate_students() -> Dict[str, Dict]:
    kg_adj = load_kg_adj()
    skill_map = load_skill_map()
    n_skills = infer_n_skills(kg_adj, skill_map)

    model, model_status = try_load_current_kgsakt(kg_adj, n_skills)
    print(f"[Info] recommendation backend: {'KG-SAKT' if model is not None else 'Heuristic'} ({model_status})")
    print(f"[Info] skills: {n_skills}, kg_edges: {sum(len(v) for v in kg_adj.values())}")

    students = {
        "学生A（基础薄弱）": {
            "skills": [1, 1, 1, 1, 12, 12, 12, 1, 1, 95],
            "corrects": [0, 0, 1, 0, 0, 1, 0, 1, 1, 0],
        },
        "学生B（稳步进阶）": {
            "skills": [27, 27, 27, 10, 10, 10, 84, 84, 84, 84],
            "corrects": [1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
        },
        "学生C（拔尖挑战）": {
            "skills": [80, 80, 110, 110, 110, 86, 86, 86, 91, 91],
            "corrects": [1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
        },
    }

    results = {}
    for name, h in students.items():
        level, acc = student_level(h["corrects"])
        mastery = predict_mastery_distribution(model, h["skills"], h["corrects"], n_skills, kg_adj)
        recs = recommend_resources(mastery, h["skills"], kg_adj, skill_map, top_k=TOP_K)
        results[name] = {"level": level, "recent_accuracy": round(acc, 4), "recommendations": recs}
    return results


def print_table(results: Dict[str, Dict]):
    # Fixed widths + line wrapping keep output aligned while preserving full text.
    w_name, w_info, w_skill, w_prob, w_score, w_reason = 18, 18, 30, 10, 8, 38

    line_width = w_name + w_info + w_skill + w_prob + w_score + w_reason + 3 * 6 + 2
    print("\n" + "=" * line_width)
    header = (
        f" {format_cell('学生画像', w_name)} | {format_cell('当前水平(近5次)', w_info)} | "
        f"{format_cell('推荐知识点', w_skill)} | {format_cell('掌握概率', w_prob)} | "
        f"{format_cell('推荐分数', w_score)} | {format_cell('推荐理由', w_reason)}"
    )
    print(header)
    print("-" * line_width)

    for student_name, info in results.items():
        recs = info["recommendations"]
        info_str = f"{info['level']}(正确率:{info['recent_accuracy']:.0%})"
        if not recs:
            row = (
                f" {format_cell(student_name, w_name)} | {format_cell(info_str, w_info)} | "
                f"{format_cell('无可推荐项', w_skill)} | {format_cell('-', w_prob)} | "
                f"{format_cell('-', w_score)} | {format_cell('建议回顾历史薄弱点后重试', w_reason)}"
            )
            print(row)
            print("-" * line_width)
            continue

        for i, rec in enumerate(recs):
            n_display = student_name if i == 0 else ""
            i_display = info_str if i == 0 else ""
            mastery_text = f"{rec['mastery_prob']:.4f}"
            score_text = f"{rec['score']:.4f}"

            # Wrap potentially long fields so we keep full content.
            skill_lines = wrap_text_by_width(rec["skill_name"], w_skill)
            reason_lines = wrap_text_by_width(rec["reason"], w_reason)
            sub_rows = max(len(skill_lines), len(reason_lines))

            for r in range(sub_rows):
                name_cell = n_display if r == 0 else ""
                info_cell = i_display if r == 0 else ""
                skill_cell = skill_lines[r] if r < len(skill_lines) else ""
                prob_cell = mastery_text if r == 0 else ""
                score_cell = score_text if r == 0 else ""
                reason_cell = reason_lines[r] if r < len(reason_lines) else ""

                row = (
                    f" {format_cell(name_cell, w_name)} | {format_cell(info_cell, w_info)} | "
                    f"{format_cell(skill_cell, w_skill)} | {format_cell(prob_cell, w_prob)} | "
                    f"{format_cell(score_cell, w_score)} | {format_cell(reason_cell, w_reason)}"
                )
                print(row)
        print("-" * line_width)
    print("=" * line_width)


def main():
    results = simulate_students()
    print_table(results)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[Saved] recommendation simulation json -> {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()
