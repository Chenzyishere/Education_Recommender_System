import json
import os
import re
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch

# Ensure project root is importable when running: python utils/inference_recommend.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
BASE_DIR = PROJECT_ROOT
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

KG_JSON_PATH = os.path.join(DATA_DIR, "kg_adj_list.json")
SKILL_MAP_JSON_PATH = os.path.join(DATA_DIR, "skill_map.json")
MODEL_WEIGHTS = os.path.join(DATA_DIR, "kg_sakt_model.pth")
OUTPUT_JSON_PATH = os.path.join(DATA_DIR, "recommendation_simulation.json")

MAX_SEQ = 100
TOP_K = 3
MASTERY_HIGH = 0.85
READINESS_MIN = 0.60
MASTERY_GATE_THRESHOLD = 0.45
ZPD_CENTER = 0.60
ZPD_HALF_WIDTH = 0.30
ZPD_MIN = 0.30
ZPD_MAX = 0.60
COGNITIVE_LOAD_LAMBDA = 0.20


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


def safe_console_text(text: str) -> str:
    """
    Make text printable on Windows consoles with legacy encodings (e.g., GBK).
    Non-encodable chars are replaced to avoid UnicodeEncodeError.
    """
    s = str(text)
    try:
        s.encode("gbk")
        return s
    except Exception:
        return s.encode("gbk", errors="replace").decode("gbk", errors="replace")


def to_chinese_skill_name(raw_name: str, skill_id: str) -> str:
    """
    将技能名规范为中文展示：
    1. 优先提取括号中的中文解释（如 `xxx (中文名)` 或 `xxx（中文名）`）
    2. 若整体已含中文，则直接使用原名
    3. 若无中文信息，则退化为 `技能{id}`
    """
    name = str(raw_name).strip()
    match = re.search(r"[（(]([^）)]*[\u4e00-\u9fff][^）)]*)[）)]", name)
    if match:
        return match.group(1).strip()
    if re.search(r"[\u4e00-\u9fff]", name):
        return name
    return f"技能{skill_id}"


def load_skill_map() -> Dict[str, str]:
    """Load skill mapping from JSON only."""
    if os.path.exists(SKILL_MAP_JSON_PATH):
        with open(SKILL_MAP_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {str(k): to_chinese_skill_name(v, str(k)) for k, v in data.items()}

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


def mastery_gate_pass(
    skill_id: int,
    mastery: np.ndarray,
    kg_adj: Dict[str, List[int]],
    threshold: float = MASTERY_GATE_THRESHOLD,
) -> bool:
    """
    Mastery-learning gate:
    recommend next skill only when all prerequisites are above threshold.
    """
    prereqs = kg_adj.get(str(skill_id), [])
    if len(prereqs) == 0:
        return True
    for p in prereqs:
        if p <= 0 or p >= len(mastery):
            continue
        if float(mastery[p]) < threshold:
            return False
    return True


def estimate_cognitive_load(
    history_corrects: List[int],
    response_time_ms: List[float] = None,
) -> float:
    """
    Simple cognitive-load proxy in [0,1]:
    - higher recent error rate -> higher load
    - higher recent volatility -> higher load
    - longer response time (if provided) -> higher load
    """
    if not history_corrects:
        return 0.5

    recent = history_corrects[-5:]
    arr = np.array(recent, dtype=np.float32)
    error_rate = 1.0 - float(np.mean(arr))
    volatility = float(np.std(arr))

    time_load = 0.0
    if response_time_ms:
        recent_time = np.array(response_time_ms[-5:], dtype=np.float32)
        mean_ms = float(np.mean(recent_time))
        # Normalize around 15s baseline, cap at [0,1].
        time_load = float(np.clip((mean_ms - 15000.0) / 15000.0, 0.0, 1.0))

    load = 0.5 * error_rate + 0.3 * volatility + 0.2 * time_load
    return float(np.clip(load, 0.0, 1.0))


def get_mastered_prereq_names(
    skill_id: int,
    mastery: np.ndarray,
    kg_adj: Dict[str, List[int]],
    skill_map: Dict[str, str],
    threshold: float = 0.5,
) -> List[str]:
    """
    Return prerequisite skill names that are already mastered by the student.
    Mastered is defined as predicted mastery >= threshold.
    """
    prereqs = kg_adj.get(str(skill_id), [])
    names = []
    for p in prereqs:
        if 0 < p < len(mastery) and float(mastery[p]) >= threshold:
            names.append(skill_map.get(str(p), f"技能{p}"))
    return names


def generate_reason(
    skill_name: str,
    mastered_prereq_names: List[str],
    skill_id: int,
    prob: float,
    readiness: float,
    prereq_mean: float,
    kg_adj: Dict[str, List[int]],
) -> str:
    # 中文解释模板：
    # “因为你已掌握前置知识A/B，所以推荐学习C。”
    if mastered_prereq_names:
        prereq_text = " / ".join(mastered_prereq_names[:3]) + (" 等" if len(mastered_prereq_names) > 3 else "")
    elif str(skill_id) in kg_adj:
        level = "高" if readiness >= 0.8 else "中" if readiness >= 0.6 else "低"
        prereq_text = f"部分前置基础（覆盖度={level}，均值={prereq_mean:.2f}）"
    elif 0.45 <= prob <= 0.75:
        prereq_text = "当前阶段基础知识"
    elif prob > 0.75:
        prereq_text = "较稳固的核心基础"
    else:
        prereq_text = "仍需巩固的基础"
    return f"因为你已掌握{prereq_text}，所以推荐学习{skill_name}。"


def recommend_resources(
    mastery: np.ndarray,
    history_skills: List[int],
    history_corrects: List[int],
    kg_adj: Dict[str, List[int]],
    skill_map: Dict[str, str],
    response_time_ms: List[float] = None,
    top_k: int = TOP_K,
) -> List[Dict]:
    n_skills = len(mastery) - 1
    history_counter = {}
    for s in history_skills:
        history_counter[s] = history_counter.get(s, 0) + 1
    max_repeat = max(history_counter.values()) if history_counter else 1
    global_load = estimate_cognitive_load(history_corrects, response_time_ms)

    candidates = []
    for s in range(1, n_skills + 1):
        prob = float(mastery[s])
        if prob >= MASTERY_HIGH:
            continue

        # Mastery learning: prerequisite skills must be sufficiently mastered.
        if not mastery_gate_pass(s, mastery, kg_adj, threshold=MASTERY_GATE_THRESHOLD):
            continue

        readiness, prereq_mean = prereq_readiness(s, mastery, kg_adj)
        if readiness < READINESS_MIN:
            continue

        zpd = zpd_score(prob)
        novelty = 1.0 - history_counter.get(s, 0) / max_repeat
        # Cognitive-load penalty:
        # harder-than-ZPD candidates are penalized more under high load.
        overload = max(0.0, ZPD_MIN - prob)
        cognitive_penalty = global_load + overload
        score = 0.55 * zpd + 0.35 * readiness + 0.10 * novelty - COGNITIVE_LOAD_LAMBDA * cognitive_penalty
        skill_name = skill_map.get(str(s), f"技能{s}")
        mastered_prereq_names = get_mastered_prereq_names(
            skill_id=s, mastery=mastery, kg_adj=kg_adj, skill_map=skill_map, threshold=0.5
        )
        mastered_prereq_text = " / ".join(mastered_prereq_names) if mastered_prereq_names else "无"

        candidates.append(
            {
                "skill_id": s,
                "skill_name": skill_name,
                "mastery_prob": round(prob, 4),
                "readiness": round(readiness, 4),
                "score": round(float(score), 4),
                "cognitive_load": round(global_load, 4),
                "cognitive_penalty": round(float(cognitive_penalty), 4),
                "mastery_gate_threshold": MASTERY_GATE_THRESHOLD,
                "mastered_prereqs": mastered_prereq_names,
                "mastered_prereqs_text": mastered_prereq_text,
                "reason": generate_reason(
                    skill_name=skill_name,
                    mastered_prereq_names=mastered_prereq_names,
                    skill_id=s,
                    prob=prob,
                    readiness=readiness,
                    prereq_mean=prereq_mean,
                    kg_adj=kg_adj,
                ),
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
    print(f"[信息] 推荐后端: {'KG-SAKT' if model is not None else '启发式'}（{model_status}）")
    print(f"[信息] 知识点数: {n_skills}，图谱边数: {sum(len(v) for v in kg_adj.values())}")
    students = {
        "学生A（基础薄弱）": {
            "skills": [1, 1, 1, 1, 12, 12, 12, 1, 1, 95],
            "corrects": [0, 0, 1, 0, 0, 1, 0, 1, 1, 0],
            "response_ms": [28000, 26000, 25000, 27000, 24000, 23000, 25000, 22000, 21000, 26000],
        },
        "学生B（稳步进阶）": {
            "skills": [27, 27, 27, 10, 10, 10, 84, 84, 84, 84],
            "corrects": [1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
            "response_ms": [17000, 16000, 15000, 18000, 16000, 15000, 14500, 16500, 15000, 14800],
        },
        "学生C（拔尖挑战）": {
            "skills": [80, 80, 110, 110, 110, 86, 86, 86, 91, 91],
            "corrects": [1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
            "response_ms": [12000, 11000, 11500, 10500, 10000, 9800, 12500, 10800, 9900, 10100],
        },
        "学生D（高正确率慢节奏）": {
            "skills": [5, 5, 19, 19, 34, 34, 58, 58, 75, 75],
            "corrects": [1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
            "response_ms": [34000, 32000, 31000, 30000, 33000, 36000, 34000, 33500, 32000, 31500],
        },
        "学生E（速度快但不稳定）": {
            "skills": [3, 14, 3, 14, 28, 28, 62, 62, 14, 3],
            "corrects": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "response_ms": [7000, 6500, 6800, 6200, 6600, 6400, 6700, 6100, 6500, 6000],
        },
        "学生F（几何偏科）": {
            "skills": [70, 71, 72, 73, 74, 75, 40, 41, 42, 43],
            "corrects": [1, 1, 1, 0, 1, 1, 0, 0, 1, 0],
            "response_ms": [16000, 15000, 14500, 17000, 15000, 14800, 21000, 22000, 20000, 23000],
        },
        "学生G（代数偏科）": {
            "skills": [20, 21, 22, 23, 24, 25, 90, 91, 92, 93],
            "corrects": [1, 1, 0, 1, 1, 0, 0, 1, 0, 0],
            "response_ms": [14000, 13000, 15000, 12500, 12000, 15500, 21000, 19500, 22000, 23000],
        },
        "学生H（恢复上升型）": {
            "skills": [8, 8, 8, 8, 15, 15, 32, 32, 32, 60],
            "corrects": [0, 0, 0, 1, 0, 1, 0, 1, 1, 1],
            "response_ms": [30000, 29000, 28000, 26000, 25000, 23000, 22000, 21000, 20000, 19000],
        },
    }
    results = {}
    for name, h in students.items():
        level, acc = student_level(h["corrects"])
        mastery = predict_mastery_distribution(model, h["skills"], h["corrects"], n_skills, kg_adj)
        recs = recommend_resources(
            mastery=mastery,
            history_skills=h["skills"],
            history_corrects=h["corrects"],
            kg_adj=kg_adj,
            skill_map=skill_map,
            response_time_ms=h.get("response_ms"),
            top_k=TOP_K,
        )
        results[name] = {"level": level, "recent_accuracy": round(acc, 4), "recommendations": recs}
    return results


def print_table(results: Dict[str, Dict]):
    # 固定列宽 + 自动换行：保证对齐，同时不丢失信息。
    w_profile, w_skill, w_prereq, w_prob, w_score, w_reason = 30, 24, 24, 10, 8, 34

    line_width = w_profile + w_skill + w_prereq + w_prob + w_score + w_reason + 4 * 5 + 2
    print(safe_console_text("\n" + "=" * line_width))
    header = (
        f" {format_cell('学生画像（当前水平）', w_profile)} | "
        f"{format_cell('推荐知识点', w_skill)} | {format_cell('已掌握前置', w_prereq)} | "
        f"{format_cell('掌握概率', w_prob)} | {format_cell('推荐分数', w_score)} | {format_cell('推荐理由', w_reason)}"
    )
    print(safe_console_text(header))
    print(safe_console_text("-" * line_width))

    for student_name, info in results.items():
        recs = info["recommendations"]
        profile_str = f"{student_name} | {info['level']}（近5次正确率:{info['recent_accuracy']:.0%}）"
        if not recs:
            row = (
                f" {format_cell(profile_str, w_profile)} | {format_cell('暂无候选', w_skill)} | "
                f"{format_cell('-', w_prereq)} | {format_cell('-', w_prob)} | "
                f"{format_cell('-', w_score)} | {format_cell('请先巩固薄弱前置知识后再推荐。', w_reason)}"
            )
            print(safe_console_text(row))
            print(safe_console_text("-" * line_width))
            continue

        for i, rec in enumerate(recs):
            p_display = profile_str if i == 0 else ""
            mastery_text = f"{rec['mastery_prob']:.4f}"
            score_text = f"{rec['score']:.4f}"

            # 对较长字段换行展示，确保完整信息可读。
            skill_lines = wrap_text_by_width(rec["skill_name"], w_skill)
            prereq_lines = wrap_text_by_width(rec.get("mastered_prereqs_text", "无"), w_prereq)
            reason_lines = wrap_text_by_width(rec["reason"], w_reason)
            sub_rows = max(len(skill_lines), len(prereq_lines), len(reason_lines))

            for r in range(sub_rows):
                profile_cell = p_display if r == 0 else ""
                skill_cell = skill_lines[r] if r < len(skill_lines) else ""
                prereq_cell = prereq_lines[r] if r < len(prereq_lines) else ""
                prob_cell = mastery_text if r == 0 else ""
                score_cell = score_text if r == 0 else ""
                reason_cell = reason_lines[r] if r < len(reason_lines) else ""

                row = (
                    f" {format_cell(profile_cell, w_profile)} | {format_cell(skill_cell, w_skill)} | "
                    f"{format_cell(prereq_cell, w_prereq)} | {format_cell(prob_cell, w_prob)} | "
                    f"{format_cell(score_cell, w_score)} | {format_cell(reason_cell, w_reason)}"
                )
                print(safe_console_text(row))
        print(safe_console_text("-" * line_width))
    print(safe_console_text("=" * line_width))


def main():
    results = simulate_students()
    print_table(results)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[已保存] 推荐仿真结果 -> {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()

