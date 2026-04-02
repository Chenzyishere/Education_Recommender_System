import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CLEAN_CSV = os.path.join(DATA_DIR, "assist9_cleaned.csv")
SKILL_MAP_JSON = os.path.join(DATA_DIR, "skill_map.json")
KG_JSON = os.path.join(DATA_DIR, "kg_adj_list.json")
KG_TRIPLE_JSON = os.path.join(DATA_DIR, "kg_triples.json")
KG_TRIPLE_CSV = os.path.join(DATA_DIR, "kg_triples.csv")
KG_GRAPH_SEMANTIC_JSON = os.path.join(DATA_DIR, "kg_graph_semantic.json")
DOMAIN_MAP_CSV = os.path.join(DATA_DIR, "skill_domain_map.csv")
CORE_PATH_JSON = os.path.join(DATA_DIR, "core_skill_paths.json")


# -----------------------------
# 1) Domain rules
# -----------------------------
# Domain keywords used for automatic categorization.
DOMAIN_KEYWORDS = {
    "Arithmetic": [
        "whole numbers", "addition", "subtraction", "multiplication", "division",
        "order of operations", "estimation", "prime number", "divisibility",
    ],
    "Fraction_Ratio_Percent": [
        "fraction", "fractions", "proportion", "percent", "unit rate", "rate",
        "least common multiple", "greatest common factor",
    ],
    "Algebra_Equation_Function": [
        "algebra", "equation", "inequal", "slope", "intercept", "quadratic",
        "polynomial", "variable", "scientific notation", "exponents",
    ],
    "Geometry_Measurement": [
        "angle", "triangle", "circle", "perimeter", "area", "surface area", "volume",
        "pythagorean", "prism", "cylinder", "cone", "sphere",
        "reflection", "rotation", "translation", "symmetry", "similar figures",
    ],
    "Statistics_Probability_Graph": [
        "table", "histogram", "stem and leaf", "box and whisker", "mean", "median",
        "probability", "scatter plot", "coordinate graph", "number line",
    ],
}


# Core chain templates per domain.
# We try to match these in order and use matched skills as the backbone chain.
CORE_CHAIN_KEYWORDS = {
    "Arithmetic": [
        "addition whole numbers",
        "subtraction whole numbers",
        "multiplication whole numbers",
        "division",
        "order of operations",
    ],
    "Fraction_Ratio_Percent": [
        "fractions addition",
        "fractions multiplication",
        "fractions division",
        "proportion",
        "percents",
        "unit rate",
    ],
    "Algebra_Equation_Function": [
        "algebraic simplification",
        "linear equations",
        "inequalities",
        "slope",
        "intercept",
        "systems of linear equations",
        "quadratic formula",
    ],
    "Geometry_Measurement": [
        "angles",
        "area",
        "surface area",
        "volume",
        "pythagorean theorem",
    ],
    "Statistics_Probability_Graph": [
        "table",
        "histogram",
        "mean",
        "median",
        "probability",
        "scatter plot",
    ],
}


# Coarse level keywords for ordering non-core skills.
LEVEL_KEYWORDS = [
    ["whole numbers", "addition", "subtraction", "table", "number line", "mean"],
    ["fraction", "percent", "ratio", "area", "perimeter", "stem and leaf", "histogram", "linear equation"],
    ["inequal", "slope", "intercept", "surface area", "volume", "probability", "scatter"],
    ["quadratic", "polynomial", "systems of linear equations", "scientific notation"],
]


def normalize_name(name: str) -> str:
    """
    Keep English portion and lowercase for robust keyword matching.
    Example: "Slope (斜率)" -> "slope"
    """
    if not isinstance(name, str):
        return ""
    name = name.strip().lower()
    # Keep text before Chinese parenthesis if present.
    name = name.split("(", 1)[0].strip()
    # Normalize spaces.
    name = re.sub(r"\s+", " ", name)
    return name


def infer_n_skills_from_clean_csv(clean_csv: str) -> int:
    df = pd.read_csv(clean_csv, usecols=["skill_id"])
    return int(df["skill_id"].max())


def load_skill_map() -> Dict[int, str]:
    if not os.path.exists(SKILL_MAP_JSON):
        return {}
    with open(SKILL_MAP_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): str(v) for k, v in raw.items() if str(k).isdigit()}


def domain_score(norm_name: str, domain: str) -> int:
    return sum(1 for kw in DOMAIN_KEYWORDS[domain] if kw in norm_name)


def classify_domain(norm_name: str) -> str:
    scores = {d: domain_score(norm_name, d) for d in DOMAIN_KEYWORDS}
    best_domain, best_score = max(scores.items(), key=lambda x: x[1])
    if best_score == 0:
        return "General_Math"
    return best_domain


def level_score(norm_name: str) -> int:
    """
    Rough complexity stage from keyword signals.
    Returns 0..len(LEVEL_KEYWORDS)-1
    """
    best_level = 0
    best_hits = 0
    for lv, keywords in enumerate(LEVEL_KEYWORDS):
        hits = sum(1 for kw in keywords if kw in norm_name)
        if hits > best_hits:
            best_hits = hits
            best_level = lv
    return best_level


def select_core_chain(domain_skills: List[Tuple[int, str]], domain: str) -> List[int]:
    """
    Pick core skills by matching ordered core keywords.
    Returns skill-id chain.
    """
    if domain not in CORE_CHAIN_KEYWORDS:
        return []
    chain = []
    used = set()
    for kw in CORE_CHAIN_KEYWORDS[domain]:
        candidates = [(sid, name) for sid, name in domain_skills if kw in name and sid not in used]
        if not candidates:
            continue
        # Prefer smaller id for stability.
        candidates.sort(key=lambda x: x[0])
        sid = candidates[0][0]
        chain.append(sid)
        used.add(sid)
    # Deduplicate while preserving order.
    out = []
    seen = set()
    for sid in chain:
        if sid not in seen:
            out.append(sid)
            seen.add(sid)
    return out


def build_domain_core_kg(
    n_skills: int,
    skill_map: Dict[int, str],
) -> Tuple[Dict[str, List[int]], pd.DataFrame, Dict[str, List[int]]]:
    """
    Build adjacency with:
    1) Domain assignment
    2) Core chain inside each domain
    3) Non-core attachment to nearest earlier core skill
    """
    rows = []
    by_domain = defaultdict(list)
    for sid in range(1, n_skills + 1):
        raw_name = skill_map.get(sid, f"Skill {sid}")
        norm = normalize_name(raw_name)
        domain = classify_domain(norm)
        lv = level_score(norm)
        rows.append({"skill_id": sid, "skill_name": raw_name, "norm_name": norm, "domain": domain, "level": lv})
        by_domain[domain].append((sid, norm))

    domain_df = pd.DataFrame(rows).sort_values(["domain", "level", "skill_id"]).reset_index(drop=True)

    core_paths = {}
    for domain, skills in by_domain.items():
        skills_sorted = sorted(skills, key=lambda x: x[0])
        core_chain = select_core_chain(skills_sorted, domain)
        # If no matched chain, keep one representative as core for that domain.
        if len(core_chain) == 0 and len(skills_sorted) > 0:
            core_chain = [skills_sorted[0][0]]
        core_paths[domain] = core_chain

    # Build adjacency: target -> [prereqs]
    kg_adj = {}

    # 1) Core chain edges
    for domain, chain in core_paths.items():
        for i in range(1, len(chain)):
            target = chain[i]
            prereq = chain[i - 1]
            kg_adj.setdefault(str(target), [])
            if prereq not in kg_adj[str(target)]:
                kg_adj[str(target)].append(prereq)

    # 2) Non-core skill attachment to nearest earlier core within same domain.
    row_lookup = {int(r.skill_id): (r.domain, int(r.level)) for r in domain_df.itertuples(index=False)}
    for sid in range(1, n_skills + 1):
        domain, lv = row_lookup[sid]
        chain = core_paths.get(domain, [])
        if sid in chain or len(chain) == 0:
            continue

        # Candidate core points: same domain, prefer level <= current, then by id closeness.
        candidates = []
        for c in chain:
            _, c_lv = row_lookup.get(c, (domain, 0))
            penalty = max(0, c_lv - lv) * 1000 + abs(sid - c)
            candidates.append((penalty, c))
        candidates.sort(key=lambda x: x[0])
        chosen = candidates[0][1]

        kg_adj.setdefault(str(sid), [])
        if chosen not in kg_adj[str(sid)]:
            kg_adj[str(sid)].append(chosen)

    # 3) Ensure deterministic ordering.
    for k in list(kg_adj.keys()):
        kg_adj[k] = sorted(set(int(x) for x in kg_adj[k]))

    return kg_adj, domain_df, core_paths


def adjacency_to_triples(kg_adj: Dict[str, List[int]], skill_map: Dict[int, str]) -> List[dict]:
    """
    Convert adjacency into triples using schema:
    (KP_source, requires, KP_target)

    Here:
    - KP_source: current skill to learn (semantic name)
    - KP_target: prerequisite skill required by KP_source (semantic name)
    """
    triples = []
    for source, prereqs in kg_adj.items():
        s = int(source)
        for p in prereqs:
            triples.append(
                {
                    "KP_source": skill_map.get(s, f"Skill {s}"),
                    "relation": "requires",
                    "KP_target": skill_map.get(int(p), f"Skill {int(p)}"),
                    # Keep id fields for reproducibility / downstream joins.
                    "KP_source_id": s,
                    "KP_target_id": int(p),
                }
            )
    # Stable order for reproducibility.
    triples.sort(key=lambda x: (x["KP_source_id"], x["KP_target_id"]))
    return triples


def triples_to_adjacency(triples: List[dict]) -> Dict[str, List[int]]:
    """Build adjacency target->prereqs from triples."""
    adj = defaultdict(list)
    for t in triples:
        s = int(t["KP_source_id"])
        p = int(t["KP_target_id"])
        adj[str(s)].append(p)
    out = {}
    for s, prereqs in adj.items():
        out[s] = sorted(set(prereqs))
    return out


def triples_to_semantic_graph(triples: List[dict]) -> Dict[str, List[str]]:
    """
    Build semantic graph dictionary in paper style:
    Graph = {kp: [prerequisites]}
    """
    graph = defaultdict(list)
    for t in triples:
        kp_source = str(t["KP_source"])
        kp_target = str(t["KP_target"])
        graph[kp_source].append(kp_target)
    out = {}
    for kp, prereqs in graph.items():
        out[kp] = sorted(set(prereqs))
    return out


def generate_kg(
    clean_csv: str = CLEAN_CSV,
    output_json: str = KG_JSON,
    mode: str = "domain_core",
    n_skills: int = None,
    output_domain_map_csv: str = DOMAIN_MAP_CSV,
    output_core_path_json: str = CORE_PATH_JSON,
    output_triple_json: str = KG_TRIPLE_JSON,
    output_triple_csv: str = KG_TRIPLE_CSV,
    output_semantic_graph_json: str = KG_GRAPH_SEMANTIC_JSON,
) -> Dict[str, List[int]]:
    """
    Unified KG generation entry.
    mode:
    - domain_core: domain-based + core-chain-based prerequisite graph (default)
    """
    if n_skills is None:
        if not os.path.exists(clean_csv):
            raise FileNotFoundError(f"Clean data not found: {clean_csv}")
        n_skills = infer_n_skills_from_clean_csv(clean_csv)

    skill_map = load_skill_map()
    if mode != "domain_core":
        raise ValueError(f"Unsupported mode: {mode}")

    kg_adj, domain_df, core_paths = build_domain_core_kg(n_skills=n_skills, skill_map=skill_map)
    kg_triples = adjacency_to_triples(kg_adj, skill_map=skill_map)
    # Rebuild adjacency from triples to ensure storage semantics are consistent.
    kg_adj = triples_to_adjacency(kg_triples)
    semantic_graph = triples_to_semantic_graph(kg_triples)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(kg_adj, f, ensure_ascii=False, indent=2)
    with open(output_triple_json, "w", encoding="utf-8") as f:
        json.dump(kg_triples, f, ensure_ascii=False, indent=2)
    pd.DataFrame(kg_triples).to_csv(output_triple_csv, index=False, encoding="utf-8-sig")
    with open(output_semantic_graph_json, "w", encoding="utf-8") as f:
        json.dump(semantic_graph, f, ensure_ascii=False, indent=2)
    domain_df.to_csv(output_domain_map_csv, index=False, encoding="utf-8-sig")
    with open(output_core_path_json, "w", encoding="utf-8") as f:
        json.dump(core_paths, f, ensure_ascii=False, indent=2)

    edge_count = sum(len(v) for v in kg_adj.values())
    print(f"[KG] mode: {mode}")
    print(f"[KG] n_skills: {n_skills}")
    print(f"[KG] targets_with_prereq: {len(kg_adj)}, edges: {edge_count}")
    print(f"[KG] triples saved: {output_triple_json}")
    print(f"[KG] triples csv saved: {output_triple_csv}")
    print(f"[KG] semantic graph saved: {output_semantic_graph_json}")
    print(f"[KG] adjacency saved: {output_json}")
    print(f"[KG] domain map saved: {output_domain_map_csv}")
    print(f"[KG] core paths saved: {output_core_path_json}")
    return kg_adj


if __name__ == "__main__":
    generate_kg()
