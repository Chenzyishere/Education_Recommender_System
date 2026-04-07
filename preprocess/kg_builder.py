import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd

# 项目路径设置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CLEAN_CSV = os.path.join(DATA_DIR, "assist9_cleaned.csv")  # 清理后的数据文件
SKILL_MAP_JSON = os.path.join(DATA_DIR, "skill_map.json")  # 技能映射文件
KG_JSON = os.path.join(DATA_DIR, "kg_adj_list.json")  # 知识图谱邻接表
KG_TRIPLE_JSON = os.path.join(DATA_DIR, "kg_triples.json")  # 知识图谱三元组
KG_GRAPH_SEMANTIC_JSON = os.path.join(DATA_DIR, "kg_graph_semantic.json")  # 语义图
DOMAIN_MAP_CSV = os.path.join(DATA_DIR, "skill_domain_map.csv")  # 技能领域映射
CORE_PATH_JSON = os.path.join(DATA_DIR, "core_skill_paths.json")  # 核心技能路径


# -----------------------------
# 1) 领域规则
# -----------------------------
# 用于自动分类的领域关键词
DOMAIN_KEYWORDS = {
    "Arithmetic": [  # 算术领域
        "whole numbers", "addition", "subtraction", "multiplication", "division",
        "order of operations", "estimation", "prime number", "divisibility",
    ],
    "Fraction_Ratio_Percent": [  # 分数、比例、百分比领域
        "fraction", "fractions", "proportion", "percent", "unit rate", "rate",
        "least common multiple", "greatest common factor",
    ],
    "Algebra_Equation_Function": [  # 代数、方程、函数领域
        "algebra", "equation", "inequal", "slope", "intercept", "quadratic",
        "polynomial", "variable", "scientific notation", "exponents",
    ],
    "Geometry_Measurement": [  # 几何、测量领域
        "angle", "triangle", "circle", "perimeter", "area", "surface area", "volume",
        "pythagorean", "prism", "cylinder", "cone", "sphere",
        "reflection", "rotation", "translation", "symmetry", "similar figures",
    ],
    "Statistics_Probability_Graph": [  # 统计、概率、图表领域
        "table", "histogram", "stem and leaf", "box and whisker", "mean", "median",
        "probability", "scatter plot", "coordinate graph", "number line",
    ],
}


# 每个领域的核心链模板
# 按顺序匹配这些模板，并使用匹配的技能作为骨干链
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


# 用于排序非核心技能的粗粒度级别关键词
# 级别从0到3，0表示最基础，3表示最复杂
LEVEL_KEYWORDS = [
    ["whole numbers", "addition", "subtraction", "table", "number line", "mean"],  # 级别0：基础
    ["fraction", "percent", "ratio", "area", "perimeter", "stem and leaf", "histogram", "linear equation"],  # 级别1：中级
    ["inequal", "slope", "intercept", "surface area", "volume", "probability", "scatter"],  # 级别2：高级
    ["quadratic", "polynomial", "systems of linear equations", "scientific notation"],  # 级别3：专家
]


def normalize_name(name: str) -> str:
    """
    规范化技能名称，保留英文部分并转为小写，以便于关键词匹配
    例如："Slope (斜率)" -> "slope"
    
    参数:
        name: 技能名称
        
    返回:
        str: 规范化后的技能名称
    """
    if not isinstance(name, str):
        return ""
    name = name.strip().lower()  # 去除首尾空格并转为小写
    # 保留中文括号前的文本
    name = name.split("(", 1)[0].strip()
    # 规范化空格
    name = re.sub(r"\s+", " ", name)
    return name


def infer_n_skills_from_clean_csv(clean_csv: str) -> int:
    """
    从清理后的数据文件中推断技能总数
    
    参数:
        clean_csv: 清理后的数据文件路径
        
    返回:
        int: 技能总数
    """
    df = pd.read_csv(clean_csv, usecols=["skill_id"])
    return int(df["skill_id"].max())


def load_skill_map() -> Dict[int, str]:
    """
    加载技能ID到技能名称的映射
    
    返回:
        Dict[int, str]: 技能ID到技能名称的映射
    """
    if not os.path.exists(SKILL_MAP_JSON):
        return {}
    with open(SKILL_MAP_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): str(v) for k, v in raw.items() if str(k).isdigit()}


def domain_score(norm_name: str, domain: str) -> int:
    """
    计算技能名称与特定领域的匹配分数
    
    参数:
        norm_name: 规范化后的技能名称
        domain: 领域名称
        
    返回:
        int: 匹配分数（匹配的关键词数量）
    """
    return sum(1 for kw in DOMAIN_KEYWORDS[domain] if kw in norm_name)


def classify_domain(norm_name: str) -> str:
    """
    对技能进行领域分类
    
    参数:
        norm_name: 规范化后的技能名称
        
    返回:
        str: 技能所属的领域
    """
    # 计算技能名称与每个领域的匹配分数
    scores = {d: domain_score(norm_name, d) for d in DOMAIN_KEYWORDS}
    # 选择分数最高的领域
    best_domain, best_score = max(scores.items(), key=lambda x: x[1])
    # 如果分数为0，归类为通用数学
    if best_score == 0:
        return "General_Math"
    return best_domain


def level_score(norm_name: str) -> int:
    """
    根据关键词信号计算技能的大致复杂度级别
    返回0..len(LEVEL_KEYWORDS)-1
    
    参数:
        norm_name: 规范化后的技能名称
        
    返回:
        int: 技能的复杂度级别（0-3）
    """
    best_level = 0
    best_hits = 0
    # 遍历每个级别的关键词，计算匹配次数
    for lv, keywords in enumerate(LEVEL_KEYWORDS):
        hits = sum(1 for kw in keywords if kw in norm_name)
        if hits > best_hits:
            best_hits = hits
            best_level = lv
    return best_level


def select_core_chain(domain_skills: List[Tuple[int, str]], domain: str) -> List[int]:
    """
    通过匹配有序的核心关键词选择核心技能
    返回技能ID链
    
    参数:
        domain_skills: 领域内的技能列表，每个元素是(技能ID, 规范化名称)的元组
        domain: 领域名称
        
    返回:
        List[int]: 核心技能ID链
    """
    if domain not in CORE_CHAIN_KEYWORDS:
        return []
    chain = []
    used = set()  # 用于记录已使用的技能ID
    # 按顺序匹配核心链关键词
    for kw in CORE_CHAIN_KEYWORDS[domain]:
        # 查找包含当前关键词且未使用的技能
        candidates = [(sid, name) for sid, name in domain_skills if kw in name and sid not in used]
        if not candidates:
            continue
        # 优先选择ID较小的技能以保持稳定性
        candidates.sort(key=lambda x: x[0])
        sid = candidates[0][0]
        chain.append(sid)
        used.add(sid)
    # 去重同时保持顺序
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
    构建邻接表，包含：
    1) 领域分配
    2) 每个领域内的核心链
    3) 非核心技能连接到最近的早期核心技能
    
    参数:
        n_skills: 技能总数
        skill_map: 技能ID到技能名称的映射
        
    返回:
        Tuple[Dict[str, List[int]], pd.DataFrame, Dict[str, List[int]]]: 
            - 知识图谱邻接表（目标技能 -> [前置技能]）
            - 包含技能领域信息的DataFrame
            - 每个领域的核心技能路径
    """
    rows = []
    by_domain = defaultdict(list)  # 按领域分组的技能
    # 处理每个技能
    for sid in range(1, n_skills + 1):
        raw_name = skill_map.get(sid, f"Skill {sid}")  # 获取技能名称
        norm = normalize_name(raw_name)  # 规范化技能名称
        domain = classify_domain(norm)  # 分类领域
        lv = level_score(norm)  # 计算复杂度级别
        rows.append({"skill_id": sid, "skill_name": raw_name, "norm_name": norm, "domain": domain, "level": lv})
        by_domain[domain].append((sid, norm))

    # 创建领域DataFrame并排序
    domain_df = pd.DataFrame(rows).sort_values(["domain", "level", "skill_id"]).reset_index(drop=True)

    # 为每个领域选择核心链
    core_paths = {}
    for domain, skills in by_domain.items():
        skills_sorted = sorted(skills, key=lambda x: x[0])  # 按技能ID排序
        core_chain = select_core_chain(skills_sorted, domain)  # 选择核心链
        # 如果没有匹配的链，保留一个代表性技能作为该领域的核心
        if len(core_chain) == 0 and len(skills_sorted) > 0:
            core_chain = [skills_sorted[0][0]]
        core_paths[domain] = core_chain

    # 构建邻接表：目标技能 -> [前置技能]
    kg_adj = {}

    # 1) 添加核心链边
    for domain, chain in core_paths.items():
        for i in range(1, len(chain)):
            target = chain[i]  # 当前技能
            prereq = chain[i - 1]  # 前一个技能作为前置技能
            kg_adj.setdefault(str(target), [])
            if prereq not in kg_adj[str(target)]:
                kg_adj[str(target)].append(prereq)

    # 2) 将非核心技能连接到同一领域中最近的早期核心技能
    # 创建技能ID到(领域, 级别)的映射
    row_lookup = {int(r.skill_id): (r.domain, int(r.level)) for r in domain_df.itertuples(index=False)}
    for sid in range(1, n_skills + 1):
        domain, lv = row_lookup[sid]
        chain = core_paths.get(domain, [])
        # 如果是核心技能或领域没有核心技能，则跳过
        if sid in chain or len(chain) == 0:
            continue

        # 候选核心点：同一领域，优先级别<=当前级别，然后按ID接近度
        candidates = []
        for c in chain:
            _, c_lv = row_lookup.get(c, (domain, 0))
            # 计算 penalty：级别差*1000 + ID差
            penalty = max(0, c_lv - lv) * 1000 + abs(sid - c)
            candidates.append((penalty, c))
        # 按 penalty 排序，选择最小的
        candidates.sort(key=lambda x: x[0])
        chosen = candidates[0][1]

        # 添加到邻接表
        kg_adj.setdefault(str(sid), [])
        if chosen not in kg_adj[str(sid)]:
            kg_adj[str(sid)].append(chosen)

    # 3) 确保确定性排序
    for k in list(kg_adj.keys()):
        kg_adj[k] = sorted(set(int(x) for x in kg_adj[k]))

    return kg_adj, domain_df, core_paths


def adjacency_to_triples(kg_adj: Dict[str, List[int]], skill_map: Dict[int, str]) -> List[dict]:
    """
    将邻接表转换为三元组，使用模式：
    (KP_source, requires, KP_target)

    其中：
    - KP_source: 当前要学习的技能（语义名称）
    - KP_target: KP_source所需的前置技能（语义名称）
    
    参数:
        kg_adj: 知识图谱邻接表
        skill_map: 技能ID到技能名称的映射
        
    返回:
        List[dict]: 三元组列表
    """
    triples = []
    for source, prereqs in kg_adj.items():
        s = int(source)  # 转换为整数
        for p in prereqs:
            triples.append(
                {
                    "KP_source": skill_map.get(s, f"Skill {s}"),  # 源技能名称
                    "relation": "requires",  # 关系类型
                    "KP_target": skill_map.get(int(p), f"Skill {int(p)}"),  # 目标技能名称
                    # 保留ID字段用于可重复性/下游连接
                    "KP_source_id": s,
                    "KP_target_id": int(p),
                }
            )
    # 为了可重复性，按源ID和目标ID排序
    triples.sort(key=lambda x: (x["KP_source_id"], x["KP_target_id"]))
    return triples


def triples_to_adjacency(triples: List[dict]) -> Dict[str, List[int]]:
    """
    从三元组构建邻接表（目标技能->前置技能）
    
    参数:
        triples: 三元组列表
        
    返回:
        Dict[str, List[int]]: 知识图谱邻接表
    """
    adj = defaultdict(list)
    for t in triples:
        s = int(t["KP_source_id"])
        p = int(t["KP_target_id"])
        adj[str(s)].append(p)
    out = {}
    for s, prereqs in adj.items():
        out[s] = sorted(set(prereqs))  # 去重并排序
    return out


def triples_to_semantic_graph(triples: List[dict]) -> Dict[str, List[str]]:
    """
    构建论文风格的语义图字典：
    Graph = {kp: [prerequisites]}
    
    参数:
        triples: 三元组列表
        
    返回:
        Dict[str, List[str]]: 语义图
    """
    graph = defaultdict(list)
    for t in triples:
        kp_source = str(t["KP_source"])
        kp_target = str(t["KP_target"])
        graph[kp_source].append(kp_target)
    out = {}
    for kp, prereqs in graph.items():
        out[kp] = sorted(set(prereqs))  # 去重并排序
    return out


def generate_kg(
    clean_csv: str = CLEAN_CSV,
    output_json: str = KG_JSON,
    mode: str = "domain_core",
    n_skills: int = None,
    output_domain_map_csv: str = DOMAIN_MAP_CSV,
    output_core_path_json: str = CORE_PATH_JSON,
    output_triple_json: str = KG_TRIPLE_JSON,
    output_semantic_graph_json: str = KG_GRAPH_SEMANTIC_JSON,
) -> Dict[str, List[int]]:
    """
    统一的知识图谱生成入口
    
    模式：
    - domain_core: 基于领域+核心链的前置技能图（默认）
    
    参数:
        clean_csv: 清理后的数据文件路径
        output_json: 邻接表输出路径
        mode: 生成模式
        n_skills: 技能总数（如果为None，则从数据文件推断）
        output_domain_map_csv: 领域映射输出路径
        output_core_path_json: 核心路径输出路径
        output_triple_json: 三元组输出路径
        output_semantic_graph_json: 语义图输出路径
        
    返回:
        Dict[str, List[int]]: 知识图谱邻接表
    """
    # 如果未指定技能总数，从数据文件推断
    if n_skills is None:
        if not os.path.exists(clean_csv):
            raise FileNotFoundError(f"Clean data not found: {clean_csv}")
        n_skills = infer_n_skills_from_clean_csv(clean_csv)

    # 加载技能映射
    skill_map = load_skill_map()
    # 检查模式是否支持
    if mode != "domain_core":
        raise ValueError(f"Unsupported mode: {mode}")

    # 构建领域核心知识图谱
    kg_adj, domain_df, core_paths = build_domain_core_kg(n_skills=n_skills, skill_map=skill_map)
    # 转换为三元组
    kg_triples = adjacency_to_triples(kg_adj, skill_map=skill_map)
    # 从三元组重建邻接表，确保存储语义一致
    kg_adj = triples_to_adjacency(kg_triples)
    # 构建语义图
    semantic_graph = triples_to_semantic_graph(kg_triples)

    # 保存结果
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(kg_adj, f, ensure_ascii=False, indent=2)
    with open(output_triple_json, "w", encoding="utf-8") as f:
        json.dump(kg_triples, f, ensure_ascii=False, indent=2)
    with open(output_semantic_graph_json, "w", encoding="utf-8") as f:
        json.dump(semantic_graph, f, ensure_ascii=False, indent=2)
    domain_df.to_csv(output_domain_map_csv, index=False, encoding="utf-8-sig")
    with open(output_core_path_json, "w", encoding="utf-8") as f:
        json.dump(core_paths, f, ensure_ascii=False, indent=2)

    # 计算边数
    edge_count = sum(len(v) for v in kg_adj.values())
    # 打印结果信息
    print(f"[KG] mode: {mode}")
    print(f"[KG] n_skills: {n_skills}")
    print(f"[KG] targets_with_prereq: {len(kg_adj)}, edges: {edge_count}")
    print(f"[KG] triples saved: {output_triple_json}")
    print(f"[KG] semantic graph saved: {output_semantic_graph_json}")
    print(f"[KG] adjacency saved: {output_json}")
    print(f"[KG] domain map saved: {output_domain_map_csv}")
    print(f"[KG] core paths saved: {output_core_path_json}")
    return kg_adj


if __name__ == "__main__":
    """
    主函数：生成知识图谱
    """
    generate_kg()