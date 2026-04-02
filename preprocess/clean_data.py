import os
import sys

import pandas as pd

# Make project root importable when running this file directly.
# 这部分代码用于设置项目路径，确保可以直接运行此文件时导入项目根目录的模块
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件所在目录
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)  # 获取项目根目录
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)  # 将项目根目录添加到系统路径

from preprocess.kg_builder import generate_kg  # 导入知识图谱生成模块

def build_time_gap(df: pd.DataFrame):
    """
    尝试从可用的时间戳列构建每个用户的时间间隔特征（秒）。
    当没有可用的时间戳时返回None。
    
    参数:
        df: 包含用户交互数据的DataFrame
        
    返回:
        pd.Series或None: 包含时间间隔的序列，或当没有可用时间戳时返回None
    """
    # 可能的时间戳列名称列表
    timestamp_candidates = ["event_time", "timestamp", "start_time", "time"]
    # 查找第一个存在的时间戳列
    timestamp_col = next((col for col in timestamp_candidates if col in df.columns), None)
    if timestamp_col is None:
        return None

    # 解析时间戳，无效值转换为NaT
    parsed_time = pd.to_datetime(df[timestamp_col], errors="coerce")
    # 如果所有时间戳都无效，返回None
    if parsed_time.isna().all():
        return None

    # 计算每个用户内的时间差，转换为秒，填充缺失值为0，并确保非负
    return (
        parsed_time.groupby(df["user_id"])
        .diff()  # 计算相邻时间戳的差值
        .dt.total_seconds()  # 转换为秒
        .fillna(0.0)  # 填充第一个交互的时间差为0
        .clip(lower=0.0)  # 确保时间差非负
    )


def clean_assist9_data(raw_path: str, save_path: str, map_save_path: str) -> int:
    """
    清理原始ASSIST风格的数据，并保留序列顺序字段用于前提条件挖掘。
    返回重新映射后的唯一技能数量。
    
    参数:
        raw_path: 原始数据文件路径
        save_path: 清理后数据保存路径
        map_save_path: 技能ID映射保存路径
        
    返回:
        int: 重新映射后的唯一技能数量
    """
    print(f"[Clean] Reading raw file: {raw_path}")
    # 读取原始数据，使用ISO-8859-1编码以处理特殊字符
    df = pd.read_csv(raw_path, encoding="ISO-8859-1", low_memory=False)

    # 规范化列名并仅保留必要/可选的有用列
    df.columns = [c.strip() for c in df.columns]  # 去除列名中的空白字符
    core_cols = ["user_id", "skill_id", "correct", "order_id"]  # 核心列
    # 收集存在的可选列
    optional_cols = [
        col
        for col in ["event_time", "timestamp", "start_time", "time", "ms_first_response"]
        if col in df.columns
    ]
    # 选择列并删除skill_id为空的行
    df = df[core_cols + optional_cols].dropna(subset=["skill_id"])

    # 去重并排序以保留交互顺序
    df = df.drop_duplicates()  # 删除重复行
    df = df.sort_values(by=["user_id", "order_id"], kind="stable")  # 按用户ID和顺序ID排序

    # 显式添加用户内序列索引（对时间推理很重要）
    df["sequence_idx"] = df.groupby("user_id").cumcount()  # 为每个用户的交互分配连续索引

    # 构建可选的时间特征
    time_gap = build_time_gap(df)  # 计算时间间隔
    if time_gap is not None:
        df["time_gap"] = time_gap

    # 将稀疏/原始技能ID重新映射到连续ID [1..N]
    unique_skills = sorted(df["skill_id"].unique())  # 获取唯一技能并排序
    skill_map = {old: i + 1 for i, old in enumerate(unique_skills)}  # 创建映射
    df["skill_id"] = df["skill_id"].map(skill_map)  # 应用映射

    # 保存映射以用于可解释性和推理
    skill_map_df = pd.DataFrame(list(skill_map.items()), columns=["old_id", "new_id"])
    skill_map_df.to_csv(map_save_path, index=False)
    print(f"[Clean] Saved skill map to: {map_save_path}")

    # 保存带有顺序信息的清理后数据集
    save_cols = ["user_id", "skill_id", "correct", "order_id", "sequence_idx"]
    if "time_gap" in df.columns:
        save_cols.append("time_gap")
    if "ms_first_response" in df.columns:
        save_cols.append("ms_first_response")
    df[save_cols].to_csv(save_path, index=False)

    n_skills = len(unique_skills)
    print(f"[Clean] Finished. Skills: {n_skills}, Rows: {len(df)}")
    print(f"[Clean] Saved cleaned data to: {save_path}")
    if "time_gap" not in df.columns:
        print("[Clean] No parseable timestamp column found, time_gap not generated.")
    return n_skills


def build_kg_after_clean(data_dir: str):
    """
    使用“学科领域 + 核心知识点链路”规则在清理后立即构建知识图谱。
    
    参数:
        data_dir: 数据目录路径
    """
    clean_csv = os.path.join(data_dir, "assist9_cleaned.csv")  # 清理后的数据文件路径
    kg_json = os.path.join(data_dir, "kg_adj_list.json")  # 知识图谱保存路径
    generate_kg(clean_csv=clean_csv, output_json=kg_json, mode="domain_core")
    print(f"[Pipeline] Domain-core KG built and saved to: {kg_json}")


if __name__ == "__main__":
    """
    主函数：执行数据清理和知识图谱构建流程
    """
    # 定义数据目录路径
    data_dir = os.path.join(PROJECT_ROOT, "data")
    # 如果数据目录不存在，创建它
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"[Init] Created missing directory: {data_dir}")

    # 定义输入输出文件路径
    input_csv = os.path.join(data_dir, "skill_builder_data.csv")  # 原始数据路径
    output_clean = os.path.join(data_dir, "assist9_cleaned.csv")  # 清理后数据保存路径
    output_map = os.path.join(data_dir, "skill_map.csv")  # 技能映射保存路径

    try:
        # 执行数据清理和知识图谱构建
        clean_assist9_data(input_csv, output_clean, output_map)
        build_kg_after_clean(data_dir)
    except FileNotFoundError:
        # 处理文件不存在的情况
        print("\n[Error] Raw data file not found.")
        print(f"[Hint] Please place the raw file at: {input_csv}")
