import pandas as pd
import os


def clean_assist9_data(raw_path, save_path, map_save_path):
    # 1. 读取原始数据 (注意编码格式)
    print(f"正在读取原始文件: {raw_path}")
    df = pd.read_csv(raw_path, encoding='ISO-8859-1', low_memory=False)

    # 2. 筛选核心列并删除 skill_id 为空的行
    cols = ['user_id', 'skill_id', 'correct', 'order_id']
    # 检查原始数据中是否存在这些列名（ASSISTments 2009 有时列名带空格）
    df.columns = [c.strip() for c in df.columns]
    df = df[cols].dropna(subset=['skill_id'])

    # 3. 数据预处理：去重与排序
    df = df.drop_duplicates()
    df = df.sort_values(by=['user_id', 'order_id'])

    # 4. 重新映射连续的 Skill ID (1 到 N)
    unique_skills = sorted(df['skill_id'].unique())
    skill_map = {old: i + 1 for i, old in enumerate(unique_skills)}
    df['skill_id'] = df['skill_id'].map(skill_map)

    # 5. 保存映射关系
    skill_map_df = pd.DataFrame(list(skill_map.items()), columns=['old_id', 'new_id'])
    skill_map_df.to_csv(map_save_path, index=False)
    print(f"已保存知识点映射表至: {map_save_path}")

    # 6. 保存清洗后的数据
    df[['user_id', 'skill_id', 'correct']].to_csv(save_path, index=False)

    n_skills = len(unique_skills)
    print(f"清洗完成！总知识点数: {n_skills}, 清洗后数据条数: {len(df)}")
    print(f"已保存清洗后的数据至: {save_path}")
    return n_skills


if __name__ == "__main__":
    # --- 自动定位路径 ---
    # 获取当前脚本所在目录的上一级 (即项目根目录)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    # 检查 data 目录是否存在，不存在则创建
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"已自动创建缺失的目录: {DATA_DIR}")

    INPUT_CSV = os.path.join(DATA_DIR, "skill_builder_data.csv")
    OUTPUT_CLEAN = os.path.join(DATA_DIR, "assist9_cleaned.csv")
    OUTPUT_MAP = os.path.join(DATA_DIR, "skill_map.csv")

    try:
        clean_assist9_data(INPUT_CSV, OUTPUT_CLEAN, OUTPUT_MAP)
    except FileNotFoundError:
        print(f"\n[错误]: 找不到原始数据文件！")
        print(f"请确认你的原始文件存放在: {INPUT_CSV}")