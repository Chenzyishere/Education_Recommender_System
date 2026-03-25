import os

import pandas as pd


def build_time_gap(df):
    timestamp_candidates = ["event_time", "timestamp", "start_time", "time"]
    timestamp_col = next((col for col in timestamp_candidates if col in df.columns), None)
    if timestamp_col is None:
        return None

    parsed_time = pd.to_datetime(df[timestamp_col], errors="coerce")
    if parsed_time.isna().all():
        return None

    return (
        parsed_time.groupby(df["user_id"])
        .diff()
        .dt.total_seconds()
        .fillna(0.0)
        .clip(lower=0.0)
    )


def clean_assist9_data(raw_path, save_path, map_save_path):
    print(f"Reading raw file: {raw_path}")
    df = pd.read_csv(raw_path, encoding="ISO-8859-1", low_memory=False)

    df.columns = [c.strip() for c in df.columns]
    core_cols = ["user_id", "skill_id", "correct", "order_id"]
    optional_cols = [
        col
        for col in ["event_time", "timestamp", "start_time", "time", "ms_first_response"]
        if col in df.columns
    ]
    df = df[core_cols + optional_cols].dropna(subset=["skill_id"])

    df = df.drop_duplicates()
    df = df.sort_values(by=["user_id", "order_id"])

    time_gap = build_time_gap(df)
    if time_gap is not None:
        df["time_gap"] = time_gap

    unique_skills = sorted(df["skill_id"].unique())
    skill_map = {old: i + 1 for i, old in enumerate(unique_skills)}
    df["skill_id"] = df["skill_id"].map(skill_map)

    skill_map_df = pd.DataFrame(list(skill_map.items()), columns=["old_id", "new_id"])
    skill_map_df.to_csv(map_save_path, index=False)
    print(f"Saved skill map to: {map_save_path}")

    save_cols = ["user_id", "skill_id", "correct"]
    if "time_gap" in df.columns:
        save_cols.append("time_gap")
    df[save_cols].to_csv(save_path, index=False)

    n_skills = len(unique_skills)
    print(f"Cleaning finished. Skills: {n_skills}, Rows: {len(df)}")
    print(f"Saved cleaned data to: {save_path}")
    if "time_gap" not in df.columns:
        print("No parseable timestamp column was found, so time_gap was not generated.")
    return n_skills


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created missing directory: {data_dir}")

    input_csv = os.path.join(data_dir, "skill_builder_data.csv")
    output_clean = os.path.join(data_dir, "assist9_cleaned.csv")
    output_map = os.path.join(data_dir, "skill_map.csv")

    try:
        clean_assist9_data(input_csv, output_clean, output_map)
    except FileNotFoundError:
        print("\n[Error]: Raw data file not found.")
        print(f"Please place the raw file at: {input_csv}")
