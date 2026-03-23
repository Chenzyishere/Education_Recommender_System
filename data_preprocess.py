import pandas as pd
import numpy as np
import joblib  # 用于保存处理好的中间结果，需要 pip install joblib


def prepare_sequences(file_path, max_seq=50):
    # 1. 加载数据
    df = pd.read_csv(file_path)

    # 2. 知识点 ID 连续化 (从 1 开始，0 留给 Padding)
    # 这一步非常重要，否则模型 Embedding 层会报错
    skills = df['skill_id'].unique()
    skill2idx = {skill: i + 1 for i, skill in enumerate(skills)}
    df['skill_idx'] = df['skill_id'].map(skill2idx)

    num_skills = len(skills)
    print(f"检测到知识点数量: {num_skills}")

    # 3. 按用户分组
    group = df.groupby('user_id')

    seqs = []
    for user_id, data in group:
        s = data['skill_idx'].values
        a = data['correct'].values

        # 4. 切割或填充序列
        # 如果长度超过 max_seq，取最后 max_seq 个
        # 如果长度不足，前面补 0
        if len(s) > max_seq:
            s = s[-max_seq:]
            a = a[-max_seq:]
        else:
            # 补 0 操作
            pad_len = max_seq - len(s)
            s = np.pad(s, (pad_len, 0), 'constant', constant_values=0)
            a = np.pad(a, (pad_len, 0), 'constant', constant_values=-1)  # 答案补 -1 区分 0 分

        seqs.append((s, a))

    # 保存映射关系，以后推荐时要用到
    joblib.dump(skill2idx, 'data/skill2idx.pkl')

    return seqs, num_skills


if __name__ == "__main__":
    # 测试一下
    sequences, n_skills = prepare_sequences('data/raw_data.csv')
    print(f"处理了 {len(sequences)} 个用户的序列")
    print(f"样例序列 (前5个知识点): {sequences[0][0][:5]}")