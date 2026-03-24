import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import panda as pd

class Assist9Dataset(Dataset):
    def __init__(self, df, n_skills, max_seq=100):
        self.max_seq = max_seq
        self.n_skills = n_skills

        # 按用户聚合序列
        self.user_data = df.groupby('user_id').apply(
            lambda x: (x['skill_id'].values, x['correct'].values)
        ).values

    def __len__(self):
        return len(self.user_data)

    def __getitem__(self, index):
        skills, corrects = self.user_data[index]
        seq_len = len(skills)

        # 构造 SAKT 交互 ID:
        # 规则：skill_id + (correct * n_skills)
        # 这样正确和错误的同一个知识点会被赋予不同的 Embedding ID
        interactions = skills + (corrects * self.n_skills)

        # 截断或填充
        if seq_len > self.max_seq:
            # 取最后 max_seq + 1 个，用于划分输入和目标
            skills = skills[-(self.max_seq + 1):]
            interactions = interactions[-(self.max_seq + 1):]
            corrects = corrects[-(self.max_seq + 1):]
        else:
            pad_len = (self.max_seq + 1) - seq_len
            skills = np.pad(skills, (pad_len, 0), 'constant', constant_values=0)
            interactions = np.pad(interactions, (pad_len, 0), 'constant', constant_values=0)
            corrects = np.pad(corrects, (pad_len, 0), 'constant', constant_values=-1)

        # SAKT 输入输出划分：
        # q: 当前预测的目标知识点 (1 to end)
        # x: 用于 Transformer 的 Key/Value 的历史交互 (0 to end-1)
        # target: 预测目标 (1 to end)
        return {
            'q': torch.LongTensor(skills[1:]),
            'x': torch.LongTensor(interactions[:-1]),
            'target': torch.FloatTensor(corrects[1:])
        }


def get_assist9_loader(file_path, n_skills, batch_size=64):
    df = pd.read_csv(file_path)
    dataset = Assist9Dataset(df, n_skills)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)