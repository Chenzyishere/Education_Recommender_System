import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def bucketize_time_gaps(time_gaps):
    clipped = np.clip(time_gaps, a_min=0.0, a_max=7 * 24 * 60 * 60 * 1000.0)
    log_gaps = np.log1p(clipped)
    boundaries = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0], dtype=np.float32)
    bucket_ids = np.searchsorted(boundaries, log_gaps, side="right") + 1
    bucket_ids[clipped <= 0] = 0
    return bucket_ids


class Assist9Dataset(Dataset):
    def __init__(self, df, n_skills, max_seq=100):
        self.max_seq = max_seq
        self.n_skills = n_skills
        self.user_data = []
        self.has_time_gap = "time_gap" in df.columns

        for uid, group in df.groupby("user_id"):
            skills = group["skill_id"].to_numpy(dtype=np.int64)
            corrects = group["correct"].to_numpy(dtype=np.int64)
            if self.has_time_gap:
                time_gaps = group["time_gap"].fillna(0.0).to_numpy(dtype=np.float32)
            else:
                time_gaps = np.zeros(len(group), dtype=np.float32)
            if len(skills) < 2:
                continue
            self.user_data.append((int(uid), skills, corrects, time_gaps))

    def __len__(self):
        return len(self.user_data)

    def __getitem__(self, index):
        uid, skills, corrects, time_gaps = self.user_data[index]
        if len(skills) > self.max_seq + 1:
            skills = skills[-(self.max_seq + 1) :]
            corrects = corrects[-(self.max_seq + 1) :]
            time_gaps = time_gaps[-(self.max_seq + 1) :]

        interactions = skills[:-1] + (corrects[:-1] * self.n_skills)
        queries = skills[1:]
        targets = corrects[1:].astype(np.float32)
        time_buckets = bucketize_time_gaps(time_gaps[1:])

        pad_len = self.max_seq - len(queries)
        interactions = np.pad(interactions, (pad_len, 0), constant_values=0)
        queries = np.pad(queries, (pad_len, 0), constant_values=0)
        targets = np.pad(targets, (pad_len, 0), constant_values=-1.0)
        time_buckets = np.pad(time_buckets, (pad_len, 0), constant_values=0)

        return {
            "user_id": torch.tensor(uid, dtype=torch.long),
            "x": torch.tensor(interactions, dtype=torch.long),
            "q": torch.tensor(queries, dtype=torch.long),
            "target": torch.tensor(targets, dtype=torch.float32),
            "time_bucket": torch.tensor(time_buckets, dtype=torch.long),
        }


def get_assist9_loader(file_path, n_skills, batch_size=64, max_seq=100, shuffle=True):
    df = pd.read_csv(file_path)
    dataset = Assist9Dataset(df, n_skills=n_skills, max_seq=max_seq)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
