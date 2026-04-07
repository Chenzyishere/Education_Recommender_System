import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# 导入必要的库
# numpy: 用于数值计算和数组操作
# pandas: 用于数据处理和分析
# torch: 用于深度学习
# DataLoader, Dataset: 用于构建和加载数据集


def bucketize_time_gaps(time_gaps: np.ndarray) -> np.ndarray:
    """
    将原始时间间隔转换为粗粒度的桶，用于嵌入查找。
    
    为什么使用桶：
    - 时间间隔可能高度倾斜。
    - 对数变换 + 桶化比直接使用原始值更稳定。
    
    参数:
        time_gaps: 原始时间间隔数组
        
    返回:
        np.ndarray: 桶化后的时间间隔ID数组
    """
    # 限制非常大的时间间隔（7天，以毫秒为单位）以减少异常值的影响
    clipped = np.clip(time_gaps, a_min=0.0, a_max=7 * 24 * 60 * 60 * 1000.0)
    # 对时间间隔进行对数变换，使分布更均匀
    log_gaps = np.log1p(clipped)

    # 7个内部边界 => 桶ID在[0..8]之间，其中0表示"无间隔/填充"
    boundaries = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0], dtype=np.float32)
    # 使用二分查找确定每个时间间隔所属的桶
    bucket_ids = np.searchsorted(boundaries, log_gaps, side="right") + 1
    # 将非正的时间间隔设置为桶0
    bucket_ids[clipped <= 0] = 0
    return bucket_ids


class Assist9Dataset(Dataset):
    """
    用于DKT/SAKT/KG-SAKT训练的序列数据集。
    
    对于每个用户：
    - 输入x使用t-1之前的交互
    - 查询q和目标y与步骤t对齐
    """

    def __init__(self, df: pd.DataFrame, n_skills: int, max_seq: int = 100):
        """
        初始化数据集
        
        参数:
            df: 包含用户交互数据的DataFrame
            n_skills: 技能总数
            max_seq: 最大序列长度，默认为100
        """
        self.max_seq = max_seq  # 最大序列长度
        self.n_skills = n_skills  # 技能总数
        self.user_data = []  # 存储用户数据的列表
        self.has_time_gap = "time_gap" in df.columns  # 检查是否有时间间隔列

        # 按用户分组构建每个用户的序列
        for uid, group in df.groupby("user_id"):
            skills = group["skill_id"].to_numpy(dtype=np.int64)  # 技能ID数组
            corrects = group["correct"].to_numpy(dtype=np.int64)  # 正确与否数组
            if self.has_time_gap:
                # 处理时间间隔，填充缺失值为0
                time_gaps = group["time_gap"].fillna(0.0).to_numpy(dtype=np.float32)
            else:
                # 无时间间隔时，创建全0数组
                time_gaps = np.zeros(len(group), dtype=np.float32)

            # 需要至少2个交互才能形成(历史->目标)对
            if len(skills) < 2:
                continue
            # 添加用户数据到列表
            self.user_data.append((int(uid), skills, corrects, time_gaps))

    def __len__(self) -> int:
        """
        返回数据集的长度（用户数量）
        
        返回:
            int: 数据集的长度
        """
        return len(self.user_data)

    def __getitem__(self, index: int) -> dict:
        """
        获取指定索引的样本
        
        参数:
            index: 样本索引
            
        返回:
            dict: 包含用户ID、输入、查询、目标和时间桶的字典
        """
        uid, skills, corrects, time_gaps = self.user_data[index]

        # 只保留最近的max_seq+1个点：
        # max_seq用于模型输入，+1用于下一步目标对齐
        if len(skills) > self.max_seq + 1:
            skills = skills[-(self.max_seq + 1) :]
            corrects = corrects[-(self.max_seq + 1) :]
            time_gaps = time_gaps[-(self.max_seq + 1) :]

        # 构建下一步预测元组
        # 交互 = 技能ID + (是否正确 * 技能总数)，用于区分同一技能的正确和错误情况
        interactions = skills[:-1] + (corrects[:-1] * self.n_skills)
        # 查询是下一步的技能ID
        queries = skills[1:]
        # 目标是下一步的正确与否
        targets = corrects[1:].astype(np.float32)
        # 对时间间隔进行桶化
        time_buckets = bucketize_time_gaps(time_gaps[1:])

        # 左填充到固定长度以进行批量训练
        pad_len = self.max_seq - len(queries)
        interactions = np.pad(interactions, (pad_len, 0), constant_values=0)
        queries = np.pad(queries, (pad_len, 0), constant_values=0)
        # target=-1表示填充位置，稍后在损失/指标计算中被屏蔽
        targets = np.pad(targets, (pad_len, 0), constant_values=-1.0)
        time_buckets = np.pad(time_buckets, (pad_len, 0), constant_values=0)

        return {
            "user_id": torch.tensor(uid, dtype=torch.long),  # 用户ID
            "x": torch.tensor(interactions, dtype=torch.long),  # 输入交互序列
            "q": torch.tensor(queries, dtype=torch.long),  # 查询序列
            "target": torch.tensor(targets, dtype=torch.float32),  # 目标序列
            "time_bucket": torch.tensor(time_buckets, dtype=torch.long),  # 时间桶序列
        }


def get_assist9_loader(
    file_path: str,
    n_skills: int,
    batch_size: int = 64,
    max_seq: int = 100,
    shuffle: bool = True,
) -> DataLoader:
    """
    从清理后的CSV创建torch DataLoader的便捷包装器。
    
    参数:
        file_path: 清理后的数据文件路径
        n_skills: 技能总数
        batch_size: 批次大小，默认为64
        max_seq: 最大序列长度，默认为100
        shuffle: 是否打乱数据，默认为True
        
    返回:
        DataLoader: 用于训练的DataLoader
    """
    # 读取清理后的数据
    df = pd.read_csv(file_path)
    # 创建数据集
    dataset = Assist9Dataset(df, n_skills=n_skills, max_seq=max_seq)
    # 创建并返回DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)