import torch
import torch.nn as nn


class DKTModel(nn.Module):
    """
    DKT baseline model.

    输入:
    - x: 交互序列编码, shape [batch, seq]
      编码方式通常是 skill_id + correct * n_skills，0 作为 padding。

    输出:
    - logits: shape [batch, seq, n_skills + 1]
      每个时间步对所有知识点的掌握预测（未过 sigmoid）。
    """

    def __init__(self, n_skills: int, hidden_dim: int = 128, num_layers: int = 1):
        super().__init__()
        self.n_skills = n_skills
        # 交互词表大小: 2 * n_skills + 1，其中 0 留给 padding。
        self.input_dim = 2 * n_skills + 1

        # 1) 交互嵌入层
        self.embedding = nn.Embedding(self.input_dim, hidden_dim, padding_idx=0)
        # 2) 时序建模层
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        # 3) 输出层（映射到全部知识点）
        self.fc = nn.Linear(hidden_dim, n_skills + 1)

    def forward(self, q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # q 在 DKT 中不直接使用，保留仅用于统一训练接口。
        del q
        embedded = self.embedding(x)
        hidden, _ = self.lstm(embedded)
        return self.fc(hidden)
