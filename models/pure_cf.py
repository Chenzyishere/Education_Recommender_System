import torch
import torch.nn as nn


class PureCFModel(nn.Module):
    """
    Matrix-Factorization style collaborative filtering baseline.

    输入:
    - u_ids: 用户 id, shape [batch]
    - s_ids: 知识点 id, shape [batch]

    输出:
    - prob: 对应 user-skill 的掌握概率, shape [batch]
    """

    def __init__(self, n_users: int, n_skills: int, embed_dim: int = 64):
        super().__init__()
        # 用户/知识点潜向量
        self.u_embed = nn.Embedding(n_users + 1, embed_dim)
        self.s_embed = nn.Embedding(n_skills + 1, embed_dim)

        # 用户偏置、知识点偏置与全局偏置
        self.u_bias = nn.Embedding(n_users + 1, 1)
        self.s_bias = nn.Embedding(n_skills + 1, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self, u_ids: torch.Tensor, s_ids: torch.Tensor) -> torch.Tensor:
        u_vec = self.u_embed(u_ids)
        s_vec = self.s_embed(s_ids)

        # 点积表示 user-skill 相关性
        dot = torch.sum(u_vec * s_vec, dim=1)
        logits = dot + self.u_bias(u_ids).squeeze() + self.s_bias(s_ids).squeeze() + self.global_bias
        return torch.sigmoid(logits)
