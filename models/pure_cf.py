import torch
import torch.nn as nn


class PureCFModel(nn.Module):
    """
    基于矩阵分解的协同过滤模型 (Baseline)
    输入: user_id, skill_id
    输出: 预测该学生掌握该知识点的概率
    """

    def __init__(self, n_users, n_skills, embed_dim=64):
        super(PureCFModel, self).__init__()
        # 用户嵌入层：记录每个用户的潜在水平
        self.u_embed = nn.Embedding(n_users + 1, embed_dim)
        # 知识点嵌入层：记录每个知识点的潜在难度/属性
        self.s_embed = nn.Embedding(n_skills + 1, embed_dim)

        # 偏置项
        self.u_bias = nn.Embedding(n_users + 1, 1)
        self.s_bias = nn.Embedding(n_skills + 1, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self, u_ids, s_ids):
        # u_ids: [batch], s_ids: [batch]
        u = self.u_embed(u_ids)
        s = self.s_embed(s_ids)

        # 点积计算相似度/掌握度
        dot = torch.sum(u * s, dim=1)

        # 加上偏置项
        u_b = self.u_bias(u_ids).squeeze()
        s_b = self.s_bias(s_ids).squeeze()

        logits = dot + u_b + s_b + self.global_bias
        return torch.sigmoid(logits)