import torch
import torch.nn as nn


class PureCFModel(nn.Module):
    """
    基于矩阵分解的协同过滤基线模型
    
    该模型使用矩阵分解技术来学习用户和技能的潜在表示，
    通过计算用户向量和技能向量的点积来预测用户对技能的掌握概率。

    输入:
    - u_ids: 用户ID, 形状为 [batch]
    - s_ids: 知识点ID, 形状为 [batch]

    输出:
    - prob: 对应 user-skill 对的掌握概率, 形状为 [batch]
    """

    def __init__(self, n_users: int, n_skills: int, embed_dim: int = 64):
        """
        初始化模型参数
        
        参数:
            n_users: 用户总数
            n_skills: 技能总数
            embed_dim: 嵌入维度，默认为64
        """
        super().__init__()
        # 用户和知识点的潜在向量
        # +1 是因为ID可能从1开始，0通常作为填充值
        self.u_embed = nn.Embedding(n_users + 1, embed_dim)  # 用户嵌入
        self.s_embed = nn.Embedding(n_skills + 1, embed_dim)  # 技能嵌入

        # 偏置项：捕捉用户和技能的固有特性
        self.u_bias = nn.Embedding(n_users + 1, 1)  # 用户偏置
        self.s_bias = nn.Embedding(n_skills + 1, 1)  # 技能偏置
        self.global_bias = nn.Parameter(torch.zeros(1))  # 全局偏置

    def forward(self, u_ids: torch.Tensor, s_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算
        
        参数:
            u_ids: 用户ID张量
            s_ids: 技能ID张量
            
        返回:
            torch.Tensor: 用户对技能的掌握概率
        """
        # 获取用户和技能的嵌入向量
        u_vec = self.u_embed(u_ids)  # [batch, embed_dim]
        s_vec = self.s_embed(s_ids)  # [batch, embed_dim]

        # 计算用户向量和技能向量的点积，表示用户与技能的相关性
        dot = torch.sum(u_vec * s_vec, dim=1)  # [batch]
        
        # 计算最终的logits，包括点积和各种偏置
        logits = dot + self.u_bias(u_ids).squeeze() + self.s_bias(s_ids).squeeze() + self.global_bias
        
        # 通过sigmoid函数将logits转换为概率（0-1之间）
        return torch.sigmoid(logits)