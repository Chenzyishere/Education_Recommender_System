import torch
import torch.nn as nn


class SAKTModel(nn.Module):
    def __init__(self, n_skills, embed_dim=128, num_heads=4, max_seq=100):
        super(SAKTModel, self).__init__()
        self.n_skills = n_skills
        self.embed_dim = embed_dim

        # 编码练习 (知识点 + 结果)
        self.exercise_embed = nn.Embedding(2 * n_skills + 1, embed_dim, padding_idx=0)
        # 编码查询 (知识点本身)
        self.query_embed = nn.Embedding(n_skills + 1, embed_dim, padding_idx=0)
        # 位置编码
        self.pos_embed = nn.Embedding(max_seq + 1, embed_dim)

        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.fc = nn.Linear(embed_dim, 1)  # 输出针对当前问题的预测值

    def forward(self, q, x):
        # x: [batch, seq] 历史练习
        # q: [batch, seq] 目标知识点
        device = x.device
        batch_size, seq_len = x.size()

        # 准备 Embedding
        e_emb = self.exercise_embed(x)
        q_emb = self.query_embed(q)
        pos = torch.arange(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)
        p_emb = self.pos_embed(pos)

        # 加入位置信息
        e_emb += p_emb

        # 因果遮罩 (Causal Mask)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

        # 注意力计算: Query 是目标知识点，Key/Value 是历史练习
        attn_out, _ = self.attention(q_emb, e_emb, e_emb, attn_mask=mask)

        # 映射到 0-1 的原始分数 (Logits)
        # 输出维度: [batch, seq]
        res = self.fc(attn_out).squeeze(-1)
        return res