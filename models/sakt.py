import torch
import torch.nn as nn


class SAKTModel(nn.Module):
    """
    Standard SAKT model.

    核心思想:
    - Query: 当前目标知识点序列 q
    - Key/Value: 历史交互序列 x
    - 通过因果注意力避免“看见未来”
    """

    def __init__(self, n_skills: int, embed_dim: int = 128, num_heads: int = 4, max_seq: int = 100):
        super().__init__()
        self.n_skills = n_skills
        self.embed_dim = embed_dim

        # 交互嵌入（知识点 + 作答结果）
        self.exercise_embed = nn.Embedding(2 * n_skills + 1, embed_dim, padding_idx=0)
        # 查询嵌入（仅知识点）
        self.query_embed = nn.Embedding(n_skills + 1, embed_dim, padding_idx=0)
        # 位置嵌入
        self.pos_embed = nn.Embedding(max_seq + 1, embed_dim)

        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # 每个时间步输出一个二分类 logit
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq], 历史交互
        # q: [batch, seq], 当前目标知识点
        device = x.device
        batch_size, seq_len = x.size()

        # 1) Embedding
        e_emb = self.exercise_embed(x)
        q_emb = self.query_embed(q)

        # 2) 位置编码注入历史交互
        pos = torch.arange(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)
        e_emb = e_emb + self.pos_embed(pos)

        # 3) 因果遮罩: 上三角为 True，表示被遮蔽（未来信息不可见）
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

        # 4) 注意力: Query=q_emb, Key/Value=e_emb
        attn_out, _ = self.attention(q_emb, e_emb, e_emb, attn_mask=causal_mask)

        # 5) 输出 shape [batch, seq]
        return self.fc(attn_out).squeeze(-1)
