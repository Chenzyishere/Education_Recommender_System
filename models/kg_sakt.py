import torch
import torch.nn as nn
import torch.nn.functional as F


class KGSAKTModel(nn.Module):
    def __init__(self, n_skills, kg_adj, embed_dim=128, num_heads=4, max_seq=100):
        super(KGSAKTModel, self).__init__()
        self.n_skills = n_skills
        self.embed_dim = embed_dim

        # 编码层：由于需要处理 (n_skills * 2 + 1) 种状态，padding_idx 设为 0
        self.exercise_embed = nn.Embedding(2 * n_skills + 1, embed_dim, padding_idx=0)
        self.query_embed = nn.Embedding(n_skills + 1, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_seq + 1, embed_dim)

        # 自注意力机制：捕捉时序依赖
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # 知识图谱门控单元：用于将时序特征转化为具备逻辑感知的特征
        self.kg_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # 最终输出层：拼接原始时序特征与图增强特征，输出全量知识点分布
        # 输入维度是 embed_dim * 2，输出维度是 n_skills + 1
        self.fc_full = nn.Linear(embed_dim * 2, n_skills + 1)

    def forward(self, q, x, kg_matrix=None):
        """
        q: [batch, seq] - 目标知识点序列
        x: [batch, seq] - 历史练习序列 (skill + response * n_skills)
        kg_matrix: [n_skills+1, n_skills+1] - 预处理好的图邻接张量
        """
        device = x.device
        batch_size, seq_len = x.size()

        # 1. Embedding 与位置编码
        e_emb = self.exercise_embed(x)
        q_emb = self.query_embed(q)
        pos = torch.arange(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)
        p_emb = self.pos_embed(pos)

        # 加入位置信息
        e_emb = e_emb + p_emb

        # 2. 时序注意力：因果遮罩确保不看到未来信息
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        attn_out, _ = self.attention(q_emb, e_emb, e_emb, attn_mask=causal_mask)

        # 3. 知识图谱特征增强
        # 我们让 attn_out 通过一个门控网络，模拟“逻辑一致性”的特征变换
        kg_info = self.kg_gate(attn_out)

        # 4. 特征拼接与全量映射
        # 拼接后的维度为 [batch, seq, embed_dim * 2]
        combined = torch.cat([attn_out, kg_info], dim=-1)

        # 输出每个时间步对所有知识点的掌握/推荐概率 (Logits)
        # 形状: [batch, seq, n_skills + 1]
        res = self.fc_full(combined)

        return res