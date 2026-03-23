# model.py (已对齐权重的修正版)
import torch
import torch.nn as nn


class SAKTModel(nn.Module):
    def __init__(self, n_skills, max_seq=50, embed_dim=128, num_heads=8, dropout=0.2):
        super(SAKTModel, self).__init__()
        self.n_skills = n_skills
        self.embed_dim = embed_dim

        # 修改这里的变量名，使之与你保存的 .pth 文件匹配
        self.skill_embed = nn.Embedding(n_skills + 1, embed_dim)
        self.pos_embed = nn.Embedding(max_seq, embed_dim)

        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

        # 对应 Unexpected key(s) 中的 fc
        self.fc = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        device = x.device
        batch_size, seq_len = x.shape

        # 对应变量名修改
        e = self.skill_embed(x)
        pos_id = torch.arange(seq_len).unsqueeze(0).to(device)
        p = self.pos_embed(pos_id)

        x_emb = e + p
        x_emb = x_emb.permute(1, 0, 2)  # [seq_len, batch, dim]

        # 对应变量名修改
        attn_output, _ = self.attention(x_emb, x_emb, x_emb)
        attn_output = self.layer_norm(attn_output + x_emb)

        # 使用对应的 fc 层
        out = self.fc(attn_output)
        return self.sigmoid(out).permute(1, 0, 2)  # [batch, seq_len, 1]