import torch
import torch.nn as nn


class KGSAKTModel(nn.Module):
    def __init__(
        self,
        n_skills,
        kg_adj,
        embed_dim=128,
        num_heads=4,
        max_seq=100,
        dropout=0.1,
        num_time_buckets=8,
        use_time_feature=False,
    ):
        super().__init__()
        self.n_skills = n_skills
        self.embed_dim = embed_dim
        self.use_time_feature = use_time_feature

        self.exercise_embed = nn.Embedding(2 * n_skills + 1, embed_dim, padding_idx=0)
        self.query_embed = nn.Embedding(n_skills + 1, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_seq, embed_dim)
        self.graph_skill_embed = nn.Embedding(n_skills + 1, embed_dim, padding_idx=0)
        self.time_embed = nn.Embedding(num_time_buckets + 1, embed_dim, padding_idx=0)

        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.graph_norm = nn.LayerNorm(embed_dim)
        self.output_norm = nn.LayerNorm(embed_dim)

        self.kg_gate = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.fc_full = nn.Linear(embed_dim, n_skills + 1)
        self.graph_mix_logit = nn.Parameter(torch.tensor(0.0))

        kg_matrix = torch.zeros(n_skills + 1, n_skills + 1, dtype=torch.float32)
        for skill, prereqs in kg_adj.items():
            skill_idx = int(skill)
            if not 0 <= skill_idx <= n_skills:
                continue
            for prereq in prereqs:
                prereq_idx = int(prereq)
                if 0 <= prereq_idx <= n_skills:
                    kg_matrix[skill_idx, prereq_idx] = 1.0

        degree = kg_matrix.sum(dim=1, keepdim=True).clamp_min(1.0)
        self.register_buffer("kg_matrix", kg_matrix)
        self.register_buffer("kg_row_norm", kg_matrix / degree)
        self.register_buffer("kg_two_hop", torch.matmul(self.kg_row_norm, self.kg_row_norm))

    def forward(self, q, x, kg_matrix=None, time_bucket=None):
        del kg_matrix
        device = x.device
        _, seq_len = x.size()

        e_emb = self.exercise_embed(x)
        q_emb = self.query_embed(q)
        if time_bucket is None or (not self.use_time_feature):
            time_bucket = torch.zeros_like(q)
        time_emb = self.time_embed(time_bucket)
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        e_emb = e_emb + self.pos_embed(pos_ids)
        q_emb = q_emb + time_emb

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )
        key_padding_mask = x.eq(0)

        attn_out, _ = self.attention(
            q_emb,
            e_emb,
            e_emb,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        attn_out = self.attn_norm(q_emb + self.dropout(attn_out))

        one_hop = torch.matmul(self.kg_row_norm, self.graph_skill_embed.weight)
        two_hop = torch.matmul(self.kg_two_hop, self.graph_skill_embed.weight)
        mix = torch.sigmoid(self.graph_mix_logit)
        graph_table = mix * one_hop + (1.0 - mix) * two_hop
        kg_context = graph_table[q]
        kg_context = self.graph_norm(kg_context)

        gate_input = torch.cat([attn_out, q_emb, kg_context, time_emb], dim=-1)
        gate = self.kg_gate(gate_input)
        fused = gate * attn_out + (1.0 - gate) * kg_context
        fused = fused + self.dropout(self.ffn(fused))
        fused = self.output_norm(fused)

        return self.fc_full(fused)
