import torch
import torch.nn as nn


class KGSAKTModel(nn.Module):
    """
    KG-enhanced SAKT model.

    结构概览:
    1) SAKT 时序注意力分支（建模历史交互）
    2) KG 图扩散分支（1-hop + 2-hop）
    3) 门控融合（动态平衡时序特征与图特征）
    4) 输出所有知识点 logits（用于训练内逻辑约束）
    """

    def __init__(
        self,
        n_skills: int,
        kg_adj: dict,
        embed_dim: int = 128,
        num_heads: int = 4,
        max_seq: int = 100,
        dropout: float = 0.1,
        num_time_buckets: int = 8,
        use_time_feature: bool = False,
    ):
        super().__init__()
        self.n_skills = n_skills
        self.embed_dim = embed_dim
        self.use_time_feature = use_time_feature

        # ===== Embedding blocks =====
        self.exercise_embed = nn.Embedding(2 * n_skills + 1, embed_dim, padding_idx=0)
        self.query_embed = nn.Embedding(n_skills + 1, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_seq, embed_dim)
        self.graph_skill_embed = nn.Embedding(n_skills + 1, embed_dim, padding_idx=0)
        self.time_embed = nn.Embedding(num_time_buckets + 1, embed_dim, padding_idx=0)

        # ===== SAKT attention branch =====
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.graph_norm = nn.LayerNorm(embed_dim)
        self.output_norm = nn.LayerNorm(embed_dim)

        # ===== Fusion blocks =====
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

        # Learnable mixing ratio for one-hop vs two-hop graph diffusion.
        self.graph_mix_logit = nn.Parameter(torch.tensor(0.0))

        # ===== Build dense KG matrix from adjacency list =====
        kg_matrix = torch.zeros(n_skills + 1, n_skills + 1, dtype=torch.float32)
        for target_skill, prereqs in kg_adj.items():
            target_idx = int(target_skill)
            if not 0 <= target_idx <= n_skills:
                continue
            for prereq in prereqs:
                prereq_idx = int(prereq)
                if 0 <= prereq_idx <= n_skills:
                    kg_matrix[target_idx, prereq_idx] = 1.0

        # Row-normalized adjacency used for diffusion.
        degree = kg_matrix.sum(dim=1, keepdim=True).clamp_min(1.0)
        kg_row_norm = kg_matrix / degree
        kg_two_hop = torch.matmul(kg_row_norm, kg_row_norm)

        # Buffers are part of state_dict but not trainable parameters.
        self.register_buffer("kg_matrix", kg_matrix)
        self.register_buffer("kg_row_norm", kg_row_norm)
        self.register_buffer("kg_two_hop", kg_two_hop)

    def forward(
        self,
        q: torch.Tensor,
        x: torch.Tensor,
        kg_matrix: torch.Tensor = None,
        time_bucket: torch.Tensor = None,
    ) -> torch.Tensor:
        # Keep signature compatible with training pipeline.
        del kg_matrix

        device = x.device
        _, seq_len = x.size()

        # ===== Prepare input representations =====
        e_emb = self.exercise_embed(x)  # history interaction embedding
        q_emb = self.query_embed(q)  # query skill embedding

        if time_bucket is None or (not self.use_time_feature):
            time_bucket = torch.zeros_like(q)
        time_emb = self.time_embed(time_bucket)

        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        e_emb = e_emb + self.pos_embed(pos_ids)
        q_emb = q_emb + time_emb

        # ===== SAKT branch (causal attention) =====
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )
        key_padding_mask = x.eq(0)

        attn_out, _ = self.attention(
            query=q_emb,
            key=e_emb,
            value=e_emb,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        attn_out = self.attn_norm(q_emb + self.dropout(attn_out))

        # ===== KG branch (one-hop + two-hop diffusion) =====
        one_hop = torch.matmul(self.kg_row_norm, self.graph_skill_embed.weight)
        two_hop = torch.matmul(self.kg_two_hop, self.graph_skill_embed.weight)
        mix = torch.sigmoid(self.graph_mix_logit)
        graph_table = mix * one_hop + (1.0 - mix) * two_hop
        kg_context = self.graph_norm(graph_table[q])

        # ===== Gate fusion =====
        gate_input = torch.cat([attn_out, q_emb, kg_context, time_emb], dim=-1)
        gate = self.kg_gate(gate_input)
        fused = gate * attn_out + (1.0 - gate) * kg_context

        # Residual FFN
        fused = fused + self.dropout(self.ffn(fused))
        fused = self.output_norm(fused)

        # Full logits for all skills at each timestep.
        return self.fc_full(fused)
