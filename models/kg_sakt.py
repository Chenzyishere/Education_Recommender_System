import torch
import torch.nn as nn


class KGSAKTModel(nn.Module):
    """
    知识图谱增强的SAKT模型

    结构概览:
    1) SAKT 时序注意力分支（建模历史交互）
    2) KG 图扩散分支（1-hop + 2-hop）
    3) 门控融合（动态平衡时序特征与图特征）
    4) 输出所有知识点 logits（用于训练内逻辑约束）
    """

    def __init__(
        self,
        n_skills: int,  # 技能总数
        kg_adj: dict,  # 知识图谱邻接表
        embed_dim: int = 128,  # 嵌入维度
        num_heads: int = 4,  # 注意力头数
        max_seq: int = 100,  # 最大序列长度
        dropout: float = 0.1,  #  dropout率
        num_time_buckets: int = 8,  # 时间桶数量
        use_time_feature: bool = False,  # 是否使用时间特征
    ):
        super().__init__()
        self.n_skills = n_skills
        self.embed_dim = embed_dim
        self.use_time_feature = use_time_feature

        # ===== 嵌入模块 =====
        # 练习嵌入：2 * n_skills + 1 是因为每个技能有正确和错误两种状态
        self.exercise_embed = nn.Embedding(2 * n_skills + 1, embed_dim, padding_idx=0)
        # 查询嵌入：用于查询技能的表示
        self.query_embed = nn.Embedding(n_skills + 1, embed_dim, padding_idx=0)
        # 位置嵌入：编码序列位置信息
        self.pos_embed = nn.Embedding(max_seq, embed_dim)
        # 图技能嵌入：用于知识图谱扩散
        self.graph_skill_embed = nn.Embedding(n_skills + 1, embed_dim, padding_idx=0)
        # 时间嵌入：编码时间间隔信息
        self.time_embed = nn.Embedding(num_time_buckets + 1, embed_dim, padding_idx=0)

        # ===== SAKT 注意力分支 =====
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,  # 嵌入维度
            num_heads=num_heads,  # 注意力头数
            dropout=dropout,  # dropout率
            batch_first=True,  # 批处理维度在前
        )
        self.dropout = nn.Dropout(dropout)  # dropout层
        self.attn_norm = nn.LayerNorm(embed_dim)  # 注意力输出的层归一化
        self.graph_norm = nn.LayerNorm(embed_dim)  # 图特征的层归一化
        self.output_norm = nn.LayerNorm(embed_dim)  # 最终输出的层归一化

        # ===== 融合模块 =====
        # 知识图谱门控：动态平衡时序特征与图特征
        self.kg_gate = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),  # 输入是四个嵌入的拼接
            nn.ReLU(),  # 激活函数
            nn.Linear(embed_dim, embed_dim),  # 降维到嵌入维度
            nn.Sigmoid(),  # 输出门控值（0-1）
        )
        # 前馈网络：处理融合后的特征
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),  # 扩展维度
            nn.ReLU(),  # 激活函数
            nn.Dropout(dropout),  # dropout
            nn.Linear(embed_dim * 2, embed_dim),  # 降回原维度
        )
        # 全连接层：输出所有技能的logits
        self.fc_full = nn.Linear(embed_dim, n_skills + 1)

        # 可学习的混合比例：控制1-hop和2-hop图扩散的权重
        self.graph_mix_logit = nn.Parameter(torch.tensor(0.0))

        # ===== 从邻接表构建密集知识图谱矩阵 =====
        # 初始化知识图谱矩阵
        kg_matrix = torch.zeros(n_skills + 1, n_skills + 1, dtype=torch.float32)
        # 填充矩阵：target_skill -> prereqs
        for target_skill, prereqs in kg_adj.items():
            target_idx = int(target_skill)
            if not 0 <= target_idx <= n_skills:
                continue
            for prereq in prereqs:
                prereq_idx = int(prereq)
                if 0 <= prereq_idx <= n_skills:
                    kg_matrix[target_idx, prereq_idx] = 1.0

        # 行归一化邻接矩阵，用于扩散
        degree = kg_matrix.sum(dim=1, keepdim=True).clamp_min(1.0)  # 计算度数
        kg_row_norm = kg_matrix / degree  # 行归一化
        kg_two_hop = torch.matmul(kg_row_norm, kg_row_norm)  # 计算2-hop邻接矩阵

        # 注册为缓冲区：属于状态字典但不是可训练参数
        self.register_buffer("kg_matrix", kg_matrix)
        self.register_buffer("kg_row_norm", kg_row_norm)
        self.register_buffer("kg_two_hop", kg_two_hop)

    def forward(
        self,
        q: torch.Tensor,  # 查询序列 [batch_size, seq_len]
        x: torch.Tensor,  # 输入交互序列 [batch_size, seq_len]
        kg_matrix: torch.Tensor = None,  # 兼容训练管道的签名
        time_bucket: torch.Tensor = None,  # 时间桶序列 [batch_size, seq_len]
    ) -> torch.Tensor:  # 返回所有技能的logits [batch_size, seq_len, n_skills+1]
        # 保持与训练管道的签名兼容
        del kg_matrix

        device = x.device
        _, seq_len = x.size()

        # ===== 准备输入表示 =====
        e_emb = self.exercise_embed(x)  # 历史交互嵌入 [batch_size, seq_len, embed_dim]
        q_emb = self.query_embed(q)  # 查询技能嵌入 [batch_size, seq_len, embed_dim]

        # 处理时间特征
        if time_bucket is None or (not self.use_time_feature):
            time_bucket = torch.zeros_like(q)  # 无时间特征时使用全零
        time_emb = self.time_embed(time_bucket)  # 时间嵌入 [batch_size, seq_len, embed_dim]

        # 添加位置编码和时间编码
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]
        e_emb = e_emb + self.pos_embed(pos_ids)  # 添加位置编码
        q_emb = q_emb + time_emb  # 添加时间编码

        # ===== SAKT 分支（因果注意力） =====
        # 创建因果掩码：上三角矩阵，确保只能关注历史信息
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )
        # 创建填充掩码：标记填充位置
        key_padding_mask = x.eq(0)

        # 注意力计算
        attn_out, _ = self.attention(
            query=q_emb,  # 查询
            key=e_emb,  # 键
            value=e_emb,  # 值
            attn_mask=causal_mask,  # 因果掩码
            key_padding_mask=key_padding_mask,  # 填充掩码
            need_weights=False,  # 不需要注意力权重
        )
        # 注意力输出的残差连接和层归一化
        attn_out = self.attn_norm(q_emb + self.dropout(attn_out))

        # ===== KG 分支（1-hop + 2-hop 扩散） =====
        # 1-hop 图扩散
        one_hop = torch.matmul(self.kg_row_norm, self.graph_skill_embed.weight)
        # 2-hop 图扩散
        two_hop = torch.matmul(self.kg_two_hop, self.graph_skill_embed.weight)
        # 计算混合权重
        mix = torch.sigmoid(self.graph_mix_logit)
        # 混合1-hop和2-hop特征
        graph_table = mix * one_hop + (1.0 - mix) * two_hop
        # 根据查询技能获取图上下文
        kg_context = self.graph_norm(graph_table[q])

        # ===== 门控融合 =====
        # 拼接注意力输出、查询嵌入、图上下文和时间嵌入
        gate_input = torch.cat([attn_out, q_emb, kg_context, time_emb], dim=-1)
        # 计算门控值
        gate = self.kg_gate(gate_input)
        # 门控融合：动态平衡时序特征和图特征
        fused = gate * attn_out + (1.0 - gate) * kg_context

        # 残差前馈网络
        fused = fused + self.dropout(self.ffn(fused))
        fused = self.output_norm(fused)

        # 输出所有技能的logits
        return self.fc_full(fused)