import torch
import torch.nn as nn


class DKTModel(nn.Module):
    def __init__(self, n_skills, hidden_dim=128, num_layers=1):
        super(DKTModel, self).__init__()
        self.n_skills = n_skills
        # 输入编码: 2*n_skills + 1 (做对、做错、Padding)
        self.input_dim = 2 * n_skills + 1

        self.embedding = nn.Embedding(self.input_dim, hidden_dim, padding_idx=0)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_skills + 1)

    def forward(self, q, x):
        # x: [batch, seq]
        embedded = self.embedding(x)
        out, _ = self.lstm(embedded)

        # 映射到所有知识点
        # 输出维度: [batch, seq, n_skills + 1]
        res = self.fc(out)
        return res