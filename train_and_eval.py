import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, mean_squared_error
import json
import os

# 导入适配后的模型
from models.dkt import DKTModel
from models.sakt import SAKTModel
from models.kg_sakt import KGSAKTModel
from models.pure_cf import PureCFModel

# ==========================================
# 1. 环境与路径配置
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CLEAN_DATA_PATH = os.path.join(DATA_DIR, "assist9_cleaned.csv")
KG_JSON_PATH = os.path.join(DATA_DIR, "kg_adj_list.json")

BATCH_SIZE = 64
MAX_SEQ = 100
EPOCHS = 20
LOGIC_LAMBDA = 0.2  # 逻辑一致性损失权重


# ==========================================
# 2. 数据集类
# ==========================================
class AssistDataset(Dataset):
    def __init__(self, df, n_skills, max_seq=100):
        self.n_skills = n_skills
        self.max_seq = max_seq
        groups = df.groupby('user_id')
        self.user_data = []
        for uid, group in groups:
            s = group['skill_id'].values.astype(int)
            c = group['correct'].values.astype(int)
            if len(s) < 2: continue
            self.user_data.append((uid, s, c))

    def __len__(self):
        return len(self.user_data)

    def __getitem__(self, index):
        uid, skills, corrects = self.user_data[index]
        if len(skills) > self.max_seq + 1:
            skills, corrects = skills[-(self.max_seq + 1):], corrects[-(self.max_seq + 1):]

        x = skills[:-1] + (corrects[:-1] * self.n_skills)
        q, y = skills[1:], corrects[1:].astype(float)

        pad_len = self.max_seq - len(q)
        x = np.pad(x, (pad_len, 0), 'constant', constant_values=0)
        q = np.pad(q, (pad_len, 0), 'constant', constant_values=0)
        y = np.pad(y, (pad_len, 0), 'constant', constant_values=-1)

        return torch.LongTensor(x), torch.LongTensor(q), torch.FloatTensor(y), torch.LongTensor([uid])


# ==========================================
# 3. 增强型评测逻辑 (Path Compliance 核心逻辑)
# ==========================================
def evaluate_metrics(name, model, kg_matrix, loader, n_skills):
    model.eval()
    y_true, y_pred = [], []
    compliance_hits, total_checks = 0, 0
    MASTERY_THRESHOLD = 0.45

    with torch.no_grad():
        for x, q, y, uids in loader:
            x, q, uids = x.to(DEVICE), q.to(DEVICE), uids.to(DEVICE).squeeze(-1)

            # 模型推理
            if name == "Pure-CF":
                pred_prob = model(uids, q[:, -1])
                full_dist = None
            elif name == "KG-SAKT":
                out = model(q, x, kg_matrix=kg_matrix)
                full_dist = torch.sigmoid(out[:, -1, :])
                pred_prob = full_dist.gather(1, q[:, -1].unsqueeze(1)).squeeze()
            else:
                out = model(q, x)
                if out.dim() == 3:
                    full_dist = torch.sigmoid(out[:, -1, :])
                    pred_prob = full_dist.gather(1, q[:, -1].unsqueeze(1)).squeeze()
                elif out.dim() == 2:
                    pred_prob = torch.sigmoid(out[:, -1])
                    full_dist = None
                else:
                    pred_prob = torch.sigmoid(out)
                    full_dist = None

            y_pred.extend(pred_prob.cpu().tolist())
            y_true.extend(y[:, -1].cpu().tolist())

            # Path Compliance 校验
            if full_dist is not None:
                rec_skills = torch.argmax(full_dist, dim=-1)
                pre_mask = kg_matrix[rec_skills]
                mastery_mask = (full_dist >= MASTERY_THRESHOLD)
                # 违规：有前置要求(1) 且 没掌握(~mastery_mask)
                violations = (pre_mask == 1) & (~mastery_mask)
                batch_hits = (violations.sum(dim=1) == 0).sum().item()
                compliance_hits += batch_hits
                total_checks += x.size(0)

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = (y_true != -1)
    auc = roc_auc_score(y_true[mask], y_pred[mask]) if len(np.unique(y_true[mask])) > 1 else 0.5
    rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
    comp = (compliance_hits / max(1, total_checks)) * 100 if total_checks > 0 else "无法计算"

    return auc, rmse, comp


# ==========================================
# 4. 主训练程序
# ==========================================
def main():
    torch.cuda.empty_cache()
    df = pd.read_csv(CLEAN_DATA_PATH)
    n_skills = int(df['skill_id'].max()) + 1
    n_users = int(df['user_id'].max()) + 1

    # 构建 GPU 知识图谱矩阵 (n_skills + 1 兼容 Padding)
    with open(KG_JSON_PATH, 'r') as f:
        kg_adj = json.load(f)
    kg_matrix = torch.zeros((n_skills + 1, n_skills + 1), device=DEVICE)
    for s, pre_list in kg_adj.items():
        s_idx = int(s)
        if s_idx < n_skills + 1:
            for p in pre_list:
                p_idx = int(p)
                if p_idx < n_skills + 1:
                    kg_matrix[s_idx, p_idx] = 1.0

    uids = df['user_id'].unique()
    np.random.shuffle(uids)
    split = int(0.8 * len(uids))
    train_loader = DataLoader(AssistDataset(df[df['user_id'].isin(uids[:split])], n_skills), batch_size=BATCH_SIZE,
                              shuffle=True)
    test_loader = DataLoader(AssistDataset(df[df['user_id'].isin(uids[split:])], n_skills), batch_size=BATCH_SIZE)

    models_list = [
        ("Pure-CF", PureCFModel(n_users=n_users, n_skills=n_skills)),
        ("DKT", DKTModel(n_skills=n_skills)),
        ("SAKT", SAKTModel(n_skills=n_skills)),
        ("KG-SAKT", KGSAKTModel(n_skills=n_skills, kg_adj=kg_adj))
    ]

    final_results = []

    for name, model in models_list:
        print(f"\n🚀 模型训练启动: {name}")
        model = model.to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss() if name == "Pure-CF" else nn.BCEWithLogitsLoss()

        for epoch in range(1, EPOCHS + 1):
            model.train()
            losses = []
            for bx, bq, by, bu in train_loader:
                bx, bq, by, bu = bx.to(DEVICE), bq.to(DEVICE), by.to(DEVICE), bu.to(DEVICE).squeeze(-1)
                optimizer.zero_grad()

                # --- KG-SAKT 专用 Logic Loss 逻辑 ---
                if name == "KG-SAKT":
                    out = model(bq, bx, kg_matrix=kg_matrix)
                    full_logits = out[:, -1, :]
                    prob = full_logits.gather(1, bq[:, -1].unsqueeze(1)).squeeze()

                    base_loss = criterion(prob, by[:, -1])
                    # 计算逻辑违规代价
                    preds = torch.sigmoid(full_logits)
                    pre_reqs = kg_matrix[bq[:, -1]]  # 当前目标的前置要求
                    # 逻辑：(是前置点) 且 (预测掌握度低) -> 产生罚分
                    logic_penalty = (pre_reqs * (1 - preds)).sum(dim=1).mean()
                    loss = base_loss + LOGIC_LAMBDA * logic_penalty
                else:
                    # 通用模型逻辑
                    out = model(bq, bx) if name != "Pure-CF" else model(bu, bq[:, -1])
                    if name == "Pure-CF":
                        prob = out
                    else:
                        prob = out[:, -1, :].gather(1, bq[:, -1].unsqueeze(1)).squeeze() if out.dim() == 3 else out[
                            :, -1]
                    loss = criterion(prob, by[:, -1])

                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            if epoch % 10 == 0 or epoch == 1:
                auc, _, _ = evaluate_metrics(name, model, kg_matrix, test_loader, n_skills)
                print(f"  [Epoch {epoch:02d}] Loss: {np.mean(losses):.4f} | AUC: {auc:.4f}")

        print(f"🏁 {name} 评估中...")
        final_auc, final_rmse, final_comp = evaluate_metrics(name, model, kg_matrix, test_loader, n_skills)
        final_results.append((name, final_auc, final_rmse, final_comp))

    print("\n" + "📊 实验结果统计".center(75, "="))
    print(f"{'模型 (Models)':<15} | {'AUC↑':<10} | {'RMSE↓':<10} | {'Path Comp.%↑'}")
    print("-" * 80)
    for n, a, r, c in final_results:
        c_str = f"{c:<12.2f}" if isinstance(c, float) else f"{c:<12}"
        print(f"{n:<15} | {a:<10.4f} | {r:<10.4f} | {c_str}")
    print("=" * 80)


if __name__ == "__main__":
    main()