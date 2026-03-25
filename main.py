import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_score
import json
import numpy as np

# 导入你之前创建的模块
# from models.sakt import SAKTModel
# from preprocess.dataset_loader import get_assist9_loader

# --- 1. 超参数配置 (对应论文 4.3.2) ---
config = {
    'batch_size': 64,
    'n_skills': 123,  # 根据 clean_data.py 输出的实际数量修改
    'max_seq': 100,
    'embed_dim': 128,
    'num_heads': 4,
    'lr': 1e-3,
    'epochs': 20,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'kg_threshold': 0.7  # 论文中提到的前置知识掌握阈值 theta
}


# --- 2. 逻辑约束推荐函数 (对应论文 3.3.3) ---
def kg_constrained_recommend(model, user_input, kg_adj, candidates):
    """
    model: 训练好的 SAKT 模型
    user_input: 用户历史交互序列 (x)
    kg_adj: 加载的 json 邻接表
    candidates: 待推荐的知识点 ID 列表
    """
    model.eval()
    with torch.no_grad():
        # 获取模型预测的全量掌握概率 P_master
        # 假设 model 输出最后一个时间步的预测向量 [1, n_skills]
        output_probs = model(user_input)
        p_master = output_probs[0].cpu().numpy()  # 转化为数组

    final_recommendation = []
    path_violations = 0

    for kp in candidates:
        prereqs = kg_adj.get(str(kp), [])  # 从 JSON 读取前置

        # 核心逻辑校验：校验前置知识掌握概率是否均大于阈值
        is_valid = True
        for pre in prereqs:
            if p_master[pre - 1] < config['kg_threshold']:  # pre-1 是因为数组索引从0开始
                is_valid = False
                path_violations += 1
                break

        if is_valid:
            final_recommendation.append((kp, p_master[kp - 1]))

    # 按预测掌握度排序（最近发展区策略：推难度适中且逻辑通顺的）
    final_recommendation.sort(key=lambda x: x[1], reverse=True)
    return final_recommendation[:5], path_violations


# --- 3. 训练与评估主循环 ---
def train():
    print(f"正在使用设备: {config['device']}")

    # 加载数据 (需先运行 clean_data.py)
    train_loader = get_assist9_loader("data/assist9_cleaned.csv", config['n_skills'], config['batch_size'])

    # 加载知识图谱 (需先运行 kg_builder.py)
    with open("data/kg_adj_list.json", 'r') as f:
        kg_adj = json.load(f)

    # 初始化模型 (这里以 SAKT 为例)
    # model = SAKTModel(config['n_skills'], config['embed_dim'], config['num_heads'], config['max_seq']).to(config['device'])
    # optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    # criterion = nn.BCELoss()

    # --- 模拟训练循环 ---
    for epoch in range(config['epochs']):
        # model.train()
        total_loss = 0
        all_targets = []
        all_preds = []

        for batch in train_loader:
            q = batch['q'].to(config['device'])
            x = batch['x'].to(config['device'])
            target = batch['target'].to(config['device'])

            # 1. 前向传播
            # preds = model(q, x) # 形状: [batch, seq_len]

            # 2. 掩码处理（只计算非填充部分的 Loss）
            # mask = target != -1
            # loss = criterion(preds[mask], target[mask])

            # 3. 反向传播
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # total_loss += loss.item()
            # all_targets.extend(target[mask].cpu().detach().numpy())
            # all_preds.extend(preds[mask].cpu().detach().numpy())
            pass

        # 计算本轮 AUC (对应论文 4.4.1)
        # epoch_auc = roc_auc_score(all_targets, all_preds)
        # print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, AUC: {epoch_auc:.4f}")

    # --- 4. 推荐实验与对比 (对应论文 4.4.2) ---
    print("\n--- 开始推荐效果评估 ---")
    # 模拟一个用户进行推荐测试
    test_candidates = [5, 12, 18, 25, 30]  # 假设的候选集
    # recs, violations = kg_constrained_recommend(model, some_user_input, kg_adj, test_candidates)
    # print(f"KG-SAKT 推荐列表: {recs}")
    # print(f"路径违例数: {violations}")


if __name__ == "__main__":
    train()