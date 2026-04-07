import json
import os
import random
import sys

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, roc_auc_score
from torch.utils.data import DataLoader

# Ensure project root is importable when running: python utils/train_and_eval.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.dkt import DKTModel
from models.kg_sakt import KGSAKTModel
from models.pure_cf import PureCFModel
from models.sakt import SAKTModel
from preprocess.dataset_loader import Assist9Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = PROJECT_ROOT
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RENDERED_DIR = os.path.join(PROJECT_ROOT, "rendered")
CLEAN_DATA_PATH = os.path.join(DATA_DIR, "assist9_cleaned.csv")
KG_JSON_PATH = os.path.join(DATA_DIR, "kg_adj_list.json")
KG_MODEL_SAVE_PATH = os.path.join(DATA_DIR, "kg_sakt_model.pth")

SEED = 42
BATCH_SIZE = 64
MAX_SEQ = 100
EPOCHS = 30
LEARNING_RATE = 5e-4
KG_LEARNING_RATE = 7e-4
LOGIC_LAMBDA_MAX = 0.02
LOGIC_WARMUP_RATIO = 0.5
LOGIC_MARGIN = 0.02
MASTERY_THRESHOLD = 0.45
EARLY_STOPPING_PATIENCE = 5
VAL_RATIO = 0.1
USE_TIME_GAP = False


def set_seed(seed):
    """
    设置随机种子，确保实验的可重复性
    
    参数:
        seed: 随机种子值
    """
    # 设置Python内置random模块的种子
    random.seed(seed)
    # 设置NumPy的种子
    np.random.seed(seed)
    # 设置PyTorch的种子
    torch.manual_seed(seed)
    # 如果使用CUDA，设置所有GPU的种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_kg_matrix(kg_adj, n_skills, device):
    """
    构建密集的前置技能矩阵
    
    矩阵结构：
    - 行 = 目标技能
    - 列 = 前置技能
    - 值为1表示"列技能是行技能的前置技能"
    
    参数:
        kg_adj: 知识图谱邻接表，字典格式，键为技能ID，值为其前置技能列表
        n_skills: 技能总数
        device: 计算设备（CPU或GPU）
        
    返回:
        torch.Tensor: 密集的前置技能矩阵，形状为(n_skills+1, n_skills+1)
    """
    # 初始化全零矩阵，大小为(n_skills+1) x (n_skills+1)，类型为float32
    kg_matrix = torch.zeros((n_skills + 1, n_skills + 1), dtype=torch.float32, device=device)
    
    # 遍历知识图谱中的每个技能及其前置技能
    for skill, prereqs in kg_adj.items():
        # 将技能ID转换为整数
        skill_idx = int(skill)
        # 检查技能ID是否在有效范围内
        if not 0 <= skill_idx <= n_skills:
            continue
        # 遍历该技能的每个前置技能
        for prereq in prereqs:
            # 将前置技能ID转换为整数
            prereq_idx = int(prereq)
            # 检查前置技能ID是否在有效范围内
            if 0 <= prereq_idx <= n_skills:
                # 标记前置关系：kg_matrix[目标技能][前置技能] = 1.0
                kg_matrix[skill_idx, prereq_idx] = 1.0
    
    return kg_matrix


def logic_lambda_for_epoch(epoch_idx, total_epochs, max_lambda, warmup_ratio):
    """
    为每个训练epoch计算逻辑正则化的权重lambda
    
    使用warm-up策略：从较小的lambda值开始，然后逐渐增加到最大值
    
    参数:
        epoch_idx: 当前epoch索引（从1开始）
        total_epochs: 总训练epoch数
        max_lambda: 逻辑正则化的最大权重
        warmup_ratio: warm-up阶段占总epoch的比例
        
    返回:
        float: 当前epoch的逻辑正则化权重
    """
    # 计算warm-up阶段的epoch数，至少为1
    warmup_epochs = max(1, int(total_epochs * warmup_ratio))
    
    # 在warm-up阶段，lambda值从max_lambda * 0.1线性增加到max_lambda * 0.1
    if epoch_idx <= warmup_epochs:
        return max_lambda * 0.1 * epoch_idx / warmup_epochs

    # 在warm-up之后，lambda值从max_lambda * 0.1继续增加到max_lambda
    progress = (epoch_idx - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return max_lambda * (0.1 + 0.9 * progress)


def model_selection_score(name, auc, path_comp):
    """
    计算模型选择分数，用于模型评估和选择
    
    优先考虑AUC指标，对于KG-SAKT模型，会添加一个小的路径合规性奖励
    
    参数:
        name: 模型名称
        auc: AUC（ROC曲线下面积）指标值，用于评估分类性能
        path_comp: 路径合规性指标值，用于评估KG-SAKT模型的逻辑一致性
        
    返回:
        float: 模型选择分数，用于比较不同模型的性能
    """
    # 在模型选择过程中优先考虑AUC指标
    # 对于KG-SAKT模型，保留一个小的路径合规性奖励，但比以前弱很多
    if name != "KG-SAKT":
        # 非KG-SAKT模型直接返回AUC值
        return auc
    # 对于KG-SAKT模型，计算路径合规性奖励
    # 如果path_comp为NaN，则奖励为0
    path_bonus = 0.0 if np.isnan(path_comp) else 0.0005 * path_comp
    # 返回AUC加上路径奖励
    return auc + path_bonus


def masked_targets(q, y):
    """
    计算有效目标的掩码
    
    有效目标是指非填充的技能ID和非填充的标签
    - 技能ID为0表示填充位置
    - 标签为-1表示填充位置
    
    参数:
        q: 查询序列，包含技能ID，形状为[batch, seq]
        y: 目标序列，包含标签，形状为[batch, seq]
        
    返回:
        torch.Tensor: 有效目标的掩码，形状为[batch, seq]，值为True表示有效，False表示无效
    """
    # 计算有效掩码：技能ID不为0且标签不为-1
    valid_mask = (q != 0) & (y != -1)
    return valid_mask


def compute_skill_depths_from_kg(kg_adj, n_skills):
    """
    估算DAG状知识图谱中每个技能的前置技能深度
    
    深度计算规则：
    - 根技能（无前置技能）的深度为0
    - 其他技能的深度为1加上其所有前置技能的最大深度
    
    此函数用于计算推荐深度一致性（RDC）指标
    
    参数:
        kg_adj: 知识图谱邻接表，字典格式，键为技能ID，值为其前置技能列表
        n_skills: 技能总数
        
    返回:
        torch.Tensor: 包含每个技能深度的张量，形状为(n_skills+1,)
    """
    # 记忆化字典，用于存储已计算的技能深度，避免重复计算
    memo = {}
    # 访问集合，用于检测循环依赖
    visiting = set()

    def dfs(skill):
        """
        深度优先搜索计算技能深度
        
        参数:
            skill: 技能ID
            
        返回:
            float: 技能的深度值
        """
        # 如果技能深度已计算，直接返回
        if skill in memo:
            return memo[skill]
        # 防御性循环处理：如果意外存在循环，返回深度0
        if skill in visiting:
            return 0
        # 标记当前技能为正在访问
        visiting.add(skill)
        # 获取当前技能的有效前置技能列表
        # 过滤掉无效的前置技能ID（小于等于0或大于n_skills）
        prereqs = [int(p) for p in kg_adj.get(str(skill), []) if 0 <= int(p) <= n_skills]
        # 如果没有前置技能，深度为0
        if len(prereqs) == 0:
            depth = 0
        else:
            # 深度为1加上所有前置技能的最大深度
            depth = 1 + max(dfs(p) for p in prereqs)
        # 标记当前技能为已访问
        visiting.remove(skill)
        # 存储计算结果到记忆化字典
        memo[skill] = depth
        return depth

    # 初始化深度张量，大小为(n_skills+1)
    depths = torch.zeros(n_skills + 1, dtype=torch.float32)
    # 计算每个技能的深度
    for s in range(1, n_skills + 1):
        depths[s] = float(dfs(s))
    return depths


def compute_sequence_loss(name, model, batch, criterion, kg_matrix, logic_lambda):
    """
    计算序列模型的损失
    
    参数:
        name: 模型名称
        model: 模型实例
        batch: 批次数据
        criterion: 损失函数
        kg_matrix: 知识图谱矩阵
        logic_lambda: 逻辑正则化权重
        
    返回:
        torch.Tensor: 计算得到的损失值
    """
    # 将批次数据移至指定设备
    x = batch["x"].to(DEVICE)  # 输入交互序列
    q = batch["q"].to(DEVICE)  # 查询序列
    y = batch["target"].to(DEVICE)  # 目标序列
    user_ids = batch["user_id"].to(DEVICE)  # 用户ID
    # 时间桶序列（如果使用时间间隔特征）
    time_bucket = batch["time_bucket"].to(DEVICE) if USE_TIME_GAP else None
    # 计算有效目标掩码
    valid_mask = masked_targets(q, y)

    # 处理Pure-CF模型（直接预测用户-技能对）
    if name == "Pure-CF":
        # 提取有效用户、技能和目标
        valid_users = user_ids.unsqueeze(1).expand_as(q)[valid_mask]
        valid_skills = q[valid_mask]
        valid_targets = y[valid_mask]
        # 模型预测
        probs = model(valid_users, valid_skills)
        # 计算损失
        loss = criterion(probs, valid_targets)
        return loss

    # 处理其他模型（DKT、SAKT、KG-SAKT）
    # 根据模型类型调用不同的前向传播方法
    outputs = model(q, x) if name != "KG-SAKT" else model(q, x, kg_matrix=kg_matrix, time_bucket=time_bucket)
    
    # 处理不同输出维度的情况
    if outputs.dim() == 2:
        # 二维输出直接作为目标logits
        target_logits = outputs
    else:
        # 三维输出需要根据查询技能索引提取对应logits
        target_logits = outputs.gather(dim=-1, index=q.unsqueeze(-1)).squeeze(-1)
    
    # 计算基础损失（仅考虑有效目标）
    base_loss = criterion(target_logits[valid_mask], y[valid_mask])

    # 只有KG-SAKT使用训练时的逻辑正则化
    # DKT/SAKT的输出形状不包含每个时间步的完整技能分布
    if name != "KG-SAKT":
        return base_loss

    # 防御性回退：如果出现意外形状，保持训练稳定
    if outputs.dim() != 3:
        return base_loss

    # KG训练时损失：基于margin的逻辑一致性
    # 将logits转换为概率
    probs = torch.sigmoid(outputs)
    # 提取目标技能的概率
    target_probs = probs.gather(dim=-1, index=q.unsqueeze(-1)).squeeze(-1)
    # 获取目标技能的前置技能掩码
    prereq_mask = kg_matrix[q]
    # 计算margin违反：目标技能概率不应显著高于其前置技能
    margin_violation = torch.relu(LOGIC_MARGIN + target_probs.unsqueeze(-1) - probs)
    # 只考虑前置技能的违反
    prereq_violation = margin_violation * prereq_mask
    # 计算每个样本的前置技能数量（至少为1）
    prereq_count = prereq_mask.sum(dim=-1).clamp_min(1.0)
    # 计算逻辑惩罚（平均每个前置技能的违反程度）
    logic_penalty = (prereq_violation.sum(dim=-1) / prereq_count)
    # 只考虑有效目标的逻辑惩罚
    logic_penalty = logic_penalty[valid_mask].mean()
    
    # 返回基础损失加上逻辑正则化损失
    return base_loss + logic_lambda * logic_penalty


def infer_sakt_full_dist_last(model, q, x, last_idx, n_skills, chunk_size=32):
    """
    为SAKT模型构建每个样本最后一个有效时间步的完整技能概率分布
    
    SAKT模型只返回查询技能的[batch, seq]对数概率，因此需要分块查询所有技能
    
    参数:
        model: SAKT模型实例
        q: 查询序列，形状为[batch, seq]
        x: 输入交互序列，形状为[batch, seq]
        last_idx: 每个样本的最后一个有效时间步索引，形状为[batch]
        n_skills: 技能总数
        chunk_size: 分块大小，默认为32，用于避免内存溢出
        
    返回:
        torch.Tensor: 完整的技能概率分布，形状为[batch, n_skills+1]
    """
    # 获取设备信息和批次大小
    device = x.device
    batch_size = x.size(0)
    # 初始化完整概率分布矩阵，形状为[batch, n_skills+1]
    full_dist = torch.zeros((batch_size, n_skills + 1), dtype=torch.float32, device=device)

    # 生成技能ID列表（从1到n_skills）
    skill_ids = list(range(1, n_skills + 1))
    # 分块处理技能，避免内存溢出
    for start in range(0, len(skill_ids), chunk_size):
        # 获取当前块的技能ID
        chunk = skill_ids[start:start + chunk_size]
        # 当前块的大小
        c = len(chunk)

        # 技能主瓦片：[skill1 * batch, skill2 * batch, ...]
        # 重复输入序列c次，形状变为[c*batch, seq]
        x_rep = x.repeat(c, 1)
        # 重复查询序列c次，形状变为[c*batch, seq]
        q_rep = q.repeat(c, 1)
        # 重复最后时间步索引c次，形状变为[c*batch]
        last_rep = last_idx.repeat(c)
        # 生成行索引，用于后续索引操作
        row_idx = torch.arange(batch_size * c, device=device)
        # 生成查询技能张量，每个技能重复batch_size次
        query_skill = torch.tensor(chunk, device=device, dtype=torch.long).repeat_interleave(batch_size)
        # 将查询序列的最后一个有效时间步设置为当前块的技能ID
        q_rep[row_idx, last_rep] = query_skill

        # 模型前向传播，获取对数概率，形状为[c*batch, seq]
        logits = model(q_rep, x_rep)
        # 提取每个样本最后一个有效时间步的对数概率，形状为[c*batch]
        logits_last = logits[row_idx, last_rep]
        # 将对数概率转换为概率，并调整形状为[batch, c]
        probs = torch.sigmoid(logits_last).view(c, batch_size).transpose(0, 1)
        # 将当前块的概率填充到完整概率分布矩阵中
        full_dist[:, chunk] = probs

    return full_dist


def evaluate_metrics(name, model, kg_adj, loader):
    """
    评估模型的预测指标和逻辑/路径指标
    
    逻辑诊断指标：
    - PVR: 前置技能违反率（Prerequisite Violation Rate）
    - APC: 平均前置技能覆盖率（Average Prerequisite Coverage）
    - VS: 违反严重程度（Violation Severity）
    - RDC: 推荐深度一致性（Recommendation Depth Consistency）
    
    参数:
        name: 模型名称
        model: 模型实例
        kg_adj: 知识图谱邻接表
        loader: 数据加载器
        
    返回:
        tuple: 包含AUC、RMSE、Path、PVR、APC、VS、RDC的元组
    """
    # 设置模型为评估模式
    model.eval()
    
    # 初始化预测指标变量
    y_true, y_pred = [], []  # 真实值和预测值
    compliance_hits, total_checks = 0, 0  # 路径合规性统计
    
    # 边缘级逻辑计数器
    total_prereq_edges = 0.0  # 总前置技能边数
    total_prereq_violations = 0.0  # 总前置技能违反数
    total_prereq_coverage = 0.0  # 总前置技能覆盖率
    total_coverage_cases = 0.0  # 覆盖率统计案例数
    total_violation_severity = 0.0  # 总违反严重程度
    total_depth_consistent = 0.0  # 总深度一致性
    total_depth_cases = 0.0  # 深度一致性统计案例数

    # 禁用梯度计算
    with torch.no_grad():
        for batch in loader:
            # 将批次数据移至指定设备
            x = batch["x"].to(DEVICE)
            q = batch["q"].to(DEVICE)
            y = batch["target"].to(DEVICE)
            user_ids = batch["user_id"].to(DEVICE)
            time_bucket = batch["time_bucket"].to(DEVICE) if USE_TIME_GAP else None

            # 计算有效目标掩码
            valid_mask = masked_targets(q, y)
            # 计算每个序列的有效长度
            seq_lengths = valid_mask.sum(dim=1)
            # 筛选出有效长度大于0的样本
            keep_rows = seq_lengths > 0
            if not keep_rows.any():
                continue

            # 计算每个序列的最后一个有效位置
            time_index = torch.arange(q.size(1), device=DEVICE).unsqueeze(0).expand_as(q)
            last_positions = torch.where(valid_mask, time_index, torch.full_like(time_index, -1))
            last_idx = last_positions[keep_rows].max(dim=1).values
            # 获取最后一个有效位置的技能和目标
            q_last = q[keep_rows].gather(1, last_idx.unsqueeze(1)).squeeze(1)
            y_last = y[keep_rows].gather(1, last_idx.unsqueeze(1)).squeeze(1)

            # 根据模型类型获取预测概率
            if name == "Pure-CF":
                # Pure-CF直接预测用户-技能对
                pred_prob = model(user_ids[keep_rows], q_last)
                full_dist = None  # 没有完整的技能分布
            else:
                # 其他模型的预测
                outputs = model(q, x) if name != "KG-SAKT" else model(q, x, kg_matrix=None, time_bucket=time_bucket)
                if outputs.dim() == 2:
                    # SAKT返回查询技能的logits [batch, seq]
                    pred_prob = torch.sigmoid(outputs[keep_rows, last_idx])
                    if name == "SAKT":
                        # 为SAKT构建完整的技能概率分布
                        full_dist = infer_sakt_full_dist_last(
                            model=model,
                            q=q[keep_rows],
                            x=x[keep_rows],
                            last_idx=last_idx,
                            n_skills=model.n_skills,
                        )
                    else:
                        full_dist = None
                else:
                    # 三维输出获取最后一个位置的完整分布
                    last_logits = outputs[keep_rows, last_idx, :]
                    full_dist = torch.sigmoid(last_logits)
                    # 根据最后一个技能获取对应的概率
                    pred_prob = full_dist.gather(1, q_last.unsqueeze(1)).squeeze(1)

            # 收集真实值和预测值
            y_true.extend(y_last.cpu().tolist())
            y_pred.extend(pred_prob.cpu().tolist())

            # 只有当有完整技能分布时才计算逻辑指标
            if full_dist is not None:
                # "推荐技能"代理 = 非填充技能中预测掌握度最高的技能
                rec_skills = (torch.argmax(full_dist[:, 1:], dim=-1) + 1).cpu().tolist()
                full_dist_np = full_dist.cpu().numpy()

                # 对每个样本计算逻辑指标
                for i, rec_skill in enumerate(rec_skills):
                    # 获取推荐技能的前置技能
                    prereqs = kg_adj.get(str(int(rec_skill)), [])
                    # 获取技能掌握度分布
                    mastery = full_dist_np[i]
                    # 掌握度掩码（高于阈值的技能）
                    mastery_mask = mastery >= MASTERY_THRESHOLD
                    # 推荐技能的掌握度概率
                    rec_prob = float(mastery[int(rec_skill)])

                    total_checks += 1
                    # 如果没有前置技能，则视为合规
                    if not prereqs:
                        # 对于路径指标，无前置技能的推荐是合规的
                        # 对于RDC，我们跳过这些样本以避免夸大深度一致性
                        compliance_hits += 1
                        continue

                    # 统计前置技能的满足情况
                    prereq_count = 0
                    satisfied_count = 0
                    violation_count = 0
                    violation_severity = 0.0

                    # 检查每个前置技能
                    for p in prereqs:
                        p_idx = int(p)
                        # 跳过无效的前置技能索引
                        if p_idx <= 0 or p_idx >= len(mastery):
                            continue
                        prereq_count += 1
                        # 前置技能的掌握度概率
                        p_prob = float(mastery[p_idx])
                        # 检查前置技能是否被掌握
                        if mastery_mask[p_idx]:
                            satisfied_count += 1
                        else:
                            violation_count += 1
                        # 计算违反严重程度
                        violation_severity += max(0.0, LOGIC_MARGIN + rec_prob - p_prob)

                    # 如果没有有效的前置技能，则视为合规
                    if prereq_count == 0:
                        # 与无前置技能的处理相同：路径合规，但排除在RDC分母之外
                        compliance_hits += 1
                        continue

                    # 更新逻辑指标计数器
                    total_prereq_edges += prereq_count
                    total_prereq_violations += violation_count
                    total_prereq_coverage += satisfied_count / prereq_count
                    total_coverage_cases += 1
                    total_violation_severity += violation_severity
                    # 如果没有违反，则视为合规
                    if violation_count == 0:
                        compliance_hits += 1

                    # ---------- RDC（前置技能门控）----------
                    # 我们仅在具有有效前置技能边的样本上评估深度一致性
                    # 得分 = 1 仅当同时满足：
                    # 1) 前置技能约束得到满足
                    # 2) 推荐深度不超过已掌握深度太多
                    # 这避免了在频繁违反前置技能的情况下RDC被误导性地提高
                    mastered_skills = [
                        s for s in range(1, len(mastery))
                        if mastery_mask[s]
                    ]
                    # 计算允许的推荐深度
                    if mastered_skills:
                        # 已掌握技能的平均深度
                        mean_depth = float(np.mean([SKILL_DEPTHS[s] for s in mastered_skills]))
                        # 允许的深度为平均深度加1
                        allowed_depth = mean_depth + 1.0
                    else:
                        # 如果没有已掌握技能，允许深度为1
                        allowed_depth = 1.0
                    # 获取推荐技能的深度
                    rec_depth = SKILL_DEPTHS.get(int(rec_skill), 0.0)
                    # 检查前置技能是否满足
                    prereq_ok = 1.0 if violation_count == 0 else 0.0
                    # 检查推荐深度是否在允许范围内
                    depth_ok = 1.0 if rec_depth <= allowed_depth else 0.0
                    # 只有当前置技能满足且深度在范围内时，才认为深度一致
                    total_depth_consistent += prereq_ok * depth_ok
                    total_depth_cases += 1

    # 转换为 numpy 数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 计算预测指标
    # AUC：ROC曲线下面积（分类性能指标）
    auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.5
    # RMSE：均方根误差（回归性能指标）
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # 计算逻辑指标
    # Path：路径合规率
    comp = (compliance_hits / max(1, total_checks)) * 100 if total_checks > 0 else float("nan")
    # PVR：前置技能违反率
    pvr = (total_prereq_violations / max(1.0, total_prereq_edges)) * 100.0 if total_checks > 0 else float("nan")
    # APC：平均前置技能覆盖率
    apc = (total_prereq_coverage / max(1.0, total_coverage_cases)) * 100.0 if total_coverage_cases > 0 else float("nan")
    # VS：违反严重程度
    vs = total_violation_severity / max(1.0, total_prereq_edges) if total_checks > 0 else float("nan")
    # RDC：推荐深度一致性
    rdc = (total_depth_consistent / max(1.0, total_depth_cases)) * 100.0 if total_depth_cases > 0 else float("nan")
    
    return auc, rmse, comp, pvr, apc, vs, rdc


def create_loaders(df, n_skills):
    """
    创建训练、验证和测试数据加载器
    
    函数将用户数据按照8:1:1的比例划分为训练集、验证集和测试集
    
    参数:
        df: 包含用户交互数据的DataFrame
        n_skills: 技能总数
        
    返回:
        tuple: 包含训练、验证和测试数据加载器的元组
    """
    # 获取唯一用户ID并打乱顺序
    user_ids = df["user_id"].unique().copy()
    np.random.shuffle(user_ids)
    
    # 计算数据集划分的边界索引
    # 测试集占20%
    test_start = int(len(user_ids) * (1.0 - 0.2))
    # 验证集占训练集的VAL_RATIO比例
    val_start = int(test_start * (1.0 - VAL_RATIO))

    # 划分用户ID到不同数据集
    train_users = user_ids[:val_start]  # 训练集用户
    val_users = user_ids[val_start:test_start]  # 验证集用户
    test_users = user_ids[test_start:]  # 测试集用户

    # 根据用户ID划分数据
    train_df = df[df["user_id"].isin(train_users)].copy()
    val_df = df[df["user_id"].isin(val_users)].copy()
    test_df = df[df["user_id"].isin(test_users)].copy()

    # 创建数据集实例
    train_dataset = Assist9Dataset(train_df, n_skills=n_skills, max_seq=MAX_SEQ)
    val_dataset = Assist9Dataset(val_df, n_skills=n_skills, max_seq=MAX_SEQ)
    test_dataset = Assist9Dataset(test_df, n_skills=n_skills, max_seq=MAX_SEQ)

    # 创建数据加载器
    # 训练集启用打乱，验证集和测试集不打乱
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader


def save_metrics_and_plots(final_results, csv_dir, chart_dir):
    """
    导出最终指标表到CSV并绘制逻辑相关图表
    
    生成的文件：
    - logic_metrics_comparison.csv：包含所有模型的评估指标
    - logic_metrics_bar.png：逻辑指标的条形图
    - logic_metrics_radar.png：逻辑指标的雷达图
    
    参数:
        final_results: 包含模型评估结果的列表，每个元素是一个元组
        csv_dir: CSV文件保存目录
        chart_dir: 图表保存目录
    """
    # 创建保存目录（如果不存在）
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(chart_dir, exist_ok=True)

    # 将结果转换为DataFrame
    df = pd.DataFrame(
        final_results,
        columns=["Model", "BestEp", "AUC", "RMSE", "Path", "PVR", "APC", "VS", "RDC"],
    )
    # 保存为CSV文件
    csv_path = os.path.join(csv_dir, "logic_metrics_comparison.csv")
    df.to_csv(csv_path, index=False)

    # 只选择有逻辑输出的模型进行图表绘制
    logic_df = df.dropna(subset=["Path", "PVR", "APC", "VS", "RDC"]).copy()
    if logic_df.empty:
        print(f"[Export] CSV saved to: {csv_path}")
        print("[Export] No valid logic metrics to plot.")
        return

    # ---------- 条形图 ----------
    # 混合方向指标：Path/APC/RDC越高越好；PVR/VS越低越好
    plot_df = logic_df.copy()
    # 对PVR和VS取负值，使所有指标方向一致（越高越好）
    plot_df["PVR_inv"] = -plot_df["PVR"]
    plot_df["VS_inv"] = -plot_df["VS"]
    # 要绘制的指标
    bar_metrics = ["Path", "APC", "RDC", "PVR_inv", "VS_inv"]
    # 指标标签
    bar_labels = ["Path(+) ", "APC(+) ", "RDC(+) ", "PVR(-)", "VS(-)"]

    # 绘制条形图
    x = np.arange(len(plot_df["Model"]))
    width = 0.14
    plt.figure(figsize=(12, 6))
    for i, metric in enumerate(bar_metrics):
        plt.bar(x + (i - 2) * width, plot_df[metric], width=width, label=bar_labels[i])
    plt.xticks(x, plot_df["Model"])
    plt.ylabel("Metric Value (PVR/VS are sign-inverted)")
    plt.title("Logic Metrics Comparison (Bar)")
    plt.legend()
    plt.tight_layout()
    # 保存条形图
    bar_path = os.path.join(chart_dir, "logic_metrics_bar.png")
    plt.savefig(bar_path, dpi=200)
    plt.close()

    # ---------- 雷达图 ----------
    # 对于雷达图，我们将每个指标归一化到[0, 1]并统一方向为"越高越好"
    radar_metrics = ["Path", "APC", "RDC", "PVR", "VS"]
    # 指标方向：True表示越高越好，False表示越低越好
    higher_better = {"Path": True, "APC": True, "RDC": True, "PVR": False, "VS": False}
    radar_data = {}

    # 归一化每个指标
    for metric in radar_metrics:
        col = plot_df[metric].to_numpy(dtype=float)
        min_v, max_v = np.min(col), np.max(col)
        if np.isclose(max_v, min_v):
            # 如果所有值相同，归一化为1
            norm = np.ones_like(col)
        else:
            if higher_better[metric]:
                # 越高越好的指标：(值-最小值)/(最大值-最小值)
                norm = (col - min_v) / (max_v - min_v)
            else:
                # 越低越好的指标：(最大值-值)/(最大值-最小值)
                norm = (max_v - col) / (max_v - min_v)
        radar_data[metric] = norm

    # 准备雷达图数据
    categories = ["Path", "APC", "RDC", "PVR(inv)", "VS(inv)"]
    # 计算角度
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合雷达图

    # 绘制雷达图
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    for i, model_name in enumerate(plot_df["Model"].tolist()):
        # 准备每个模型的数据
        values = [
            radar_data["Path"][i],
            radar_data["APC"][i],
            radar_data["RDC"][i],
            radar_data["PVR"][i],
            radar_data["VS"][i],
        ]
        values += values[:1]  # 闭合雷达图
        # 绘制线条和填充
        ax.plot(angles, values, linewidth=2, label=model_name)
        ax.fill(angles, values, alpha=0.15)

    # 设置雷达图属性
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_yticklabels([])
    ax.set_title("Logic Metrics Radar (Normalized)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()
    # 保存雷达图
    radar_path = os.path.join(chart_dir, "logic_metrics_radar.png")
    plt.savefig(radar_path, dpi=200)
    plt.close()

    # 打印保存路径
    print(f"[Export] CSV saved to: {csv_path}")
    print(f"[Export] Bar chart saved to: {bar_path}")
    print(f"[Export] Radar chart saved to: {radar_path}")


def main():
    """
    主函数：训练和评估多个模型，比较它们的性能
    
    流程：
    1. 设置随机种子和清空CUDA缓存
    2. 加载数据和知识图谱
    3. 构建知识图谱矩阵和计算技能深度
    4. 创建数据加载器
    5. 定义模型列表
    6. 训练和评估每个模型
    7. 保存最佳模型（特别是KG-SAKT）
    8. 输出实验结果
    9. 保存指标和图表
    """
    # 设置随机种子，确保实验的可重复性
    set_seed(SEED)
    # 清空CUDA缓存，避免内存溢出
    torch.cuda.empty_cache()

    # 加载清理后的数据
    df = pd.read_csv(CLEAN_DATA_PATH)
    # 获取技能总数和用户总数
    n_skills = int(df["skill_id"].max())
    n_users = int(df["user_id"].max()) + 1

    # 加载知识图谱邻接表
    with open(KG_JSON_PATH, "r", encoding="utf-8") as f:
        kg_adj = json.load(f)
    # 构建知识图谱矩阵
    kg_matrix = build_kg_matrix(kg_adj, n_skills, DEVICE)
    # 预计算技能深度，用于RDC指标
    skill_depths = compute_skill_depths_from_kg(kg_adj, n_skills)
    # 全局变量，存储技能深度
    global SKILL_DEPTHS
    SKILL_DEPTHS = {i: float(skill_depths[i].item()) for i in range(len(skill_depths))}

    # 创建训练、验证和测试数据加载器
    train_loader, val_loader, test_loader = create_loaders(df, n_skills)

    # 定义要训练的模型列表
    models_list = [
        ("Pure-CF", PureCFModel(n_users=n_users, n_skills=n_skills)),  # 纯协同过滤模型
        ("DKT", DKTModel(n_skills=n_skills)),  # 深度知识追踪模型
        ("SAKT", SAKTModel(n_skills=n_skills, max_seq=MAX_SEQ)),  # 自注意力知识追踪模型
        ("KG-SAKT", KGSAKTModel(n_skills=n_skills, kg_adj=kg_adj, max_seq=MAX_SEQ, use_time_feature=USE_TIME_GAP)),  # 带知识图谱的SAKT模型
    ]

    # 存储最终结果
    final_results = []

    # 遍历每个模型进行训练和评估
    for name, model in models_list:
        print(f"\n[Training] {name}")
        # 将模型移至指定设备
        model = model.to(DEVICE)
        # 根据模型类型选择优化器
        if name == "KG-SAKT":
            # KG-SAKT使用AdamW优化器，学习率更高，带有权重衰减
            optimizer = optim.AdamW(model.parameters(), lr=KG_LEARNING_RATE, weight_decay=1e-5)
        else:
            # 其他模型使用Adam优化器
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        # 根据模型类型选择损失函数
        # Pure-CF直接输出概率，使用BCELoss
        # 其他模型输出对数概率，使用BCEWithLogitsLoss
        criterion = nn.BCELoss() if name == "Pure-CF" else nn.BCEWithLogitsLoss()
        # 初始化最佳状态和指标
        best_state = None  # 最佳模型状态
        best_epoch = 0  # 最佳epoch
        best_metrics = None  # 最佳验证指标
        best_score = float("-inf")  # 最佳模型选择分数
        patience_counter = 0  # 早停计数器

        # 训练循环
        for epoch in range(1, EPOCHS + 1):
            # 设置模型为训练模式
            model.train()
            epoch_losses = []  # 存储每个批次的损失
            # 计算当前epoch的逻辑正则化权重
            current_logic_lambda = logic_lambda_for_epoch(
                epoch_idx=epoch,
                total_epochs=EPOCHS,
                max_lambda=LOGIC_LAMBDA_MAX,
                warmup_ratio=LOGIC_WARMUP_RATIO,
            )

            # 遍历训练数据
            for batch in train_loader:
                # 清零梯度
                optimizer.zero_grad()
                # 计算损失
                loss = compute_sequence_loss(
                    name=name,
                    model=model,
                    batch=batch,
                    criterion=criterion,
                    kg_matrix=kg_matrix,
                    logic_lambda=current_logic_lambda,
                )
                # 反向传播
                loss.backward()
                # 梯度裁剪，避免梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                # 更新参数
                optimizer.step()
                # 记录损失
                epoch_losses.append(loss.item())

            # 在验证集上评估模型
            val_auc, val_rmse, val_comp, val_pvr, val_apc, val_vs, val_rdc = evaluate_metrics(name, model, kg_adj, val_loader)
            # 计算模型选择分数
            val_score = model_selection_score(name, val_auc, val_comp)
            # 如果当前分数更好，更新最佳状态
            if val_score > best_score:
                best_score = val_score
                best_epoch = epoch
                best_metrics = (val_auc, val_rmse, val_comp)
                # 保存模型状态
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                # 重置早停计数器
                patience_counter = 0
            else:
                # 早停计数器加1
                patience_counter += 1

            # 每5个epoch或第一个epoch打印一次结果
            if epoch == 1 or epoch % 5 == 0:
                # 格式化输出逻辑指标
                comp_text = f"{val_comp:.2f}" if not np.isnan(val_comp) else "N/A"
                pvr_text = f"{val_pvr:.2f}" if not np.isnan(val_pvr) else "N/A"
                apc_text = f"{val_apc:.2f}" if not np.isnan(val_apc) else "N/A"
                vs_text = f"{val_vs:.4f}" if not np.isnan(val_vs) else "N/A"
                rdc_text = f"{val_rdc:.2f}" if not np.isnan(val_rdc) else "N/A"
                # 打印当前epoch的结果
                print(
                    f"  Epoch {epoch:02d} | Loss {np.mean(epoch_losses):.4f} | "
                    f"ValAUC {val_auc:.4f} | ValRMSE {val_rmse:.4f} | ValPath {comp_text} | "
                    f"ValPVR {pvr_text} | ValAPC {apc_text} | ValVS {vs_text} | ValRDC {rdc_text} | "
                    f"LogicLambda {current_logic_lambda:.4f}"
                )

            # 早停检查
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"  Early stopping at epoch {epoch:02d} | Best epoch {best_epoch:02d}")
                break

        # 如果有最佳状态，加载它
        if best_state is not None:
            model.load_state_dict(best_state)
            # 保存最佳KG-SAKT模型，用于推理模拟
            if name == "KG-SAKT":
                torch.save(best_state, KG_MODEL_SAVE_PATH)
                print(f"  [Saved] Best KG-SAKT weights -> {KG_MODEL_SAVE_PATH}")

        # 打印最佳验证结果
        best_val_auc, best_val_rmse, best_val_comp = best_metrics
        best_val_comp_text = f"{best_val_comp:.2f}" if not np.isnan(best_val_comp) else "N/A"
        print(
            f"  Best validation | Epoch {best_epoch:02d} | "
            f"AUC {best_val_auc:.4f} | RMSE {best_val_rmse:.4f} | Path {best_val_comp_text}"
        )

        # 在测试集上评估模型
        final_auc, final_rmse, final_comp, final_pvr, final_apc, final_vs, final_rdc = evaluate_metrics(name, model, kg_adj, test_loader)
        # 将结果添加到最终结果列表
        final_results.append((name, best_epoch, final_auc, final_rmse, final_comp, final_pvr, final_apc, final_vs, final_rdc))

    # 打印实验结果
    print("\n" + "Experiment Results".center(76, "="))
    # 打印表头，↑表示指标越高越好，↓表示指标越低越好
    print(
        f"{'Model':<10} | {'BestEp':<6} | {'AUC↑':<7} | {'RMSE↓':<7} | "
        f"{'Path%↑':<7} | {'PVR%↓':<7} | {'APC%↑':<7} | {'VS↓':<7} | {'RDC%↑':<7}"
    )
    print("-" * 76)
    # 打印每个模型的结果
    for name, best_epoch, auc, rmse, comp, pvr, apc, vs, rdc in final_results:
        # 格式化输出逻辑指标
        comp_text = f"{comp:.2f}" if not np.isnan(comp) else "N/A"
        pvr_text = f"{pvr:.2f}" if not np.isnan(pvr) else "N/A"
        apc_text = f"{apc:.2f}" if not np.isnan(apc) else "N/A"
        vs_text = f"{vs:.4f}" if not np.isnan(vs) else "N/A"
        rdc_text = f"{rdc:.2f}" if not np.isnan(rdc) else "N/A"
        # 打印结果
        print(
            f"{name:<10} | {best_epoch:<6d} | {auc:<7.4f} | {rmse:<7.4f} | "
            f"{comp_text:<7} | {pvr_text:<7} | {apc_text:<7} | {vs_text:<7} | {rdc_text:<7}"
        )
    print("=" * 76)

    # 导出CSV和图表，用于报告和可视化
    save_metrics_and_plots(final_results, csv_dir=DATA_DIR, chart_dir=RENDERED_DIR)


if __name__ == "__main__":
    main()