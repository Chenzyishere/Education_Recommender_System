import torch
import json
import numpy as np
import os
from models.kg_sakt import KGSAKTModel

# ==========================================
# 1. 环境与路径配置
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

KG_JSON_PATH = os.path.join(DATA_DIR, "kg_adj_list.json")
SKILL_MAP_PATH = os.path.join(DATA_DIR, "skill_map.json")
MODEL_WEIGHTS = os.path.join(DATA_DIR, "kg_sakt_model.pth")

# 如果根目录有模型，也尝试读取
if not os.path.exists(MODEL_WEIGHTS):
    MODEL_WEIGHTS = os.path.join(BASE_DIR, "kg_sakt_model.pth")

MAX_SEQ = 100


# ==========================================
# 2. 辅助工具函数
# ==========================================
def get_width(text):
    """计算中英文字符混合宽度"""
    return sum(2 if '\u4e00' <= char <= '\u9fff' else 1 for char in str(text))


def align_text(text, width):
    """动态对齐填充"""
    return str(text) + " " * (width - get_width(text))


def generate_reason(skill_id, prob, kg_adj, skill_map, prob_dict):
    """根据知识图谱和预测概率生成教学解释"""
    pre_skills = kg_adj.get(str(skill_id), [])
    if pre_skills:
        # 寻找已掌握的前置知识点名称
        for p_id in pre_skills:
            if prob_dict.get(int(p_id), 0) > 0.6:
                name = skill_map.get(str(p_id), f"ID:{p_id}")
                return f"已掌握[{name}]，符合进阶逻辑"

    if 0.45 <= prob <= 0.75:
        return "处于最近发展区(ZPD)，最易产生突破"
    elif prob > 0.75:
        return "预估掌握度高，可作为信心巩固练习"
    else:
        return "基础概念待强化，建议通过本题查漏补缺"


# ==========================================
# 3. 资源加载引擎
# ==========================================
def load_resources():
    # A. 加载知识点名字映射
    skill_map = {}
    if os.path.exists(SKILL_MAP_PATH):
        with open(SKILL_MAP_PATH, 'r', encoding='utf-8') as f:
            skill_map = json.load(f)
        print(f"📖 成功导入 {len(skill_map)} 个知识点映射。")

    # B. 加载知识图谱
    kg_adj = {}
    if os.path.exists(KG_JSON_PATH):
        with open(KG_JSON_PATH, 'r', encoding='utf-8') as f:
            kg_adj = json.load(f)
        print(f"🕸️ 成功导入知识图谱关系。")

    # C. 加载模型权重并自动适配维度
    if not os.path.exists(MODEL_WEIGHTS):
        print(f"❌ 错误: 未找到权重文件 {MODEL_WEIGHTS}")
        return None, kg_adj, skill_map

    state_dict = torch.load(MODEL_WEIGHTS, map_location=DEVICE)
    # 自动探测训练时的 Embedding 维度 (例如报错中提到的 624)
    trained_n_skills = state_dict['skill_embed.weight'].shape[0]

    # 初始化模型 (注意：传入 trained_n_skills-1 是为了抵消模型内部的 +1 逻辑)
    model = KGSAKTModel(n_skills=trained_n_skills - 1, kg_adj=kg_adj).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"🎉 模型加载成功 (维度适配: {trained_n_skills})。")

    return model, kg_adj, skill_map


# ==========================================
# 4. 推荐核心算法 (长序列版)
# ==========================================
def get_recommendation(model, kg_adj, skill_map, history):
    skills, corrects = history["skills"], history["corrects"]

    # 1. 诊断现状：计算最近 5 次练习的正确率作为水平参考
    recent_corrects = corrects[-5:]
    avg_score = sum(recent_corrects) / len(recent_corrects)
    level = "初级" if avg_score < 0.4 else "中级" if avg_score < 0.7 else "高级"

    # 2. 构造模型输入 x (Offset 设为 623 以匹配 1247 维度的 inter_embed)
    x_input = np.array(skills) + (np.array(corrects) * 623)
    pad_len = MAX_SEQ - len(x_input)
    x_tensor_input = np.pad(x_input, (pad_len, 0), 'constant', constant_values=0)
    x_tensor = torch.LongTensor(x_tensor_input).unsqueeze(0).to(DEVICE)

    # 3. 预测所有潜在知识点的掌握概率
    prob_results = []
    with torch.no_grad():
        for skill_id in range(1, 125):
            q = np.zeros(MAX_SEQ);
            q[-1] = skill_id
            q_tensor = torch.LongTensor(q).unsqueeze(0).to(DEVICE)
            output = model(q_tensor, x_tensor)
            prob_results.append((skill_id, output[0, -1].item()))

    prob_dict = {sid: p for sid, p in prob_results}

    # 4. 逻辑过滤与解释生成
    recommendations = []
    for skill_id, p in prob_results:
        if p > 0.88: continue  # 过滤掉基本已掌握的内容

        is_logical = True
        # KG 强约束：前置掌握度必须 > 0.5
        if str(skill_id) in kg_adj:
            for pre in kg_adj[str(skill_id)]:
                if prob_dict.get(int(pre), 0) < 0.5:
                    is_logical = False;
                    break

        if is_logical and p > 0.3:
            reason = generate_reason(skill_id, p, kg_adj, skill_map, prob_dict)
            recommendations.append((skill_id, p, reason))

    # 按概率降序排列，取前 2 个
    return sorted(recommendations, key=lambda x: x[1], reverse=True)[:2], level, avg_score


# ==========================================
# 5. 执行主程序
# ==========================================
def main():
    model, kg_adj, skill_map = load_resources()
    if model is None: return

    # 长序列学生数据 (更有说服力)
    test_students = {
        "学生 A (基础薄弱型)": {
            "skills": [1, 1, 1, 1, 12, 12, 12, 1, 1, 95],
            "corrects": [0, 0, 1, 0, 0, 1, 0, 1, 1, 0]
        },
        "学生 B (稳步进阶型)": {
            "skills": [27, 27, 27, 10, 10, 10, 84, 84, 84, 84],
            "corrects": [1, 1, 1, 0, 1, 1, 1, 0, 1, 1]
        },
        "学生 C (拔尖挑战型)": {
            "skills": [80, 80, 110, 110, 110, 86, 86, 86, 91, 91],
            "corrects": [1, 1, 1, 1, 1, 1, 0, 1, 1, 1]
        }
    }

    # 动态列宽配置
    w_name, w_info, w_skill, w_prob, w_reason = 18, 16, 28, 10, 40

    print("\n" + "=" * 116)
    header = f" {align_text('学生画像', w_name)} | {align_text('当前水平(近5次)', w_info)} | {align_text('推荐知识点', w_skill)} | {align_text('预测掌握度', w_prob)} | {align_text('教学推荐理由', w_reason)}"
    print(header)
    print("-" * 116)

    for name, history in test_students.items():
        recs, level, score = get_recommendation(model, kg_adj, skill_map, history)
        info_str = f"{level}(得分:{score:.0%})"

        for i, (sid, prob, reason) in enumerate(recs):
            skill_name = skill_map.get(str(sid), f"Skill ID: {sid}")
            n_display = name if i == 0 else ""
            i_display = info_str if i == 0 else ""

            line = f" {align_text(n_display, w_name)} | {align_text(i_display, w_info)} | {align_text(skill_name, w_skill)} | {prob:.4f}   | {align_text(reason, w_reason)}"
            print(line)
        print("-" * 116)

    print("=" * 116)
    print("💡 结论: 该模型成功将长序列历史与知识图谱结合，实现了具备可解释性的教学决策。")


if __name__ == "__main__":
    main()