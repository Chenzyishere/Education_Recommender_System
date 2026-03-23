# utils.py

class KnowledgeManager:
    def __init__(self):
        # 定义核心知识图谱（基于你查到的高频ID：11, 70, 311, 47, 277）
        # 结构：{ p_id: {"name": 名字, "pre": [前置ID列表]} }
        self.graph = {
            11: {"name": "整数四则运算", "pre": []},
            47: {"name": "分数的定义", "pre": [11]},
            70: {"name": "简单方程求解", "pre": [11]},
            311: {"name": "线性函数与图像", "pre": [70]},
            277: {"name": "几何图形属性", "pre": [47]},
            307: {"name": "不等式基础", "pre": [70]}
        }

    def get_student_profiles(self):
        """定义三个不同程度的模拟学生"""
        return {
            "struggling": {
                "label": "基础薄弱型 (Student_A)",
                "history": [11],
                "performance": [0.45], # 45% 正确率
                "desc": "仅尝试过基础运算，且掌握不牢固。"
            },
            "average": {
                "label": "稳健进步型 (Student_B)",
                "history": [11, 70],
                "performance": [0.85, 0.80],
                "desc": "已掌握基础与方程，逻辑链条完整。"
            },
            "top": {
                "label": "学霸进阶型 (Student_C)",
                "history": [11, 70, 311, 47],
                "performance": [0.98, 0.95, 0.92, 0.90],
                "desc": "学科知识储备丰富，已进入高阶函数阶段。"
            }
        }

    def get_learnable_skills(self, learned_ids):
        """逻辑过滤：找出前置已满足且未学过的知识点"""
        candidates = []
        for k_id, info in self.graph.items():
            if k_id in learned_ids:
                continue
            # 检查是否所有前置都在已学列表中
            if not info['pre'] or all(p in learned_ids for p in info['pre']):
                candidates.append({
                    'p_id': k_id,
                    'name': info['name'],
                    'pre_names': [self.graph[p]['name'] for p in info['pre']] if info['pre'] else ["学科基础"]
                })
        return candidates