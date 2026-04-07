# KG-SAKT 教育推荐系统（中文）

本项目研究在 SAKT 中引入知识图谱约束，以提升学习资源推荐的预测效果与路径逻辑性。

## 目录结构
```text
Education_Recommender_System/
├── data/
├── models/
├── preprocess/
├── utils/
│   ├── train_and_eval.py
│   ├── inference_recommend.py
│   ├── case_study_viz.py
│   ├── render_all.py
│   └── plot_results.py
├── documents/
│   ├── README.md
│   ├── readme_zhCN.md
│   ├── Algorithm_Overview.md
│   ├── log.md
│   └── chapter34_thesis_merged.md
└── main.py
```

## 运行流程
```powershell
.\.venv\Scripts\python.exe preprocess\clean_data.py
.\.venv\Scripts\python.exe utils\train_and_eval.py
.\.venv\Scripts\python.exe utils\inference_recommend.py
.\.venv\Scripts\python.exe utils\case_study_viz.py
```

## 项目运转顺序
1. 数据清洗：
   - 执行 `preprocess/clean_data.py`
   - 输入：`data/skill_builder_data.csv`
   - 输出：`data/assist9_cleaned.csv`、`data/skill_id_remap.json`
2. 知识图谱构建（清洗后自动触发）：
   - 使用 `preprocess/kg_builder.py` 的“学科领域 + 核心链路”构图
   - 三元组主存储：`(KP_source, requires, KP_target)`
   - 输出：
     - `data/kg_triples.json`
     - `data/kg_adj_list.json`（由三元组回写，供训练兼容使用）
3. 模型训练与对比：
   - 执行 `utils/train_and_eval.py`
   - 训练 `Pure-CF`、`DKT`、`SAKT`、`KG-SAKT`
   - 输出：指标表/图 与 `data/kg_sakt_model.pth`
4. 推荐模拟：
   - 执行 `utils/inference_recommend.py`
   - 读取模型权重 + KG + 技能映射
   - 输出：`data/recommendation_simulation.json`
5. 案例可视化：
   - 执行 `utils/case_study_viz.py`
   - 输出：`rendered/case_study_scores.png`、`rendered/case_study_paths.png`

## 主要输出
- `data/assist9_cleaned.csv`
- `data/kg_triples.json`
- `data/kg_adj_list.json`
- `data/kg_sakt_model.pth`
- `data/logic_metrics_comparison.csv`
- `data/recommendation_simulation.json`
- `rendered/kg_adjacency_matrix.png`
- `rendered/four_models_comparison.png`
- `rendered/student_recommendation_example.png`

