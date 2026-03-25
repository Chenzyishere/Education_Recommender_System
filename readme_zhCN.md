# KG-SAKT 教育推荐系统

本项目研究：在 SAKT 中引入知识图谱（Knowledge Graph, KG）约束后，是否能在保证预测能力的同时，给出更符合先修逻辑的学习资源推荐。

## 研究重点

- 在统一训练协议下对比 `Pure-CF`、`DKT`、`SAKT`、`KG-SAKT`。
- 同时评估预测质量与教育逻辑一致性。
- `Time Gap` 作为可选消融项，不作为默认主线优化方向。

## 当前 KG-SAKT 优化点

- 在自注意力后进行图结构增强表示融合。
- 使用一阶 + 二阶先修扩散，并通过可学习权重融合。
- 使用 margin 形式的逻辑一致性损失，鼓励 `P(prereq) >= P(target) - margin`。
- 基于验证集早停，并回载最佳轮次参数。

## 评估指标

- `AUC`（越高越好）：下一题作答正确性的区分能力。
- `RMSE`（越低越好）：预测概率与真实标签的偏差。
- `Path Compliance`（越高越好）：推荐知识点是否满足 KG 先修关系。

## 项目结构

```text
Education_Recommender_System/
├── data/
│   ├── assist9_cleaned.csv
│   ├── kg_adj_list.json
│   └── skill_builder_data.csv
├── models/
│   ├── pure_cf.py
│   ├── dkt.py
│   ├── sakt.py
│   └── kg_sakt.py
├── preprocess/
│   ├── clean_data.py
│   └── dataset_loader.py
├── train_and_eval.py
├── Algorithm_Overview.md
├── README.md
├── readme_zhCN.md
└── log.md
```

## 运行环境

- Python 3.8+
- PyTorch 2.0+
- pandas、numpy、scikit-learn

安装依赖：

```bash
pip install torch pandas numpy scikit-learn
```

## 运行实验

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe train_and_eval.py
```

脚本会自动训练四个模型并输出基于最佳轮次的结果表。

## 关键配置（train_and_eval.py）

- `USE_TIME_GAP = False`（默认关闭）
- `LOGIC_LAMBDA_MAX = 0.02`
- `LOGIC_MARGIN = 0.02`
- `LEARNING_RATE = 5e-4`（通用）
- `KG_LEARNING_RATE = 7e-4`（KG-SAKT）
- `EPOCHS = 30`
- `EARLY_STOPPING_PATIENCE = 5`

如需进行 Time Gap 消融，请设置：

- `USE_TIME_GAP = True`

仅当清洗后的数据包含有效 `time_gap` 列时建议开启。

## 数据准备

1. 将原始 ASSIST 风格数据放到 `data/skill_builder_data.csv`。
2. 执行：

```powershell
.\.venv\Scripts\python.exe preprocess\clean_data.py
```

清洗脚本默认输出 `user_id`、`skill_id`、`correct`，并在可解析到时间信息时追加 `time_gap`。

## 说明

- 主实验路线是 KG 优先，Time Gap 可选。
- 每轮优化记录在 `log.md`。
