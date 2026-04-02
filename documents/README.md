# KG-SAKT Education Recommender System

This project studies KG-enhanced SAKT for educational recommendation.

## Project Structure
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

## Run Pipeline
```powershell
.\.venv\Scripts\python.exe preprocess\clean_data.py
.\.venv\Scripts\python.exe utils\train_and_eval.py
.\.venv\Scripts\python.exe utils\inference_recommend.py
.\.venv\Scripts\python.exe utils\case_study_viz.py
```

## Project Execution Order
1. Data cleaning:
   - Run `preprocess/clean_data.py`
   - Input: `data/skill_builder_data.csv`
   - Output: `data/assist9_cleaned.csv`, `data/skill_map.csv`
2. KG construction (triggered by cleaning):
   - Uses domain-core pipeline in `preprocess/kg_builder.py`
   - Triple storage schema: `(KP_source, requires, KP_target)`
   - Outputs:
     - `data/kg_triples.json`
     - `data/kg_triples.csv`
     - `data/kg_adj_list.json` (derived from triples for training compatibility)
3. Model training and comparison:
   - Run `utils/train_and_eval.py`
   - Trains `Pure-CF`, `DKT`, `SAKT`, `KG-SAKT`
   - Output: metric tables/figures and `data/kg_sakt_model.pth`
4. Recommendation simulation:
   - Run `utils/inference_recommend.py`
   - Uses trained KG-SAKT checkpoint + KG + skill map
   - Output: `data/recommendation_simulation.json`
5. Case-study visualization:
   - Run `utils/case_study_viz.py`
   - Output: `rendered/case_study_scores.png`, `rendered/case_study_paths.png`

## Key Outputs
- `data/assist9_cleaned.csv`
- `data/kg_triples.json`
- `data/kg_triples.csv`
- `data/kg_adj_list.json`
- `data/kg_sakt_model.pth`
- `data/logic_metrics_comparison.csv`
- `data/recommendation_simulation.json`
- `rendered/kg_adjacency_matrix.png`
- `rendered/four_models_comparison.png`
- `rendered/student_recommendation_example.png`
