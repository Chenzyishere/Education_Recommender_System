import json
import os
import random

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, roc_auc_score
from torch.utils.data import DataLoader

from models.dkt import DKTModel
from models.kg_sakt import KGSAKTModel
from models.pure_cf import PureCFModel
from models.sakt import SAKTModel
from preprocess.dataset_loader import Assist9Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_kg_matrix(kg_adj, n_skills, device):
    # Build dense prerequisite matrix:
    # row = target skill, col = prerequisite skill, value 1 means "col is prereq of row".
    kg_matrix = torch.zeros((n_skills + 1, n_skills + 1), dtype=torch.float32, device=device)
    for skill, prereqs in kg_adj.items():
        skill_idx = int(skill)
        if not 0 <= skill_idx <= n_skills:
            continue
        for prereq in prereqs:
            prereq_idx = int(prereq)
            if 0 <= prereq_idx <= n_skills:
                kg_matrix[skill_idx, prereq_idx] = 1.0
    return kg_matrix


def logic_lambda_for_epoch(epoch_idx, total_epochs, max_lambda, warmup_ratio):
    # Warm-up schedule for logic regularization:
    # start with tiny logic pressure, then increase gradually.
    warmup_epochs = max(1, int(total_epochs * warmup_ratio))
    if epoch_idx <= warmup_epochs:
        return max_lambda * 0.1 * epoch_idx / warmup_epochs

    progress = (epoch_idx - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return max_lambda * (0.1 + 0.9 * progress)


def model_selection_score(name, auc, path_comp):
    # We now prioritize AUC during model selection.
    # For KG-SAKT we keep a small path bonus, but much weaker than before.
    if name != "KG-SAKT":
        return auc
    path_bonus = 0.0 if np.isnan(path_comp) else 0.0005 * path_comp
    return auc + path_bonus


def masked_targets(q, y):
    # Valid targets are non-padding skill ids and non-padding labels.
    valid_mask = (q != 0) & (y != -1)
    return valid_mask


def compute_skill_depths_from_kg(kg_adj, n_skills):
    """
    Estimate each skill's prerequisite depth in the DAG-like KG.
    depth(skill) = 0 for root skills (no prereq), otherwise 1 + max(depth(prereq)).
    This is used for Recommendation Depth Consistency (RDC).
    """
    memo = {}
    visiting = set()

    def dfs(skill):
        if skill in memo:
            return memo[skill]
        # Defensive cycle handling: if cycle exists unexpectedly, break with depth 0.
        if skill in visiting:
            return 0
        visiting.add(skill)
        prereqs = [int(p) for p in kg_adj.get(str(skill), []) if 0 <= int(p) <= n_skills]
        if len(prereqs) == 0:
            depth = 0
        else:
            depth = 1 + max(dfs(p) for p in prereqs)
        visiting.remove(skill)
        memo[skill] = depth
        return depth

    depths = torch.zeros(n_skills + 1, dtype=torch.float32)
    for s in range(1, n_skills + 1):
        depths[s] = float(dfs(s))
    return depths


def compute_sequence_loss(name, model, batch, criterion, kg_matrix, logic_lambda):
    x = batch["x"].to(DEVICE)
    q = batch["q"].to(DEVICE)
    y = batch["target"].to(DEVICE)
    user_ids = batch["user_id"].to(DEVICE)
    time_bucket = batch["time_bucket"].to(DEVICE) if USE_TIME_GAP else None
    valid_mask = masked_targets(q, y)

    if name == "Pure-CF":
        # Pure-CF predicts user-skill pairs directly (no sequence logits).
        valid_users = user_ids.unsqueeze(1).expand_as(q)[valid_mask]
        valid_skills = q[valid_mask]
        valid_targets = y[valid_mask]
        probs = model(valid_users, valid_skills)
        loss = criterion(probs, valid_targets)
        return loss

    outputs = model(q, x) if name != "KG-SAKT" else model(q, x, kg_matrix=kg_matrix, time_bucket=time_bucket)
    if outputs.dim() == 2:
        target_logits = outputs
    else:
        target_logits = outputs.gather(dim=-1, index=q.unsqueeze(-1)).squeeze(-1)
    base_loss = criterion(target_logits[valid_mask], y[valid_mask])

    if name != "KG-SAKT":
        return base_loss

    # Margin-based logic loss:
    # enforce P(prereq) >= P(target) - margin for prerequisite edges.
    probs = torch.sigmoid(outputs)
    target_probs = probs.gather(dim=-1, index=q.unsqueeze(-1)).squeeze(-1)
    prereq_mask = kg_matrix[q]
    margin_violation = torch.relu(LOGIC_MARGIN + target_probs.unsqueeze(-1) - probs)
    prereq_violation = margin_violation * prereq_mask
    prereq_count = prereq_mask.sum(dim=-1).clamp_min(1.0)
    logic_penalty = (prereq_violation.sum(dim=-1) / prereq_count)
    logic_penalty = logic_penalty[valid_mask].mean()
    return base_loss + logic_lambda * logic_penalty


def evaluate_metrics(name, model, kg_matrix, loader):
    """
    Return both prediction metrics and logic/path metrics.
    Added logic diagnostics:
    - PVR: Prerequisite Violation Rate
    - APC: Average Prerequisite Coverage
    - VS: Violation Severity
    - RDC: Recommendation Depth Consistency
    """
    model.eval()
    y_true, y_pred = [], []
    compliance_hits, total_checks = 0, 0
    # Edge-level logic counters.
    total_prereq_edges = 0.0
    total_prereq_violations = 0.0
    total_prereq_coverage = 0.0
    total_coverage_cases = 0.0
    total_violation_severity = 0.0
    total_depth_consistent = 0.0
    total_depth_cases = 0.0

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(DEVICE)
            q = batch["q"].to(DEVICE)
            y = batch["target"].to(DEVICE)
            user_ids = batch["user_id"].to(DEVICE)
            time_bucket = batch["time_bucket"].to(DEVICE) if USE_TIME_GAP else None

            valid_mask = masked_targets(q, y)
            seq_lengths = valid_mask.sum(dim=1)
            keep_rows = seq_lengths > 0
            if not keep_rows.any():
                continue

            time_index = torch.arange(q.size(1), device=DEVICE).unsqueeze(0).expand_as(q)
            last_positions = torch.where(valid_mask, time_index, torch.full_like(time_index, -1))
            last_idx = last_positions[keep_rows].max(dim=1).values
            q_last = q[keep_rows].gather(1, last_idx.unsqueeze(1)).squeeze(1)
            y_last = y[keep_rows].gather(1, last_idx.unsqueeze(1)).squeeze(1)

            if name == "Pure-CF":
                pred_prob = model(user_ids[keep_rows], q_last)
                full_dist = None
            else:
                outputs = (
                    model(q, x)
                    if name != "KG-SAKT"
                    else model(q, x, kg_matrix=kg_matrix, time_bucket=time_bucket)
                )
                if outputs.dim() == 2:
                    pred_prob = torch.sigmoid(outputs[keep_rows, last_idx])
                    full_dist = None
                else:
                    last_logits = outputs[keep_rows, last_idx, :]
                    full_dist = torch.sigmoid(last_logits)
                    pred_prob = full_dist.gather(1, q_last.unsqueeze(1)).squeeze(1)

            y_true.extend(y_last.cpu().tolist())
            y_pred.extend(pred_prob.cpu().tolist())

            if full_dist is not None:
                # "Recommended skill" proxy = highest predicted mastery among non-padding skills.
                rec_skills = torch.argmax(full_dist[:, 1:], dim=-1) + 1
                prereq_mask = kg_matrix[rec_skills]
                mastery_mask = full_dist >= MASTERY_THRESHOLD
                violations = (prereq_mask == 1) & (~mastery_mask)
                compliance_hits += (violations.sum(dim=1) == 0).sum().item()
                total_checks += rec_skills.size(0)

                # ---------- PVR / APC ----------
                prereq_count = prereq_mask.sum(dim=1)  # number of prereq edges for each recommended skill
                satisfied_count = ((prereq_mask == 1) & mastery_mask).sum(dim=1)
                violation_count = ((prereq_mask == 1) & (~mastery_mask)).sum(dim=1)

                total_prereq_edges += prereq_count.sum().item()
                total_prereq_violations += violation_count.sum().item()

                valid_cov = prereq_count > 0
                if valid_cov.any():
                    cov_ratio = (satisfied_count[valid_cov] / prereq_count[valid_cov]).sum().item()
                    total_prereq_coverage += cov_ratio
                    total_coverage_cases += valid_cov.sum().item()

                # ---------- VS ----------
                # Severity uses the same margin idea as training-time logic loss.
                rec_prob = full_dist.gather(1, rec_skills.unsqueeze(1))
                edge_margin_violation = torch.relu(LOGIC_MARGIN + rec_prob - full_dist) * prereq_mask
                total_violation_severity += edge_margin_violation.sum().item()

                # ---------- RDC ----------
                # If student already masters deeper skills, deeper recommendations are acceptable.
                # Otherwise recommending too deep skills is considered inconsistent.
                mastered_mask = mastery_mask[:, 1:]  # remove padding skill 0
                mastered_depth = SKILL_DEPTHS_DEVICE[1:].unsqueeze(0) * mastered_mask.float()
                mastered_count = mastered_mask.sum(dim=1)
                mastered_mean_depth = mastered_depth.sum(dim=1) / mastered_count.clamp_min(1)
                # For users with no mastered skills yet, use root-level baseline (depth 0 -> allow depth <=1).
                allowed_depth = torch.where(mastered_count > 0, mastered_mean_depth + 1.0, torch.ones_like(mastered_mean_depth))
                rec_depth = SKILL_DEPTHS_DEVICE[rec_skills]
                depth_consistent = (rec_depth <= allowed_depth).float()
                total_depth_consistent += depth_consistent.sum().item()
                total_depth_cases += rec_skills.size(0)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.5
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    comp = (compliance_hits / max(1, total_checks)) * 100 if total_checks > 0 else float("nan")
    pvr = (total_prereq_violations / max(1.0, total_prereq_edges)) * 100.0 if total_checks > 0 else float("nan")
    apc = (total_prereq_coverage / max(1.0, total_coverage_cases)) * 100.0 if total_coverage_cases > 0 else float("nan")
    vs = total_violation_severity / max(1.0, total_prereq_edges) if total_checks > 0 else float("nan")
    rdc = (total_depth_consistent / max(1.0, total_depth_cases)) * 100.0 if total_depth_cases > 0 else float("nan")
    return auc, rmse, comp, pvr, apc, vs, rdc


def create_loaders(df, n_skills):
    user_ids = df["user_id"].unique().copy()
    np.random.shuffle(user_ids)
    test_start = int(len(user_ids) * (1.0 - 0.2))
    val_start = int(test_start * (1.0 - VAL_RATIO))

    train_users = user_ids[:val_start]
    val_users = user_ids[val_start:test_start]
    test_users = user_ids[test_start:]

    train_df = df[df["user_id"].isin(train_users)].copy()
    val_df = df[df["user_id"].isin(val_users)].copy()
    test_df = df[df["user_id"].isin(test_users)].copy()

    train_dataset = Assist9Dataset(train_df, n_skills=n_skills, max_seq=MAX_SEQ)
    val_dataset = Assist9Dataset(val_df, n_skills=n_skills, max_seq=MAX_SEQ)
    test_dataset = Assist9Dataset(test_df, n_skills=n_skills, max_seq=MAX_SEQ)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader


def save_metrics_and_plots(final_results, output_dir):
    """
    Export final metrics table to CSV and draw logic-focused charts.
    Generated files:
    - logic_metrics_comparison.csv
    - logic_metrics_bar.png
    - logic_metrics_radar.png
    """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(
        final_results,
        columns=["Model", "BestEp", "AUC", "RMSE", "Path", "PVR", "APC", "VS", "RDC"],
    )
    csv_path = os.path.join(output_dir, "logic_metrics_comparison.csv")
    df.to_csv(csv_path, index=False)

    # Only models with logic outputs can be plotted on logic charts.
    logic_df = df.dropna(subset=["Path", "PVR", "APC", "VS", "RDC"]).copy()
    if logic_df.empty:
        print(f"[Export] CSV saved to: {csv_path}")
        print("[Export] No valid logic metrics to plot.")
        return

    # ---------- Bar chart ----------
    # Mixed direction metrics: Path/APC/RDC higher is better; PVR/VS lower is better.
    plot_df = logic_df.copy()
    plot_df["PVR_inv"] = -plot_df["PVR"]
    plot_df["VS_inv"] = -plot_df["VS"]
    bar_metrics = ["Path", "APC", "RDC", "PVR_inv", "VS_inv"]
    bar_labels = ["Path(+) ", "APC(+) ", "RDC(+) ", "PVR(-)", "VS(-)"]

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
    bar_path = os.path.join(output_dir, "logic_metrics_bar.png")
    plt.savefig(bar_path, dpi=200)
    plt.close()

    # ---------- Radar chart ----------
    # For radar we normalize each metric to [0, 1] and unify direction to "higher is better".
    radar_metrics = ["Path", "APC", "RDC", "PVR", "VS"]
    higher_better = {"Path": True, "APC": True, "RDC": True, "PVR": False, "VS": False}
    radar_data = {}

    for metric in radar_metrics:
        col = plot_df[metric].to_numpy(dtype=float)
        min_v, max_v = np.min(col), np.max(col)
        if np.isclose(max_v, min_v):
            norm = np.ones_like(col)
        else:
            if higher_better[metric]:
                norm = (col - min_v) / (max_v - min_v)
            else:
                norm = (max_v - col) / (max_v - min_v)
        radar_data[metric] = norm

    categories = ["Path", "APC", "RDC", "PVR(inv)", "VS(inv)"]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    for i, model_name in enumerate(plot_df["Model"].tolist()):
        values = [
            radar_data["Path"][i],
            radar_data["APC"][i],
            radar_data["RDC"][i],
            radar_data["PVR"][i],
            radar_data["VS"][i],
        ]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=model_name)
        ax.fill(angles, values, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_yticklabels([])
    ax.set_title("Logic Metrics Radar (Normalized)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()
    radar_path = os.path.join(output_dir, "logic_metrics_radar.png")
    plt.savefig(radar_path, dpi=200)
    plt.close()

    print(f"[Export] CSV saved to: {csv_path}")
    print(f"[Export] Bar chart saved to: {bar_path}")
    print(f"[Export] Radar chart saved to: {radar_path}")


def main():
    set_seed(SEED)
    torch.cuda.empty_cache()

    df = pd.read_csv(CLEAN_DATA_PATH)
    n_skills = int(df["skill_id"].max())
    n_users = int(df["user_id"].max()) + 1

    with open(KG_JSON_PATH, "r", encoding="utf-8") as f:
        kg_adj = json.load(f)
    kg_matrix = build_kg_matrix(kg_adj, n_skills, DEVICE)
    # Precompute skill depth once for RDC metric.
    skill_depths = compute_skill_depths_from_kg(kg_adj, n_skills)
    global SKILL_DEPTHS_DEVICE
    SKILL_DEPTHS_DEVICE = skill_depths.to(DEVICE)

    train_loader, val_loader, test_loader = create_loaders(df, n_skills)

    models_list = [
        ("Pure-CF", PureCFModel(n_users=n_users, n_skills=n_skills)),
        ("DKT", DKTModel(n_skills=n_skills)),
        ("SAKT", SAKTModel(n_skills=n_skills, max_seq=MAX_SEQ)),
        ("KG-SAKT", KGSAKTModel(n_skills=n_skills, kg_adj=kg_adj, max_seq=MAX_SEQ, use_time_feature=USE_TIME_GAP)),
    ]

    final_results = []

    for name, model in models_list:
        print(f"\n[Training] {name}")
        model = model.to(DEVICE)
        if name == "KG-SAKT":
            optimizer = optim.AdamW(model.parameters(), lr=KG_LEARNING_RATE, weight_decay=1e-5)
        else:
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.BCELoss() if name == "Pure-CF" else nn.BCEWithLogitsLoss()
        best_state = None
        best_epoch = 0
        best_metrics = None
        best_score = float("-inf")
        patience_counter = 0

        for epoch in range(1, EPOCHS + 1):
            model.train()
            epoch_losses = []
            current_logic_lambda = logic_lambda_for_epoch(
                epoch_idx=epoch,
                total_epochs=EPOCHS,
                max_lambda=LOGIC_LAMBDA_MAX,
                warmup_ratio=LOGIC_WARMUP_RATIO,
            )

            for batch in train_loader:
                optimizer.zero_grad()
                loss = compute_sequence_loss(
                    name=name,
                    model=model,
                    batch=batch,
                    criterion=criterion,
                    kg_matrix=kg_matrix,
                    logic_lambda=current_logic_lambda,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                epoch_losses.append(loss.item())

            val_auc, val_rmse, val_comp, val_pvr, val_apc, val_vs, val_rdc = evaluate_metrics(name, model, kg_matrix, val_loader)
            val_score = model_selection_score(name, val_auc, val_comp)
            if val_score > best_score:
                best_score = val_score
                best_epoch = epoch
                best_metrics = (val_auc, val_rmse, val_comp)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch == 1 or epoch % 5 == 0:
                comp_text = f"{val_comp:.2f}" if not np.isnan(val_comp) else "N/A"
                pvr_text = f"{val_pvr:.2f}" if not np.isnan(val_pvr) else "N/A"
                apc_text = f"{val_apc:.2f}" if not np.isnan(val_apc) else "N/A"
                vs_text = f"{val_vs:.4f}" if not np.isnan(val_vs) else "N/A"
                rdc_text = f"{val_rdc:.2f}" if not np.isnan(val_rdc) else "N/A"
                print(
                    f"  Epoch {epoch:02d} | Loss {np.mean(epoch_losses):.4f} | "
                    f"ValAUC {val_auc:.4f} | ValRMSE {val_rmse:.4f} | ValPath {comp_text} | "
                    f"ValPVR {pvr_text} | ValAPC {apc_text} | ValVS {vs_text} | ValRDC {rdc_text} | "
                    f"LogicLambda {current_logic_lambda:.4f}"
                )

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"  Early stopping at epoch {epoch:02d} | Best epoch {best_epoch:02d}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)
            # Persist best KG-SAKT checkpoint for inference simulation.
            if name == "KG-SAKT":
                torch.save(best_state, KG_MODEL_SAVE_PATH)
                print(f"  [Saved] Best KG-SAKT weights -> {KG_MODEL_SAVE_PATH}")

        best_val_auc, best_val_rmse, best_val_comp = best_metrics
        best_val_comp_text = f"{best_val_comp:.2f}" if not np.isnan(best_val_comp) else "N/A"
        print(
            f"  Best validation | Epoch {best_epoch:02d} | "
            f"AUC {best_val_auc:.4f} | RMSE {best_val_rmse:.4f} | Path {best_val_comp_text}"
        )

        final_auc, final_rmse, final_comp, final_pvr, final_apc, final_vs, final_rdc = evaluate_metrics(name, model, kg_matrix, test_loader)
        final_results.append((name, best_epoch, final_auc, final_rmse, final_comp, final_pvr, final_apc, final_vs, final_rdc))

    print("\n" + "Experiment Results".center(76, "="))
    print(
        f"{'Model':<10} | {'BestEp':<6} | {'AUC':<7} | {'RMSE':<7} | "
        f"{'Path%':<7} | {'PVR%':<7} | {'APC%':<7} | {'VS':<7} | {'RDC%':<7}"
    )
    print("-" * 76)
    for name, best_epoch, auc, rmse, comp, pvr, apc, vs, rdc in final_results:
        comp_text = f"{comp:.2f}" if not np.isnan(comp) else "N/A"
        pvr_text = f"{pvr:.2f}" if not np.isnan(pvr) else "N/A"
        apc_text = f"{apc:.2f}" if not np.isnan(apc) else "N/A"
        vs_text = f"{vs:.4f}" if not np.isnan(vs) else "N/A"
        rdc_text = f"{rdc:.2f}" if not np.isnan(rdc) else "N/A"
        print(
            f"{name:<10} | {best_epoch:<6d} | {auc:<7.4f} | {rmse:<7.4f} | "
            f"{comp_text:<7} | {pvr_text:<7} | {apc_text:<7} | {vs_text:<7} | {rdc_text:<7}"
        )
    print("=" * 76)

    # Export CSV and charts after final summary for reporting/visualization.
    save_metrics_and_plots(final_results, DATA_DIR)


if __name__ == "__main__":
    main()
