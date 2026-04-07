# Algorithm Overview

## 1. Goal

Build a Knowledge Graph enhanced SAKT framework for educational recommendation that is:

- Accurate on next-response prediction.
- Consistent with prerequisite learning logic.

## 2. Baseline Evolution

- `Pure-CF`: matrix factorization baseline, no temporal modeling.
- `DKT`: LSTM-based knowledge tracing.
- `SAKT`: self-attention based sequence modeling.
- `KG-SAKT` (target model): SAKT + KG structural constraints.

## 3. KG-SAKT Core Design

### 3.1 Sequence Encoder

- Encode interaction sequence (`skill_id`, correctness) with embedding + positional encoding.
- Use causal self-attention to model historical influence.

### 3.2 KG Context Injection

- Build KG adjacency matrix over skills.
- Compute graph context from one-hop and two-hop prerequisite diffusion.
- Mix one-hop and two-hop context via a learnable scalar gate.

### 3.3 KG Gate Fusion

- Fuse attention output, query embedding, and KG context through a gate network.
- Output full mastery logits for all skills at each time step.

### 3.4 Logic Consistency Loss (Margin-based)

For each target skill `q_t` and its prerequisites `p`:

- Encourage `P(p) >= P(q_t) - m`, where `m` is margin.
- Penalize violations with hinge form:

`max(0, m + P(q_t) - P(p))`

This aligns model behavior with educational prerequisite ordering.

## 4. Training Strategy

- User-level split: train/val/test.
- Validation-driven early stopping.
- Best checkpoint restore before test reporting.
- Dynamic logic-loss weight schedule: early epochs prioritize data fitting, later epochs increase logic alignment pressure.

## 5. Evaluation

- `AUC`: next-response discrimination quality.
- `RMSE`: calibration/fit error.
- `Path Compliance`: whether recommended skills satisfy KG prerequisites.

## 6. About Time Gap

- Time Gap is not part of default mainline optimization.
- It remains an optional ablation feature.
- Default setting keeps `USE_TIME_GAP = False` to focus on pure KG contribution.

## 7. Optimization Roadmap

- Add recommendation-oriented ranking objective on top of mastery prediction.
- Replace static KG diffusion with lightweight GNN message passing.
- Add uncertainty-aware recommendation confidence for safer learning path suggestions.
