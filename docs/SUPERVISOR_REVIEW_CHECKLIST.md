# Supervisor Review Checklist

## Current architecture decisions
- Lightweight Camera CNN + Radar CNN -> fusion -> single-layer GRU -> classifier.
- One physical trajectory (`seq_index`) = one meta-learning task.
- K=5, inner_lr=0.0005, 3 inner steps.
- Full second-order MAML.
- Teacher guidance for MAML-KD is in the Query meta-objective.
- lambda_kd = 0 / 0.05 / 0.10.
- KD-init MAML starts from the clean KD Student and uses lambda_kd=0 during meta-training.
- Clean frozen Teacher for all lambda_kd > 0 runs.
- Temporal Support–Query gap = 11.

## Review questions
- Should all Student layers adapt, or only selected layers / classifier / GRU?
- Should inner_lr be fixed, layer-wise, or learned?
- Should KD remain only in the outer/query objective?
- Should KD initialization be protected by a representation-preserving regularizer during adaptation?
- Should adaptation be confidence-gated / task-aware?
- Which new external scenario or untouched split should be reserved for the next method?

## Publication caution
The current protected Meta-Test has already been opened. A new architecture should be developed on Meta-Train/Meta-Val and evaluated on a fresh held-out scenario or newly reserved test split.
