# ISAC Beam Tracking with Knowledge Distillation and MAML

Research repository for leakage-safe few-shot adaptation in multimodal sensing-aided mmWave beam tracking.

## Research question
Can a lightweight multimodal Student preserve a strong distilled representation while adapting rapidly to an unseen physical trajectory from only a few support samples?

## Core pipeline
- Reference Teacher–Student KD framework with Camera + FMCW Radar.
- One `seq_index` is one physical trajectory / meta-learning task.
- Clean split: 86 Meta-Train, 21 Meta-Val, 28 protected Meta-Test tasks.
- Temporal-blocked Support/Query evaluation with minimum gap = 11.
- Full second-order MAML with K=5 support samples.
- Fixed inner loop: `inner_lr=0.0005`, `inner_steps=3`.
- Selected outer learning rate: `meta_lr=1e-5`.
- Meta-KD ablation: `lambda_kd ∈ {0, 0.05, 0.10}`.
- Final protected test: 28 tasks / 298 Query windows, same episodes for all methods.

## Main finding
The final experiments reveal a **representation–adaptability trade-off**:
- KD-initialized MAML has the highest numerical final Macro Top-1 on protected Meta-Test: **49.13%**.
- MAML-KD with `lambda_kd=0.05` has the strongest positive adaptation gain: **+0.54 pp Macro**.
- MAML-only was the Meta-Val winner but its adaptation became negative on protected Meta-Test.
- Friedman test across six methods and 28 test tasks: `p=0.678`; no universal statistically proven winner is claimed.

## Repository map
- `docs/MAML_ARCHITECTURE.md` — exact meta-learning flow, losses, initialization, and update paths.
- `docs/EXPERIMENT_PROTOCOL.md` — clean split, leakage control, fixed adaptation budget, and selection discipline.
- `docs/RESULTS_SUMMARY.md` — Meta-Val and protected Meta-Test results.
- `docs/SUPERVISOR_REVIEW_CHECKLIST.md` — architecture decisions to review before a paper-specific next experiment.
- `docs/PAPER_DIRECTION.md` — current publishable story and next research question.
- `code/CODE_MANIFEST.md` — exact final scripts that must be copied from the experiment machine.

## Reproducibility rule
The original held-out Meta-Test was not used for hyperparameter tuning. Once it was opened, its results were analyzed only; no learning-rate, KD-weight, checkpoint-selection, epoch-count, or episode-construction setting was changed afterward.
