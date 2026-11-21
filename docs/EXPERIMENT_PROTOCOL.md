# Experiment Protocol

## Clean split
- Original training pool: 108 trajectories.
- One training trajectory was unsuitable for the initial few-shot construction.
- Usable training tasks: 107.
- Fixed-seed split: 86 Meta-Train / 21 Meta-Val.
- Original 28 held-out test trajectories remain the protected Meta-Test.

## Temporal leakage correction
Each sample spans 8 observations + 3 future targets = 11 temporal positions. Adjacent sliding windows can therefore reuse almost the same raw temporal information.

Final evaluation uses temporal-blocked episodes with minimum Support–Query distance = 11. Earlier random-window few-shot numbers are exploratory only.

## Fixed adaptation protocol
- K=5.
- inner_lr=0.0005.
- inner_steps=3.
- Primary metric: Macro Top-1.
- Secondary metric: Micro Top-1.
- Same episode JSON and evaluator for all methods.

## Tuning discipline
Inner-loop sweep on clean No-KD Meta-Val only:
- inner_lr in {1e-5, 5e-5, 1e-4, 5e-4}.
- inner steps in {1, 3}.
- selected: 0.0005 and 3 steps.

Meta learning-rate sweep:
- 1e-6 -> 54.18% after adaptation.
- 5e-6 -> 54.28%.
- 1e-5 -> 54.52%.
- selected: meta_lr=1e-5.

Meta-training:
- 5 meta-epochs max.
- best checkpoint chosen by internal Meta-Val query loss.
- official claims use the common fixed blocked evaluator.

## Protected Meta-Test
- 28 unseen tasks, IDs 109–136.
- 298 Query windows.
- Query-size range: 3–19 per task.
- 0 tasks skipped.
- Minimum Support–Query distance = 11.
- No retuning after opening Test.
