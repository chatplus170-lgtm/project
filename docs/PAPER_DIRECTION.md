# Paper Direction

## Current publishable story
The work shows that two goals do not automatically align:
1. **Representation strength** — a strong starting predictor.
2. **Few-shot adaptability** — the ability to improve from a small support set on a new physical trajectory.

Protected-test evidence shows:
- KD-init MAML has the strongest numerical starting/final representation.
- MAML-KD with lambda=0.05 has the strongest positive adaptation.
- No tested method maximizes both.

## Methodological contribution already established
- Trajectory-level meta-learning task definition.
- Discovery of temporal leakage in random Support/Query windows.
- Clean 86/21/28 split and clean retraining.
- Fixed blocked evaluation with gap=11.
- Controlled inner-loop and meta-LR selection.
- Protected Meta-Test with no retuning after opening.
- Per-task adaptation analysis.

## Strong next-paper question
How can a method preserve the strong KD representation while applying adaptation only when the support evidence is expected to help?

Candidate direction for supervisor discussion:
- KD initialization.
- Representation-preserving or selective inner adaptation.
- Confidence/task-aware gating.
- Fresh external held-out scenario for final evaluation.

This is a proposed next direction, not a result of the current experiments.
