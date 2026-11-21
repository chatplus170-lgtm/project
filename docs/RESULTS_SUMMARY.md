# Results Summary

## Meta-Val: fixed blocked K=5
| Method | Macro before | Macro after | Delta |
|---|---:|---:|---:|
| No-KD + FT | 52.59% | 53.99% | +1.40 pp |
| KD + FT | 55.01% | 54.60% | -0.41 pp |
| MAML-only | 53.30% | 54.80% | +1.50 pp |
| MAML-KD, lambda=0.05 | 52.82% | 54.52% | +1.70 pp |
| MAML-KD, lambda=0.10 | 52.18% | 53.46% | +1.28 pp |
| KD-init MAML | 54.49% | 54.33% | -0.16 pp |

## Protected Meta-Test
| Method | Macro before | Macro after | Delta Macro | Micro after |
|---|---:|---:|---:|---:|
| No-KD + FT | 48.20% | 48.31% | +0.11 pp | 50.17% |
| KD + FT | 48.85% | 48.54% | -0.31 pp | 48.74% |
| MAML-only | 47.64% | 47.16% | -0.48 pp | 48.99% |
| MAML-KD, lambda=0.05 | 46.91% | 47.44% | +0.54 pp | 49.16% |
| MAML-KD, lambda=0.10 | 46.95% | 46.88% | -0.07 pp | 48.74% |
| KD-init MAML | 49.43% | 49.13% | -0.30 pp | 49.66% |

Main numerical findings:
- Highest final Macro: KD-init MAML = 49.13%.
- Highest starting Macro: KD-init MAML = 49.43%.
- Strongest positive adaptation: MAML-KD lambda=0.05 = +0.54 pp Macro.
- MAML-only changed from +1.50 pp adaptation on Meta-Val to -0.48 pp on Meta-Test.
- Friedman test over 6 methods × 28 tasks: p=0.678.

## Per-task adaptation
| Method | Improved | Tied | Worsened |
|---|---:|---:|---:|
| No-KD + FT | 6 | 18 | 4 |
| KD + FT | 2 | 22 | 4 |
| MAML-only | 5 | 16 | 7 |
| MAML-KD, lambda=0.05 | 9 | 15 | 4 |
| MAML-KD, lambda=0.10 | 5 | 19 | 4 |
| KD-init MAML | 4 | 19 | 5 |

The evidence supports a representation–adaptability trade-off, not a universal statistically proven winner.
