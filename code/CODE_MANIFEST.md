# Code Manifest

Final publication repository should contain the exact scripts used for the clean experiments:

- train_maml_kd_v4.py
- make_temporal_blocked_meta_eval_episodes.py
- make_clean_meta_csvs.py
- maml_task_utils.py
- model_both.py
- final clean Teacher training script
- train_clean_student_nokd.py
- train_clean_student_kd.py
- diagnose_fixed_blocked_adaptation.py

Do **not** reconstruct missing scripts from descriptions. Copy the exact final versions used for the reported experiments and commit/tag them before paper submission.

Large datasets, checkpoints, blocked episode JSONs, result CSVs, PPTX/PDF files, and experiment binaries should stay in the Drive package unless a public-release decision is made.
