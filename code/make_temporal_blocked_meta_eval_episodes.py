
import argparse
import json
from pathlib import Path

import pandas as pd

SEQ_CANDIDATES = [
    "seq_index", "sequence_index", "seq_id", "sequence_id",
    "trajectory_id", "track_id", "task_id"
]

def find_seq_col(df):
    lower = {c.lower(): c for c in df.columns}
    for cand in SEQ_CANDIDATES:
        if cand in lower:
            return lower[cand]
    raise ValueError(
        f"Could not detect task/sequence column. Available columns: {list(df.columns)}"
    )

def pick_nested_supports(support_pool):
    s10 = support_pool[:10]
    idx = [0, 2, 4, 6, 9]
    s5 = s10[idx]
    return s5, s10

def main():
    p = argparse.ArgumentParser(
        description="Build temporally blocked paired K=5/K=10 Meta-Val episodes."
    )
    p.add_argument("--csv", required=True)
    p.add_argument("--split_name", required=True)
    p.add_argument("--min_gap", type=int, default=11)
    p.add_argument("--query_size", type=int, default=20)
    p.add_argument("--out_dir", default="blocked_eval_episodes")
    args = p.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path).reset_index(drop=True)
    seq_col = find_seq_col(df)

    episodes = {5: {}, 10: {}}
    audit_rows = []
    skipped = []

    for task_value, g in df.groupby(seq_col, sort=True):
        task_rows = g.index.to_numpy()
        n = len(task_rows)

        first_query_pos = 9 + args.min_gap

        if n <= first_query_pos:
            skipped.append({
                "task": str(task_value),
                "n_windows": int(n),
                "reason": f"not enough windows for K=10 with min_gap={args.min_gap}"
            })
            continue

        support_pool = task_rows[:10]
        support5, support10 = pick_nested_supports(support_pool)

        query_candidates = task_rows[first_query_pos:]
        query = query_candidates[:min(args.query_size, len(query_candidates))]

        if len(query) == 0:
            skipped.append({
                "task": str(task_value),
                "n_windows": int(n),
                "reason": "empty query after temporal gap"
            })
            continue

        for k, support in [(5, support5), (10, support10)]:
            episodes[k][str(task_value)] = {
                "support_rows": [int(x) for x in support],
                "query_rows": [int(x) for x in query],
                "n_windows": int(n),
                "k": k,
                "query_size": int(len(query)),
                "temporal_blocked": True,
                "min_gap": int(args.min_gap),
            }

        rel = {row: pos for pos, row in enumerate(task_rows)}
        s5_pos = [rel[x] for x in support5]
        s10_pos = [rel[x] for x in support10]
        q_pos = [rel[x] for x in query]

        min5 = min(abs(q - s) for q in q_pos for s in s5_pos)
        min10 = min(abs(q - s) for q in q_pos for s in s10_pos)

        audit_rows.append({
            "task": str(task_value),
            "n_windows": int(n),
            "k5_support_n": len(support5),
            "k10_support_n": len(support10),
            "query_n": len(query),
            "k5_subset_of_k10": set(support5).issubset(set(support10)),
            "same_query_k5_k10": True,
            "min_distance_k5": int(min5),
            "min_distance_k10": int(min10),
            "passes_gap_k5": bool(min5 >= args.min_gap),
            "passes_gap_k10": bool(min10 >= args.min_gap),
        })

    audit = pd.DataFrame(audit_rows)

    for k in (5, 10):
        payload = {
            "protocol_version": 3,
            "temporal_blocked": True,
            "source_csv": str(csv_path).replace("\\", "/"),
            "split_name": args.split_name,
            "task_column": seq_col,
            "k": k,
            "min_gap": args.min_gap,
            "query_size_requested": args.query_size,
            "support_policy": (
                "chronological early support; K=5 nested in K=10; "
                "both include the same latest support position"
            ),
            "query_policy": "same later chronological query block for K=5 and K=10",
            "num_tasks": len(episodes[k]),
            "episodes": episodes[k],
            "skipped_tasks": skipped,
        }
        out = out_dir / f"{args.split_name}_blocked_k{k}.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved K={k}: {out}")

    audit_path = out_dir / f"{args.split_name}_blocked_audit.csv"
    audit.to_csv(audit_path, index=False)

    print("\n=== TEMPORAL-BLOCKED CHECKS ===")
    print(f"Total original tasks: {df[seq_col].nunique()}")
    print(f"Episodes created: {len(audit)}")
    print(f"Skipped tasks: {len(skipped)}")
    if skipped:
        print("Skipped task IDs:", [x["task"] for x in skipped])

    if len(audit):
        print(f"Query size range: {audit['query_n'].min()}..{audit['query_n'].max()}")
        print(f"All K5 subset of K10: {bool(audit['k5_subset_of_k10'].all())}")
        print(f"Same queries for K5/K10: {bool(audit['same_query_k5_k10'].all())}")
        print(f"Minimum K5 support-query distance: {audit['min_distance_k5'].min()}")
        print(f"Minimum K10 support-query distance: {audit['min_distance_k10'].min()}")
        print(f"All K5 pass min_gap: {bool(audit['passes_gap_k5'].all())}")
        print(f"All K10 pass min_gap: {bool(audit['passes_gap_k10'].all())}")

    print(f"Saved audit: {audit_path}")
    print("\nNOTE: This assumes row order within each seq_index is chronological and")
    print("consecutive rows correspond to consecutive stride-1 windows.")

if __name__ == "__main__":
    main()