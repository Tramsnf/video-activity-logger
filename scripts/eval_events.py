#!/usr/bin/env python3
from __future__ import annotations
import argparse
import pandas as pd


def tiou(a_start, a_end, b_start, b_end):
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description="Evaluate event CSVs: F1 at tIoU and MAE of times")
    ap.add_argument("--pred", required=True, help="Predicted events CSV")
    ap.add_argument("--gt", required=True, help="Ground-truth events CSV")
    ap.add_argument("--activities", nargs="*", default=None, help="Activities to evaluate (default: all in GT)")
    ap.add_argument("--tiou", type=float, default=0.5)
    ap.add_argument("--by_actor", action="store_true", help="Match only within the same actor_id")
    args = ap.parse_args()

    pred = pd.read_csv(args.pred)
    gt = pd.read_csv(args.gt)
    acts = args.activities or sorted(set(gt["activity"].unique()))

    total_tp = total_fp = total_fn = 0
    maes = []
    for act in acts:
        p = pred[pred["activity"] == act].copy()
        g = gt[gt["activity"] == act].copy()
        matched_g = set()
        tp = fp = 0
        for i, pr in p.iterrows():
            candidates = g
            if args.by_actor and "actor_id" in pr and "actor_id" in g.columns:
                candidates = g[g["actor_id"] == pr["actor_id"]]
            best_j = None
            best_iou = 0.0
            for j, gt_row in candidates.iterrows():
                if j in matched_g:
                    continue
                iou = tiou(pr["start_time_s"], pr["end_time_s"], gt_row["start_time_s"], gt_row["end_time_s"])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= args.tiou and best_j is not None:
                tp += 1
                matched_g.add(best_j)
                gt_row = candidates.loc[best_j]
                # timestamp MAE per boundary
                maes.append(abs(pr["start_time_s"] - gt_row["start_time_s"]))
                maes.append(abs(pr["end_time_s"] - gt_row["end_time_s"]))
            else:
                fp += 1
        fn = len(g) - len(matched_g)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    mae = sum(maes) / len(maes) if maes else 0.0
    print(f"Activities: {acts}")
    print(f"TP={total_tp} FP={total_fp} FN={total_fn}")
    print(f"Precision={prec:.3f} Recall={rec:.3f} F1={f1:.3f} tIoU={args.tiou}")
    print(f"Boundary MAE={mae:.3f}s (averaged over matched start/end)")


if __name__ == "__main__":
    main()

