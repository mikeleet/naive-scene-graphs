import numpy as np
from typing import Callable


def compute(
    preds: list[list[tuple]],  # per-image: [(pred_idx, conf), ...]
    gt: list[set[tuple[int, int]]],  # per-image: set of (pred, subj_cls)
    k_values: tuple = (20, 50, 100),
    mode: str = "R",
) -> dict[str, float]:
    num_preds = 50
    preds = [sorted(p, key=lambda x: -x[1]) for p in preds]

    if mode == "mR":
        # compute per predicate
        hits = {k: {p: 0 for p in range(num_preds)} for k in k_values}
        totals = {p: 0 for p in range(num_preds)}

        for img_preds, img_gt in zip(preds, gt):
            gt_copy = set(img_gt)
            for p_idx, _ in img_preds:
                for t in list(gt_copy):
                    if t[0] == p_idx:
                        totals[t[0]] = totals.get(t[0], 0) + 1
                        for k in k_values:
                            if len(gt_copy) <= len(img_preds):
                                hits[k][t[0]] = hits[k].get(t[0], 0) + 1
                        gt_copy.discard(t)
                        break

        result = {}
        for k in k_values:
            vals = [hits[k][p] / max(totals[p], 1) for p in range(num_preds) if totals[p] > 0]
            result[f"mR@{k}"] = round(np.mean(vals) * 100, 2) if vals else 0.0
        return result

    # standard R@K
    total = sum(len(g) for g in gt)
    hits = {k: 0 for k in k_values}

    for img_preds, img_gt in zip(preds, gt):
        gt_remaining = set(img_gt)
        for rank, (p_idx, _) in enumerate(img_preds):
            for t in list(gt_remaining):
                if t[0] == p_idx:
                    for k in k_values:
                        if rank < k:
                            hits[k] += 1
                    gt_remaining.discard(t)
                    break

    return {f"{mode}@{k}": round(hits[k] / max(total, 1) * 100, 2) for k in k_values}
