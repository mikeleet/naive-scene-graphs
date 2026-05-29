"""run all feature combinations against paper targets."""

import json
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.naive_bayes import CategoricalNB

from trf_baseline.config import FEATURE_N_CATEGORIES


def evaluate(exist_logp, pred_logp, pairs_list, k_values=(20, 50, 100)):
    offset = 0
    img_preds, img_gts = [], []

    for img_pairs in pairs_list:
        n = len(img_pairs)
        if n == 0:
            img_preds.append([]), img_gts.append(set())
            continue
        img_exist = exist_logp[offset:offset + n]
        img_plogp = pred_logp[offset:offset + n]
        offset += n
        best_pred = np.argmax(img_plogp, axis=1)
        best_conf = img_plogp[np.arange(n), best_pred]
        combined = img_exist + best_conf
        sorted_idx = np.argsort(combined)[::-1]

        preds, gts = [], set()
        for idx in sorted_idx:
            h_local, t_local, h_cls, t_cls, gt_pid = img_pairs[idx]
            p = int(best_pred[idx])
            preds.append((h_local, t_local, h_cls, t_cls, p, float(combined[idx])))
            if gt_pid >= 0:
                gts.add((h_local, t_local, gt_pid))
        img_preds.append(preds), img_gts.append(gts)

    total_gt = sum(len(g) for g in img_gts)
    r_hits = {k: 0 for k in k_values}
    for preds, gts in zip(img_preds, img_gts):
        if not gts:
            continue
        for k in k_values:
            found = set()
            for rank, (h, t, hc, tc, p, _) in enumerate(preds[:k]):
                triplet = (h, t, p)
                if triplet in gts and triplet not in found:
                    found.add(triplet)
            r_hits[k] += len(found)
    r = {f"R@{k}": round(r_hits[k] / max(total_gt, 1) * 100, 2) for k in k_values}

    num_preds = 50
    per_pred_gt = {p: 0 for p in range(num_preds)}
    per_pred_hits = {k: {p: 0 for p in range(num_preds)} for k in k_values}
    for gts in img_gts:
        for h, t, p in gts:
            per_pred_gt[p] = per_pred_gt.get(p, 0) + 1
    for preds, gts in zip(img_preds, img_gts):
        for k in k_values:
            found = set()
            for rank, (h, t, hc, tc, p, _) in enumerate(preds[:k]):
                triplet = (h, t, p)
                if triplet in gts and triplet not in found:
                    found.add(triplet)
                    per_pred_hits[k][p] = per_pred_hits[k].get(p, 0) + 1
    mr = {}
    for k in k_values:
        vals = [per_pred_hits[k][p] / max(per_pred_gt[p], 1)
                for p in range(num_preds) if per_pred_gt[p] > 0]
        mr[f"mR@{k}"] = round(np.mean(vals) * 100, 2) if vals else 0.0

    return {**r, **mr}


def train_and_eval(X_train, y_train, exist_train, X_test, test_pairs,
                   feature_indices, cats_subset, name, paper_targets):
    print(f"\n--- {name} (features {feature_indices}) ---")

    t1 = time.time()
    B = 500000

    exist_clf = CategoricalNB(alpha=1.0, min_categories=2)
    for i in range(0, len(X_train), B):
        exist_clf.partial_fit(X_train[i:i+B, feature_indices],
                              exist_train[i:i+B], classes=[0, 1])

    pos = exist_train
    pred_clf = CategoricalNB(alpha=1.0, min_categories=cats_subset)
    X_pos = X_train[pos][:, feature_indices]
    y_pos = y_train[pos]
    for i in range(0, len(X_pos), B):
        pred_clf.partial_fit(X_pos[i:i+B], y_pos[i:i+B],
                             classes=list(range(50)))
    train_t = time.time() - t1

    all_elogp, all_plogp = [], []
    B_inf = 200000
    t2 = time.time()
    for i in range(0, len(X_test), B_inf):
        end = min(i + B_inf, len(X_test))
        chunk = np.array(X_test[i:end, feature_indices], dtype=np.int32)
        all_elogp.append(exist_clf.predict_log_proba(chunk)[:, 1])
        all_plogp.append(pred_clf.predict_log_proba(chunk))
    exist_logp = np.concatenate(all_elogp)
    pred_logp = np.concatenate(all_plogp)
    infer_t = time.time() - t2

    metrics = evaluate(exist_logp, pred_logp, test_pairs)

    r100 = metrics.get("R@100", 0)
    mr100 = metrics.get("mR@100", 0)
    pr100 = paper_targets.get("R@100", 0)
    pmr100 = paper_targets.get("mR@100", 0)

    print(f"  R@100: {r100} (paper {pr100}, delta {r100-pr100:+.2f})")
    print(f"  mR@100: {mr100} (paper {pmr100}, delta {mr100-pmr100:+.2f})")
    print(f"  train: {train_t:.1f}s, infer: {infer_t:.1f}s")

    return {"R@100": r100, "mR@100": mr100, "train_s": round(train_t, 1),
            "infer_s": round(infer_t, 1)}


def main():
    data = Path("/app/data/preprocessed")
    X_train = np.load(data / "train_X.npy").astype(np.int32)
    y_train = np.load(data / "train_y.npy").astype(np.int32)
    exist_train = np.load(data / "train_exist.npy").astype(bool)
    X_test = np.load(data / "test_X.npy", mmap_mode='r')

    with open(data / "test_pairs.pkl", "rb") as f:
        test_pairs = pickle.load(f)

    feature_names = {
        "class-only": ([0, 1], [FEATURE_N_CATEGORIES[0], FEATURE_N_CATEGORIES[1]]),
        "class+topology": ([0, 1, 2, 3],
                           [FEATURE_N_CATEGORIES[0], FEATURE_N_CATEGORIES[1],
                            FEATURE_N_CATEGORIES[2], FEATURE_N_CATEGORIES[3]]),
        "class+topo+area": ([0, 1, 2, 3, 4, 5, 6],
                            [FEATURE_N_CATEGORIES[i] for i in range(7)]),
    }

    paper = {
        "class-only": {"R@100": 50.41, "mR@100": 16.58},
        "class+topology": {"R@100": 57.37, "mR@100": 20.62},
        "class+topo+area": {"R@100": 55.34, "mR@100": 20.81},
    }

    results = {}
    for name, (feat_idx, cats) in feature_names.items():
        results[name] = train_and_eval(
            X_train, y_train, exist_train, X_test, test_pairs,
            feat_idx, cats, name, paper[name]
        )

    print("\n" + "=" * 50)
    print("summary")
    print("=" * 50)
    for name, r in results.items():
        pt = paper[name]
        r_d = r["R@100"] - pt["R@100"]
        m_d = r["mR@100"] - pt["mR@100"]
        status = "✓" if abs(r_d) < 4 and abs(m_d) < 3 else "✗"
        print(f"  {name}: R@100={r['R@100']} ({r_d:+.1f}) | "
              f"mR@100={r['mR@100']} ({m_d:+.2f}) | {status}")

    out = Path("/app/results/latest")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "feature_combinations.json", "w") as f:
        json.dump({"results": results, "paper_targets": paper}, f, indent=2)
    print(f"\nsaved to {out / 'feature_combinations.json'}")


if __name__ == "__main__":
    main()
