import json
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.naive_bayes import CategoricalNB

from trf_baseline.config import FEATURE_N_CATEGORIES


def evaluate_batched(exist_logp, pred_logp, pairs_list, k_values=(20, 50, 100)):
    offset = 0
    img_preds = []
    img_gts = []

    for img_pairs in pairs_list:
        n = len(img_pairs)
        if n == 0:
            img_preds.append([])
            img_gts.append(set())
            continue

        img_exist = exist_logp[offset:offset + n]
        img_plogp = pred_logp[offset:offset + n]
        offset += n

        best_pred = np.argmax(img_plogp, axis=1)
        best_conf = img_plogp[np.arange(n), best_pred]
        combined = img_exist + best_conf

        sorted_idx = np.argsort(combined)[::-1]

        preds = []
        gts = set()
        for local_idx in sorted_idx:
            h_local, t_local, h_cls, t_cls, gt_pid = img_pairs[local_idx]
            p = int(best_pred[local_idx])
            preds.append((h_local, t_local, h_cls, t_cls, p, float(combined[local_idx])))
            if gt_pid >= 0:
                gts.add((h_local, t_local, gt_pid))

        img_preds.append(preds)
        img_gts.append(gts)

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


def main():
    data = Path("/app/data/preprocessed")
    if not (data / "train_X.npy").exists():
        print("no preprocessed data. run 'make preprocess' first.")
        return

    print("loading...")
    t0 = time.time()

    # train
    X_train = np.load(data / "train_X.npy").astype(np.int32)
    y_train = np.load(data / "train_y.npy").astype(np.int32)
    exist_train = np.load(data / "train_exist.npy").astype(bool)
    load_t = time.time() - t0
    print(f"  train: {len(X_train)} pairs, {exist_train.sum()} rels ({load_t:.1f}s)")

    # fit in batches
    print("training...")
    t1 = time.time()
    B = 500000

    exist_clf = CategoricalNB(alpha=1.0, min_categories=2)
    for i in range(0, len(X_train), B):
        exist_clf.partial_fit(X_train[i:i+B], exist_train[i:i+B], classes=[0, 1])

    pos = exist_train
    pred_clf = CategoricalNB(alpha=1.0, min_categories=FEATURE_N_CATEGORIES)
    X_pos = X_train[pos]
    y_pos = y_train[pos]
    for i in range(0, len(X_pos), B):
        pred_clf.partial_fit(X_pos[i:i+B], y_pos[i:i+B], classes=list(range(50)))
    train_t = time.time() - t1
    print(f"  done ({train_t:.1f}s)")

    # free train data
    del X_train, y_train, exist_train

    # load test + eval in batches
    with open(data / "test_pairs.pkl", "rb") as f:
        test_pairs = pickle.load(f)
    print(f"  test: {len(test_pairs)} images")

    X_test = np.load(data / "test_X.npy", mmap_mode='r')
    print("inference + eval (batched)...")
    t2 = time.time()

    all_exist_logp, all_pred_logp = [], []
    B_inf = 200000
    for i in range(0, len(X_test), B_inf):
        end = min(i + B_inf, len(X_test))
        chunk = np.array(X_test[i:end], dtype=np.int32)
        all_exist_logp.append(exist_clf.predict_log_proba(chunk)[:, 1])
        all_pred_logp.append(pred_clf.predict_log_proba(chunk))

    exist_logp = np.concatenate(all_exist_logp)
    pred_logp = np.concatenate(all_pred_logp)
    infer_t = time.time() - t2

    print("evaluating...")
    metrics = evaluate_batched(exist_logp, pred_logp, test_pairs)

    results = {
        "implementation": "cpu-sklearn",
        "train_time_s": round(train_t, 3),
        "inference_time_s": round(infer_t, 4),
        "load_time_s": round(load_t, 3),
        "data_stats": {
            "train_pairs": int(len(X_test)),
            "test_images": int(len(test_pairs)),
        },
        "metrics": metrics,
        "paper_target": {"R@100": 55.34, "mR@100": 20.81},
    }

    out = Path("/app/results/latest")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "cpu_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\npaper target: R@100=55.34, mR@100=20.81")
    match = (abs(metrics.get("R@100", 0) - 55.34) < 3 and
             abs(metrics.get("mR@100", 0) - 20.81) < 3)
    print(f"match: {'close' if match else 'off — needs investigation'}")


if __name__ == "__main__":
    main()
