"""Held-out permutation dependence for all 45 sensory targets.

The original five splits and tree parameters are retained. Permutations act
on validation inputs only, including each feature's missing indicator.
Correlated modules are learned separately from each training fold.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
from common import SEED, data_dir, load_xy

OUT = HERE / "results/q2_permutation"
REPEATS = 5


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def components(training):
    correlation = training.corr(method="spearman").fillna(0).to_numpy()
    adjacency = np.abs(correlation) >= .7
    remaining = set(range(training.shape[1]))
    groups = []
    while remaining:
        current = [min(remaining)]
        remaining.remove(current[0])
        group = []
        while current:
            j = current.pop()
            group.append(j)
            neighbors = sorted(remaining.intersection(np.flatnonzero(adjacency[j])))
            remaining.difference_update(neighbors)
            current.extend(neighbors)
        groups.append(sorted(group))
    assert sorted(j for group in groups for j in group) == list(range(training.shape[1]))
    return groups


def predict_permutations(tree, base, groups, columns, permutation, truth, baseline):
    result = np.empty((len(groups), truth.shape[1]))
    for start in range(0, len(groups), 32):
        batch = groups[start:start + 32]
        stacked = np.tile(base, (len(batch), 1))
        n = len(base)
        for k, group in enumerate(batch):
            selected = [col for j in group for col in columns[j]]
            stacked[k*n:(k+1)*n, selected] = base[np.ix_(permutation, selected)]
        prediction = tree.predict(stacked).reshape(len(batch), n, truth.shape[1])
        result[start:start + len(batch)] = ((prediction - truth) ** 2).mean(axis=1) - baseline
    return result


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    q1 = load_module("q1_permutation_baseline", "02_q1_baseline.py")
    q2 = load_module("q2_permutation_candidates", "03_q2_network.py")
    meta, x, y = load_xy()
    features, targets = list(x), list(y)
    candidates = [j for j, feature in enumerate(features) if feature not in q2.EXCLUDE_X]
    assert len(candidates) == 218
    y_values = y.to_numpy()
    delta = np.empty((5, REPEATS, len(features), len(targets)))
    oof = np.full(y.shape, np.nan)
    fold_ids = np.zeros(len(x), dtype=int)
    fold_sizes = []
    module_rows, membership = [], []
    check_features = ["ethyl_acetate", "X4VG", "ethanol..v.v."]
    check_predictions = np.empty((5, REPEATS, len(check_features), 50, len(targets)))
    check_rows = []
    original = pd.read_csv(ROOT / "outputs/q1/oof_extra_trees.csv")
    assert original.beer.tolist() == meta.beer.tolist()
    for fold, (train, test) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(x)):
        fold_ids[test] = fold + 1
        fold_sizes.append(len(test))
        assert original.iloc[test].fold.eq(fold + 1).all()
        model = q1.tree_pipeline(SEED + fold + 1).fit(x.iloc[train], y.iloc[train])
        imputer, tree = model.named_steps["imputer"], model.named_steps["model"]
        np.testing.assert_allclose(imputer.statistics_, np.nanmedian(x.iloc[train], axis=0))
        base = imputer.transform(x.iloc[test])
        oof[test] = tree.predict(base)
        truth = y_values[test]
        baseline = ((truth - oof[test]) ** 2).mean(axis=0)
        columns = [[j] for j in range(len(features))]
        for k, j in enumerate(imputer.indicator_.features_):
            columns[j].append(len(features) + k)
        groups = components(x.iloc[train])
        group_delta = np.empty((REPEATS, len(groups), len(targets)))
        for g, group in enumerate(groups):
            membership.append({"fold": fold + 1, "module": g + 1, "size": len(group),
                               "members": " | ".join(features[j] for j in group)})
        for repeat in range(REPEATS):
            permutation = np.random.default_rng(SEED + 50000 + 100*fold + repeat).permutation(len(test))
            delta[fold, repeat] = predict_permutations(
                tree, base, [[j] for j in range(len(features))], columns, permutation, truth, baseline)
            group_delta[repeat] = predict_permutations(
                tree, base, groups, columns, permutation, truth, baseline)
            for k, feature in enumerate(check_features):
                j = features.index(feature)
                raw = x.iloc[test].copy()
                raw.loc[:, feature] = x.iloc[test].iloc[permutation][feature].to_numpy()
                predicted = model.predict(raw)
                check_predictions[fold, repeat, k] = predicted
                independent = ((predicted - truth) ** 2).mean(axis=0) - baseline
                np.testing.assert_allclose(independent, delta[fold, repeat, j], rtol=1e-10, atol=1e-12)
        for g, group in enumerate(groups):
            for k, target in enumerate(targets):
                module_rows.append({"fold": fold + 1, "module": g + 1, "size": len(group),
                                    "target": target, "delta_mse": group_delta[:, g, k].mean(),
                                    "repeat_sd": group_delta[:, g, k].std(ddof=1)})
        check_rows.append(test)
        print(f"Permutation fold {fold + 1}/5: {len(groups)} training modules; held-out rows={len(test)}", flush=True)
    assert np.isfinite(delta).all() and np.isfinite(oof).all()
    by_fold = delta.mean(axis=1)
    mean = np.average(by_fold, axis=0, weights=fold_sizes)
    sd = by_fold.std(axis=0, ddof=1)
    positive = (by_fold > 0).sum(axis=0)
    overall, fold_rows, top_rows, comparisons = [], [], [], []
    original_top = pd.read_csv(ROOT / "outputs/q2/top3_compounds.csv")
    for k, target in enumerate(targets):
        ordered = sorted(candidates, key=lambda j: (-mean[j, k], j))
        ranks = {j: rank for rank, j in enumerate(ordered, 1)}
        chosen = ordered[:3]
        existing = original_top.loc[original_top.sensory_target.eq(target)].sort_values("rank").compound.tolist()
        chosen_names = [features[j] for j in chosen]
        common = set(chosen_names) & set(existing)
        comparisons.append({"sensory_target": target, "spearman_top3": " | ".join(existing),
                            "permutation_top3": " | ".join(chosen_names), "overlap": len(common),
                            "jaccard": len(common) / len(set(chosen_names) | set(existing)),
                            "all_three_positive_mean": bool((mean[chosen, k] > 0).all())})
        for j, feature in enumerate(features):
            row = {"sensory_target": target, "feature": feature, "chemical_candidate": j in ranks,
                   "candidate_rank": ranks.get(j), "delta_mse": mean[j, k],
                   "fold_sd": sd[j, k], "positive_folds": int(positive[j, k])}
            overall.append(row)
            if j in chosen:
                top_rows.append(row)
            for fold in range(5):
                fold_rows.append({"fold": fold + 1, "sensory_target": target, "feature": feature,
                                  "delta_mse": by_fold[fold, j, k],
                                  "repeat_sd": delta[fold, :, j, k].std(ddof=1)})
    for name, rows in [("importance", overall), ("importance_by_fold", fold_rows),
                       ("top3", top_rows), ("top3_comparison", comparisons),
                       ("module_importance_by_fold", module_rows), ("module_membership", membership)]:
        pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False, lineterminator="\n", float_format="%.12g")
    pd.concat([meta, pd.DataFrame({"fold": fold_ids}), pd.DataFrame(oof, columns=targets)], axis=1).to_csv(
        OUT / "oof.csv", index=False, lineterminator="\n")
    np.savez_compressed(OUT / "permutation_draws.npz", delta_mse=delta,
                        fold_ids=fold_ids, features=np.array(features), targets=np.array(targets),
                        check_predictions=check_predictions, check_features=np.array(check_features),
                        check_test_rows=np.array(check_rows))
    summary = {"scope": "Supplementary post-hoc model-dependence analysis; not causal effects or a new test dataset.",
               "seed": SEED, "folds": 5, "repeats": REPEATS, "features": len(features), "targets": len(targets),
               "chemical_candidates": len(candidates), "rmse": q1.scores(y_values, oof)["rmse"],
               "frozen_oof_max_abs_difference": float(np.max(np.abs(oof-original[targets].to_numpy()))),
               "frozen_oof_rmse": q1.scores(y_values, original[targets].to_numpy())["rmse"],
               "modules_per_fold": [sum(r["fold"] == f for r in membership) for f in range(1, 6)],
               "mean_jaccard_with_spearman": float(np.mean([r["jaccard"] for r in comparisons])),
               "targets_with_no_top3_overlap": sum(r["overlap"] == 0 for r in comparisons),
               "targets_with_all_positive_top3_means": sum(r["all_three_positive_mean"] for r in comparisons),
               "input_sha256": {name: hashlib.sha256((data_dir()/name).read_bytes()).hexdigest() for name in ["1.csv", "4.csv"]},
               "elapsed_seconds": time.perf_counter() - started,
               "limitations": ["Same development beers; follow-up designed after viewing the main results.",
                               "Correlated and derived inputs can substitute for each other; individual permutation may break chemical consistency.",
                               "Training-fold correlation modules differ across folds and are reported separately, not pooled as identical objects.",
                               "Positive folds and fold SD describe stability, not independent significance tests or causal contributions."]}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
