"""Read-only numerical checks for the supplementary analyses."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
from common import SEED, data_dir, load_xy


def module(filename):
    spec = importlib.util.spec_from_file_location("supplement_check_" + filename, ROOT / "src" / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-10)


def scores(actual, predicted):
    return dict(mae=mean_absolute_error(actual, predicted),
                rmse=np.sqrt(mean_squared_error(actual, predicted)),
                macro_r2=r2_score(actual, predicted, multioutput="uniform_average"))


def main():
    meta, x, y = load_xy()
    targets = list(y)
    checks = []
    q1 = HERE / "results/q1"
    metrics = pd.read_csv(q1 / "metrics_overall.csv")
    balanced = pd.read_csv(q1 / "balanced_group_metrics_overall.csv")
    tested = 0
    for path in q1.glob("oof_*.csv"):
        frame = pd.read_csv(path)
        assert frame.beer.tolist() == meta.beer.tolist()
        predicted = frame[targets].to_numpy()
        assert np.isfinite(predicted).all()
        name = path.stem.removeprefix("oof_")
        if "balanced" in name:
            model = "extra_trees" if name.endswith("extra_trees") else "mean"
            row = balanced.loc[balanced.model.eq(model)].iloc[0]
        else:
            protocol = "unseen_style_3fold" if name.startswith("unseen_style") else "random_3fold"
            model = name.removeprefix(protocol + "_")
            row = metrics.loc[metrics.protocol.eq(protocol) & metrics.model.eq(model)].iloc[0]
        for key, value in scores(y, predicted).items():
            close(row[key], value)
        for fold in sorted(frame.fold.unique()):
            test = frame.fold.eq(fold).to_numpy()
            if "unseen_style" in name:
                assert not set(meta.loc[test, "tasting_category_fine"]) & set(meta.loc[~test, "tasting_category_fine"])
            if model == "mean":
                close(predicted[test], np.tile(y.loc[~test].mean().to_numpy(), (test.sum(), 1)))
        tested += 1
    assert tested == 8
    draws = np.load(q1 / "test_mask_predictions.npz")
    for rate, field in [(.1, "rate10"), (.2, "rate20")]:
        assert draws[field].shape == (20, 250, 45)
        for repeat, predicted in enumerate(draws[field], 1):
            row = metrics.loc[metrics.protocol.eq("random_3fold_test_mask") & metrics.rate.eq(rate)
                              & metrics["repeat"].eq(repeat)].iloc[0]
            for key, value in scores(y, predicted).items():
                close(row[key], value)
    checks.append({"name": "q1_oof_metrics_and_group_exclusion", "oof_tables": tested, "mask_draws": 40})

    q2 = HERE / "results/q2_stratified"
    helper = module("03_q2_network.py")
    saved = np.load(q2 / "draw_indices_and_selections.npz")
    chemical = pd.read_csv(data_dir() / "1.csv")
    sensory = pd.read_csv(data_dir() / "4.csv")
    merged = chemical.merge(sensory[["beer", *targets]], on="beer", validate="one_to_one")
    candidates = saved["candidate_order"].tolist()
    candidate_targets = saved["target_order"].tolist()
    xx, yy = merged[candidates].to_numpy(), merged[candidate_targets].to_numpy()
    assert saved["beer_order"].tolist() == merged.beer.tolist()
    labels = merged.tasting_category_fine.to_numpy()
    counts = np.zeros((218, 45), dtype=int)
    for rows, winners in zip(saved["row_indices"], saved["selected_feature_indices"]):
        assert pd.Series(labels[rows]).value_counts().to_dict() == pd.Series(labels).value_counts().to_dict()
        counts[winners, np.arange(45)] += 1
    for i in [0, 99, 199]:
        rows = saved["row_indices"][i]
        np.testing.assert_array_equal(helper.top3(helper.cross_correlation(xx[rows], yy[rows], "spearman")),
                                      saved["selected_feature_indices"][i])
    table = pd.read_csv(q2 / "all_candidates_comparison.csv")
    for k, target in enumerate(candidate_targets):
        subset = table.loc[table.sensory_target.eq(target)].set_index("compound").loc[candidates]
        close(subset.stratified_top3_count, counts[:, k])
        close(subset.stratified_top3_frequency, counts[:, k] / 200)
    checks.append({"name": "q2_stratified_draws_and_frequencies", "draws": 200, "candidate_target_pairs": 9810})

    perm = HERE / "results/q2_permutation"
    arrays = np.load(perm / "permutation_draws.npz")
    delta = arrays["delta_mse"]
    assert delta.shape == (5, 5, 231, 45) and np.isfinite(delta).all()
    frame = pd.read_csv(perm / "oof.csv")
    assert frame.beer.tolist() == meta.beer.tolist()
    importance = pd.read_csv(perm / "importance.csv")
    fold_means = delta.mean(axis=1)
    for k, target in enumerate(targets):
        part = importance.loc[importance.sensory_target.eq(target)].set_index("feature").loc[x.columns]
        close(part.delta_mse, fold_means[:, :, k].mean(axis=0))
        close(part.fold_sd, fold_means[:, :, k].std(axis=0, ddof=1))
        close(part.positive_folds, (fold_means[:, :, k] > 0).sum(axis=0))
    membership = pd.read_csv(perm / "module_membership.csv")
    for fold, (train, test) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(x)):
        np.testing.assert_array_equal(arrays["check_test_rows"][fold], test)
        assert frame.iloc[test].fold.eq(fold + 1).all()
        truth = y.iloc[test].to_numpy()
        baseline = ((truth - frame.iloc[test][targets].to_numpy()) ** 2).mean(axis=0)
        for k, feature in enumerate(arrays["check_features"]):
            j = list(x).index(feature)
            independent = ((arrays["check_predictions"][fold, :, k] - truth) ** 2).mean(axis=1) - baseline
            close(independent, delta[fold, :, j])
        corr = x.iloc[train].corr(method="spearman").fillna(0).to_numpy()
        _, labels = connected_components(np.abs(corr) >= .7, directed=False)
        independent_groups = {frozenset(x.columns[labels == label]) for label in set(labels)}
        recorded_groups = {frozenset(r.split(" | ")) for r in membership.loc[membership.fold.eq(fold+1), "members"]}
        assert independent_groups == recorded_groups
    top = pd.read_csv(perm / "top3.csv")
    for target in targets:
        eligible = importance.loc[importance.sensory_target.eq(target) & importance.chemical_candidate].copy()
        order = {feature: i for i, feature in enumerate(x.columns)}
        eligible["order"] = eligible.feature.map(order)
        expected = eligible.sort_values(["delta_mse", "order"], ascending=[False, True]).head(3).feature.tolist()
        actual = top.loc[top.sensory_target.eq(target)].sort_values("candidate_rank").feature.tolist()
        assert actual == expected
    checks.append({"name": "held_out_permutation_metrics_and_training_modules", "feature_target_pairs": 10395,
                   "folds": 5, "repeats": 5, "independent_raw_feature_checks": 75})
    for source in [q1 / "manifest.json", q2 / "summary.json", perm / "summary.json"]:
        record = json.loads(source.read_text(encoding="utf-8"))
        hashes = record.get("input_sha256", record.get("input_sha256_before", {}))
        for name, expected in hashes.items():
            path = ROOT.parent / name if "/" in name else data_dir() / name
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    report = {"status": "PASS", "checks": checks,
              "scope": "Saved supplementary calculations, fold separation and source integrity; no causal or independent-population validation."}
    (HERE / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
