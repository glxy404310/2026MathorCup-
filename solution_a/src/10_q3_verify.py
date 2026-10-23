"""Recompute question-3 provenance, saved metrics and transfer constraints."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from threadpoolctl import threadpool_limits

from common import KEYS, SEED, SOLUTION, data_dir, load_xy


OUT = SOLUTION / "outputs" / "q3"
DERIVED = {"kcalperc", "esters_sum", "aroma_hops_sum", "acids_sum", "sulfur_sum"}
DEFECT_FIELDS = ("DMS", "X2.3.butanedione", "X4VG", "X.E..2.nonenal")
GENERATION_STYLES = ("Tripel", "Kriek", "Lambic", "Pils/Lager")
TOL = 1e-10


def close(actual, expected, label="numeric comparison", *, atol=1e-11):
    try:
        np.testing.assert_allclose(np.asarray(actual, dtype=float),
                                   np.asarray(expected, dtype=float),
                                   rtol=1e-9, atol=atol, equal_nan=True)
    except AssertionError as error:
        raise AssertionError(label) from error


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def score(y, prediction):
    return {
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
        "macro_r2": float(r2_score(y, prediction, multioutput="uniform_average")),
    }


def same_table(actual, expected, keys, values, label):
    assert not actual.duplicated(keys).any(), f"Duplicate rows: {label}"
    left = actual.set_index(keys).sort_index()
    right = expected.set_index(keys).sort_index()
    assert left.index.equals(right.index), f"Row keys differ: {label}"
    close(left[values], right[values], label)


def check_hashes():
    table = pd.read_csv(SOLUTION / "outputs/audit/source_files.csv")
    assert len(table) == 9 and table.source.is_unique
    for row in table.itertuples(index=False):
        raw = (data_dir() / row.source).read_bytes()
        assert len(raw) == row.bytes, f"Source byte count changed: {row.source}"
        assert hashlib.sha256(raw).hexdigest() == row.sha256, f"Source changed: {row.source}"
    return {"files": len(table), "hash_algorithm": "sha256", "unchanged": True}


def load_contract():
    meta, original, y = load_xy()
    calibrated = pd.read_excel(data_dir() / "8.xlsx", sheet_name="Calibrated chemical dataset")
    assert calibrated.beer.is_unique and set(calibrated.beer) == set(meta.beer)
    calibrated = calibrated.set_index("beer").loc[meta.beer].reset_index()
    assert calibrated.tasting_category_fine.equals(meta.tasting_category_fine)
    features = [column for column in original if column not in DERIVED]
    calibrated_features = [column for column in calibrated if column not in set(KEYS) | DERIVED]
    assert len(features) == 226 and len(calibrated_features) == 69
    assert set(calibrated_features).issubset(features)
    expected = original[features].copy()
    negative_count = 0
    for feature in calibrated_features:
        values = calibrated[feature].copy()
        negative_count += int(values.lt(0).sum())
        expected[feature] = values.mask(values.lt(0)).to_numpy()
    assert expected.shape == (250, 226) and y.shape == (250, 45)
    folder = OUT / "sensory"
    stored = pd.read_csv(folder / "hybrid_features.csv")
    assert stored.columns.tolist() == features
    close(stored, expected, "Saved hybrid features differ from original attachments")
    saved_meta = pd.read_csv(folder / "meta.csv", dtype={"beer_id": str})
    pd.testing.assert_frame_equal(saved_meta, meta, check_dtype=False)
    close(pd.read_csv(folder / "y45.csv"), y, "Saved targets differ from attachment 4")
    manifest = pd.read_csv(folder / "feature_manifest.csv")
    assert manifest.feature.tolist() == features and manifest.position.tolist() == list(range(226))
    assert manifest.representation.value_counts().to_dict() == {"relative": 157, "calibrated": 69}
    assert set(manifest.loc[manifest.representation.eq("calibrated"), "feature"]) == set(calibrated_features)
    assert not set(features) & DERIVED
    for row in manifest.itertuples(index=False):
        raw = calibrated[row.feature] if row.feature in calibrated_features else original[row.feature]
        invalid = int(raw.lt(0).sum()) if row.feature in calibrated_features else 0
        assert row.original_missing_count == int(raw.isna().sum())
        assert row.original_1csv_missing_count == int(original[row.feature].isna().sum())
        assert row.negative_invalid_count == invalid
        assert row.final_missing_count == int(expected[row.feature].isna().sum())
        assert row.zero_count == int(expected[row.feature].eq(0).sum())
    assert int(manifest.negative_invalid_count.sum()) == negative_count
    assert not np.isinf(expected.to_numpy()).any()
    calibrated_values = expected[calibrated_features].to_numpy()
    assert (calibrated_values[np.isfinite(calibrated_values)] >= 0).all()
    assert stored.isna().equals(expected.isna())
    context = {"meta": meta, "x": expected, "y": y,
               "features": features, "calibrated_features": calibrated_features}
    return context, {"rows": len(expected), "features": len(features),
                     "calibrated": len(calibrated_features), "relative": 157,
                     "negative_calibrated_invalidated": negative_count,
                     "missing_cells_preserved": int(expected.isna().sum().sum())}


def supported(values, empirical):
    positions = np.searchsorted(empirical, values)
    left = empirical[np.clip(positions - 1, 0, len(empirical) - 1)]
    right = empirical[np.clip(positions, 0, len(empirical) - 1)]
    return np.isclose(values, left, rtol=1e-9, atol=1e-11) | np.isclose(
        values, right, rtol=1e-9, atol=1e-11)


def check_generation(context):
    x, features = context["x"], context["features"]
    folder = OUT / "generation"
    summary = read_json(folder / "summary.json")
    model = joblib.load(folder / "fitted_model.joblib")
    assert model.columns_ == features and model.n_samples_ == len(x)
    close(model.median_, np.nanmedian(x, axis=0), "Full-fit median differs from training data")
    training = x.fillna(pd.Series(model.median_, index=features)).to_numpy()
    for j in range(len(features)):
        close(model.empirical_values_[j], np.unique(training[:, j]), "Empirical support differs from training")
    tables, copies, rows = [], 0, 0
    for style in GENERATION_STYLES:
        frame = pd.read_csv(folder / f"generated_{style.replace('/', '-')}.csv")
        assert len(frame) == 300 and frame.sample_id.is_unique
        assert frame.target_style.eq(style).all()
        assert frame.columns.tolist() == ["sample_id", "target_style"] + features
        fake = frame[features].to_numpy()
        assert np.isfinite(fake).all()
        assert frame[context["calibrated_features"]].ge(0).all().all()
        assert (fake >= training.min(axis=0) - TOL).all()
        assert (fake <= training.max(axis=0) + TOL).all()
        for j in range(len(features)):
            assert supported(fake[:, j], model.empirical_values_[j]).all(), f"Outside empirical support: {style}/{features[j]}"
        copied = np.any(np.all(np.isclose(fake[:, None, :], training[None, :, :],
                                        rtol=1e-10, atol=1e-12), axis=2), axis=1)
        copies += int(copied.sum())
        rows += len(frame)
        assert not copied.any(), f"Whole training-row copy generated for {style}"
        tables.append(frame)
    generated = pd.concat(tables, ignore_index=True)
    assert generated.sample_id.is_unique and rows == summary["full_generation_rows"]
    close(copies / rows, summary["full_generation_copy_rate"], "Generation copy-rate summary")
    diagnostics = pd.read_csv(folder / "generated_diagnostics.csv")
    assert set(diagnostics.sample_id) == set(generated.sample_id)
    assert not diagnostics.exact_training_copy.any()
    context["generated"] = generated
    return {"styles": list(GENERATION_STYLES), "rows_per_style": 300, "rows": rows,
            "whole_training_row_copies": copies, "calibrated_nonnegative": True,
            "marginal_training_support": True, "discrete_empirical_support": True}


def check_generation_cv(context):
    folder = OUT / "generation"
    meta, x = context["meta"], context["x"]
    summary = read_json(folder / "summary.json")
    assignments = pd.read_csv(folder / "cv_assignments.csv")
    metrics = pd.read_csv(folder / "cv_metrics.csv")
    assert assignments.role.eq("test").all()
    assert not assignments.duplicated(["repeat", "row_index"]).any()
    assert set(assignments.repeat) == {1, 2} and set(assignments.fold) == {1, 2, 3}
    assert len(assignments) == 2 * len(meta)
    assert (assignments.groupby(["repeat", "row_index"]).size() == 1).all()
    assert assignments.beer.tolist() == assignments.row_index.map(meta.beer).tolist()
    assert assignments["style"].tolist() == assignments.row_index.map(meta.tasting_category_fine).tolist()
    methods = {"conditional_copula", "style_bootstrap", "unconditional_copula"}
    keys = ["repeat", "fold", "mc", "style", "method"]
    assert not metrics.duplicated(keys).any()
    assert set(metrics.method) == methods and set(metrics["style"]) == set(GENERATION_STYLES)
    assert set(metrics.mc) == {0, 1, 2}
    assert len(metrics) == 2 * 3 * 3 * len(GENERATION_STYLES) * len(methods)
    assert (metrics.groupby(["repeat", "fold", "style", "method"]).size() == 3).all()
    assert metrics.generated_n.eq(300).all()
    assert metrics.mean_strength.eq(summary["mean_strength"]).all()
    assert metrics.cov_strength.eq(summary["cov_strength"]).all()
    for repeat, seed in enumerate(summary["cv_repeat_seeds"], 1):
        splitter = StratifiedKFold(3, shuffle=True, random_state=seed)
        for fold, (train, test) in enumerate(splitter.split(x, meta.tasting_category_fine), 1):
            stored = assignments[assignments.repeat.eq(repeat) & assignments.fold.eq(fold)]
            assert set(stored.row_index) == set(test)
            group = metrics[metrics.repeat.eq(repeat) & metrics.fold.eq(fold)]
            assert group.train_n.eq(len(train)).all() and group.split_seed.eq(seed).all()
            for style in GENERATION_STYLES:
                part = group[group["style"].eq(style)]
                assert part.style_train_n.eq(meta.tasting_category_fine.iloc[train].eq(style).sum()).all()
                assert part.holdout_style_n.eq(meta.tasting_category_fine.iloc[test].eq(style).sum()).all()
    fold_table = pd.read_csv(folder / "cv_fold_means.csv")
    fold_keys = ["repeat", "fold", "style", "method"]
    value_columns = [column for column in fold_table if column not in fold_keys]
    assert np.isfinite(metrics[value_columns].to_numpy()).all()
    expected_fold = metrics.groupby(fold_keys, as_index=False)[value_columns].mean()
    same_table(fold_table, expected_fold, fold_keys, value_columns, "CV Monte Carlo fold means")
    expected_summary = expected_fold.groupby(["style", "method"])[value_columns].agg(["mean", "std"])
    expected_summary.columns = [f"{metric}_{stat}" for metric, stat in expected_summary.columns]
    expected_summary = expected_summary.reset_index()
    stored_summary = pd.read_csv(folder / "cv_summary.csv")
    same_table(stored_summary, expected_summary, ["style", "method"],
               [column for column in expected_summary if column not in {"style", "method"}],
               "CV six-split summary")
    assert summary["validation_records"] == len(metrics)
    return {"repeats": 2, "folds": 3, "mc_per_method_fold": 3,
            "every_sample_test_once_per_repeat": True, "metric_records": len(metrics),
            "fold_and_overall_summaries_reconciled": True,
            "scope": "Saved split assignments, schema and aggregation; no claim of independent real experiments"}


def check_sensory(context):
    meta, x, y = context["meta"], context["x"], context["y"]
    folder = OUT / "sensory"
    overall = pd.read_csv(folder / "metrics_overall.csv")
    by_target = pd.read_csv(folder / "metrics_by_target.csv")
    by_fold = pd.read_csv(folder / "metrics_by_fold.csv")
    expected_folds = np.zeros(len(x), dtype=int)
    for fold, (_, test) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(x), 1):
        expected_folds[test] = fold
    targets, folds = [], []
    for row in overall.itertuples(index=False):
        frame = pd.read_csv(folder / f"oof_{row.model}.csv", dtype={"beer_id": str})
        pd.testing.assert_frame_equal(frame[KEYS], meta, check_dtype=False)
        np.testing.assert_array_equal(frame.fold, expected_folds)
        predictions = frame[y.columns].to_numpy()
        assert np.isfinite(predictions).all()
        values = score(y, predictions)
        close([row.mae, row.rmse, row.macro_r2], list(values.values()), f"OOF metrics: {row.model}")
        for j, target in enumerate(y):
            targets.append({"model": row.model, "target": target,
                            **score(y.iloc[:, j], predictions[:, j]),
                            "spearman": float(spearmanr(y.iloc[:, j], predictions[:, j]).statistic)})
        for fold in range(1, 6):
            test = expected_folds == fold
            folds.append({"model": row.model, "fold": fold,
                          **score(y.loc[test], predictions[test])})
            if row.model == "mean":
                close(predictions[test], np.broadcast_to(y.loc[~test].mean(), predictions[test].shape),
                      "Mean OOF baseline must use training-fold targets")
    same_table(by_target, pd.DataFrame(targets), ["model", "target"],
               ["mae", "rmse", "macro_r2", "spearman"], "Sensory target OOF metrics")
    same_table(by_fold, pd.DataFrame(folds), ["model", "fold"],
               ["mae", "rmse", "macro_r2"], "Sensory fold OOF metrics")
    sources = np.flatnonzero(meta.tasting_category_fine.eq("Pils/Lager"))
    train = np.flatnonzero(~meta.tasting_category_fine.eq("Pils/Lager"))
    proxy = joblib.load(folder / "transfer_proxy.joblib")
    ensemble = joblib.load(folder / "stale_bootstrap.joblib")
    np.testing.assert_array_equal(proxy["train_global_indices"], train)
    np.testing.assert_array_equal(proxy["heldout_global_indices"], sources)
    np.testing.assert_array_equal(ensemble["train_global_indices"], train)
    np.testing.assert_array_equal(ensemble["excluded_global_indices"], sources)
    assert len(train) == 239 and len(sources) == 11 and len(ensemble["models"]) == 40
    assert proxy["features"] == ensemble["features"] == x.columns.tolist()
    assert proxy["targets"] == y.columns.tolist()
    indices = np.asarray(ensemble["bootstrap_indices"])
    assert indices.shape == (40, len(train)) and np.isin(indices, train).all()
    assert not np.isin(indices, sources).any()
    assert ensemble["selection_model_indices"] == list(range(30))
    assert ensemble["audit_model_indices"] == list(range(30, 40))
    with np.load(folder / "stale_bootstrap_indices.npz") as saved:
        np.testing.assert_array_equal(saved["indices"], indices)
        np.testing.assert_array_equal(saved["source_beer"], meta.beer)
    heldout_prediction = proxy["pipeline"].predict(x.iloc[sources])
    heldout = pd.read_csv(folder / "transfer_pils_heldout_predictions.csv")
    np.testing.assert_array_equal(heldout.beer, meta.beer.iloc[sources])
    close(heldout[y.columns], heldout_prediction, "Transfer proxy saved source predictions")
    heldout_metrics = pd.read_csv(folder / "transfer_pils_heldout_metrics.csv").set_index("model")
    for name, prediction in [("transfer_proxy", heldout_prediction),
                             ("non_pils_mean", np.broadcast_to(y.iloc[train].mean(), heldout_prediction.shape))]:
        close(heldout_metrics.loc[name, ["mae", "rmse", "macro_r2"]],
              list(score(y.iloc[sources], prediction).values()), "Held-out Pils sensory metrics")
    context.update({"source_rows": sources, "transfer_train": train,
                    "proxy": proxy, "ensemble": ensemble})
    stale = pd.DataFrame(targets)
    stale = stale[stale.target.eq("stale_hops")].set_index("model")
    return {"oof_models": len(overall), "oof_folds": 5, "targets": len(y.columns),
            "excluded_sources": len(sources), "transfer_training_rows": len(train),
            "bootstrap_models": len(ensemble["models"]),
            "bootstrap_resample_shape": list(indices.shape),
            "stale_hops_oof_r2": float(stale.loc["extra_trees", "macro_r2"]),
            "stale_hops_oof_rmse": float(stale.loc["extra_trees", "rmse"]),
            "stale_hops_oof_mean_rmse": float(stale.loc["mean", "rmse"])}


def chemical_caps(source, thresholds, kind):
    result = {}
    for feature in DEFECT_FIELDS:
        value = float(source[feature].iloc[0])
        threshold = thresholds.at[feature, f"threshold_{kind}"]
        result[feature] = max(value, float(threshold)) if pd.notna(threshold) else value
    return result


def chemical_flags(frame, caps):
    values = frame[list(caps)].to_numpy()
    limits = np.array(list(caps.values()))
    return np.isfinite(values).all(axis=1) & (values >= 0).all(axis=1) & (values <= limits + TOL).all(axis=1)


def check_transfer(context):
    meta, x, y = context["meta"], context["x"], context["y"]
    source_rows, train = context["source_rows"], context["transfer_train"]
    folder = OUT / "transfer"
    summary = read_json(folder / "summary.json")
    saved = joblib.load(folder / "transfer_copula.joblib")
    copula = saved["copula"]
    np.testing.assert_array_equal(saved["training_rows"], train)
    np.testing.assert_array_equal(saved["source_rows"], source_rows)
    assert copula.columns_ == x.columns.tolist() and copula.n_samples_ == len(train)
    close(copula.median_, np.nanmedian(x.iloc[train], axis=0), "Source-excluded copula medians")
    thresholds = pd.read_excel(data_dir() / "8.xlsx", sheet_name="Sensory thresholds").set_index("field_name_R")
    assert pd.isna(thresholds.at["X.E..2.nonenal", "threshold_median"])
    target_rows = np.flatnonzero(meta.tasting_category_fine.eq(summary["target_style"]))
    target = x.iloc[target_rows]
    target_z = copula.transform(target)
    dimensions = x.shape[1]
    target_pairs = cdist(target_z, target_z) / np.sqrt(dimensions)
    np.fill_diagonal(target_pairs, np.inf)
    loo_nn = target_pairs.min(axis=1)
    support_limit = float(np.quantile(loo_nn, summary["support"]["quantile"]))
    close(support_limit, summary["support"]["limit"], "Transfer target support limit")
    support_csv = pd.read_csv(folder / "target_support_reference.csv")
    np.testing.assert_array_equal(support_csv.beer, meta.beer.iloc[target_rows])
    close(support_csv.loo_nn_rms, loo_nn, "Tripel leave-one-out support reference")
    report = pd.read_csv(folder / "transfer_summary.csv")
    spectra = pd.read_csv(folder / "transferred_spectra.csv").set_index("transfer_id")
    candidates = pd.read_csv(folder / "candidate_diagnostics.csv")
    paired_csv = pd.read_csv(folder / "stale_paired_deltas.csv")
    chemicals = pd.read_csv(folder / "defect_checks.csv")
    profiles = pd.read_csv(folder / "sensory_before_after.csv")
    source_spectra = pd.read_csv(folder / "source_spectra.csv")
    np.testing.assert_array_equal(source_spectra.beer, meta.beer.iloc[source_rows])
    close(source_spectra[x.columns], x.iloc[source_rows], "Transfer raw source spectra")
    assert not report.duplicated(["source_beer", "mode", "lambda"]).any()
    assert spectra.index.is_unique
    selected_report = report[report.candidate_index.notna()].copy()
    assert set(spectra.index) == set(selected_report.transfer_id)
    assert np.isfinite(spectra[x.columns].to_numpy()).all()
    assert spectra[context["calibrated_features"]].ge(0).all().all()
    assert set(report.source_beer) == set(meta.beer.iloc[source_rows])
    assert set(report["lambda"]) == set(summary["lambdas"])
    assert report.is_default_policy.equals(report["lambda"].eq(summary["default_lambda"]))
    models = context["ensemble"]["models"]
    source_frame = x.iloc[source_rows].fillna(x.iloc[train].median())
    source_predictions = np.vstack([model.predict(source_frame) for model in models])
    selected_predictions = np.vstack([model.predict(spectra[x.columns]) for model in models])
    source_lookup = {int(meta.beer.iloc[index]): position for position, index in enumerate(source_rows)}
    center = target_z.mean(axis=0)
    training_min, training_max = x.iloc[train].min().to_numpy(), x.iloc[train].max().to_numpy()
    max_quantile_error = 0.0
    max_distance_error = 0.0
    max_nn_error = 0.0
    argmin_checks = 0
    for source_beer, group in candidates.groupby("source_beer"):
        assert len(group) == summary["n_candidates_per_source"]
        assert set(group.candidate_index) == set(range(summary["n_candidates_per_source"]))
        flags = (group.support_ok & group.nontrivial & group.target_improvement_ok
                 & group.chemical_median_ok & group.selection_proxy_ok)
        assert flags.equals(group.selection_feasible)
        assert group.selection_proxy_ok.equals(group.selection_paired_delta_q95.le(TOL))
        assert group.nontrivial.equals(group.change_latent_rms.ge(summary["minimum_latent_rms_change"]))
        assert group.target_improvement_ok.equals(group.target_improvement.ge(summary["minimum_target_improvement"]))
        assert group.support_ok.equals(group.target_nn_rms.le(support_limit + TOL) & group.marginal_support)
    for _, row in selected_report.iterrows():
        selected = spectra.loc[[row.transfer_id], x.columns]
        source_position = source_lookup[int(row.source_beer)]
        source = source_frame.iloc[[source_position]]
        source_z, selected_z = copula.transform(source), copula.transform(selected)
        original_source = x.iloc[[source_rows[source_position]]]
        assert original_source[list(DEFECT_FIELDS)].notna().all().all()
        caps_median = chemical_caps(original_source, thresholds, "median")
        caps_mean = chemical_caps(original_source, thresholds, "mean")
        chemical_median = bool(chemical_flags(selected, caps_median)[0])
        chemical_mean = bool(chemical_flags(selected, caps_mean)[0])
        assert chemical_median == row.chemical_median_ok and chemical_mean == row.chemical_mean_ok
        position = spectra.index.get_loc(row.transfer_id)
        deltas = selected_predictions[:, position] - source_predictions[:, source_position]
        selection_q95, audit_q95 = float(np.quantile(deltas[:30], .95)), float(np.quantile(deltas[30:], .95))
        max_quantile_error = max(max_quantile_error, abs(selection_q95 - row.selection_paired_delta_q95),
                                 abs(audit_q95 - row.audit_paired_delta_q95))
        close([selection_q95, audit_q95],
              [row.selection_paired_delta_q95, row.audit_paired_delta_q95], "Selected paired quantiles")
        assert bool(audit_q95 <= TOL) == row.audit_proxy_ok
        for ensemble_name, values in [("selection", deltas[:30]), ("audit", deltas[30:])]:
            saved_deltas = paired_csv[paired_csv.transfer_id.eq(row.transfer_id)
                                     & paired_csv.ensemble.eq(ensemble_name)].sort_values("model_index")
            np.testing.assert_array_equal(saved_deltas.model_index, np.arange(len(values)))
            close(saved_deltas.paired_delta, values, "Saved per-model paired differences")
        before = float(copula.distance(source, summary["target_style"])[0])
        after = float(copula.distance(selected, summary["target_style"])[0])
        near = float(cdist(selected_z, target_z).min() / np.sqrt(dimensions))
        change = float(np.linalg.norm(selected_z - source_z) / np.sqrt(dimensions))
        improvement = 1 - after / max(before, 1e-8)
        max_distance_error = max(max_distance_error, abs(after - row.target_distance_after))
        max_nn_error = max(max_nn_error, abs(near - row.target_nn_after))
        close([before, after, near, change, improvement],
              [row.target_distance_before, row.target_distance_after, row.target_nn_after,
               row.change_latent_rms, row.target_improvement], "Selected transfer geometry")
        assert near <= support_limit + TOL
        assert (selected.to_numpy() >= training_min - TOL).all()
        assert (selected.to_numpy() <= training_max + TOL).all()
        assert improvement >= summary["minimum_target_improvement"]
        assert change >= summary["minimum_latent_rms_change"]
        scale = max(float(np.linalg.norm(source_z[0] - center) / np.sqrt(dimensions)), 1e-8)
        close(after / max(before, 1e-8) + row["lambda"] * change / scale, row.objective,
              "Selected fixed objective")
        if row["mode"] == "protected":
            assert chemical_median and selection_q95 <= TOL
            assert bool(audit_q95 <= TOL) == row.all_model_checks_passed
            expected_status = "model_checks_passed" if audit_q95 <= TOL else "resampling_audit_failed"
            assert row.status == expected_status
            feasible = candidates[candidates.source_beer.eq(row.source_beer) & candidates.selection_feasible]
            objective = feasible.target_distance / max(before, 1e-8) + row["lambda"] * feasible.change_latent_rms / scale
            expected_index = int(feasible.iloc[np.argmin(objective)].candidate_index)
            assert int(row.candidate_index) == expected_index, "Audit changed the selected candidate"
            argmin_checks += 1
        else:
            assert row.status == "baseline_only" and not row.all_model_checks_passed
        check_rows = chemicals[chemicals.transfer_id.eq(row.transfer_id)].set_index("feature")
        assert set(check_rows.index) == set(DEFECT_FIELDS)
        for feature in DEFECT_FIELDS:
            close([check_rows.at[feature, "before"], check_rows.at[feature, "after"],
                   check_rows.at[feature, "median_upper_bound"], check_rows.at[feature, "mean_upper_bound"]],
                  [original_source[feature].iloc[0], selected[feature].iloc[0],
                   caps_median[feature], caps_mean[feature]], "Defect concentration audit")
            for kind in ["median", "mean"]:
                threshold = thresholds.at[feature, f"threshold_{kind}"]
                for phase, value in [("before", original_source[feature].iloc[0]),
                                     ("after", selected[feature].iloc[0])]:
                    excess = max(value / threshold - 1, 0) if pd.notna(threshold) else np.nan
                    close(check_rows.at[feature, f"{kind}_excess_{phase}"], excess, "Defect excess audit")
        for phase, values in [("source", source), ("transferred", selected)]:
            stored_profile = profiles[profiles.transfer_id.eq(row.transfer_id) & profiles.phase.eq(phase)]
            assert len(stored_profile) == 1
            close(stored_profile[y.columns], context["proxy"]["pipeline"].predict(values),
                  "Transfer relative sensory profile")
    assert len(paired_csv) == len(spectra) * 40 and len(chemicals) == len(spectra) * 4
    assert len(profiles) == len(spectra) * 2
    for _, row in report[report.candidate_index.isna()].iterrows():
        if row.status == "no_candidate_found":
            assert not candidates[candidates.source_beer.eq(row.source_beer)].selection_feasible.any()
        elif row.status == "source_defect_unverifiable":
            original = x.iloc[[source_rows[source_lookup[int(row.source_beer)]]]][list(DEFECT_FIELDS)]
            assert original.isna().any().any() or original.lt(0).any().any()
        else:
            raise AssertionError(f"Unrecognized missing-candidate status: {row.status}")
    default = report[report.is_default_policy & report["mode"].eq("protected")]
    default_counts = {
        "n_sources": len(default),
        "n_candidates_selected": int(default.status.isin(["model_checks_passed", "resampling_audit_failed"]).sum()),
        "n_model_checks_passed": int(default.all_model_checks_passed.sum()),
        "n_audit_failed": int(default.status.eq("resampling_audit_failed").sum()),
        "n_no_candidate": int(default.status.eq("no_candidate_found").sum()),
        "n_source_unverifiable": int(default.status.eq("source_defect_unverifiable").sum()),
        "n_all_checks_also_pass_mean_thresholds_same_spectra": int(
            (default.all_model_checks_passed & default.chemical_mean_ok.fillna(False)).sum()),
    }
    assert default_counts == summary["default_policy"], "Default transfer summary differs from CSV"
    assert len(default) == len(source_rows)
    context["default_transfer_counts"] = default_counts
    return {"selected_spectra_checked": len(spectra), "source_training_rows": len(train),
            "protected_fixed_argmin_checks": argmin_checks,
            "selection_models": 30, "audit_models": 10,
            "maximum_paired_quantile_error": max_quantile_error,
            "maximum_target_distance_error": max_distance_error,
            "maximum_full_space_nn_error": max_nn_error,
            "default_policy": default_counts,
            "scope": "Chemical constraints and uncalibrated model diagnostics; no actual taste guarantee"}


def check_report_artifacts(context):
    path = OUT / "generated_sensory_profiles.csv"
    assert path.is_file(), f"Missing deliverable: {path}"
    report = SOLUTION / "docs/q3_results.md"
    assert report.is_file() and report.stat().st_size > 100, f"Missing or empty deliverable: {report}"
    generated, y = context["generated"], context["y"]
    profiles = pd.read_csv(path)
    assert len(profiles) == len(generated) and profiles.sample_id.is_unique
    assert set(profiles.sample_id) == set(generated.sample_id)
    assert set(y.columns).issubset(profiles.columns)
    final = joblib.load(OUT / "sensory/sensory_model.joblib")
    assert final["features"] == context["features"] and final["targets"] == y.columns.tolist()
    expected = final["pipeline"].predict(generated[context["features"]])
    actual = profiles.set_index("sample_id").loc[generated.sample_id, y.columns]
    assert np.isfinite(actual.to_numpy()).all()
    close(actual, expected, "Generated frozen-model sensory profiles")
    return {"generated_sensory_rows": len(profiles), "sensory_dimensions": len(y.columns),
            "frozen_model_predictions_reconciled": True, "results_report_exists": True}


def main():
    start = perf_counter()
    checks = []
    result = {"status": "running", "checks": checks,
              "scope": "Source provenance, saved-output integrity and independently recomputed numerical consistency"}
    active_check = "source_hashes_before"
    try:
        checks.append({"name": active_check, "status": "passed", "details": check_hashes()})
        active_check = "hybrid_data_contract"
        context, details = load_contract()
        checks.append({"name": active_check, "status": "passed", "details": details})
        for name, operation in [("generation_training_support", check_generation),
                                ("generation_repeated_cv", check_generation_cv),
                                ("sensory_oof_and_source_exclusion", check_sensory),
                                ("transfer_constraints_and_selection", check_transfer),
                                ("report_and_generated_sensory", check_report_artifacts)]:
            active_check = name
            details = operation(context)
            checks.append({"name": name, "status": "passed", "details": details})
            print(f"Verified: {name}", flush=True)
        active_check = "source_hashes_after"
        checks.append({"name": active_check, "status": "passed", "details": check_hashes()})
        result["status"] = "passed"
        result["limitations"] = [
            "Numerical consistency does not establish model adequacy or physical brewing feasibility.",
            "Bootstrap selection and audit groups share real training data and have no calibrated outcome coverage.",
            "Sensory values are relative attachment-standardized predictions, not recovered raw 0-6 panel scores.",
        ]
    except Exception as error:
        result["status"] = "failed"
        result["failed_check"] = active_check
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        result["elapsed_seconds"] = perf_counter() - start
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "verification.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
