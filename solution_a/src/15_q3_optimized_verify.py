"""Verify preserved baselines, nested selection and robust transfer outputs."""
from __future__ import annotations

import hashlib
import importlib
import json
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from threadpoolctl import threadpool_limits

from common import SOLUTION, data_dir

BASE = importlib.import_module("10_q3_verify")
OUT = SOLUTION / "outputs/q3_optimized"
TOL = 1e-10


def score(truth, prediction):
    truth, prediction = np.asarray(truth, float), np.asarray(prediction, float)
    tv, pv = np.var(truth), np.var(prediction)
    tr = np.quantile(truth, .95) - np.quantile(truth, .05)
    pr = np.quantile(prediction, .95) - np.quantile(prediction, .05)
    return {"mae": mean_absolute_error(truth, prediction),
            "rmse": np.sqrt(mean_squared_error(truth, prediction)),
            "r2": r2_score(truth, prediction),
            "spearman": spearmanr(truth, prediction).statistic if pv > 1e-20 else np.nan,
            "prediction_variance": pv, "target_variance": tv,
            "prediction_variance_ratio": pv / tv,
            "prediction_std_ratio": np.sqrt(pv / tv),
            "prediction_range90": pr, "target_range90": tr,
            "prediction_range90_ratio": pr / tr, "mean_bias": np.mean(prediction - truth)}


def check_baseline():
    manifest = BASE.read_json(OUT / "baseline_hashes.json")
    baseline = SOLUTION / "outputs/q3"
    expected = {row["path"] for row in manifest["files"]}
    actual = {path.relative_to(baseline).as_posix() for path in baseline.rglob("*") if path.is_file()}
    assert actual == expected and len(actual) == manifest["file_count"]
    for row in manifest["files"]:
        raw = (baseline / row["path"]).read_bytes()
        assert len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"], row["path"]
    return {"unchanged_files": len(actual), "hash_algorithm": "sha256"}


def winner(rows):
    qualified = rows[rows.nonconstant_qualified]
    accepted = qualified if len(qualified) else rows
    return accepted.sort_values(["rmse", "candidate"]).iloc[0]


def check_pipeline_training(pipeline, training, calibrated):
    chemical = pipeline.named_steps["chemical"]
    names = chemical.selected_features_
    values = training[names].to_numpy().copy()
    scales = np.ones(len(names))
    for j, name in enumerate(names):
        if name in calibrated:
            positives = values[:, j][np.isfinite(values[:, j]) & (values[:, j] > 0)]
            scales[j] = np.median(positives) if len(positives) else 1.0
            values[:, j] = np.log1p(values[:, j] / scales[j])
    BASE.close(chemical.positive_scales_, scales, "Training-only positive scales")
    bounds = np.nanquantile(values, [.01, .99], axis=0)
    BASE.close(chemical.bounds_, bounds, "Training-only clipping bounds")
    clipped = np.clip(values, bounds[0], bounds[1])
    medians = np.nanmedian(clipped, axis=0)
    medians = np.nan_to_num(medians, nan=0.0)
    imputer = pipeline.named_steps["imputer"]
    BASE.close(imputer.statistics_, medians, "Training-only imputation medians")
    indicators = np.flatnonzero(np.isnan(clipped).any(axis=0))
    np.testing.assert_array_equal(imputer.indicator_.features_, indicators)
    filled = np.where(np.isnan(clipped), medians, clipped)
    filled = np.column_stack([filled, np.isnan(clipped)[:, indicators]])
    BASE.close(pipeline.named_steps["scale"].mean_, filled.mean(axis=0), "Training-only scaling means")
    BASE.close(pipeline.named_steps["scale"].var_, filled.var(axis=0), "Training-only scaling variances")


def check_stale(context):
    folder = OUT / "stale"
    meta, x, y45 = context["meta"], context["x"], context["y"]
    y = y45.stale_hops
    manifest = BASE.read_json(folder / "run_manifest.json")
    summary = BASE.read_json(folder / "summary.json")
    oof = pd.read_csv(folder / "oof_predictions.csv")
    stored_metrics = pd.read_csv(folder / "metrics.csv")
    stored_folds = pd.read_csv(folder / "metrics_by_fold.csv")
    choices = pd.read_csv(folder / "nested_selected_configs.csv")
    inner = pd.read_csv(folder / "nested_inner_cv_results.csv")
    specs = {spec["name"]: spec for spec in manifest["candidate_specs"]}
    assert manifest["features"] == x.columns.tolist() and len(specs) == summary["candidate_count"]
    assert not oof.duplicated(["repeat", "method", "beer"]).any()
    methods = sorted(oof.method.unique())
    metrics, folds = [], []
    for repeat, seed in enumerate(manifest["outer_seeds"], 1):
        expected_fold = np.zeros(len(x), int)
        for fold, (_, test) in enumerate(KFold(manifest["outer_folds"], shuffle=True, random_state=seed).split(x), 1):
            expected_fold[test] = fold
        for method in methods:
            group = oof[oof.repeat.eq(repeat) & oof.method.eq(method)].set_index("beer").loc[meta.beer]
            assert len(group) == len(x) and group.outer_seed.eq(seed).all()
            np.testing.assert_array_equal(group.fold, expected_fold)
            BASE.close(group.observed, y, "Repeated OOF observed labels")
            assert np.isfinite(group.prediction).all()
            metrics.append({"scope": "per_seed", "repeat": repeat, "method": method,
                            **score(group.observed, group.prediction)})
            for fold in range(1, manifest["outer_folds"] + 1):
                test = expected_fold == fold
                folds.append({"repeat": repeat, "fold": fold, "method": method,
                              **score(y.loc[test], group.prediction.to_numpy()[test])})
                if method == "mean":
                    BASE.close(group.prediction.to_numpy()[test], y.loc[~test].mean(), "Fold-mean stale baseline")
    for method in methods:
        group = oof[oof.method.eq(method)]
        metrics.append({"scope": "pooled_repeated_oof", "repeat": 0, "method": method,
                        **score(group.observed, group.prediction)})
    columns = list(score(y, y))
    BASE.same_table(stored_metrics, pd.DataFrame(metrics), ["scope", "repeat", "method"], columns,
                    "Repeated nested stale metrics")
    BASE.same_table(stored_folds, pd.DataFrame(folds), ["repeat", "fold", "method"], columns,
                    "Nested stale fold metrics")
    gate = manifest["nonconstant_variance_ratio_min"]
    assert inner.nonconstant_qualified.equals(inner.prediction_variance_ratio.ge(gate))
    assert set(inner.candidate) == set(specs)
    for row in choices.itertuples(index=False):
        candidates = inner[inner.repeat.eq(row.repeat) & inner.outer_fold.eq(row.outer_fold)]
        if row.method.startswith("nested_family_"):
            candidates = candidates[candidates.family.eq(row.method.removeprefix("nested_family_"))]
        selected = winner(candidates)
        assert row.selected_candidate == selected.candidate
        assert row.nonconstant_qualified == selected.nonconstant_qualified
        BASE.close(row.inner_oof_rmse, selected.rmse, "Training-only nested candidate choice")
        assert row.outer_seed in manifest["outer_seeds"]
    excluded = np.flatnonzero(meta.tasting_category_fine.eq("Pils/Lager"))
    training = np.flatnonzero(~meta.tasting_category_fine.eq("Pils/Lager"))
    artifact = joblib.load(folder / "transfer_stale_model.joblib")
    ensemble = joblib.load(folder / "bootstrap_ensemble.joblib")
    transfer_inner = pd.read_csv(folder / "transfer_inner_cv_results.csv")
    selected = winner(transfer_inner)
    assert artifact["selected_config"] == specs[selected.candidate] == ensemble["selected_config"]
    inner_oof = pd.read_csv(folder / "transfer_selected_inner_oof.csv")
    np.testing.assert_array_equal(inner_oof.beer, meta.beer.iloc[training])
    BASE.close(inner_oof.observed, y.iloc[training], "Transfer inner observed labels")
    selected_score = score(inner_oof.observed, inner_oof.selected_inner_oof_prediction)
    BASE.close([selected[name] for name in columns], [selected_score[name] for name in columns],
               "Transfer selected inner OOF metrics")
    expected_mean = np.empty(len(training))
    for train, test in KFold(manifest["inner_folds"], shuffle=True, random_state=manifest["transfer_cv_seed"]).split(training):
        expected_mean[test] = y.iloc[training[train]].mean()
    BASE.close(inner_oof.mean_inner_oof_prediction, expected_mean, "Transfer inner fold-mean baseline")
    mean_rmse = score(y.iloc[training], expected_mean)["rmse"]
    qualified = bool(selected.prediction_variance_ratio >= gate and selected.rmse < mean_rmse)
    assert qualified == artifact["qualified_for_proxy"] == ensemble["qualified_for_proxy"]
    assert qualified == summary["transfer_qualified_for_proxy"]
    for saved in [artifact, ensemble]:
        np.testing.assert_array_equal(saved["train_global_indices"], training)
        np.testing.assert_array_equal(saved["excluded_global_indices"], excluded)
        assert saved["features"] == x.columns.tolist()
    check_pipeline_training(artifact["pipeline"], x.iloc[training], context["calibrated_features"])
    indices = np.asarray(ensemble["bootstrap_indices"])
    assert indices.shape == (60, len(training)) and not np.isin(indices, excluded).any()
    assert np.isin(indices, training).all()
    assert ensemble["selection_model_indices"] == list(range(40))
    assert ensemble["audit_model_indices"] == list(range(40, 60))
    assert ensemble["hyperparameter_uncertainty_included"] is False
    with np.load(folder / "bootstrap_indices.npz") as saved:
        np.testing.assert_array_equal(saved["indices"], indices)
        np.testing.assert_array_equal(saved["bootstrap_seeds"], ensemble["bootstrap_seeds"])
        np.testing.assert_array_equal(saved["estimator_seeds"], ensemble["estimator_seeds"])
    for i, (model, index, seed) in enumerate(zip(ensemble["models"], indices, ensemble["bootstrap_seeds"])):
        expected_seed = manifest["selection_bootstrap_seed"] + i if i < 40 else manifest["audit_bootstrap_seed"] + i - 40
        assert seed == expected_seed
        expected_index = np.random.default_rng(seed).choice(training, size=len(training), replace=True)
        np.testing.assert_array_equal(index, expected_index)
        check_pipeline_training(model, x.iloc[index], context["calibrated_features"])
    heldout = pd.read_csv(folder / "pils_style_holdout_predictions.csv")
    np.testing.assert_array_equal(heldout.beer, meta.beer.iloc[excluded])
    BASE.close(heldout.observed, y.iloc[excluded], "Pils holdout observed labels")
    BASE.close(heldout.prediction, np.asarray(artifact["pipeline"].predict(x.iloc[excluded])).reshape(-1),
               "Pils holdout frozen model predictions")
    holdout_metrics = pd.read_csv(folder / "pils_style_holdout_metrics.csv").set_index("model")
    for method, pred in [("optimized_single_target", heldout.prediction),
                         ("non_pils_mean", heldout.mean_baseline_prediction)]:
        BASE.close(holdout_metrics.loc[method, columns], list(score(heldout.observed, pred).values()),
                   "Pils source holdout metrics")
    context.update({"optimized_stale": artifact, "optimized_ensemble": ensemble,
                    "optimized_train": training, "optimized_sources": excluded})
    main = stored_metrics[stored_metrics.scope.eq("pooled_repeated_oof")].set_index("method")
    return {"outer_seeds": manifest["outer_seeds"], "outer_folds": manifest["outer_folds"],
            "candidate_count": len(specs), "nested_choices_recomputed": len(choices),
            "transfer_config": artifact["selected_config"], "qualified_for_proxy": qualified,
            "training_rows": len(training), "excluded_sources": len(excluded),
            "bootstrap_shape": list(indices.shape), "preprocessing_training_checks": 61,
            "nested_selected_rmse": float(main.loc["nested_selected", "rmse"]),
            "old_multioutput_same_split_rmse": float(main.loc["old_multioutput_extra_trees", "rmse"]),
            "fixed_single_tree_rmse": float(main.loc["fixed_single_extra_trees", "rmse"]),
            "pils_holdout_rmse": float(holdout_metrics.loc["optimized_single_target", "rmse"]),
            "pils_holdout_mean_rmse": float(holdout_metrics.loc["non_pils_mean", "rmse"])}


def check_generation(context):
    folder = OUT / "generation"
    meta, x = context["meta"], context["x"]
    summary = BASE.read_json(folder / "summary.json")
    assignments = pd.read_csv(folder / "cv_assignments.csv")
    metrics = pd.read_csv(folder / "cv_metrics.csv")
    inner = pd.read_csv(folder / "inner_scores_outer.csv")
    choices = pd.read_csv(folder / "inner_selection_outer.csv")
    assert not assignments.duplicated(["repeat", "row_index"]).any()
    assert len(assignments) == len(meta) * len(summary["outer_seeds"])
    assert not metrics.duplicated(["repeat", "fold", "style", "method", "mc"]).any()
    expected_n = len(summary["outer_seeds"]) * summary["outer_folds"] * 4 * 5 * summary["mc_repeats"]
    assert len(metrics) == expected_n == summary["cv_records"]
    for repeat, seed in enumerate(summary["outer_seeds"], 1):
        for fold, (train, test) in enumerate(StratifiedKFold(3, shuffle=True, random_state=seed).split(x, meta.tasting_category_fine), 1):
            assigned = assignments[assignments.repeat.eq(repeat) & assignments.fold.eq(fold)]
            assert set(assigned.row_index) == set(test)
            assert assigned.beer.tolist() == assigned.row_index.map(meta.beer).tolist()
            subset = metrics[metrics.repeat.eq(repeat) & metrics.fold.eq(fold)]
            assert subset.train_n.eq(len(train)).all()
            assert subset.generated_n.eq(summary["generated_n_each_draw"]).all()
            assert set(subset.mc) == set(range(summary["mc_repeats"]))
            for style in BASE.GENERATION_STYLES:
                assert subset.loc[subset["style"].eq(style), "holdout_style_n"].eq(
                    meta.tasting_category_fine.iloc[test].eq(style).sum()).all()
    gates = summary["novelty_gates"]
    expected_eligible = (inner.copy_rate.eq(0) & inner.observed_training_copy_rate.eq(0)
                         & inner.nn_ratio.ge(gates["minimum_inner_nn_ratio"])
                         & inner.nearest_equal_feature_fraction.le(gates["maximum_inner_nearest_equal_fraction"]))
    assert inner.eligible.equals(expected_eligible)
    selected_lookup = {}
    for (repeat, fold, style), group in inner.groupby(["outer_repeat", "outer_fold", "style"]):
        assert len(group) == 2 * len(summary["candidate_methods"])
        grouped = group.groupby("method").agg(energy_pca5=("energy_pca5", "mean"),
                                              eligible=("eligible", "all"))
        acceptable = grouped[grouped.eligible]
        selected = acceptable.energy_pca5.idxmin() if len(acceptable) else "legacy_copula"
        actual = choices[choices.repeat.eq(repeat) & choices.fold.eq(fold) & choices["style"].eq(style)]
        assert len(actual) == len(summary["candidate_methods"]) and actual.selected.sum() == 1
        assert actual.loc[actual.selected, "method"].iloc[0] == selected
        BASE.close(actual.set_index("method").loc[grouped.index, "energy_pca5"], grouped.energy_pca5,
                   "Generation training-only inner energy")
        selected_lookup[(repeat, fold, style)] = selected
    folds = pd.read_csv(folder / "cv_fold_means.csv")
    keys = ["repeat", "fold", "style", "method"]
    columns = [column for column in folds if column not in keys]
    for row in metrics[metrics.method.eq("nested_selected")].itertuples(index=False):
        method = selected_lookup[(row.repeat, row.fold, row.style)]
        assert row.selected_method == method
        match = metrics[metrics.repeat.eq(row.repeat) & metrics.fold.eq(row.fold)
                        & metrics["style"].eq(row.style) & metrics.mc.eq(row.mc) & metrics.method.eq(method)]
        assert len(match) == 1
        BASE.close([getattr(row, column) for column in columns], match[columns].iloc[0],
                   "Nested outer outcome must equal its training-selected candidate")
    expected_fold = metrics.groupby(keys, as_index=False)[columns].mean()
    BASE.same_table(folds, expected_fold, keys, columns, "Optimized generation MC fold means")
    expected_summary = expected_fold.groupby(["style", "method"])[columns].agg(["mean", "std"])
    expected_summary.columns = [f"{metric}_{stat}" for metric, stat in expected_summary.columns]
    expected_summary = expected_summary.reset_index()
    BASE.same_table(pd.read_csv(folder / "cv_summary.csv"), expected_summary, ["style", "method"],
                    [column for column in expected_summary if column not in {"style", "method"}],
                    "Optimized generation summary")
    model = joblib.load(folder / "fullmodel.joblib")
    assert model.columns_ == x.columns.tolist()
    reference = model.fitted_candidates_["legacy_copula"]
    BASE.close(reference.median_, np.nanmedian(x, axis=0), "Full optimized reference medians")
    generated_tables, copy_count = [], 0
    for i, style in enumerate(BASE.GENERATION_STYLES):
        generated = pd.read_csv(folder / f"generated_{style.replace('/', '-')}.csv")
        assert len(generated) == summary["generated_n_each_draw"] and generated.sample_id.is_unique
        assert generated.target_style.eq(style).all()
        fake = generated[x.columns].to_numpy()
        assert np.isfinite(fake).all() and generated[context["calibrated_features"]].ge(0).all().all()
        selected = model.selected_methods_[style]
        assert selected == summary["full_selected_methods"][style]
        candidate = model.fitted_candidates_[selected]
        support = candidate.empirical_values_ if selected == "legacy_copula" else candidate.tables_[style][0]
        for j in range(x.shape[1]):
            assert BASE.supported(fake[:, j], support[j]).all(), f"Unsupported optimized value: {style}/{x.columns[j]}"
        BASE.close(fake, model.sample(style, len(generated), summary["outer_seeds"][0] + 900000 + i),
                   "Saved optimized generation replay")
        raw = x.to_numpy()
        copied = np.any(np.all(np.isclose(fake[:, None, :], raw[None, :, :], rtol=1e-10, atol=1e-12)
                              | np.isnan(raw)[None, :, :], axis=2), axis=1)
        copy_count += int(copied.sum())
        assert not copied.any(), f"Observed-field training-row copy generated: {style}"
        generated_tables.append(generated)
    assert copy_count == 0 and summary["full_generation_observed_copy_rate"] == 0
    context["optimized_generated"] = pd.concat(generated_tables, ignore_index=True)
    return {"cv_records": len(metrics), "inner_records": len(inner),
            "inner_choices_recomputed": len(selected_lookup), "generated_rows": sum(map(len, generated_tables)),
            "observed_training_row_copies": copy_count, "full_selected_methods": summary["full_selected_methods"],
            "training_empirical_support": True, "nested_outcomes_match_selected_methods": True}


def check_fixed_stale(context):
    folder = OUT / "stale_fixed"
    meta, x, y = context["meta"], context["x"], context["y"].stale_hops
    manifest = BASE.read_json(folder / "run_manifest.json")
    summary = BASE.read_json(folder / "summary.json")
    assert manifest["parameter_tuning"] is False and manifest["not_independent_external_confirmation"] is True
    oof = pd.read_csv(folder / "oof_predictions.csv")
    rows, fold_rows = [], []
    columns = list(score(y, y))
    for repeat, seed in enumerate(manifest["outer_seeds"], 1):
        fold_ids = np.zeros(len(x), int)
        for fold, (_, test) in enumerate(KFold(5, shuffle=True, random_state=seed).split(x), 1):
            fold_ids[test] = fold
        for method in oof.method.unique():
            group = oof[oof.repeat.eq(repeat) & oof.method.eq(method)].set_index("beer").loc[meta.beer]
            assert len(group) == len(x)
            np.testing.assert_array_equal(group.fold, fold_ids)
            BASE.close(group.observed, y, "Fixed forest OOF labels")
            rows.append({"scope": "per_seed", "repeat": repeat, "method": method,
                         **score(group.observed, group.prediction)})
            for fold in range(1, 6):
                test = fold_ids == fold
                fold_rows.append({"repeat": repeat, "fold": fold, "method": method,
                                  **score(y.loc[test], group.prediction.to_numpy()[test])})
                if method == "mean":
                    BASE.close(group.prediction.to_numpy()[test], y.loc[~test].mean(), "Fixed study fold mean")
    for method in oof.method.unique():
        group = oof[oof.method.eq(method)]
        rows.append({"scope": "pooled_repeated_oof", "repeat": 0, "method": method,
                     **score(group.observed, group.prediction)})
    metrics = pd.read_csv(folder / "metrics.csv")
    BASE.same_table(metrics, pd.DataFrame(rows), ["scope", "repeat", "method"], columns, "Fixed forest OOF metrics")
    BASE.same_table(pd.read_csv(folder / "metrics_by_fold.csv"), pd.DataFrame(fold_rows),
                    ["repeat", "fold", "method"], columns, "Fixed forest fold metrics")
    artifact = joblib.load(folder / "transfer_stale_model.joblib")
    ensemble = joblib.load(folder / "bootstrap_ensemble.joblib")
    train = np.flatnonzero(~meta.tasting_category_fine.eq("Pils/Lager"))
    excluded = np.flatnonzero(meta.tasting_category_fine.eq("Pils/Lager"))
    for saved in [artifact, ensemble]:
        np.testing.assert_array_equal(saved["train_global_indices"], train)
        np.testing.assert_array_equal(saved["excluded_global_indices"], excluded)
        assert saved["features"] == x.columns.tolist() and saved["parameter_tuning"] is False
    inner = pd.read_csv(folder / "transfer_inner_oof.csv")
    np.testing.assert_array_equal(inner.beer, meta.beer.iloc[train])
    BASE.close(inner.observed, y.iloc[train], "Fixed transfer inner observed")
    inner_scores = score(inner.observed, inner.fixed_inner_oof_prediction)
    mean = np.empty(len(train))
    for tr, te in KFold(3, shuffle=True, random_state=manifest["transfer_cv_seed"]).split(train):
        mean[te] = y.iloc[train[tr]].mean()
    BASE.close(inner.mean_inner_oof_prediction, mean, "Fixed transfer fold means")
    qualified = bool(inner_scores["prediction_variance_ratio"] >= .05
                     and inner_scores["rmse"] < score(y.iloc[train], mean)["rmse"])
    assert qualified == artifact["qualified_for_proxy"] == ensemble["qualified_for_proxy"]
    indices = np.asarray(ensemble["bootstrap_indices"])
    assert indices.shape == (60, len(train)) and not np.isin(indices, excluded).any()
    assert np.isin(indices, train).all()
    assert ensemble["selection_model_indices"] == list(range(40))
    assert ensemble["audit_model_indices"] == list(range(40, 60))
    assert artifact["pipeline"].named_steps["model"].n_estimators == 300
    all_models = [(artifact["pipeline"], train)] + list(zip(ensemble["models"], indices))
    for i, (model, training_indices) in enumerate(all_models):
        imputer = model.named_steps["imputer"]
        BASE.close(imputer.statistics_, np.nanmedian(x.iloc[training_indices], axis=0), "Fixed model source-excluded medians")
        np.testing.assert_array_equal(imputer.indicator_.features_, np.flatnonzero(x.iloc[training_indices].isna().any(axis=0)))
        tree = model.named_steps["model"]
        assert tree.min_samples_leaf == 3 and tree.max_features == .7
        assert tree.n_estimators == (300 if i == 0 else 120)
    with np.load(folder / "bootstrap_indices.npz") as saved:
        np.testing.assert_array_equal(saved["indices"], indices)
        np.testing.assert_array_equal(saved["bootstrap_seeds"], ensemble["bootstrap_seeds"])
    for i, seed in enumerate(ensemble["bootstrap_seeds"]):
        expected_seed = manifest["selection_bootstrap_seed"] + i if i < 40 else manifest["audit_bootstrap_seed"] + i - 40
        assert seed == expected_seed
        np.testing.assert_array_equal(indices[i], np.random.default_rng(seed).choice(train, len(train), replace=True))
    heldout = pd.read_csv(folder / "pils_style_holdout_predictions.csv")
    BASE.close(heldout.prediction, np.asarray(artifact["pipeline"].predict(x.iloc[excluded])).reshape(-1), "Fixed source holdout")
    holdout = pd.read_csv(folder / "pils_style_holdout_metrics.csv").set_index("model")
    for name, column in [("fixed_single_extra_trees", "prediction"), ("non_pils_mean", "mean_baseline_prediction"),
                         ("old_multioutput_transfer_proxy", "old_multioutput_prediction")]:
        BASE.close(holdout.loc[name, columns], list(score(heldout.observed, heldout[column]).values()), "Fixed holdout metrics")
    context.update({"fixed_stale": artifact, "fixed_ensemble": ensemble,
                    "optimized_train": train, "optimized_sources": excluded})
    pooled = metrics[metrics.scope.eq("pooled_repeated_oof")].set_index("method")
    cluster = BASE.read_json(folder / "paired_cluster_bootstrap.json")
    errors = oof.assign(error_squared=(oof.prediction - oof.observed) ** 2)
    per_beer = errors.pivot_table(index="beer", columns="method", values="error_squared", aggfunc="mean").sort_index()
    fixed_errors = per_beer.fixed_single_extra_trees.to_numpy()
    old_errors = per_beer.old_multioutput_extra_trees.to_numpy()
    draws = np.random.default_rng(cluster["seed"]).integers(0, len(per_beer), (cluster["repeats"], len(per_beer)))
    fixed_rmse = np.sqrt(fixed_errors[draws].mean(axis=1))
    old_rmse = np.sqrt(old_errors[draws].mean(axis=1))
    differences, reductions = fixed_rmse - old_rmse, 1 - fixed_rmse / old_rmse
    BASE.close(cluster["rmse_difference_percentile95"], np.quantile(differences, [.025, .975]), "Beer-cluster RMSE difference interval")
    BASE.close(cluster["relative_rmse_reduction_percentile95"], np.quantile(reductions, [.025, .975]), "Beer-cluster relative change interval")
    BASE.close(cluster["bootstrap_fraction_rmse_difference_below_zero"], np.mean(differences < 0), "Beer-cluster difference fraction")
    BASE.close(cluster["point_rmse_difference"], np.sqrt(fixed_errors.mean()) - np.sqrt(old_errors.mean()), "Beer-cluster point difference")
    return {"extra_internal_seeds": manifest["outer_seeds"], "parameter_tuning": False,
            "fixed_single_oof_rmse": float(pooled.loc["fixed_single_extra_trees", "rmse"]),
            "same_split_old_multioutput_rmse": float(pooled.loc["old_multioutput_extra_trees", "rmse"]),
            "same_split_mean_rmse": float(pooled.loc["mean", "rmse"]),
            "transfer_qualified": qualified, "bootstrap_shape": list(indices.shape),
            "beer_cluster_bootstrap_repeats": cluster["repeats"],
            "rmse_difference_interval": cluster["rmse_difference_percentile95"],
            "pils_holdout_rmse": float(holdout.loc["fixed_single_extra_trees", "rmse"]),
            "pils_holdout_mean_rmse": float(holdout.loc["non_pils_mean", "rmse"])}


def check_transfer(context):
    folder = OUT / "transfer"
    meta, x = context["meta"], context["x"]
    train, sources = context["optimized_train"], context["optimized_sources"]
    summary = BASE.read_json(folder / "summary.json")
    scenarios = BASE.read_json(folder / "threshold_scenarios.json")
    thresholds = pd.read_excel(data_dir() / "8.xlsx", sheet_name="Sensory thresholds").set_index("field_name_R")
    assert len(scenarios) == 10
    expected_ids = {f"{statistic}_x{factor:.1f}" for statistic in ["median", "mean"] for factor in [.8, .9, 1, 1.1, 1.2]}
    assert {scenario["scenario"] for scenario in scenarios} == expected_ids
    for scenario in scenarios:
        for field in BASE.DEFECT_FIELDS:
            raw = thresholds.at[field, "threshold_" + scenario["statistic"]]
            actual = scenario["thresholds"][field]
            if pd.isna(raw):
                assert actual is None and field == "X.E..2.nonenal"
            else:
                BASE.close(actual, raw * scenario["factor"], "Threshold stress scenario")
    assert scenarios == summary["threshold_scenarios"]
    geometry = joblib.load(SOLUTION / "outputs/q3/transfer/transfer_copula.joblib")
    reference = geometry["copula"]
    np.testing.assert_array_equal(geometry["training_rows"], train)
    generator_artifact = joblib.load(folder / "transfer_generator.joblib")
    np.testing.assert_array_equal(generator_artifact["training_rows"], train)
    np.testing.assert_array_equal(generator_artifact["source_rows"], sources)
    generator = generator_artifact["generator"]
    BASE.close(generator.fitted_candidates_["legacy_copula"].median_, np.nanmedian(x.iloc[train], axis=0), "Source-excluded new generator medians")
    old_draws = pd.read_csv(folder / "old_base_draws.csv")
    new_draws = pd.read_csv(folder / "optimized_base_draws.csv")
    assert old_draws.columns.tolist() == new_draws.columns.tolist() == x.columns.tolist()
    BASE.close(old_draws, reference.sample("Tripel", len(old_draws), 20261023 + 301), "Frozen old generator base draws")
    BASE.close(new_draws, generator.sample("Tripel", len(new_draws), 20261023 + 301), "Source-excluded new generator base draws")
    assert bool(np.array_equal(old_draws.to_numpy(), new_draws.to_numpy())) == summary["base_draws_equal"]
    report = pd.read_csv(folder / "transfer_summary.csv")
    diagnostics = pd.read_csv(folder / "candidate_diagnostics.csv")
    spectra = pd.read_csv(folder / "transferred_spectra.csv").set_index("transfer_id")
    paired = pd.read_csv(folder / "stale_paired_deltas.csv")
    sensitivity = pd.read_csv(folder / "threshold_sensitivity.csv")
    profiles = pd.read_csv(folder / "sensory_before_after.csv")
    policies = pd.read_csv(folder / "policy_summary.csv")
    ensemble = context["fixed_ensemble"]
    old_ensemble = joblib.load(SOLUTION / "outputs/q3/sensory/stale_bootstrap.joblib")
    profile_model = joblib.load(SOLUTION / "outputs/q3/sensory/transfer_proxy.joblib")
    target = x.loc[meta.tasting_category_fine.eq("Tripel")]
    tz = reference.transform(target)
    dimension = x.shape[1]
    neighbors = cdist(tz, tz) / np.sqrt(dimension)
    np.fill_diagonal(neighbors, np.inf)
    support_limit = np.quantile(neighbors.min(axis=1), .95)
    BASE.close(support_limit, summary["support_nn_q95"], "Unchanged support limit")
    assert summary["target_improvement_minimum"] == .1 and summary["latent_rms_change_minimum"] == .05
    assert summary["selection_models"] == 40 and summary["audit_models"] == 20
    medians = x.iloc[train].median()
    minimum, maximum = x.iloc[train].min().to_numpy(), x.iloc[train].max().to_numpy()
    target_median = target.median().to_numpy()
    p_selected = report[report.candidate_index.notna()]
    assert set(p_selected.transfer_id) == set(spectra.index) and spectra.index.is_unique
    selected_predictions = np.vstack([model.predict(spectra[x.columns]) for model in ensemble["models"]])
    old_predictions = np.vstack([model.predict(spectra[x.columns]) for model in old_ensemble["models"]])
    selected_profiles = profile_model["pipeline"].predict(spectra[x.columns])
    max_quantile_error, argmin_checks, scenario_checks = 0.0, 0, 0
    for source_index in sources:
        beer = int(meta.beer.iloc[source_index])
        raw_source = x.iloc[[source_index]]
        source = raw_source.fillna(medians)
        caps = {}
        for field in BASE.DEFECT_FIELDS:
            levels = [scenario["thresholds"][field] for scenario in scenarios if scenario["thresholds"][field] is not None]
            value = float(raw_source[field].iloc[0])
            caps[field] = max(value, min(levels)) if levels else value
            if levels:
                BASE.close(caps[field], min(max(value, level) for level in levels), "Robust minimum-of-scenario cap")
        arrays, families, gammas, draws = [], [], [], []
        for family, base in [("old_copula", old_draws), ("optimized", new_draws)]:
            for gamma in [.25, .5, .75, 1.]:
                arrays.append((1-gamma) * source.to_numpy()[0] + gamma * base.to_numpy())
                families.extend([family] * len(base)); gammas.extend([gamma] * len(base)); draws.extend(range(len(base)))
        for gamma in [.25, .5, .75, 1.]:
            arrays.append(((1-gamma) * source.to_numpy()[0] + gamma * target_median)[None])
            families.append("shared_target_median"); gammas.append(gamma); draws.append(-1)
        values = pd.DataFrame(np.vstack(arrays), columns=x.columns)
        for field, cap in caps.items():
            values[field] = values[field].clip(lower=0, upper=cap)
        diagnostic = diagnostics[diagnostics.source_beer.eq(beer)].sort_values("candidate_index")
        assert len(diagnostic) == len(values)
        np.testing.assert_array_equal(diagnostic.candidate_index, np.arange(len(values)))
        assert diagnostic.family.tolist() == families
        BASE.close(diagnostic.gamma, gammas, "Reconstructed interpolation fractions")
        np.testing.assert_array_equal(diagnostic.draw, draws)
        z, sz = reference.transform(values), reference.transform(source)
        distance, before = reference.distance(values, "Tripel"), float(reference.distance(source, "Tripel")[0])
        nn = cdist(z, tz).min(axis=1) / np.sqrt(dimension)
        change = np.linalg.norm(z - sz, axis=1) / np.sqrt(dimension)
        improvement = 1 - distance / max(before, 1e-8)
        support = (nn <= support_limit + TOL) & (values.to_numpy() >= minimum-TOL).all(axis=1) & (values.to_numpy() <= maximum+TOL).all(axis=1)
        chemical = BASE.chemical_flags(values, caps)
        source_pred = np.array([float(np.asarray(model.predict(source)).reshape(-1)[0]) for model in ensemble["models"]])
        delta = np.column_stack([np.asarray(model.predict(values)).reshape(-1)-source_pred[b] for b, model in enumerate(ensemble["models"][:40])])
        q95 = np.quantile(delta, .95, axis=1)
        nonzero = np.any(np.abs(delta) > 1e-8, axis=1)
        feasible = support & chemical & (change >= .05) & (improvement >= .1) & (q95 <= TOL) & nonzero
        BASE.close(diagnostic[["target_distance", "target_improvement", "target_nn_rms", "change_latent_rms", "selection_q95"]],
                   np.column_stack([distance, improvement, nn, change, q95]), "Independently rebuilt candidate geometry and screening quantiles")
        np.testing.assert_array_equal(diagnostic.selection_feasible, feasible)
        np.testing.assert_array_equal(diagnostic.support_ok, support)
        np.testing.assert_array_equal(diagnostic.all_thresholds_chemical_ok, chemical)
        np.testing.assert_array_equal(diagnostic.nonzero_paired_signal, nonzero)
        center_scale = max(float(np.linalg.norm(sz[0] - tz.mean(axis=0)) / np.sqrt(dimension)), 1e-8)
        masks = {"old_generator": np.isin(families, ["old_copula", "shared_target_median"]),
                 "optimized_generator": np.isin(families, ["optimized", "shared_target_median"]),
                 "combined_pool": np.ones(len(values), bool)}
        old_source_pred = np.array([float(np.asarray(model.predict(source)).reshape(-1)[0]) for model in old_ensemble["models"]])
        for _, row in report[report.source_beer.eq(beer)].iterrows():
            mask = masks[row.pool]
            allowed = feasible & mask
            assert row.n_candidates == mask.sum() and row.n_selection_feasible == allowed.sum()
            variation = np.std(delta[mask].mean(axis=1))
            zero_fraction = np.mean(np.all(np.abs(delta[mask]) <= 1e-8, axis=1))
            BASE.close([row.candidate_mean_delta_sd, row.candidate_zero_paired_fraction], [variation, zero_fraction], "Proxy responsiveness diagnostics")
            if not allowed.any():
                assert pd.isna(row.candidate_index) and row.status == "no_candidate_found"
                continue
            objective = distance / max(before, 1e-8) + row["lambda"] * change / center_scale
            chosen = int(np.flatnonzero(allowed)[np.argmin(objective[allowed])])
            assert chosen == int(row.candidate_index), "Candidate differs from fixed selection-only argmin"
            argmin_checks += 1
            final = spectra.loc[[row.transfer_id], x.columns]
            BASE.close(final, values.iloc[[chosen]], "Selected spectrum reconstructed from base draws")
            saved_position = spectra.index.get_loc(row.transfer_id)
            all_delta = selected_predictions[:, saved_position]-source_pred
            old_delta = old_predictions[:, saved_position]-old_source_pred
            quantities = [np.quantile(all_delta[:40], .95), np.quantile(all_delta[40:], .95),
                          np.quantile(old_delta[:30], .95), np.quantile(old_delta[30:], .95)]
            actual = [row.selection_q95, row.audit_q95, row.old_selection_q95, row.old_audit_q95]
            BASE.close(actual, quantities, "Selected and frozen-architecture audit quantiles")
            max_quantile_error = max(max_quantile_error, float(np.max(np.abs(np.asarray(actual)-quantities))))
            informative = bool(variation > 1e-6 and nonzero[chosen])
            qualified = bool(context["fixed_stale"]["qualified_for_proxy"])
            passed = bool(qualified and informative and quantities[1] <= TOL)
            old_consensus = bool(quantities[2] <= TOL and quantities[3] <= TOL)
            assert passed == row.all_model_checks_passed and old_consensus == row.old_proxy_consensus
            assert bool(passed and old_consensus) == row.cross_proxy_checks_passed
            expected_status = "proxy_not_qualified" if not qualified else "proxy_uninformative" if not informative else "model_checks_passed" if quantities[1] <= TOL else "resampling_audit_failed"
            assert row.status == expected_status
            for label, differences in [("selection", all_delta[:40]), ("audit", all_delta[40:]),
                                       ("old_selection", old_delta[:30]), ("old_audit", old_delta[30:])]:
                recorded = paired[paired.transfer_id.eq(row.transfer_id) & paired.ensemble.eq(label)].sort_values("model_index")
                np.testing.assert_array_equal(recorded.model_index, np.arange(len(differences)))
                BASE.close(recorded.paired_delta, differences, "Per-model screening/audit paired differences")
            check_rows = sensitivity[sensitivity.transfer_id.eq(row.transfer_id)]
            assert len(check_rows) == len(scenarios) * len(BASE.DEFECT_FIELDS)
            for check in check_rows.itertuples(index=False):
                scenario = next(item for item in scenarios if item["scenario"] == check.scenario)
                threshold = scenario["thresholds"][check.feature]
                original, transferred = float(raw_source[check.feature].iloc[0]), float(final[check.feature].iloc[0])
                cap = max(original, threshold) if threshold is not None else original
                BASE.close([check.source, check.transferred, check.upper_bound, check.robust_upper_bound],
                           [original, transferred, cap, caps[check.feature]], "All-scenario concentration audit")
                assert bool(0 <= transferred <= cap + TOL) == check.passed
                scenario_checks += 1
            source_profile = profile_model["pipeline"].predict(source)[0]
            after_profile = selected_profiles[saved_position].copy()
            position = context["y"].columns.get_loc("stale_hops")
            source_profile[position] = float(np.asarray(context["fixed_stale"]["pipeline"].predict(source)).reshape(-1)[0])
            after_profile[position] = float(np.asarray(context["fixed_stale"]["pipeline"].predict(final)).reshape(-1)[0])
            for phase, profile in [("source", source_profile), ("transferred", after_profile)]:
                recorded = profiles[profiles.transfer_id.eq(row.transfer_id) & profiles.phase.eq(phase)]
                assert len(recorded) == 1
                BASE.close(recorded[context["y"].columns], profile[None], "Mixed-model sensory profile")
        print(f"Verified robust candidates: source {beer}", flush=True)
    expected_policies = []
    for (pool, weight), group in report.groupby(["pool", "lambda"]):
        expected_policies.append({"pool": pool, "lambda": weight, "sources": len(group),
                                 "selected": int(group.candidate_index.notna().sum()),
                                 "model_checks_passed": int(group.all_model_checks_passed.sum()),
                                 "cross_proxy_checks_passed": int(group.cross_proxy_checks_passed.sum()),
                                 "audit_failed": int(group.status.eq("resampling_audit_failed").sum()),
                                 "no_candidate": int(group.status.eq("no_candidate_found").sum())})
    BASE.same_table(policies, pd.DataFrame(expected_policies), ["pool", "lambda"],
                    ["sources", "selected", "model_checks_passed", "cross_proxy_checks_passed", "audit_failed", "no_candidate"], "Robust policy summary")
    default = report[report.is_default_policy]
    assert default.pool.eq(summary["default_pool"]).all() and default["lambda"].eq(summary["default_lambda"]).all()
    counts = {"sources": len(default), "selected": int(default.candidate_index.notna().sum()),
              "model_checks_passed_all_thresholds": int(default.all_model_checks_passed.sum()),
              "cross_proxy_checks_passed_all_thresholds": int(default.cross_proxy_checks_passed.sum()),
              "audit_failed": int(default.status.eq("resampling_audit_failed").sum()),
              "no_candidate": int(default.status.eq("no_candidate_found").sum())}
    assert counts == summary["default_policy"]
    archived = pd.read_csv(folder / "archived_threshold_recheck.csv")
    old_spectra = pd.read_csv(SOLUTION / "outputs/q3/transfer/transferred_spectra.csv").set_index("transfer_id")
    for row in archived.itertuples(index=False):
        source = x.loc[meta.beer.eq(row.source_beer)]
        selected = old_spectra.loc[[row.transfer_id], x.columns]
        scenario = next(item for item in scenarios if item["scenario"] == row.scenario)
        caps = {field: max(float(source[field].iloc[0]), level) if level is not None else float(source[field].iloc[0])
                for field, level in scenario["thresholds"].items()}
        assert bool(BASE.chemical_flags(selected, caps)[0]) == row.chemical_passed
    original_passed = archived[archived.old_model_checks_passed].groupby("source_beer").chemical_passed.all()
    assert int(original_passed.sum()) == summary["archived_baseline"]["model_checks_passed_all_thresholds_same_spectra"]
    return {"scenario_count": len(scenarios), "sources": len(sources), "reconstructed_candidates": len(diagnostics),
            "fixed_selection_argmin_checks": argmin_checks, "concentration_scenario_checks": scenario_checks,
            "maximum_paired_quantile_error": max_quantile_error,
            "base_draws_equal": summary["base_draws_equal"], "default_policy": counts}


def check_report(context):
    generated = context["optimized_generated"]
    meta, x, y = context["meta"], context["x"], context["y"]
    profiles = pd.read_csv(OUT / "generated_sensory_profiles.csv")
    assert len(profiles) == len(generated) == 1200 and profiles.sample_id.is_unique
    assert set(profiles.sample_id) == set(generated.sample_id)
    old = joblib.load(SOLUTION / "outputs/q3/sensory/sensory_model.joblib")
    expected = old["pipeline"].predict(generated[x.columns])
    expected[:, y.columns.get_loc("stale_hops")] = np.asarray(
        context["fixed_stale"]["pipeline"].predict(generated[x.columns])).reshape(-1)
    aligned = profiles.set_index("sample_id").loc[generated.sample_id]
    assert aligned.target_style.tolist() == generated.target_style.tolist()
    BASE.close(aligned[y.columns], expected, "Optimized generated 44+1 sensory profiles")
    summary_rows = []
    for style in BASE.GENERATION_STYLES:
        positions = generated.target_style.eq(style).to_numpy()
        for j, target in enumerate(y):
            values = expected[positions, j]
            summary_rows.append({"style": style, "target": target,
                                 "predicted_mean": values.mean(), "predicted_p05": np.quantile(values, .05),
                                 "predicted_p95": np.quantile(values, .95),
                                 "real_observed_mean": y.loc[meta.tasting_category_fine.eq(style), target].mean()})
    sensory_summary = pd.read_csv(OUT / "generated_sensory_summary.csv")
    BASE.same_table(sensory_summary, pd.DataFrame(summary_rows), ["style", "target"],
                    ["predicted_mean", "predicted_p05", "predicted_p95", "real_observed_mean"],
                    "Optimized generated sensory summary")
    assert sensory_summary.scale.eq("attachment_4_standardized").all()
    assert sensory_summary.interval_meaning.eq("generated_spectrum_spread_not_confidence_interval").all()
    folder = OUT / "transfer"
    report = pd.read_csv(folder / "transfer_summary.csv")
    default = report[report.is_default_policy]
    current = pd.read_csv(folder / "threshold_sensitivity.csv")
    archived = pd.read_csv(folder / "archived_threshold_recheck.csv")
    expected_counts = []
    for scenario in BASE.read_json(folder / "threshold_scenarios.json"):
        name = scenario["scenario"]
        old_rows = archived[archived.scenario.eq(name)]
        passing = current[current.scenario.eq(name)].groupby("transfer_id").passed.all()
        chemical = default.transfer_id.map(passing).fillna(False)
        expected_counts.append({"scenario": name, "statistic": scenario["statistic"], "factor": scenario["factor"],
                                "old_model_checks_and_chemical": int((old_rows.old_model_checks_passed & old_rows.chemical_passed).sum()),
                                "new_chemical_only": int(chemical.sum()),
                                "new_model_checks_and_chemical": int((chemical & default.all_model_checks_passed).sum()),
                                "cross_proxy_checks_and_chemical": int((chemical & default.cross_proxy_checks_passed).sum())})
    BASE.same_table(pd.read_csv(folder / "threshold_pass_counts.csv"), pd.DataFrame(expected_counts), ["scenario"],
                    ["factor", "old_model_checks_and_chemical", "new_chemical_only", "new_model_checks_and_chemical", "cross_proxy_checks_and_chemical"],
                    "All-scenario pass-count report")
    results = BASE.read_json(OUT / "results_summary.json")
    cv = pd.read_csv(OUT / "generation/cv_summary.csv")
    gains = {}
    for style in BASE.GENERATION_STYLES:
        rows = cv[cv["style"].eq(style)].set_index("method")
        gains[style] = float(1 - rows.loc["nested_selected", "energy_pca5_mean"] / rows.loc["legacy_copula", "energy_pca5_mean"])
        BASE.close(results["generation_energy_reduction_vs_matched_legacy"][style], gains[style], "Generation report gains")
    fixed = BASE.read_json(OUT / "stale_fixed/summary.json")
    transfer = BASE.read_json(folder / "summary.json")
    assert results["default_transfer"] == transfer["default_policy"]
    assert results["base_draws_equal"] == transfer["base_draws_equal"]
    assert results["source_style_prediction_improved"] == fixed["pils_holdout_improves_old"]
    BASE.close(results["fixed_stale_rmse_reduction_vs_matched_old"], fixed["rmse_reduction_vs_old_multioutput"], "Fixed-proxy report change")
    BASE.close(results["paired_rmse_difference_interval"], fixed["paired_cluster_bootstrap"]["rmse_difference_percentile95"], "Reported paired uncertainty interval")
    text = (SOLUTION / "docs/q3_optimization_results.md").read_text(encoding="utf-8")
    assert "尚未证实稳定的预测提升" in text
    assert f"**{transfer['default_policy']['model_checks_passed_all_thresholds']}/11**" in text
    assert f"**{transfer['default_policy']['cross_proxy_checks_passed_all_thresholds']}/11**" in text
    for style in ["Kriek", "Lambic"]:
        assert f"{100*gains[style]:.1f}%" in text
    figures = ["generation_and_prediction", "threshold_stability", "sensory_fingerprint_comparison"]
    for name in figures:
        for suffix in ["png", "svg"]:
            path = OUT / "figures" / f"{name}.{suffix}"
            assert path.is_file() and path.stat().st_size > 100
    return {"generated_sensory_rows": len(profiles), "sensory_targets": len(y.columns),
            "profile_model": "44 original full-data outputs plus source-excluded fixed stale target",
            "scenario_count_rows": len(expected_counts), "summary_values_recomputed": True,
            "figures_with_png_and_svg": len(figures), "results_report_exists": True}


def main():
    start = perf_counter()
    result = {"status": "running", "checks": [],
              "scope": "Frozen baseline integrity, repeated internal validation and robust candidate model checks"}
    active = "preserved_baseline_before"
    try:
        result["checks"].append({"name": active, "details": check_baseline()})
        active = "raw_sources_and_hybrid_contract"
        provenance = BASE.check_hashes()
        context, contract = BASE.load_contract()
        result["checks"].append({"name": active, "details": {"sources": provenance, "contract": contract}})
        for active, operation in [("optimized_stale_nested_validation", check_stale),
                                  ("optimized_generation_nested_validation", check_generation),
                                  ("fixed_single_target_followup", check_fixed_stale),
                                  ("robust_threshold_transfer", check_transfer),
                                  ("report_profiles_and_threshold_counts", check_report)]:
            details = operation(context)
            result["checks"].append({"name": active, "details": details})
            print(f"Verified: {active}", flush=True)
        active = "preserved_baseline_after"
        result["checks"].append({"name": active, "details": check_baseline()})
        result["status"] = "passed"
        result["limitations"] = [
            "Repeated and nested validation reuse the same real samples; new seeds are not fresh independent data.",
            "Bootstrap audit shares real training data and does not provide calibrated post-search coverage.",
            "Numerical and chemical checks do not establish actual sensory improvement or brewing feasibility.",
        ]
    except Exception as error:
        result.update({"status": "failed", "failed_check": active,
                       "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        result["elapsed_seconds"] = perf_counter() - start
        (OUT / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                                          allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
