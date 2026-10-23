"""Recompute source diagnosis, grouped proxy selection and balanced transfer."""
from __future__ import annotations

import hashlib
import importlib
import json
from decimal import Decimal
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from common import SOLUTION, data_dir

BASE = importlib.import_module("10_q3_verify")
DOMAIN = importlib.import_module("19_q3_domain_verify")
OUT = SOLUTION / "outputs/q3_refined"
TOL = 1e-10


def check_frozen():
    details = DOMAIN.check_frozen()
    folder = SOLUTION / "outputs/q3_domain"
    manifest = BASE.read_json(OUT / "previous_hashes.json")
    actual = {str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file()}
    assert actual == {r["path"] for r in manifest["files"]}
    for row in manifest["files"]:
        raw = (folder / row["path"]).read_bytes()
        assert len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"], row["path"]
    details.append({"directory": "outputs/q3_domain", "unchanged_files": len(actual)})
    return details


def check_beer81(context):
    from q3_stale_optimized import FoldChemicalTransform
    folder = OUT / "beer81"
    meta, x, y45 = context["meta"], context["x"], context["y"]
    y = y45.stale_hops
    train = np.flatnonzero(~meta.tasting_category_fine.eq("Pils/Lager"))
    sources = np.flatnonzero(meta.tasting_category_fine.eq("Pils/Lager"))
    index = np.flatnonzero(meta.beer.eq(81)).item()
    summary = BASE.read_json(folder / "summary.json")
    for row in BASE.read_json(folder / "source_manifest.json"):
        assert hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest() == row["sha256"]
    identity = pd.read_csv(folder / "identity_join_audit.csv")
    assert len(identity) == 4 and identity.beer81_matches.eq(1).all()
    assert identity.beer_unique.all() and identity.style_matches_1csv.all() and identity.id_matches_printed_precision.all()
    assert identity.rows_after_exact_beer_join.eq(250).all()
    originals = {f:pd.read_csv(data_dir()/f,dtype={"beer_id":str}) for f in ["1.csv","4.csv","7.csv"]}
    originals["8.xlsx/calibrated"] = pd.read_excel(data_dir()/"8.xlsx",sheet_name="Calibrated chemical dataset",dtype={"beer_id":str})
    reference_id = Decimal(originals["1.csv"].set_index("beer").at[81,"beer_id"])
    for name, frame in originals.items():
        assert frame.beer.is_unique and set(frame.beer)==set(meta.beer)
        row = frame.set_index("beer").loc[81]
        assert row.tasting_category_fine=="Pils/Lager"
        printed = Decimal(row.beer_id)
        resolution = Decimal(10)**printed.as_tuple().exponent
        assert abs(reference_id-printed)<=resolution/2
        actual = identity.set_index("source").loc[name]
        assert actual.beer_id_unique_count==frame.beer_id.nunique()
        assert bool(actual.exact_id_matches_1csv)==(reference_id==printed)
    names = pd.read_excel(data_dir()/"8.xlsx", sheet_name="Sensory names")
    saved_names = pd.read_csv(folder/"sensory_target_metadata.csv")
    assert set(saved_names.R_name) == {"stale_hops", "barnyard", "t2n", "dms"}
    assert names.set_index("R_name").loc[["stale_hops", "barnyard"], "DataType"].eq("numeric").all()
    profile = pd.read_csv(folder/"chemical_profile_beer81.csv").set_index("feature").loc[x.columns]
    BASE.close(profile.value, x.iloc[index], "Beer81 hybrid measurements")
    BASE.close(profile[["nonpils_min","nonpils_q01","nonpils_median","nonpils_q99","nonpils_max"]],
               np.column_stack([x.iloc[train].min(),x.iloc[train].quantile(.01),x.iloc[train].median(),x.iloc[train].quantile(.99),x.iloc[train].max()]))
    outside = (x.iloc[index] < x.iloc[train].min())|(x.iloc[index] > x.iloc[train].max())
    np.testing.assert_array_equal(profile.outside_nonpils_range,outside)
    assert summary["sample_profile"]["outside_nonpils_feature_ranges"]==outside.sum()
    assert not x.iloc[index].isna().any() and x.iloc[index][context["calibrated_features"]].ge(0).all()
    assert summary["label_error_established"] is False and summary["analytical_lod_loq_available"] is False
    BASE.close(summary["stale_observed"], y.iloc[index])
    assert summary["stale_descending_rank_all250"] == int(y.rank(ascending=False, method="min").iloc[index])
    high = train[y.iloc[train].to_numpy() >= y.iloc[index]]
    assert summary["nonpils_stale_at_least_beer81"] == len(high)
    assert set(summary["nonpils_high_tail_beers"]) == set(meta.beer.iloc[high])
    raw = x.to_numpy(float).copy()
    for j, field in enumerate(x):
        if field in context["calibrated_features"]:
            positive = raw[train, j]
            positive = positive[np.isfinite(positive) & (positive > 0)]
            raw[:, j] = np.log1p(raw[:, j] / (np.median(positive) if len(positive) else 1.))
    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(raw[train])
    filled = imp.transform(raw)
    standard = StandardScaler().fit(filled[train]).transform(filled)
    aging = BASE.read_json(OUT/"stale/run_manifest.json")["aging_features"]
    positions = [x.columns.get_loc(c) for c in aging]
    spaces = {"unclipped_log_scaled_226": standard, "unclipped_log_scaled_aging24": standard[:, positions]}
    for name, subset in [("clipped_log_scaled_226", None), ("clipped_log_scaled_aging24", aging)]:
        z = FoldChemicalTransform(x.columns.tolist(), context["calibrated_features"], subset).fit(x.iloc[train]).transform(x)
        imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(z[train])
        z = imputer.transform(z)
        spaces[name] = StandardScaler().fit(z[train]).transform(z)
    reference = joblib.load(SOLUTION/"outputs/q3/transfer/transfer_copula.joblib")["copula"]
    spaces["old_copula_226"] = reference.transform(x)
    spaces["old_copula_aging24"] = spaces["old_copula_226"][:, positions]
    local = joblib.load(SOLUTION/"outputs/q3_domain/stale/transfer_stale_model.joblib")["pipeline"]
    spaces["frozen_domain_aging_pca8"] = local[:-1].transform(x)
    saved = pd.read_csv(folder/"chemical_ood_all_pils.csv").set_index(["space", "query_beer"])
    neighbors = pd.read_csv(folder/"nearest_training_neighbors.csv")
    full81 = pd.read_csv(folder/"beer81_all_training_neighbor_ranks.csv")
    assert len(saved) == 7*len(sources) and len(neighbors) == 7*len(sources)*25 and len(full81) == 7*len(train)
    for name, z in spaces.items():
        training_d = cdist(z[train], z[train])/np.sqrt(z.shape[1])
        np.fill_diagonal(training_d, np.inf)
        loo = training_d.min(axis=1)
        query_d = cdist(z[sources], z[train])/np.sqrt(z.shape[1])
        for query, distances in zip(sources, query_d):
            beer = int(meta.beer.iloc[query]); row = saved.loc[(name, beer)]
            BASE.close(row[["nearest_nonpils_distance", "nonpils_loo_nn_percentile", "training_loo_nn_q95"]],
                       [distances.min(), np.mean(loo <= distances.min()), np.quantile(loo, .95)])
            order = np.argsort(distances)
            part = neighbors[neighbors.space.eq(name)&neighbors.query_beer.eq(beer)].sort_values("neighbor_rank")
            np.testing.assert_array_equal(part.neighbor_beer, meta.beer.iloc[train[order[:25]]])
            BASE.close(part.distance, distances[order[:25]])
            BASE.close(part.neighbor_stale, y.iloc[train[order[:25]]])
            if beer == 81:
                part = full81[full81.space.eq(name)].sort_values("neighbor_rank")
                np.testing.assert_array_equal(part.neighbor_beer, meta.beer.iloc[train[order]])
                BASE.close(part.distance, distances[order])
    local25 = neighbors[neighbors.space.eq("frozen_domain_aging_pca8")&neighbors.query_beer.eq(81)]
    assert summary["beer81_outside_local_top25_target_convex_hull"] == bool(y.iloc[index] > local25.neighbor_stale.max())
    errors = pd.read_csv(folder/"all_saved_pils_prediction_errors.csv")
    influence = pd.read_csv(folder/"beer81_sse_contribution.csv").set_index(["stage", "method"])
    deletion = pd.read_csv(folder/"leave_one_source_out_influence.csv").set_index(["stage", "method", "omitted_beer"])
    for keys, part in errors.groupby(["stage", "method"]):
        assert len(part) == len(sources) and set(part.beer) == set(meta.beer.iloc[sources])
        BASE.close(part.observed, part.beer.map(pd.Series(y.to_numpy(), index=meta.beer)))
        residual = part.prediction.to_numpy()-part.observed.to_numpy()
        BASE.close(part.residual, residual); BASE.close(part.squared_error, residual**2)
        BASE.close(part.sse_share, residual**2/np.sum(residual**2))
        r81 = part[part.beer.eq(81)].iloc[0]
        BASE.close(influence.loc[keys, ["full_rmse", "beer81_residual", "beer81_sse_share"]],
                   [np.sqrt(np.mean(residual**2)), r81.residual, r81.sse_share])
        for row in part.itertuples():
            BASE.close(deletion.loc[(*keys, row.beer), "fixed_prediction_rmse_without_one_source"],
                       np.sqrt(part.loc[part.beer.ne(row.beer), "squared_error"].mean()))
    forum = pd.read_csv(data_dir()/"7.csv").set_index("beer")
    terms = pd.read_csv(folder/"forum_all_terms_beer81.csv")
    assert len(terms) == 186
    for row in terms.itertuples():
        BASE.close(row.beer81_value, forum.at[81, row.term])
        values = forum[row.term].to_numpy(float)
        values = values[np.isfinite(values)]
        value = forum.at[81, row.term]
        BASE.close(row.all_percentile, (np.sum(values < value)+.5*np.sum(values==value))/len(values) if pd.notna(value) else np.nan)
    return {"source": 81, "chemical_features": len(profile), "ood_spaces": len(spaces),
            "retained_sources": len(sources), "stale_rank": summary["stale_descending_rank_all250"],
            "high_tail_training": len(high), "label_error_established": False,
            "fixed_prediction_influence_groups": len(influence)}


def check_fixed_transform(transform, training, calibrated):
    selected = transform.selected_features_
    values = training[selected].to_numpy(float).copy()
    missing = np.isnan(values)
    scales = np.ones(len(selected))
    for j, field in enumerate(selected):
        if field in calibrated:
            positive = values[np.isfinite(values[:, j])&(values[:, j] > 0), j]
            scales[j] = np.median(positive) if len(positive) else 1.
            values[:, j] = np.log1p(values[:, j]/scales[j])
    bounds = np.nanquantile(values, [.01, .99], axis=0)
    values = np.clip(values, *bounds)
    medians = np.nan_to_num(np.nanmedian(values, axis=0), nan=0.)
    filled = np.where(np.isnan(values), medians, values)
    means, std = filled.mean(axis=0), filled.std(axis=0)
    std[std < 1e-12] = 1.
    for actual, expected in [(transform.positive_scales_, scales), (transform.bounds_, bounds),
                             (transform.medians_, medians), (transform.means_, means), (transform.scales_, std)]:
        BASE.close(actual, expected, "Fold-local fixed-width chemical transform")
    assert transform.indicator_weight == .2 and transform.n_output_features_ == 2*len(selected)
    result = np.column_stack([(filled-means)/std, .2*missing])
    BASE.close(transform.transform(training), result, "Fixed unscaled missing flags")
    return result


def check_model_training(model, training, y45, styles, calibrated):
    assert model.n_features_in_ == 226 and model.n_outputs_ == 1
    if hasattr(model, "members_"):
        for member in model.members_:
            check_model_training(member, training, y45, styles, calibrated)
        return
    if hasattr(model, "chemical_"):
        z = check_fixed_transform(model.chemical_, training, calibrated)
        target = y45.stale_hops.to_numpy()
        BASE.close(model.tail_training_threshold_, np.quantile(target, .8))
        if hasattr(model, "auxiliary_target_indices_"):
            assert model.auxiliary_target_names_ == [c for c in y45 if c != "stale_hops"]
            assert len(model.auxiliary_target_indices_) == 44 and model.target_position_ not in model.auxiliary_target_indices_
            w = DOMAIN.style_weights(styles)
            aux = y45[model.auxiliary_target_names_].to_numpy()
            BASE.close(model.auxiliary_mean_, np.average(aux, axis=0, weights=w))
        if model.config["family"] == "ridge":
            weights = DOMAIN.style_weights(styles)
            if model.config.get("tail", False):
                weights *= np.where(target >= np.quantile(target, .8), 2., 1.)
                weights *= len(weights)/weights.sum()
            fitted = Ridge(alpha=model.config["alpha"], solver="svd").fit(z, target, sample_weight=weights)
            BASE.close(model.model_.coef_, fitted.coef_, "Training-only weighted ridge coefficients")
            BASE.close(model.model_.intercept_, fitted.intercept_)


def select_from_tables(table, stability):
    baseline = table[table.method.eq("mean_global")].iloc[0]
    eligible = table.selectable & (table.prediction_variance_ratio >= .05)
    eligible &= (table.macro_style_rmse < baseline.macro_style_rmse)&(table.rmse < baseline.rmse)
    np.testing.assert_array_equal(eligible, table.eligible_for_one_se)
    candidates = table[eligible]
    fallback = candidates.empty
    best = (table[table.selectable] if fallback else candidates).sort_values(["selection_score", "method"]).iloc[0]
    one_se = [best.method] if fallback else candidates.loc[candidates.selection_score <= best.selection_score+best.style_score_se, "method"].tolist()
    assert set(table.loc[table.within_one_se, "method"]) == set(one_se) == set(stability.method)
    selected = stability.sort_values(["response_noise_ratio", "priority", "method"]).iloc[0]
    qualified = bool(not fallback and selected.pair_delta_signal_rms > 1e-4 and selected.response_noise_ratio <= 1.)
    return selected.method, qualified, best


def check_stale(context):
    from q3_stale_refined import RefinedStaleModel, FixedChemicalTransform, refined_specs
    folder = OUT/"stale"
    meta, x, y45 = context["meta"], context["x"], context["y"]
    y = y45.stale_hops
    train = np.flatnonzero(~meta.tasting_category_fine.eq("Pils/Lager"))
    sources = np.flatnonzero(meta.tasting_category_fine.eq("Pils/Lager"))
    manifest, decision, summary = [BASE.read_json(folder/f) for f in ("run_manifest.json", "selection_result.json", "summary.json")]
    specs = {s["name"]: s for s in manifest["candidate_specs"]}
    assert list(specs.values()) == refined_specs() and len(specs) == 12
    assert manifest["pils_labels_used_for_selection_or_intercept"] is False and manifest["no_beer81_exclusion"]
    assert manifest["features"] == x.columns.tolist()
    assert manifest["auxiliary_target_names"] == [c for c in y45 if c != "stale_hops"]
    np.testing.assert_array_equal(manifest["training_indices"], train)
    np.testing.assert_array_equal(manifest["excluded_indices"], sources)
    def fit(spec, rows, seed):
        return RefinedStaleModel(x.columns.tolist(), context["calibrated_features"], manifest["aging_features"],
                                 y45.columns.tolist(), spec, seed).fit(x.iloc[rows], y45.iloc[rows], meta.tasting_category_fine.iloc[rows])
    predictions = pd.read_csv(folder/"loso_predictions.csv")
    audit = pd.read_csv(folder/"loso_split_audit.csv")
    by_style = pd.read_csv(folder/"loso_metrics_by_style.csv")
    overall = pd.read_csv(folder/"loso_metrics_overall.csv")
    assert len(predictions) == len(train)*len(specs) and not predictions.duplicated(["method", "global_index"]).any()
    calculated_styles = []
    for fold, part in audit.groupby("fold"):
        heldout = part.heldout_style.unique().item()
        te = train[meta.tasting_category_fine.iloc[train].eq(heldout)]
        tr = train[~meta.tasting_category_fine.iloc[train].eq(heldout)]
        assert set(part.loc[part.role.eq("train"), "global_index"]) == set(tr)
        assert set(part.loc[part.role.eq("test"), "global_index"]) == set(te)
        assert len(part) == len(train) and not part.global_index.duplicated().any()
        fitted = fit(decision["config"], tr, manifest["final_seed"]+int(fold))
        check_model_training(fitted, x.iloc[tr], y45.iloc[tr], meta.tasting_category_fine.iloc[tr], context["calibrated_features"])
        for method in specs:
            rows = predictions[predictions.method.eq(method)&predictions.fold.eq(fold)].set_index("global_index").loc[te]
            assert rows.heldout_style.eq(heldout).all()
            BASE.close(rows.observed, y.iloc[te])
            BASE.close(rows.tail_threshold_training_q90, np.quantile(y.iloc[tr], .9))
            np.testing.assert_array_equal(rows.is_tail_eval, rows.observed >= rows.tail_threshold_training_q90)
            if method == decision["selected_method"]:
                BASE.close(rows.prediction, fitted.predict(x.iloc[te]), "Selected LOSO refit excluding held-out style")
            if method == "mean_global": BASE.close(rows.prediction, y.iloc[tr].mean())
            calculated_styles.append({"method": method, "heldout_style": heldout, "fold": fold,
                                      **DOMAIN.score(rows.observed, rows.prediction), "bias": (rows.prediction-rows.observed).mean()})
    metrics = list(DOMAIN.score(y, y))
    BASE.same_table(by_style, pd.DataFrame(calculated_styles), ["method", "heldout_style", "fold"], metrics+["bias"], "LOSO style statistics")
    calculated = []
    for method in specs:
        rows = predictions[predictions.method.eq(method)].set_index("global_index").loc[train]
        bs = by_style[by_style.method.eq(method)]
        losses = bs.rmse+.25*bs.bias.abs()
        tail = rows.is_tail_eval
        calculated.append({"method": method, **DOMAIN.score(rows.observed, rows.prediction),
                           "macro_style_rmse": bs.rmse.mean(), "macro_style_abs_bias": bs.bias.abs().mean(),
                           "selection_score": losses.mean(), "style_score_se": losses.std(ddof=1)/np.sqrt(len(losses)),
                           "tail_n": tail.sum(), "tail_rmse": np.sqrt(np.mean((rows.loc[tail, "prediction"]-rows.loc[tail, "observed"])**2))})
    BASE.same_table(overall, pd.DataFrame(calculated), ["method"], list(calculated[0])[1:], "LOSO pooled and selection statistics")
    stability = pd.read_csv(folder/"selection_stability.csv")
    selected, qualified, best = select_from_tables(overall, stability)
    assert selected == decision["selected_method"] == summary["selected_method"] and qualified == decision["qualified_for_proxy"]
    BASE.close([decision["lowest_selection_score"], decision["one_se_width"], decision["one_se_threshold"]],
               [best.selection_score, best.style_score_se, best.selection_score+best.style_score_se])
    pairs_transform = FixedChemicalTransform(x.columns.tolist(), context["calibrated_features"], manifest["aging_features"]).fit(x.iloc[train])
    z = check_fixed_transform(pairs_transform, x.iloc[train], context["calibrated_features"])
    d = cdist(z, z); np.fill_diagonal(d, np.inf)
    pairs = train[np.argsort(d, axis=1)[:, 2]]
    pair_table = pd.read_csv(folder/"selection_chemical_pairs.csv").set_index("global_index").loc[train]
    np.testing.assert_array_equal(pair_table.paired_global_index, pairs)
    for row in stability.itertuples():
        delta = []
        for b in range(manifest["stability_draws"]):
            seed = manifest["stability_seed"]+b
            resample = np.random.default_rng(seed).choice(train, len(train), replace=True)
            fitted = fit(specs[row.method], resample, seed+10000)
            delta.append(fitted.predict(x.iloc[train])-fitted.predict(x.iloc[pairs]))
        delta = np.asarray(delta)
        noise, signal = np.sqrt(np.mean(np.var(delta, axis=0, ddof=1))), np.sqrt(np.mean(delta.mean(axis=0)**2))
        BASE.close([row.pair_delta_noise_rms, row.pair_delta_signal_rms, row.response_noise_ratio], [noise, signal, noise/max(signal, 1e-8)])
    artifact = joblib.load(folder/"transfer_stale_model.joblib")
    ensemble = joblib.load(folder/"bootstrap_ensemble.joblib")
    for saved in [artifact, ensemble]:
        np.testing.assert_array_equal(saved["train_global_indices"], train)
        np.testing.assert_array_equal(saved["excluded_global_indices"], sources)
        assert saved["pils_used_for_selection"] is False and saved["selected_config"] == specs[selected]
        assert saved["qualified_for_proxy"] == qualified and saved["features"] == x.columns.tolist()
    check_model_training(artifact["pipeline"], x.iloc[train], y45.iloc[train], meta.tasting_category_fine.iloc[train], context["calibrated_features"])
    indices = np.asarray(ensemble["bootstrap_indices"])
    assert indices.shape == (60, len(train)) and not np.isin(indices, sources).any()
    assert ensemble["selection_model_indices"] == list(range(40)) and ensemble["audit_model_indices"] == list(range(40,60))
    with np.load(folder/"bootstrap_indices.npz") as saved:
        np.testing.assert_array_equal(saved["indices"], indices)
        np.testing.assert_array_equal(saved["bootstrap_seeds"], ensemble["bootstrap_seeds"])
    for b, (model, rows, seed) in enumerate(zip(ensemble["models"], indices, ensemble["bootstrap_seeds"])):
        assert seed == (manifest["selection_bootstrap_seed"]+b if b < 40 else manifest["audit_bootstrap_seed"]+b-40)
        np.testing.assert_array_equal(rows, np.random.default_rng(seed).choice(train, len(train), replace=True))
        assert model.seed == seed+10000
        check_model_training(model, x.iloc[rows], y45.iloc[rows], meta.tasting_category_fine.iloc[rows], context["calibrated_features"])
    archive = pd.read_csv(folder/"candidate_model_manifest.csv")
    assert set(archive.method) == {name for name, spec in specs.items() if spec["selectable"]}
    for row in archive.itertuples():
        saved = joblib.load(folder/row.file)
        np.testing.assert_array_equal(saved["train_global_indices"], train)
        assert saved["pils_used_for_selection"] is False and saved["selected_config"] == specs[row.method]
        check_model_training(saved["pipeline"], x.iloc[train], y45.iloc[train], meta.tasting_category_fine.iloc[train], context["calibrated_features"])
    diagnostic = pd.read_csv(folder/"pils_diagnostic_predictions.csv")
    dm = pd.read_csv(folder/"pils_diagnostic_metrics.csv")
    assert len(diagnostic) == len(specs)*len(sources)
    for method, part in diagnostic.groupby("method"):
        part = part.set_index("global_index").loc[sources]
        BASE.close(part.observed, y.iloc[sources])
        BASE.close(dm.set_index("method").loc[method, metrics], list(DOMAIN.score(part.observed, part.prediction).values()))
        if method == selected: BASE.close(part.prediction, artifact["pipeline"].predict(x.iloc[sources]))
        if specs[method]["selectable"]:
            saved = joblib.load(folder/f"candidate_models/{method}.joblib")
            BASE.close(part.prediction, saved["pipeline"].predict(x.iloc[sources]), "All archived candidate Pils diagnostics")
            refitted = fit(specs[method], train, manifest["final_seed"])
            BASE.close(saved["pipeline"].predict(x), refitted.predict(x), "Candidate refit on non-Pils training only")
    comparisons = pd.read_csv(folder/"bootstrap_response_comparison.csv").set_index("method")
    old_models = joblib.load(SOLUTION/"outputs/q3_domain/stale/bootstrap_ensemble.joblib")["models"]
    for label, models in [("refined_selected", ensemble["models"]), ("domain_local_previous", old_models)]:
        delta = np.asarray([m.predict(x.iloc[train])-m.predict(x.iloc[pairs]) for m in models])
        noise, signal = np.sqrt(np.mean(np.var(delta, axis=0, ddof=1))), np.sqrt(np.mean(delta.mean(axis=0)**2))
        BASE.close(comparisons.loc[label, ["pair_delta_noise_rms", "pair_delta_signal_rms", "response_noise_ratio"]], [noise, signal, noise/max(signal, 1e-8)])
    nested = pd.read_csv(folder/"nested_predictions.csv")
    choices = pd.read_csv(folder/"nested_choices.csv").set_index("outer_fold")
    outer_audit = pd.read_csv(folder/"nested_outer_split_audit.csv")
    inner_audit = pd.read_csv(folder/"nested_inner_split_audit.csv")
    inner_metrics = pd.read_csv(folder/"nested_inner_metrics.csv")
    inner_stability = pd.read_csv(folder/"nested_selection_stability.csv")
    assert len(nested) == len(train)*4 and not nested.duplicated(["method", "global_index"]).any()
    groups = meta.tasting_category_fine.iloc[train].to_numpy()
    for fold, (tr, te) in enumerate(GroupKFold(4, shuffle=True, random_state=manifest["nested_seed"]).split(train, groups=groups), 1):
        tr, te = train[tr], train[te]
        for role, rows in [("train", tr), ("test", te)]:
            actual = outer_audit[outer_audit.outer_fold.eq(fold)&outer_audit.role.eq(role)]
            np.testing.assert_array_equal(actual.global_index, rows)
        assert not set(meta.tasting_category_fine.iloc[tr])&set(meta.tasting_category_fine.iloc[te])
        for _, part in inner_audit[inner_audit.outer_fold.eq(fold)].groupby("fold"):
            style = part.heldout_style.unique().item()
            assert set(part.global_index) == set(tr)
            assert set(part.loc[part.role.eq("test"), "global_index"]) == set(tr[meta.tasting_category_fine.iloc[tr].eq(style)])
            assert not set(part.global_index)&set(te)
        method, ok, _ = select_from_tables(inner_metrics[inner_metrics.outer_fold.eq(fold)], inner_stability[inner_stability.outer_fold.eq(fold)])
        assert method == choices.at[fold, "selected_method"] and ok == choices.at[fold, "qualified_for_proxy"]
        rows = nested[nested.outer_fold.eq(fold)&nested.method.eq("nested_stability_rule")].set_index("global_index").loc[te]
        fitted = fit(specs[method], tr, manifest["nested_seed"]+fold)
        BASE.close(rows.prediction, fitted.predict(x.iloc[te]), "Outer grouped selected predictor")
        BASE.close(rows.observed, y.iloc[te]); BASE.close(rows.tail_threshold_training_q90, np.quantile(y.iloc[tr], .9))
    nm = pd.read_csv(folder/"nested_metrics.csv").set_index("method")
    for method, rows in nested.groupby("method"):
        BASE.close(nm.loc[method, metrics], list(DOMAIN.score(rows.observed, rows.prediction).values()))
    context.update(refined_artifact=artifact, refined_ensemble=ensemble)
    return {"candidate_models": len(specs), "archived_selectable_models": len(archive), "loso_styles": len(audit.fold.unique()),
            "loso_predictions": len(predictions), "nested_outer_folds": len(choices), "bootstrap_models": len(indices),
            "source_excluded": True, "selected_method": selected, "qualified_internal": qualified,
            "pils_rmse": float(dm.set_index("method").at[selected, "rmse"]),
            "pooled_loso_rmse": float(overall.set_index("method").at[selected, "rmse"]),
            "nested_selected_rmse": float(nm.at["nested_stability_rule", "rmse"])}


def paired(models, candidate, source):
    return np.column_stack([np.asarray(m.predict(candidate)).reshape(-1)-np.asarray(m.predict(source)).reshape(-1)[0] for m in models])


def check_transfer(context):
    from q3_data import read_defect_thresholds
    folder = OUT/"transfer"
    meta, x, y45 = context["meta"], context["x"], context["y"]
    features = x.columns.tolist(); dim = len(features)
    train = np.flatnonzero(~meta.tasting_category_fine.eq("Pils/Lager"))
    sources = np.flatnonzero(meta.tasting_category_fine.eq("Pils/Lager"))
    target = np.flatnonzero(meta.tasting_category_fine.eq("Tripel"))
    manifest, summary = [BASE.read_json(folder/f) for f in ("run_manifest.json", "summary.json")]
    default = {"arm":"refined", "pool":"contracted_6004", "bound":"retain80", "ranking":"margin"}
    assert manifest["default"] == summary["default"] == default
    assert manifest["default_chosen_from_audit"] is False and manifest["audit_reselection"] is False
    bounds = {"absolute10":(.1,None), "absolute30":(.3,None), "absolute50":(.5,None),
              "absolute60":(.6,None), "retain80":(.1,.8), "retain100":(.1,1.)}
    assert manifest["bounds"] == {k:{"absolute_minimum":v[0],"anchor_fraction":v[1]} for k,v in bounds.items()}
    assert manifest["pools"] == {"legacy_2004":2004,"legacy_6004":6004,"contracted_6004":6004}
    geometry = joblib.load(SOLUTION/"outputs/q3/transfer/transfer_copula.joblib")
    np.testing.assert_array_equal(geometry["training_rows"], train)
    reference = geometry["copula"]
    tz = reference.transform(x.iloc[target]); center = tz.mean(axis=0)
    d = cdist(tz,tz)/np.sqrt(dim); np.fill_diagonal(d,np.inf)
    limit = np.quantile(d.min(axis=1),.95)
    BASE.close(summary["support_limit"],limit)
    legacy = pd.read_csv(folder/"legacy_base_draws.csv")
    BASE.close(legacy,pd.read_csv(SOLUTION/"outputs/q3_domain/consensus/base_draws.csv"))
    assert legacy.shape == (1500,226)
    base = reference.transform(legacy.iloc[:500])
    expected_latent = np.vstack([reference.style_center_["Tripel"]+c*(base-reference.style_center_["Tripel"]) for c in [.2,.4,.6]])
    BASE.close(np.load(folder/"contracted_latents.npy"),expected_latent)
    contracted = pd.read_csv(folder/"contracted_base_draws.csv")
    BASE.close(contracted,reference.inverse_transform(expected_latent),"Contracted latent inverse")
    drawmeta = pd.read_csv(folder/"contracted_draw_metadata.csv")
    BASE.close(drawmeta, np.column_stack([np.arange(1500),np.tile(np.arange(500),3),np.repeat([.2,.4,.6],500)]))
    assert summary["contracted_targets_unique"] == len(np.unique(contracted,axis=0))
    scenarios = BASE.read_json(SOLUTION/"outputs/q3_optimized/transfer/threshold_scenarios.json")
    assert manifest["threshold_scenarios"] == scenarios and len(scenarios) == 10
    rules = read_defect_thresholds()
    original = joblib.load(SOLUTION/"outputs/q3/sensory/stale_bootstrap.joblib")
    center_old = joblib.load(SOLUTION/"outputs/q3/sensory/transfer_proxy.joblib")["pipeline"]
    fixed = joblib.load(SOLUTION/"outputs/q3_optimized/stale_fixed/transfer_stale_model.joblib")
    fixed_ensemble = joblib.load(SOLUTION/"outputs/q3_optimized/stale_fixed/bootstrap_ensemble.joblib")
    fresh = {}
    for label,filename,seed in [("old","fresh_original_audit.joblib",202104123),("fixed","fresh_fixed_audit.joblib",202105123)]:
        saved = joblib.load(folder/filename)
        np.testing.assert_array_equal(saved["train_global_indices"],train)
        np.testing.assert_array_equal(saved["excluded_global_indices"],sources)
        assert saved["features"] == features and len(saved["models"]) == 20
        for b,(model,rows,current) in enumerate(zip(saved["models"],saved["bootstrap_indices"],saved["bootstrap_seeds"])):
            assert current == seed+b
            np.testing.assert_array_equal(rows,np.random.default_rng(current).choice(train,len(train),replace=True))
            BASE.close(model.named_steps["imputer"].statistics_,x.iloc[rows].median())
            tree = model.named_steps["model"]
            assert tree.n_outputs_ == 1 and tree.n_estimators == 120 and tree.min_samples_leaf == 3 and tree.max_features == .7
            assert tree.random_state == current+10000
        fresh[label] = saved["models"]
    assert center_old.named_steps["model"].n_outputs_ == 45
    assert all(m.named_steps["model"].n_outputs_ == 1 for m in original["models"])
    arms = {"refined":(context["refined_artifact"],context["refined_ensemble"]["models"][:40],context["refined_ensemble"]["models"][40:]),
            "fixed":(fixed,fixed_ensemble["models"][:40],fresh["fixed"])}
    result = pd.read_csv(folder/"transfer_summary.csv")
    geom = pd.read_csv(folder/"candidate_geometry.csv.gz")
    model_checks = pd.read_csv(folder/"candidate_model_checks.csv.gz")
    spectra = pd.read_csv(folder/"selected_spectra.csv").set_index("transfer_id")
    deltas = pd.read_csv(folder/"paired_deltas.csv")
    thresholds = pd.read_csv(folder/"threshold_checks.csv")
    rechecks = pd.read_csv(folder/"fixed_spectra_rechecks.csv")
    assert rechecks.evaluation_pool.eq(default["pool"]).all()
    uniqueness = pd.read_csv(folder/"pool_uniqueness.csv").set_index(["source_beer","pool"])
    anchors = pd.read_csv(SOLUTION/"outputs/q3_optimized/transfer/transfer_summary.csv")
    anchors = anchors[anchors.is_default_policy].set_index("source_beer")
    anchor_spectra = pd.read_csv(SOLUTION/"outputs/q3_optimized/transfer/transferred_spectra.csv").set_index("transfer_id")
    prior = pd.read_csv(SOLUTION/"outputs/q3_domain/consensus/transfer_summary.csv")
    prior_spectra = pd.read_csv(SOLUTION/"outputs/q3_domain/consensus/transferred_spectra.csv").set_index("transfer_id")
    assert len(result) == len(arms)*3*6*2*len(sources) and not result.transfer_id.duplicated().any()
    assert len(geom) == 12004*len(sources) and len(model_checks) == len(arms)*len(geom)
    expected_deltas, expected_thresholds, expected_rechecks = [],[],[]
    medians = x.iloc[train].median(); target_median = x.iloc[target].median().to_numpy()
    minimum,maximum = x.iloc[train].min().to_numpy(),x.iloc[train].max().to_numpy()
    stale_position = y45.columns.get_loc("stale_hops")
    for index in sources:
        beer = int(meta.beer.iloc[index]); raw = x.iloc[[index]]; source = raw.fillna(medians)
        descriptions,blocks = [],[]
        for family,draws in [("legacy",legacy),("contracted",contracted)]:
            for gamma in [.25,.5,.75,1.]:
                blocks.append((1-gamma)*source.to_numpy()[0]+gamma*draws.to_numpy())
                descriptions.extend((family,gamma,j) for j in range(len(draws)))
        for gamma in [.25,.5,.75,1.]:
            blocks.append(((1-gamma)*source.to_numpy()[0]+gamma*target_median)[None])
            descriptions.append(("median",gamma,-1))
        candidates = pd.DataFrame(np.vstack(blocks),columns=features)
        desc = pd.DataFrame(descriptions,columns=["family","gamma","draw"])
        chemical = np.ones(len(candidates),bool)
        for rule in rules.values():
            field = rule["feature"]; c0 = raw[field].iloc[0]
            assert np.isfinite(c0) and c0 >= 0
            finite_t = [s["thresholds"][field] for s in scenarios if s["thresholds"][field] is not None]
            cap = max(c0,min(finite_t)) if finite_t else c0
            candidates[field] = candidates[field].clip(lower=0,upper=cap)
            for scenario in scenarios:
                threshold = scenario["thresholds"][field]
                chemical &= candidates[field].to_numpy() <= (max(c0,threshold) if threshold is not None else c0)+TOL
        assert np.isfinite(candidates.to_numpy()).all() and candidates[context["calibrated_features"]].ge(0).all().all()
        masks = {"legacy_2004":desc.family.eq("median")|(desc.family.eq("legacy")&desc.draw.lt(500)),
                 "legacy_6004":desc.family.isin(["legacy","median"]),"contracted_6004":desc.family.isin(["contracted","median"])}
        z,sz = reference.transform(candidates),reference.transform(source)
        distance = reference.distance(candidates,"Tripel"); before = reference.distance(source,"Tripel")[0]
        improvement = 1-distance/max(before,1e-8)
        change = np.linalg.norm(z-sz,axis=1)/np.sqrt(dim)
        nn = cdist(z,tz).min(axis=1)/np.sqrt(dim)
        support = (nn<=limit+TOL)&((candidates.to_numpy()>=minimum-TOL)&(candidates.to_numpy()<=maximum+TOL)).all(axis=1)
        objective = distance/max(before,1e-8)+.2*change/max(np.linalg.norm(sz[0]-center)/np.sqrt(dim),1e-8)
        geometric = support&chemical&(change>=.05)
        actual_geom = geom[geom.source_beer.eq(beer)].sort_values("candidate_index")
        BASE.close(actual_geom[["target_distance","target_improvement","change_latent_rms","target_nn_rms","support_ok","chemical_ok","geometry_objective"]],
                   np.column_stack([distance,improvement,change,nn,support,chemical,objective]),"All post-projection fixed geometry")
        assert actual_geom.family.tolist() == desc.family.tolist()
        BASE.close(actual_geom[["gamma","draw"]],desc[["gamma","draw"]])
        anchor_frame = anchor_spectra.loc[[anchors.at[beer,"transfer_id"]],features]
        anchor_gain = 1-reference.distance(anchor_frame,"Tripel")[0]/max(before,1e-8)
        unique,inverse = np.unique(candidates.to_numpy(),axis=0,return_inverse=True)
        unique = pd.DataFrame(unique,columns=features)
        old_delta = paired(original["models"][:30],unique,source)[inverse]
        old_q = np.quantile(old_delta,.95,axis=1); old_signal = np.any(np.abs(old_delta)>1e-8,axis=1)
        old_center_source = center_old.predict(source)[0,stale_position]
        old_center_delta = center_old.predict(unique)[:,stale_position][inverse]-old_center_source
        for pool,mask in masks.items():
            actual_unique = len(np.unique(candidates.to_numpy()[mask],axis=0)); nominal = int(mask.sum())
            BASE.close(uniqueness.loc[(beer,pool),["nominal_candidates","unique_candidates","duplicate_fraction","anchor_gain"]],
                       [nominal,actual_unique,1-actual_unique/nominal,anchor_gain])
        for arm,(artifact,selection,audit) in arms.items():
            new_delta = paired(selection,unique,source)[inverse]
            new_q = np.quantile(new_delta,.95,axis=1); new_signal = np.any(np.abs(new_delta)>1e-8,axis=1)
            current = artifact["pipeline"]
            center_source = np.asarray(current.predict(source)).reshape(-1)[0]
            center_delta = np.asarray(current.predict(unique)).reshape(-1)[inverse]-center_source
            worst = np.maximum(new_q,old_q)
            joint = geometric&(new_q<=TOL)&(old_q<=TOL)&new_signal&old_signal&(center_delta<=TOL)&(old_center_delta<=TOL)
            checks = model_checks[model_checks.source_beer.eq(beer)&model_checks.arm.eq(arm)].sort_values("candidate_index")
            BASE.close(checks[["new_selection_q95","old_selection_q95","worst_selection_q95","center_new_delta","center_old_delta","new_signal","old_signal","joint_feasible"]],
                       np.column_stack([new_q,old_q,worst,center_delta,old_center_delta,new_signal,old_signal,joint]),"Complete selection diagnostics without audit")
            qualified = artifact["qualified_for_proxy"]
            rows = result[result.source_beer.eq(beer)&result.arm.eq(arm)]
            cache = {}
            def audit_frame(frame):
                key = frame.to_numpy().tobytes()
                if key not in cache:
                    cache[key] = {"new_audit":paired(audit,frame,source)[0],"old_fresh_audit":paired(fresh["old"],frame,source)[0],
                                  "old_historical":paired(original["models"][30:],frame,source)[0]}
                return cache[key]
            for row in rows.itertuples():
                absolute,retention = bounds[row.bound]
                floor = max(absolute,retention*anchor_gain) if retention is not None else absolute
                mask = masks[row.pool].to_numpy()
                nsd,osd = np.std(new_delta[mask].mean(axis=1)),np.std(old_delta[mask].mean(axis=1))
                informative = bool(nsd>1e-6 and osd>1e-6)
                ids = np.flatnonzero(joint&mask&(improvement>=floor))
                BASE.close([row.anchor_gain,row.required_gain,row.candidate_count,row.feasible_count,row.new_delta_sd,row.old_delta_sd],
                           [anchor_gain,floor,mask.sum(),len(ids),nsd,osd])
                assert row.qualified == qualified and row.informative == informative
                arm_default = row.pool==default["pool"] and row.bound==default["bound"] and row.ranking==default["ranking"]
                assert row.is_arm_default == arm_default and row.is_default_policy == (arm==default["arm"] and arm_default)
                if not len(ids):
                    assert pd.isna(row.candidate_index) and row.status=="no_candidate" and not row.fresh_consistency_passed
                    continue
                chosen = ids[np.lexsort((objective[ids],worst[ids]))[0]] if row.ranking=="margin" else ids[np.argmin(objective[ids])]
                assert int(row.candidate_index)==chosen, "Frozen eligible argmin cannot use audit predictions"
                frame = candidates.iloc[[chosen]]
                BASE.close(spectra.loc[row.transfer_id,features],frame.iloc[0],"Selected spectrum reconstruction")
                assessed = audit_frame(frame); aq = {k:np.quantile(v,.95) for k,v in assessed.items()}
                newpass = bool(qualified and informative and aq["new_audit"]<=TOL)
                freshpass = bool(newpass and aq["old_fresh_audit"]<=TOL)
                historypass = bool(newpass and aq["old_historical"]<=TOL)
                status = "unqualified" if not qualified else "uninformative" if not informative else "new_audit_failed" if not newpass else "old_fresh_audit_failed" if not freshpass else "passed"
                assert row.status==status and row.new_model_passed==newpass and row.fresh_consistency_passed==freshpass and row.historical_consistency_passed==historypass
                BASE.close([row.target_distance_before,row.target_distance_after,row.target_improvement,row.retained_anchor_gain,row.change_latent_rms,
                            row.target_nn_rms,row.support_nn_limit,row.geometry_objective,row.new_selection_q95,row.old_selection_q95,row.worst_selection_q95,
                            row.center_new_delta,row.center_old_delta,row.new_audit_q95,row.old_fresh_audit_q95,row.old_historical_q95],
                           [before,distance[chosen],improvement[chosen],improvement[chosen]/anchor_gain,change[chosen],nn[chosen],limit,objective[chosen],
                            new_q[chosen],old_q[chosen],worst[chosen],center_delta[chosen],old_center_delta[chosen],aq["new_audit"],aq["old_fresh_audit"],aq["old_historical"]])
                for label,delta in {"new_selection":new_delta[chosen],"old_selection":old_delta[chosen],**assessed}.items():
                    expected_deltas.extend({"transfer_id":row.transfer_id,"ensemble":label,"model_index":i,"delta":value} for i,value in enumerate(delta))
                for scenario in scenarios:
                    for label,rule in rules.items():
                        field = rule["feature"]; threshold = scenario["thresholds"][field]; c0 = raw[field].iloc[0]
                        cap = max(c0,threshold) if threshold is not None else c0
                        expected_thresholds.append({"transfer_id":row.transfer_id,"scenario":scenario["scenario"],"defect":label,
                            "source":c0,"transferred":frame[field].iloc[0],"cap":cap,"passed":frame[field].iloc[0]<=cap+TOL})
            references = [("original_anchor",anchor_frame)]
            for pool in ("matched_2004","expanded_6004"):
                pr = prior[prior.source_beer.eq(beer)&prior.pool.eq(pool)&prior.policy.eq("joint_margin")].iloc[0]
                if pd.notna(pr.candidate_index): references.append((f"prior_{pool}",prior_spectra.loc[[pr.transfer_id],features]))
            evaluation = masks[default["pool"]].to_numpy()
            informative = np.std(new_delta[evaluation].mean(axis=1))>1e-6 and np.std(old_delta[evaluation].mean(axis=1))>1e-6
            for label,frame in references:
                pz = reference.transform(frame); gain = 1-reference.distance(frame,"Tripel")[0]/max(before,1e-8)
                shift = np.linalg.norm(pz-sz)/np.sqrt(dim)
                supported = bool(cdist(pz,tz).min()/np.sqrt(dim)<=limit+TOL and ((frame.to_numpy()>=minimum-TOL)&(frame.to_numpy()<=maximum+TOL)).all())
                chem = all(frame[r["feature"]].iloc[0] <= (max(raw[r["feature"]].iloc[0],s["thresholds"][r["feature"]]) if s["thresholds"][r["feature"]] is not None else raw[r["feature"]].iloc[0])+TOL for s in scenarios for r in rules.values())
                nd,od = paired(selection,frame,source)[0],paired(original["models"][:30],frame,source)[0]
                nq,oq = np.quantile(nd,.95),np.quantile(od,.95)
                cdn = np.asarray(current.predict(frame)).reshape(-1)[0]-center_source
                cdo = center_old.predict(frame)[0,stale_position]-old_center_source
                assessed = audit_frame(frame); aq = {k:np.quantile(v,.95) for k,v in assessed.items()}
                basepass = bool(qualified and informative and supported and chem and shift>=.05 and nq<=TOL and oq<=TOL and cdn<=TOL and cdo<=TOL and np.any(np.abs(nd)>1e-8) and np.any(np.abs(od)>1e-8))
                for bound,(absolute,retention) in bounds.items():
                    floor = max(absolute,retention*anchor_gain) if retention is not None else absolute
                    expected_rechecks.append({"source_beer":beer,"arm":arm,"reference":label,"bound":bound,"required_gain":floor,
                        "target_improvement":gain,"retained_anchor_gain":gain/anchor_gain,"change_latent_rms":shift,"support_ok":supported,"chemical_ok":chem,
                        "new_selection_q95":nq,"old_selection_q95":oq,"center_new_delta":cdn,"center_old_delta":cdo,"new_audit_q95":aq["new_audit"],
                        "old_fresh_audit_q95":aq["old_fresh_audit"],"fresh_consistency_passed":bool(basepass and gain>=floor and aq["new_audit"]<=TOL and aq["old_fresh_audit"]<=TOL)})
        print(f"Verified transfer source beer={beer}",flush=True)
    BASE.same_table(deltas,pd.DataFrame(expected_deltas),["transfer_id","ensemble","model_index"],["delta"],"Every selected paired model delta")
    BASE.same_table(thresholds,pd.DataFrame(expected_thresholds),["transfer_id","scenario","defect"],["source","transferred","cap","passed"],"All selected threshold scenarios")
    keys = ["source_beer","arm","reference","bound"]
    BASE.same_table(rechecks,pd.DataFrame(expected_rechecks),keys,[k for k in expected_rechecks[0] if k not in keys],"Same-rule frozen-spectrum rechecks")
    assert len(spectra)==result.candidate_index.notna().sum()==summary["selected"]
    assert len(thresholds)==len(spectra)*len(scenarios)*len(rules)==summary["threshold_rows"]
    frontier = pd.read_csv(folder/"frontier.csv")
    expected_frontier = []
    def counts(part):
        selected = part.candidate_index.notna(); passing = part.fresh_consistency_passed
        return {"sources":len(part),"selected":selected.sum(),"new_passed":part.new_model_passed.sum(),"fresh_consistency_passed":passing.sum(),
            "historical_consistency_passed":part.historical_consistency_passed.sum(),"no_candidate":(~selected).sum(),
            "median_improvement_selected":part.loc[selected,"target_improvement"].median(),"median_improvement_passed":part.loc[passing,"target_improvement"].median(),
            "median_retention_selected":part.loc[selected,"retained_anchor_gain"].median(),"median_change_rms_selected":part.loc[selected,"change_latent_rms"].median()}
    for (arm,pool,bound,ranking),part in result.groupby(["arm","pool","bound","ranking"]):
        expected_frontier.append({"arm":arm,"pool":pool,"bound":bound,"ranking":ranking,**counts(part)})
    BASE.same_table(frontier,pd.DataFrame(expected_frontier),["arm","pool","bound","ranking"],list(counts(result)),"Complete trade-off frontier")
    assert len(frontier)==summary["frontier_rows"]==72 and len(result)==summary["policies"]
    for key,value in counts(result[result.is_default_policy]).items():
        BASE.close(summary["default_results"][key],value)
    for arm in arms:
        for key,value in counts(result[result.arm.eq(arm)&result.is_arm_default]).items(): BASE.close(summary["arm_defaults"][arm][key],value)
    return {"candidate_geometry_rows":len(geom),"candidate_screening_rows":len(model_checks),"policies":len(result),
            "selected_spectra":len(spectra),"threshold_checks":len(thresholds),"frontier_rows":len(frontier),
            "same_rule_rechecks":len(rechecks),"default":default,"default_counts":summary["default_results"],
            "fixed_control_counts":summary["arm_defaults"]["fixed"],"all_argmins_reconstructed_without_audit":True}


def check_reports(context):
    import xml.etree.ElementTree as ET
    from PIL import Image
    report = SOLUTION/"docs/q3_refinement_results.md"
    assert report.is_file() and report.stat().st_size > 500
    text = report.read_text(encoding="utf-8")
    provenance = BASE.read_json(OUT/"report_sources.json")
    assert len(provenance)==13 and len({r["path"] for r in provenance})==len(provenance)
    for row in provenance:
        path = SOLUTION/row["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row["sha256"],row["path"]
        if row["rows"] is not None: assert len(pd.read_csv(path))==row["rows"]
    comparison = pd.read_csv(OUT/"report_tables/prediction_comparison.csv")
    loso = pd.read_csv(OUT/"stale/loso_metrics_overall.csv")
    pils = pd.read_csv(OUT/"stale/pils_diagnostic_metrics.csv")
    expected = loso.merge(pils[["method","rmse","mean_bias"]].rename(columns={"rmse":"pils_rmse","mean_bias":"pils_bias"}),on="method",validate="one_to_one")
    columns = expected.select_dtypes(include=[np.number,bool]).columns.tolist()
    BASE.same_table(comparison,expected,["method"],columns,"Report prediction table")
    transfer = pd.read_csv(OUT/"transfer/transfer_summary.csv")
    retained = pd.read_csv(OUT/"report_tables/retain80_all_policies.csv")
    expected = transfer[transfer.bound.eq("retain80")]
    columns = expected.select_dtypes(include=[np.number,bool]).columns.tolist()
    BASE.same_table(retained,expected,["transfer_id"],columns,"Report retained-gain policy table")
    rechecks = pd.read_csv(OUT/"transfer/fixed_spectra_rechecks.csv")
    expected = rechecks.groupby(["arm","reference","bound"],as_index=False).agg(available=("source_beer","size"),passed=("fresh_consistency_passed","sum"),median_gain=("target_improvement","median"))
    expected["sources_denominator"]=len(context["meta"][context["meta"].tasting_category_fine.eq("Pils/Lager")])
    BASE.same_table(pd.read_csv(OUT/"report_tables/reference_counts.csv"),expected,["arm","reference","bound"],
                    ["available","passed","median_gain","sources_denominator"],"Report same-rule historical comparison")
    summary = BASE.read_json(OUT/"report_summary.json")
    selection = BASE.read_json(OUT/"stale/selection_result.json")
    ts = BASE.read_json(OUT/"transfer/summary.json")
    audit = BASE.read_json(OUT/"beer81/summary.json")
    nested = pd.read_csv(OUT/"stale/nested_metrics.csv").set_index("method")
    metric = pils.set_index("method")
    assert summary["selected_method"]==selection["selected_method"]
    for key,value in {"selected_pils_rmse":metric.at[selection["selected_method"],"rmse"],
                      "tail_candidate_pils_rmse":metric.at["tail_ridge_aging100","rmse"],
                      "matched_old_pils_rmse":metric.at["old_multioutput_tree","rmse"],
                      "nested_rule_rmse":nested.at["nested_stability_rule","rmse"],
                      "beer81_observed":audit["stale_observed"]}.items():
        BASE.close(summary[key],value)
        assert f"{value:.4f}" in text, f"Reported number missing: {key}"
    prediction = pd.read_csv(OUT/"stale/pils_diagnostic_predictions.csv")
    selected81 = prediction[prediction.method.eq(selection["selected_method"])&prediction.beer.eq(81)].prediction.item()
    BASE.close(summary["beer81_selected_prediction"],selected81)
    assert summary["arm_defaults"]==ts["arm_defaults"] and summary["default_results"]==ts["default_results"]
    assert summary["tail_candidate_eligible"]==bool(loso.set_index("method").at["tail_ridge_aging100","eligible_for_one_se"])
    assert summary["candidate_rows"]==len(loso) and summary["frontier_rows"]==len(pd.read_csv(OUT/"transfer/frontier.csv"))
    for row in comparison.itertuples():
        assert f"|{row.method}|{row.macro_style_rmse:.4f}|{row.rmse:.4f}|{row.pils_rmse:.4f}|" in text
    frontier = pd.read_csv(OUT/"transfer/frontier.csv")
    for row in frontier[frontier.bound.eq("retain80")].itertuples():
        assert f"|{row.arm}|{row.pool}|{row.ranking}|{row.selected}|{row.fresh_consistency_passed}|" in text
    figures=[]
    for name in ["beer81_and_prediction","proximity_consistency_grid"]:
        png,svg = OUT/f"figures/{name}.png",OUT/f"figures/{name}.svg"
        with Image.open(png) as im:
            im.verify()
        with Image.open(png) as im:
            assert im.width>=1000 and im.height>=600
        root = ET.parse(svg).getroot()
        assert root.tag.endswith("svg") and len(list(root.iter()))>50
        assert f"figures/{name}.png" in text
        figures.append(name)
    assert summary["figure_pairs"]==len(figures) and summary["source_denominator"]==len(transfer[transfer.is_default_policy])
    return {"report":str(report.relative_to(SOLUTION)),"source_hashes":len(provenance),"candidate_rows":len(comparison),
            "retention_policy_rows":len(retained),"reference_rows":len(expected),"figure_pairs":len(figures),
            "report_tables_and_text_reconciled":True}


def main():
    started = perf_counter()
    checks = {}
    checks["raw_sources"] = BASE.check_hashes()
    checks["frozen_history"] = check_frozen()
    context,checks["data_contract"] = BASE.load_contract()
    checks["beer81"] = check_beer81(context)
    checks["stale_prediction"] = check_stale(context)
    checks["balanced_transfer"] = check_transfer(context)
    checks["final_artifacts"] = check_reports(context)
    assert check_frozen() == checks["frozen_history"]
    result = {"status":"PASS","checks":checks,"elapsed_seconds":perf_counter()-started,
              "scope":"Saved-data provenance, training-only internal validation, model-response constraints and predeclared finite candidate search; no external sensory validation."}
    (OUT/"verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1): main()
