"""Constrained Pils/Lager -> Tripel target-spectrum search.

Source beers are excluded from fitted copula and sensory proxy training. The
bootstrap quantiles are model diagnostics, not calibrated outcome guarantees.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from common import SEED, SOLUTION
from q3_copula import ConditionalGaussianCopula
from q3_data import load_q3_data, read_defect_thresholds

OUT = SOLUTION / "outputs/q3/transfer"
SOURCE_STYLE = "Pils/Lager"
TARGET_STYLE = "Tripel"
N_GENERATED = 500
LAMBDAS = [0.05, 0.2, 1.0]
DEFAULT_LAMBDA = 0.2
MIN_TARGET_IMPROVEMENT = 0.10
MIN_LATENT_RMS_CHANGE = 0.05
SUPPORT_QUANTILE = 0.95
TOL = 1e-10


def paired_deltas(models, candidates, source):
    return np.column_stack([
        np.asarray(model.predict(candidates)).reshape(-1)
        - float(np.asarray(model.predict(source)).reshape(-1)[0])
        for model in models
    ])


def defect_caps(source, defects, threshold_kind):
    caps = {}
    for name, rule in defects.items():
        value = float(source[rule["feature"]].iloc[0])
        threshold = rule[f"threshold_{threshold_kind}"]
        caps[rule["feature"]] = max(value, float(threshold)) if pd.notna(threshold) else value
    return caps


def chemical_check(frame, caps):
    good = np.ones(len(frame), dtype=bool)
    for feature, cap in caps.items():
        values = frame[feature].to_numpy(float)
        good &= np.isfinite(values) & (values >= 0) & (values <= cap + TOL)
    return good


def excess(value, threshold):
    return max(value / threshold - 1, 0) if pd.notna(threshold) else np.nan


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta, x, y, manifest = load_q3_data()
    source_rows = np.flatnonzero(meta.tasting_category_fine.eq(SOURCE_STYLE))
    training_rows = np.flatnonzero(~meta.tasting_category_fine.eq(SOURCE_STYLE))
    defects = read_defect_thresholds()
    proxy = joblib.load(SOLUTION / "outputs/q3/sensory/transfer_proxy.joblib")
    ensemble = joblib.load(SOLUTION / "outputs/q3/sensory/stale_bootstrap.joblib")
    assert proxy["features"] == x.columns.tolist() == ensemble["features"]
    assert proxy["targets"] == y.columns.tolist()
    assert len(ensemble["models"]) == 40
    selection_models, audit_models = ensemble["models"][:30], ensemble["models"][30:]
    for indices in ensemble["bootstrap_indices"]:
        assert not np.isin(indices, source_rows).any(), "Source beer entered bootstrap training"

    # All eleven transfer source rows are kept out of empirical marginals too.
    copula = ConditionalGaussianCopula(mean_strength=5.0, cov_strength=20.0)
    copula.fit(x.iloc[training_rows], meta.tasting_category_fine.iloc[training_rows])
    target_rows = np.flatnonzero(meta.tasting_category_fine.eq(TARGET_STYLE))
    target = x.iloc[target_rows]
    target_z = copula.transform(target)
    p = x.shape[1]
    target_pairs = cdist(target_z, target_z) / np.sqrt(p)
    np.fill_diagonal(target_pairs, np.inf)
    target_loo_nn = target_pairs.min(axis=1)
    support_limit = float(np.quantile(target_loo_nn, SUPPORT_QUANTILE))
    # This is an empirical support rule, not a coverage or production guarantee.
    center = target_z.mean(axis=0)
    train_min = x.iloc[training_rows].min().to_numpy()
    train_max = x.iloc[training_rows].max().to_numpy()
    base_pool = copula.sample(TARGET_STYLE, N_GENERATED, seed=SEED + 301)
    target_median = target.median().to_numpy()
    records, candidate_metrics, chemical_records = [], [], []
    spectra, profiles, changes, paired_audit_records = [], [], [], []
    source_spectra = pd.concat([meta.iloc[source_rows].reset_index(drop=True),
                               x.iloc[source_rows].reset_index(drop=True)], axis=1)
    source_spectra.to_csv(OUT / "source_spectra.csv", index=False, lineterminator="\n")

    for source_no, row_index in enumerate(source_rows):
        beer = int(meta.beer.iloc[row_index])
        raw_source = x.iloc[[row_index]].copy()
        complete_defects = all(np.isfinite(raw_source[rule["feature"]].iloc[0])
                               and raw_source[rule["feature"]].iloc[0] >= 0
                               for rule in defects.values())
        if not complete_defects:
            for weight in LAMBDAS:
                records.append({"source_beer": beer, "source_style": SOURCE_STYLE,
                                "target_style": TARGET_STYLE, "mode": "protected",
                                "status": "source_defect_unverifiable", "lambda": weight,
                                "is_default_policy": weight == DEFAULT_LAMBDA,
                                "all_model_checks_passed": False})
            continue
        # Other missing features may be filled using training medians; four
        # protected source values above must always be directly observed.
        source = raw_source.fillna(x.iloc[training_rows].median())
        source_z = copula.transform(source)
        before_distance = float(copula.distance(source, TARGET_STYLE)[0])
        before_nn = float(cdist(source_z, target_z).min() / np.sqrt(p))
        center_scale = max(float(np.linalg.norm(source_z[0] - center) / np.sqrt(p)), 1e-8)
        caps_median = defect_caps(raw_source, defects, "median")
        caps_mean = defect_caps(raw_source, defects, "mean")
        actual_target_feasible = int(chemical_check(target, caps_median).sum())

        # Fixed search design: conditional samples and interpolation towards them.
        # Same pool for all sources and lambdas; no post-result relaxation.
        source_values = source.to_numpy()[0]
        blocks, descriptions = [], []
        for gamma in [0.25, 0.5, 0.75, 1.0]:
            values = (1 - gamma) * source_values + gamma * base_pool.to_numpy()
            blocks.append(values)
            descriptions.extend(("copula_interpolation", gamma, j) for j in range(len(values)))
        for gamma in [0.25, 0.5, 0.75, 1.0]:
            blocks.append(((1 - gamma) * source_values + gamma * target_median)[None])
            descriptions.append(("target_median_interpolation", gamma, -1))
        unprojected = pd.DataFrame(np.vstack(blocks), columns=x.columns)
        projected = unprojected.copy()
        for feature, cap in caps_median.items():
            projected[feature] = projected[feature].clip(lower=0, upper=cap)

        z = copula.transform(projected)
        distances = copula.distance(projected, TARGET_STYLE)
        nn = cdist(z, target_z).min(axis=1) / np.sqrt(p)
        change = np.linalg.norm(z - source_z, axis=1) / np.sqrt(p)
        marginal_support = ((projected.to_numpy() >= train_min - TOL)
                            & (projected.to_numpy() <= train_max + TOL)).all(axis=1)
        support = (nn <= support_limit + TOL) & marginal_support
        improvement = 1 - distances / max(before_distance, 1e-8)
        chemical_ok = chemical_check(projected, caps_median)
        delta_selection = paired_deltas(selection_models, projected, source)
        selection_q95 = np.quantile(delta_selection, .95, axis=1)
        proxy_ok = selection_q95 <= TOL
        nontrivial = change >= MIN_LATENT_RMS_CHANGE
        target_ok = improvement >= MIN_TARGET_IMPROVEMENT
        eligible = support & nontrivial & target_ok & chemical_ok & proxy_ok
        mean_ok = chemical_check(projected, caps_mean)

        unprojected_z = copula.transform(unprojected)
        unprojected_dist = copula.distance(unprojected, TARGET_STYLE)
        unprojected_change = np.linalg.norm(unprojected_z - source_z, axis=1) / np.sqrt(p)
        unprojected_nn = cdist(unprojected_z, target_z).min(axis=1) / np.sqrt(p)
        unconstrained_support = (unprojected_nn <= support_limit + TOL) & (
            (unprojected.to_numpy() >= train_min - TOL)
            & (unprojected.to_numpy() <= train_max + TOL)).all(axis=1)
        baseline_eligible = (unconstrained_support & (unprojected_change >= MIN_LATENT_RMS_CHANGE)
                             & (1 - unprojected_dist / max(before_distance, 1e-8) >= MIN_TARGET_IMPROVEMENT))
        baseline_chemical = chemical_check(unprojected, caps_median)
        base_source_profile = proxy["pipeline"].predict(source)[0]
        for j in range(len(projected)):
            method, gamma, pool_row = descriptions[j]
            candidate_metrics.append({
                "source_beer": beer, "candidate_index": j, "method": method,
                "gamma": gamma, "pool_row": pool_row,
                "target_distance": float(distances[j]), "target_improvement": float(improvement[j]),
                "change_latent_rms": float(change[j]), "target_nn_rms": float(nn[j]),
                "marginal_support": bool(marginal_support[j]), "support_ok": bool(support[j]),
                "chemical_median_ok": bool(chemical_ok[j]), "chemical_mean_ok": bool(mean_ok[j]),
                "selection_paired_delta_q95": float(selection_q95[j]), "selection_proxy_ok": bool(proxy_ok[j]),
                "target_improvement_ok": bool(target_ok[j]), "nontrivial": bool(nontrivial[j]),
                "selection_feasible": bool(eligible[j]),
            })

        for weight in LAMBDAS:
            objective = distances / max(before_distance, 1e-8) + weight * change / center_scale
            baseline_objective = unprojected_dist / max(before_distance, 1e-8) + weight * unprojected_change / center_scale
            choices = [("unconstrained_baseline", baseline_eligible, baseline_objective,
                        unprojected, unprojected_dist, unprojected_nn, unprojected_change),
                       ("protected", eligible, objective, projected, distances, nn, change)]
            for mode, allowed, criterion, pool, pool_dist, pool_nn, pool_change in choices:
                label = f"beer_{beer}_{mode}_lambda_{weight:g}"
                common = {"source_beer": beer, "source_style": SOURCE_STYLE, "target_style": TARGET_STYLE,
                          "mode": mode, "lambda": weight, "is_default_policy": weight == DEFAULT_LAMBDA,
                          "n_candidates": len(pool), "n_selection_feasible": int(allowed.sum()),
                          "n_chemical_feasible": int((chemical_ok if mode == "protected" else baseline_chemical).sum()),
                          "n_support_feasible": int((support if mode == "protected" else unconstrained_support).sum()),
                          "n_real_target_chemical_feasible": actual_target_feasible,
                          "target_distance_before": before_distance, "target_nn_before": before_nn,
                          "support_nn_limit": support_limit,
                          "source_missing_nondefect": int(raw_source.isna().sum().sum()),
                          "transfer_id": label}
                if not allowed.any():
                    records.append({**common, "status": "no_candidate_found", "all_model_checks_passed": False})
                    continue
                candidate_index = int(np.flatnonzero(allowed)[np.argmin(criterion[allowed])])
                selected = pool.iloc[[candidate_index]]
                selection_delta = paired_deltas(selection_models, selected, source)[0]
                audit_delta = paired_deltas(audit_models, selected, source)[0]
                sq95 = float(np.quantile(selection_delta, .95))
                aq95 = float(np.quantile(audit_delta, .95))
                chem_median = bool(chemical_check(selected, caps_median)[0])
                chem_mean = bool(chemical_check(selected, caps_mean)[0])
                audit_ok = aq95 <= TOL
                profile_after = proxy["pipeline"].predict(selected)[0]
                # Audit models do not affect candidate selection. A failed audit
                # is retained in outputs; candidates are never reselected using it.
                status = ("model_checks_passed" if audit_ok else "resampling_audit_failed") if mode == "protected" else "baseline_only"
                records.append({**common, "candidate_index": candidate_index,
                                "status": status,
                                "all_model_checks_passed": bool(mode == "protected" and audit_ok and chem_median),
                                "target_distance_after": float(pool_dist[candidate_index]),
                                "target_improvement": float(1 - pool_dist[candidate_index] / max(before_distance, 1e-8)),
                                "target_nn_after": float(pool_nn[candidate_index]),
                                "change_latent_rms": float(pool_change[candidate_index]),
                                "objective": float(criterion[candidate_index]),
                                "chemical_median_ok": chem_median, "chemical_mean_ok": chem_mean,
                                "selection_paired_delta_q95": sq95, "audit_paired_delta_q95": aq95,
                                "audit_proxy_ok": audit_ok,
                                "stale_prediction_before": float(base_source_profile[y.columns.get_loc("stale_hops")]),
                                "stale_prediction_after": float(profile_after[y.columns.get_loc("stale_hops")])})
                spectra.append({"transfer_id": label, "source_beer": beer, "mode": mode,
                                "lambda": weight, **selected.iloc[0].to_dict()})
                for phase, profile in [("source", base_source_profile), ("transferred", profile_after)]:
                    profiles.append({"transfer_id": label, "source_beer": beer, "mode": mode,
                                     "lambda": weight, "phase": phase,
                                     **dict(zip(y.columns, map(float, profile)))})
                for k, deltas in [("selection", selection_delta), ("audit", audit_delta)]:
                    paired_audit_records.extend({"transfer_id": label, "ensemble": k,
                                                 "model_index": b, "paired_delta": float(value)}
                                                for b, value in enumerate(deltas))
                for name, rule in defects.items():
                    feature = rule["feature"]
                    before = float(raw_source[feature].iloc[0])
                    after = float(selected[feature].iloc[0])
                    chemical_records.append({"transfer_id": label, "source_beer": beer,
                                             "mode": mode, "lambda": weight, "defect": name,
                                             "feature": feature, "unit": rule["unit"],
                                             "before": before, "after": after,
                                             "threshold_median": rule["threshold_median"],
                                             "threshold_mean": rule["threshold_mean"],
                                             "median_upper_bound": caps_median[feature],
                                             "mean_upper_bound": caps_mean[feature],
                                             "median_excess_before": excess(before, rule["threshold_median"]),
                                             "median_excess_after": excess(after, rule["threshold_median"]),
                                             "mean_excess_before": excess(before, rule["threshold_mean"]),
                                             "mean_excess_after": excess(after, rule["threshold_mean"]),
                                             "concentration_change": after - before})
                for feature in x.columns:
                    before = float(source[feature].iloc[0])
                    after = float(selected[feature].iloc[0])
                    if abs(after - before) > 1e-12:
                        changes.append({"transfer_id": label, "feature": feature,
                                        "before": before, "after": after, "change": after - before})
        print(f"Transfer {source_no + 1}/{len(source_rows)} beer={beer}: "
              f"{eligible.sum()}/{len(eligible)} selection-feasible candidates", flush=True)

    tables = {
        "transfer_summary.csv": records, "candidate_diagnostics.csv": candidate_metrics,
        "transferred_spectra.csv": spectra, "sensory_before_after.csv": profiles,
        "defect_checks.csv": chemical_records, "feature_changes.csv": changes,
        "stale_paired_deltas.csv": paired_audit_records,
    }
    for filename, rows in tables.items():
        pd.DataFrame(rows).to_csv(OUT / filename, index=False, lineterminator="\n")
    result = pd.DataFrame(records)
    default = result[result["is_default_policy"].fillna(False) & result["mode"].eq("protected")]
    n_selected = int(default.status.isin(["model_checks_passed", "resampling_audit_failed"]).sum())
    n_checked = int(default.all_model_checks_passed.sum())
    mean_sensitivity = default.get("chemical_mean_ok", pd.Series(False, index=default.index))
    mean_threshold_all_checks = int((default.all_model_checks_passed
                                    & mean_sensitivity.fillna(False)).sum())
    summary = {
        "seed": SEED, "source_style": SOURCE_STYLE, "target_style": TARGET_STYLE,
        "source_beers": list(map(int, meta.beer.iloc[source_rows])), "n_training_beers": len(training_rows),
        "source_exclusion": "All Pils/Lager rows excluded from fitted copula, sensory proxy and bootstrap models",
        "n_base_generated": N_GENERATED, "n_candidates_per_source": N_GENERATED * 4 + 4,
        "lambdas": LAMBDAS, "default_lambda": DEFAULT_LAMBDA,
        "minimum_target_improvement": MIN_TARGET_IMPROVEMENT,
        "minimum_latent_rms_change": MIN_LATENT_RMS_CHANGE,
        "support": {"metric": "full-226-dimensional Gaussian-rank RMS nearest Tripel distance",
                    "reference": "Tripel training leave-one-out nearest-neighbor distances",
                    "quantile": SUPPORT_QUANTILE, "limit": support_limit,
                    "additional_rule": "All features within non-source training marginal support"},
        "constraints": defects, "selection_bootstrap_models": len(selection_models),
        "audit_bootstrap_models": len(audit_models),
        "default_policy": {"n_sources": len(source_rows), "n_candidates_selected": n_selected,
                           "n_model_checks_passed": n_checked,
                           "n_audit_failed": int(default.status.eq("resampling_audit_failed").sum()),
                           "n_no_candidate": int(default.status.eq("no_candidate_found").sum()),
                           "n_source_unverifiable": int(default.status.eq("source_defect_unverifiable").sum()),
                           "n_all_checks_also_pass_mean_thresholds_same_spectra": mean_threshold_all_checks},
        "limitations": [
            "Model checks passed does not mean verified sensory or brewing success.",
            "Bootstrap quantiles after candidate search are uncalibrated model diagnostics, not 95% outcome guarantees.",
            "The 10 audit models share real training data; separate resampling is not an independent experiment.",
            "No audit-based reselection or post-result relaxation of improvement/support thresholds.",
            "Trans-2-nonenal has no supplied threshold: its concentration may not increase.",
            "The 45 sensory outputs remain on attachment 4 standardized scale, not raw 0-6.",
            "Hybrid relative features do not imply measured absolute concentrations or a realizable brewing recipe.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    joblib.dump({"copula": copula, "features": x.columns.tolist(),
                 "training_rows": training_rows, "source_rows": source_rows}, OUT / "transfer_copula.joblib")
    pd.DataFrame({"beer": meta.beer.iloc[target_rows].to_numpy(), "loo_nn_rms": target_loo_nn}).to_csv(
        OUT / "target_support_reference.csv", index=False, lineterminator="\n")
    print(json.dumps(summary["default_policy"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
