"""Threshold-robust transfer with matched generation budgets and proxy audits."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from common import SEED, SOLUTION
from q3_data import load_q3_data, read_defect_thresholds
from q3_generator_optimized import OptimizedStyleGenerator

OUT = SOLUTION / "outputs/q3_optimized/transfer"
OLD = SOLUTION / "outputs/q3"
ROOT_OUT = SOLUTION / "outputs/q3_optimized"
N_GENERATED = 500
FACTORS = [0.8, 0.9, 1.0, 1.1, 1.2]
LAMBDAS = [0.05, 0.2, 1.0]
DEFAULT_LAMBDA = 0.2
DEFAULT_POOL = "optimized_generator"
TOL = 1e-10


def scenarios(defects):
    return [{"scenario": f"{statistic}_x{factor:.1f}", "statistic": statistic,
             "factor": factor,
             "thresholds": {rule["feature"]: (float(rule[f"threshold_{statistic}"]) * factor
                                               if rule[f"threshold_{statistic}"] is not None else None)
                            for rule in defects.values()}}
            for statistic in ["median", "mean"] for factor in FACTORS]


def robust_caps(source, defects, scenario_list):
    caps = {}
    for rule in defects.values():
        field = rule["feature"]
        observed = float(source[field].iloc[0])
        values = [item["thresholds"][field] for item in scenario_list]
        finite = [value for value in values if value is not None]
        caps[field] = max(observed, min(finite)) if finite else observed
    return caps


def chemical_feasible(frame, source, scenario):
    good = np.ones(len(frame), dtype=bool)
    for field, threshold in scenario["thresholds"].items():
        source_value = float(source[field].iloc[0])
        cap = max(source_value, threshold) if threshold is not None else source_value
        value = frame[field].to_numpy(float)
        good &= np.isfinite(value) & (value >= 0) & (value <= cap + TOL)
    return good


def paired(models, frame, source):
    return np.column_stack([
        np.asarray(model.predict(frame)).reshape(-1)
        - float(np.asarray(model.predict(source)).reshape(-1)[0])
        for model in models
    ])


def build_pool(source, old_draws, new_draws, target_median):
    blocks, descriptions = [], []
    for family, draws in [("old_copula", old_draws), ("optimized", new_draws)]:
        for gamma in [.25, .5, .75, 1.0]:
            values = (1 - gamma) * source.to_numpy()[0] + gamma * draws.to_numpy()
            blocks.append(values)
            descriptions.extend({"family": family, "gamma": gamma, "draw": j}
                                for j in range(len(values)))
    for gamma in [.25, .5, .75, 1.0]:
        blocks.append(((1 - gamma) * source.to_numpy()[0] + gamma * target_median)[None])
        descriptions.append({"family": "shared_target_median", "gamma": gamma, "draw": -1})
    return pd.DataFrame(np.vstack(blocks), columns=source.columns), pd.DataFrame(descriptions)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta, x, y, manifest = load_q3_data()
    defects = read_defect_thresholds()
    scenario_list = scenarios(defects)
    (OUT / "threshold_scenarios.json").write_text(json.dumps(scenario_list, indent=2), encoding="utf-8", newline="\n")
    source_rows = np.flatnonzero(meta.tasting_category_fine.eq("Pils/Lager"))
    train = np.flatnonzero(~meta.tasting_category_fine.eq("Pils/Lager"))
    target_rows = np.flatnonzero(meta.tasting_category_fine.eq("Tripel"))
    # Keep exactly the baseline geometry; changing the yardstick would make
    # an improvement in target distance difficult to interpret.
    geometry = joblib.load(OLD / "transfer/transfer_copula.joblib")
    np.testing.assert_array_equal(geometry["training_rows"], train)
    reference = geometry["copula"]
    old_draws = reference.sample("Tripel", N_GENERATED, seed=SEED + 301)
    generator = OptimizedStyleGenerator().fit(x.iloc[train], meta.tasting_category_fine.iloc[train])
    new_draws = generator.sample("Tripel", N_GENERATED, seed=SEED + 301)
    assert old_draws.columns.tolist() == new_draws.columns.tolist() == x.columns.tolist()
    joblib.dump({"generator": generator, "training_rows": train, "source_rows": source_rows,
                 "features": x.columns.tolist()}, OUT / "transfer_generator.joblib", compress=3)
    new_draws.to_csv(OUT / "optimized_base_draws.csv", index=False, lineterminator="\n")
    old_draws.to_csv(OUT / "old_base_draws.csv", index=False, lineterminator="\n")

    stale_model = joblib.load(ROOT_OUT / "stale_fixed/transfer_stale_model.joblib")
    ensemble = joblib.load(ROOT_OUT / "stale_fixed/bootstrap_ensemble.joblib")
    assert ensemble["features"] == stale_model["features"] == x.columns.tolist()
    assert len(ensemble["models"]) == 60
    np.testing.assert_array_equal(stale_model["train_global_indices"], train)
    for indices in ensemble["bootstrap_indices"]:
        assert not np.isin(indices, source_rows).any()
    selection_models, audit_models = ensemble["models"][:40], ensemble["models"][40:]
    qualified = bool(stale_model["qualified_for_proxy"])
    old_ensemble = joblib.load(OLD / "sensory/stale_bootstrap.joblib")
    profile_model = joblib.load(OLD / "sensory/transfer_proxy.joblib")
    target_z = reference.transform(x.iloc[target_rows])
    p = x.shape[1]
    pairwise = cdist(target_z, target_z) / np.sqrt(p)
    np.fill_diagonal(pairwise, np.inf)
    support_limit = float(np.quantile(pairwise.min(axis=1), .95))
    target_center = target_z.mean(axis=0)
    minimum, maximum = x.iloc[train].min().to_numpy(), x.iloc[train].max().to_numpy()
    medians = x.iloc[train].median()
    target_median = x.iloc[target_rows].median().to_numpy()

    records, diag_rows, spectra, threshold_rows, delta_rows, profiles = [], [], [], [], [], []
    for number, index in enumerate(source_rows, 1):
        beer = int(meta.beer.iloc[index])
        raw_source = x.iloc[[index]]
        complete = all(np.isfinite(raw_source[rule["feature"]].iloc[0])
                       and raw_source[rule["feature"]].iloc[0] >= 0 for rule in defects.values())
        if not complete:
            for pool in ["old_generator", "optimized_generator", "combined_pool"]:
                for weight in LAMBDAS:
                    records.append({"source_beer": beer, "pool": pool, "lambda": weight,
                                    "is_default_policy": pool == DEFAULT_POOL and weight == DEFAULT_LAMBDA,
                                    "status": "source_defect_unverifiable", "all_model_checks_passed": False,
                                    "cross_proxy_checks_passed": False})
            continue
        source = raw_source.fillna(medians)
        values, descriptions = build_pool(source, old_draws, new_draws, target_median)
        caps = robust_caps(raw_source, defects, scenario_list)
        for field, cap in caps.items():
            values[field] = values[field].clip(lower=0, upper=cap)
        z, source_z = reference.transform(values), reference.transform(source)
        distance = reference.distance(values, "Tripel")
        before = float(reference.distance(source, "Tripel")[0])
        nn = cdist(z, target_z).min(axis=1) / np.sqrt(p)
        change = np.linalg.norm(z - source_z, axis=1) / np.sqrt(p)
        center_scale = max(float(np.linalg.norm(source_z[0] - target_center) / np.sqrt(p)), 1e-8)
        target_improvement = 1 - distance / max(before, 1e-8)
        support = (nn <= support_limit + TOL) & (
            (values.to_numpy() >= minimum - TOL) & (values.to_numpy() <= maximum + TOL)).all(axis=1)
        chemical = np.column_stack([chemical_feasible(values, raw_source, scenario)
                                    for scenario in scenario_list]).all(axis=1)
        unique_values, inverse = np.unique(values.to_numpy(), axis=0, return_inverse=True)
        candidate_delta = paired(selection_models, pd.DataFrame(unique_values, columns=x.columns), source)[inverse]
        q95 = np.quantile(candidate_delta, .95, axis=1)
        nonzero_delta = np.any(np.abs(candidate_delta) > 1e-8, axis=1)
        eligible = (support & chemical & (change >= .05) & (target_improvement >= .10)
                    & (q95 <= TOL) & nonzero_delta)
        pool_masks = {
            "old_generator": descriptions.family.isin(["old_copula", "shared_target_median"]).to_numpy(),
            "optimized_generator": descriptions.family.isin(["optimized", "shared_target_median"]).to_numpy(),
            "combined_pool": np.ones(len(values), dtype=bool),
        }
        # Full candidate matrices are reproducible from the two saved base
        # draws, source rows, fixed interpolation fractions and robust caps.
        for j in range(len(values)):
            diag_rows.append({"source_beer": beer, "candidate_index": j,
                              **descriptions.iloc[j].to_dict(), "target_distance": float(distance[j]),
                              "target_improvement": float(target_improvement[j]),
                              "target_nn_rms": float(nn[j]), "change_latent_rms": float(change[j]),
                              "support_ok": bool(support[j]), "all_thresholds_chemical_ok": bool(chemical[j]),
                              "selection_q95": float(q95[j]), "nonzero_paired_signal": bool(nonzero_delta[j]),
                              "selection_feasible": bool(eligible[j])})

        before_profile = profile_model["pipeline"].predict(source)[0]
        before_stale = float(np.asarray(stale_model["pipeline"].predict(source)).reshape(-1)[0])
        audit_cache = {}
        for pool, pool_mask in pool_masks.items():
            variation = float(np.std(candidate_delta[pool_mask].mean(axis=1)))
            zero_fraction = float(np.mean(np.all(np.abs(candidate_delta[pool_mask]) <= 1e-8, axis=1)))
            allowed = eligible & pool_mask
            for weight in LAMBDAS:
                transfer_id = f"beer_{beer}_{pool}_lambda_{weight:g}"
                common = {"transfer_id": transfer_id, "source_beer": beer, "source_style": "Pils/Lager",
                          "target_style": "Tripel", "pool": pool, "lambda": weight,
                          "is_default_policy": pool == DEFAULT_POOL and weight == DEFAULT_LAMBDA,
                          "n_candidates": int(pool_mask.sum()), "n_selection_feasible": int(allowed.sum()),
                          "target_distance_before": before, "support_nn_limit": support_limit,
                          "proxy_qualified": qualified, "candidate_mean_delta_sd": variation,
                          "candidate_zero_paired_fraction": zero_fraction,
                          "all_model_checks_passed": False, "cross_proxy_checks_passed": False}
                if not allowed.any():
                    records.append({**common, "status": "no_candidate_found"})
                    continue
                objective = distance / max(before, 1e-8) + weight * change / center_scale
                chosen = int(np.flatnonzero(allowed)[np.argmin(objective[allowed])])
                selected = values.iloc[[chosen]]
                select_delta = candidate_delta[chosen]
                cache_key = selected.to_numpy().tobytes()
                if cache_key not in audit_cache:
                    audit_cache[cache_key] = {
                        "audit_delta": paired(audit_models, selected, source)[0],
                        "old_delta": paired(old_ensemble["models"], selected, source)[0],
                        "after_stale": float(np.asarray(stale_model["pipeline"].predict(selected)).reshape(-1)[0]),
                        "after_profile": profile_model["pipeline"].predict(selected)[0],
                    }
                cached = audit_cache[cache_key]
                audit_delta = cached["audit_delta"]
                audit_q95 = float(np.quantile(audit_delta, .95))
                audit_ok = audit_q95 <= TOL
                # The frozen first-round committee is an architecture check,
                # never a way of choosing a different candidate after the audit.
                old_delta = cached["old_delta"]
                old_selection_q95 = float(np.quantile(old_delta[:30], .95))
                old_audit_q95 = float(np.quantile(old_delta[30:], .95))
                old_consensus = old_selection_q95 <= TOL and old_audit_q95 <= TOL
                informative = variation > 1e-6 and nonzero_delta[chosen]
                passed = bool(qualified and informative and audit_ok)
                status = ("proxy_not_qualified" if not qualified else "proxy_uninformative" if not informative
                          else "model_checks_passed" if audit_ok else "resampling_audit_failed")
                after_stale = cached["after_stale"]
                records.append({**common, "candidate_index": chosen, "status": status,
                                "objective": float(objective[chosen]),
                                "target_distance_after": float(distance[chosen]),
                                "target_improvement": float(target_improvement[chosen]),
                                "target_nn_after": float(nn[chosen]), "change_latent_rms": float(change[chosen]),
                                "all_thresholds_chemical_ok": True,
                                "selection_q95": float(q95[chosen]), "audit_q95": audit_q95,
                                "all_model_checks_passed": passed,
                                "old_selection_q95": old_selection_q95, "old_audit_q95": old_audit_q95,
                                "old_proxy_consensus": bool(old_consensus),
                                "cross_proxy_checks_passed": bool(passed and old_consensus),
                                "stale_before": before_stale, "stale_after": after_stale})
                spectra.append({"transfer_id": transfer_id, "source_beer": beer,
                                "pool": pool, "lambda": weight, **selected.iloc[0].to_dict()})
                after_profile = cached["after_profile"]
                for phase, profile, stale_value in [("source", before_profile, before_stale),
                                                     ("transferred", after_profile, after_stale)]:
                    profile_values = dict(zip(y.columns, map(float, profile)))
                    profile_values["stale_hops"] = stale_value
                    profiles.append({"transfer_id": transfer_id, "phase": phase,
                                     **profile_values})
                for group, delta in [("selection", select_delta), ("audit", audit_delta),
                                     ("old_selection", old_delta[:30]), ("old_audit", old_delta[30:])]:
                    delta_rows.extend({"transfer_id": transfer_id, "ensemble": group,
                                       "model_index": k, "paired_delta": float(value)}
                                      for k, value in enumerate(delta))
                for scenario in scenario_list:
                    for name, rule in defects.items():
                        field = rule["feature"]
                        threshold = scenario["thresholds"][field]
                        old_value = float(raw_source[field].iloc[0])
                        new_value = float(selected[field].iloc[0])
                        cap = max(old_value, threshold) if threshold is not None else old_value
                        threshold_rows.append({"transfer_id": transfer_id, "source_beer": beer,
                                               "pool": pool, "lambda": weight,
                                               "scenario": scenario["scenario"], "statistic": scenario["statistic"],
                                               "factor": scenario["factor"], "defect": name,
                                               "feature": field, "unit": rule["unit"], "threshold": threshold,
                                               "source": old_value, "transferred": new_value,
                                               "upper_bound": cap, "robust_upper_bound": caps[field],
                                               "passed": new_value <= cap + TOL})
        print(f"Optimized transfer {number}/11 beer={beer}, eligible={eligible.sum()}/{len(values)}", flush=True)

    frames = {"transfer_summary.csv": records, "candidate_diagnostics.csv": diag_rows,
              "transferred_spectra.csv": spectra, "threshold_sensitivity.csv": threshold_rows,
              "stale_paired_deltas.csv": delta_rows, "sensory_before_after.csv": profiles}
    for name, rows in frames.items():
        pd.DataFrame(rows).to_csv(OUT / name, index=False, lineterminator="\n")

    # Recheck the unchanged first-round selections on the same threshold grid.
    archived = pd.read_csv(OLD / "transfer/transfer_summary.csv")
    archived = archived[archived["mode"].eq("protected") & archived.is_default_policy]
    archived_spectra = pd.read_csv(OLD / "transfer/transferred_spectra.csv").set_index("transfer_id")
    old_checks = []
    for row in archived.itertuples():
        if pd.isna(row.candidate_index):
            continue
        source = x.loc[meta.beer.eq(row.source_beer)]
        selected = archived_spectra.loc[[row.transfer_id], x.columns]
        for scenario in scenario_list:
            old_checks.append({"source_beer": row.source_beer, "transfer_id": row.transfer_id,
                               "scenario": scenario["scenario"], "old_model_checks_passed": row.all_model_checks_passed,
                               "chemical_passed": bool(chemical_feasible(selected, source, scenario)[0])})
    archived_checks = pd.DataFrame(old_checks)
    archived_checks.to_csv(OUT / "archived_threshold_recheck.csv", index=False, lineterminator="\n")
    result = pd.DataFrame(records)
    policy_rows = []
    for (pool, weight), part in result.groupby(["pool", "lambda"]):
        policy_rows.append({"pool": pool, "lambda": float(weight), "sources": len(part),
                            "selected": int(part.candidate_index.notna().sum()) if "candidate_index" in part else 0,
                            "model_checks_passed": int(part.all_model_checks_passed.sum()),
                            "cross_proxy_checks_passed": int(part.cross_proxy_checks_passed.sum()),
                            "audit_failed": int(part.status.eq("resampling_audit_failed").sum()),
                            "no_candidate": int(part.status.eq("no_candidate_found").sum())})
    pd.DataFrame(policy_rows).to_csv(OUT / "policy_summary.csv", index=False, lineterminator="\n")
    archived_robust = archived_checks.groupby("source_beer").chemical_passed.all()
    old_passed_beers = archived.loc[archived.all_model_checks_passed, "source_beer"]
    old_all_grid = int(archived_robust.reindex(old_passed_beers, fill_value=False).sum())
    default = result[result.is_default_policy]
    summary = {
        "seed": SEED, "source_beers": list(map(int, meta.beer.iloc[source_rows])), "training_n": len(train),
        "geometry": "Frozen first-round 239-source-excluded copula; unchanged full-rank NN support",
        "threshold_scenarios": scenario_list, "threshold_robustness": "simultaneous intersection of all 10 scenarios",
        "target_improvement_minimum": .1, "latent_rms_change_minimum": .05,
        "support_nn_q95": support_limit, "lambdas": LAMBDAS,
        "default_pool": DEFAULT_POOL, "default_lambda": DEFAULT_LAMBDA,
        "matched_pool_candidates": N_GENERATED * 4 + 4, "combined_candidates": N_GENERATED * 8 + 4,
        "base_draws_equal": bool(np.array_equal(old_draws.to_numpy(), new_draws.to_numpy())),
        "proxy_experiment": "fixed_single_target_extra_trees_followup",
        "proxy_qualified": qualified, "selection_models": 40, "audit_models": 20,
        "default_policy": {"sources": len(default),
                           "selected": int(default.candidate_index.notna().sum()) if "candidate_index" in default else 0,
                           "model_checks_passed_all_thresholds": int(default.all_model_checks_passed.sum()),
                           "cross_proxy_checks_passed_all_thresholds": int(default.cross_proxy_checks_passed.sum()),
                           "audit_failed": int(default.status.eq("resampling_audit_failed").sum()),
                           "no_candidate": int(default.status.eq("no_candidate_found").sum())},
        "archived_baseline": {"model_checks_passed_median_only": int(archived.all_model_checks_passed.sum()),
                              "model_checks_passed_all_thresholds_same_spectra": old_all_grid},
        "limitations": [
            "All ten chemical thresholds are enforced by construction; robustness outside this grid is not established.",
            "Proxy audit quantiles are uncalibrated and do not certify actual tasting outcomes.",
            "New random splits reuse the same 250 beers and are not independent external validation.",
            "Proxy family choice is an exploratory follow-up; repeated splits do not replace an external test set.",
            "The old committee is a frozen architecture-consistency check, never used to reselect candidates.",
            "Combined-pool gains are reported separately from matched-budget generator comparisons.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(summary["default_policy"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
