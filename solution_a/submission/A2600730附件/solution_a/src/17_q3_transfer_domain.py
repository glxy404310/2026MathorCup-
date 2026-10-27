"""Joint committee screening with fixed candidate budgets and fresh audit draws."""
from __future__ import annotations

import json
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from common import SOLUTION
from q3_data import load_q3_data, read_defect_thresholds
from q3_stale_optimized import predict_1d, raw_tree_pipeline

OUT = SOLUTION / "outputs/q3_domain/transfer"
OLD = SOLUTION / "outputs/q3"
PREVIOUS = SOLUTION / "outputs/q3_optimized"
DOMAIN = SOLUTION / "outputs/q3_domain/stale"
ADDITIONAL_DRAW_SEED = 20271223
FRESH_AUDIT_SEED = 20280123
GAMMAS = [.25, .5, .75, 1.0]
POOLS = ["matched_2004", "expanded_6004"]
POLICIES = ["single_geometry", "joint_geometry", "joint_margin"]
DEFAULT_POOL, DEFAULT_POLICY = "matched_2004", "joint_margin"
WEIGHT, TOL = .2, 1e-10


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8", newline="\n")


def paired(models, frame, source):
    return np.column_stack([predict_1d(model, frame) - predict_1d(model, source)[0]
                            for model in models])


def chemical_ok(frame, source, scenario):
    good = np.ones(len(frame), dtype=bool)
    for field, threshold in scenario["thresholds"].items():
        observed = float(source[field].iloc[0])
        cap = max(observed, threshold) if threshold is not None else observed
        values = frame[field].to_numpy(float)
        good &= np.isfinite(values) & (values >= 0) & (values <= cap + TOL)
    return good


def build_candidates(source, draws, target_median, caps):
    blocks, descriptions = [], []
    for gamma in GAMMAS:
        block = (1-gamma) * source.to_numpy()[0] + gamma * draws.to_numpy()
        blocks.append(block)
        descriptions.extend({"gamma": gamma, "draw": j, "matched_budget": j < 500}
                            for j in range(len(draws)))
    for gamma in GAMMAS:
        blocks.append(((1-gamma)*source.to_numpy()[0]+gamma*target_median)[None])
        descriptions.append({"gamma": gamma, "draw": -1, "matched_budget": True})
    values = pd.DataFrame(np.vstack(blocks), columns=source.columns)
    for field, cap in caps.items():
        values[field] = values[field].clip(lower=0, upper=cap)
    return values, pd.DataFrame(descriptions)


def policy_counts(part):
    return {"sources": len(part), "selected": int(part.candidate_index.notna().sum()),
            "new_model_checks_passed": int(part.all_model_checks_passed.sum()),
            "historical_cross_proxy_passed": int(part.cross_proxy_checks_passed.sum()),
            "fresh_cross_proxy_passed": int(part.fresh_cross_proxy_checks_passed.sum()),
            "all_committees_passed": int(part.all_committees_passed.sum()),
            "no_candidate": int(part.status.eq("no_candidate_found").sum())}


def main(output_directory=OUT, proxy_directory=DOMAIN, experiment="domain_loso",
         fresh_domain_audit_seed=None, fresh_original_audit_seed=FRESH_AUDIT_SEED):
    started = perf_counter()
    OUT, DOMAIN = output_directory, proxy_directory
    FRESH_AUDIT_SEED = fresh_original_audit_seed
    OUT.mkdir(parents=True, exist_ok=True)
    meta, x, y, _ = load_q3_data()
    train = np.flatnonzero(~meta.tasting_category_fine.eq("Pils/Lager"))
    sources = np.flatnonzero(meta.tasting_category_fine.eq("Pils/Lager"))
    target_rows = np.flatnonzero(meta.tasting_category_fine.eq("Tripel"))
    features = x.columns.tolist()
    defects = read_defect_thresholds()
    scenarios = json.loads((PREVIOUS / "transfer/threshold_scenarios.json").read_text(encoding="utf-8"))
    # This protocol is persisted before committee predictions or audit fitting.
    manifest = {"experiment":experiment,"proxy_directory":DOMAIN.relative_to(SOLUTION).as_posix(),
        "default_pool": DEFAULT_POOL, "default_policy": DEFAULT_POLICY,
        "pools": {"matched_2004": 2004, "expanded_6004": 6004}, "policies": POLICIES,
        "selection": "40 current-proxy models; first 30 frozen original models; paired Q95 <= 0",
        "joint_margin_order": "minimize max(domain selection Q95, original selection Q95), then geometry objective",
        "geometry_objective": "target_distance/source_distance + .2*change_rms/source_center_rms",
        "domain_audit": "last 20 current-proxy models, selected spectra only",
        "fresh_domain_audit_seed":fresh_domain_audit_seed,
        "historical_diagnostic": "last 10 original models, adaptively reused across previous experiments",
        "fresh_audit": "20 fresh-seed draws of original 120-tree single-target protocol, selected spectra only",
        "fresh_audit_seed": FRESH_AUDIT_SEED, "additional_draw_seed": ADDITIONAL_DRAW_SEED,
        "gammas": GAMMAS, "lambda": WEIGHT, "minimum_target_improvement": .10,
        "minimum_change_rms": .05, "threshold_scenarios": scenarios,
        "audit_reselection": False, "default_selected_from_audit": False,
        "qualification_required": True,
        "limitations": ["All audits reuse the same 239 observed training beers.",
                        "Bootstrap quantiles do not have calibrated post-search or causal coverage.",
                        "A safety-margin objective can require larger chemical changes."]}
    write_json(OUT / "run_manifest.json", manifest)
    geometry = joblib.load(OLD / "transfer/transfer_copula.joblib")
    np.testing.assert_array_equal(geometry["training_rows"], train)
    reference = geometry["copula"]
    previous_draws = pd.read_csv(PREVIOUS / "transfer/optimized_base_draws.csv")
    draws = pd.concat([previous_draws, reference.sample("Tripel", 1000, seed=ADDITIONAL_DRAW_SEED)], ignore_index=True)
    draws.to_csv(OUT / "base_draws.csv", index=False, lineterminator="\n")
    target_z = reference.transform(x.iloc[target_rows])
    dimension = x.shape[1]
    pairwise = cdist(target_z, target_z)/np.sqrt(dimension)
    np.fill_diagonal(pairwise, np.inf)
    support_limit = float(np.quantile(pairwise.min(axis=1), .95))
    target_center = target_z.mean(axis=0)
    medians = x.iloc[train].median()
    minimum, maximum = x.iloc[train].min().to_numpy(), x.iloc[train].max().to_numpy()
    target_median = x.iloc[target_rows].median().to_numpy()
    domain_model = joblib.load(DOMAIN / "transfer_stale_model.joblib")
    domain_ensemble = joblib.load(DOMAIN / "bootstrap_ensemble.joblib")
    if fresh_domain_audit_seed is not None:
        # Follow-up experiments keep the frozen 40 screening models, but do
        # not reuse their already observed audit predictions as fresh evidence.
        assert domain_model["selected_config"]["family"]=="extra_trees"
        replacement, replacement_indices, replacement_seeds = [], [], []
        for b in range(20):
            seed=fresh_domain_audit_seed+b
            index=np.random.default_rng(seed).choice(train,size=len(train),replace=True)
            replacement.append(raw_tree_pipeline(seed+10000,n_estimators=120).fit(x.iloc[index],y.stale_hops.iloc[index]))
            replacement_indices.append(index)
            replacement_seeds.append(seed)
        domain_ensemble={**domain_ensemble,
            "models":domain_ensemble["models"][:40]+replacement,
            "bootstrap_indices":list(domain_ensemble["bootstrap_indices"][:40])+replacement_indices,
            "bootstrap_seeds":list(domain_ensemble["bootstrap_seeds"][:40])+replacement_seeds,
            "estimator_seeds":list(domain_ensemble["estimator_seeds"][:40])+[s+10000 for s in replacement_seeds],
            "audit_protocol":"frozen 40 screening models; 20 new-seed audit models for a declared exploratory follow-up"}
        joblib.dump(domain_ensemble,OUT/"screening_and_audit.joblib",compress=3)
    original = joblib.load(OLD / "sensory/stale_bootstrap.joblib")
    old_center = joblib.load(OLD / "sensory/transfer_proxy.joblib")
    for artifact in [domain_model, domain_ensemble, original]:
        assert artifact["features"] == features
        np.testing.assert_array_equal(artifact["train_global_indices"], train)
    for index in domain_ensemble["bootstrap_indices"]:
        assert not np.isin(index, sources).any()
    new_select, new_audit = domain_ensemble["models"][:40], domain_ensemble["models"][40:]
    old_select, old_historical = original["models"][:30], original["models"][30:]
    assert len(new_select) == 40 and len(new_audit) == 20
    qualified = bool(domain_model["qualified_for_proxy"])
    fresh_models, fresh_indices = [], []
    for b in range(20):
        seed = FRESH_AUDIT_SEED+b
        index = np.random.default_rng(seed).choice(train, size=len(train), replace=True)
        fresh_models.append(raw_tree_pipeline(seed+10000, n_estimators=120).fit(x.iloc[index], y.stale_hops.iloc[index]))
        fresh_indices.append(index)
    joblib.dump({"models": fresh_models, "bootstrap_indices": fresh_indices,
                 "bootstrap_seeds": list(range(FRESH_AUDIT_SEED, FRESH_AUDIT_SEED+20)),
                 "features": features, "train_global_indices": train,
                 "excluded_global_indices": sources, "target": "stale_hops",
                 "protocol": "original single-target ExtraTrees 120/minleaf3/maxfeatures.7"},
                OUT / "fresh_original_protocol_audit.joblib", compress=3)
    summaries, diagnostics, spectra, deltas, thresholds, previous_checks = [], [], [], [], [], []
    previous_summary = pd.read_csv(PREVIOUS / "transfer/transfer_summary.csv")
    previous_summary = previous_summary[previous_summary.is_default_policy].set_index("source_beer")
    previous_spectra = pd.read_csv(PREVIOUS / "transfer/transferred_spectra.csv").set_index("transfer_id")
    stale_position = y.columns.get_loc("stale_hops")
    for count, index in enumerate(sources, 1):
        beer = int(meta.beer.iloc[index])
        raw_source = x.iloc[[index]]
        assert all(np.isfinite(raw_source[rule["feature"]].iloc[0]) and raw_source[rule["feature"]].iloc[0]>=0
                   for rule in defects.values())
        source = raw_source.fillna(medians)
        caps = {}
        for rule in defects.values():
            field = rule["feature"]
            finite = [scenario["thresholds"][field] for scenario in scenarios if scenario["thresholds"][field] is not None]
            caps[field] = max(float(raw_source[field].iloc[0]), min(finite)) if finite else float(raw_source[field].iloc[0])
        values, descriptions = build_candidates(source, draws, target_median, caps)
        latent, source_z = reference.transform(values), reference.transform(source)
        distance = reference.distance(values, "Tripel")
        before = float(reference.distance(source, "Tripel")[0])
        nn = cdist(latent, target_z).min(axis=1)/np.sqrt(dimension)
        change = np.linalg.norm(latent-source_z, axis=1)/np.sqrt(dimension)
        center_scale = max(float(np.linalg.norm(source_z[0]-target_center)/np.sqrt(dimension)), 1e-8)
        improvement = 1-distance/max(before, 1e-8)
        support = (nn <= support_limit+TOL) & ((values.to_numpy()>=minimum-TOL)&(values.to_numpy()<=maximum+TOL)).all(axis=1)
        chemical = np.column_stack([chemical_ok(values, raw_source, s) for s in scenarios]).all(axis=1)
        unique, inverse = np.unique(values.to_numpy(), axis=0, return_inverse=True)
        unique_frame = pd.DataFrame(unique, columns=features)
        new_delta = paired(new_select, unique_frame, source)[inverse]
        old_delta = paired(old_select, unique_frame, source)[inverse]
        new_q95, old_q95 = np.quantile(new_delta, .95, axis=1), np.quantile(old_delta, .95, axis=1)
        worst = np.maximum(new_q95, old_q95)
        new_signal = np.any(np.abs(new_delta)>1e-8, axis=1)
        old_signal = np.any(np.abs(old_delta)>1e-8, axis=1)
        base_ok = support & chemical & (improvement>=.1) & (change>=.05)
        single_ok = base_ok & (new_q95<=TOL) & new_signal
        joint_ok = single_ok & (old_q95<=TOL) & old_signal
        objective = distance/max(before, 1e-8)+WEIGHT*change/center_scale
        diagnostics.append(pd.concat([pd.DataFrame({"source_beer":beer, "candidate_index":np.arange(len(values)),
            "target_distance":distance,"target_improvement":improvement,"target_nn_rms":nn,
            "change_latent_rms":change,"support_ok":support,"all_thresholds_chemical_ok":chemical,
            "selection_new_q95":new_q95,"selection_old_q95":old_q95,"worst_selection_q95":worst,
            "new_signal":new_signal,"old_signal":old_signal,"single_feasible":single_ok,
            "joint_feasible":joint_ok,"geometry_objective":objective}), descriptions],axis=1))
        before_stale = float(predict_1d(domain_model["pipeline"], source)[0])
        before_old = float(old_center["pipeline"].predict(source)[0,stale_position])
        audit_cache = {}
        for pool in POOLS:
            pool_mask = descriptions.matched_budget.to_numpy() if pool=="matched_2004" else np.ones(len(values),bool)
            new_variation = float(np.std(new_delta[pool_mask].mean(axis=1)))
            old_variation = float(np.std(old_delta[pool_mask].mean(axis=1)))
            for policy in POLICIES:
                eligible = (single_ok if policy=="single_geometry" else joint_ok) & pool_mask
                ids = np.flatnonzero(eligible)
                transfer_id = f"beer_{beer}_{pool}_{policy}"
                common = {"transfer_id":transfer_id,"source_beer":beer,"pool":pool,"policy":policy,
                    "is_default_policy":pool==DEFAULT_POOL and policy==DEFAULT_POLICY,"n_candidates":int(pool_mask.sum()),
                    "n_selection_feasible":len(ids),"target_distance_before":before,"support_nn_limit":support_limit,
                    "proxy_qualified":qualified,"new_candidate_delta_sd":new_variation,"old_candidate_delta_sd":old_variation,
                    "all_model_checks_passed":False,"cross_proxy_checks_passed":False,
                    "fresh_cross_proxy_checks_passed":False,"all_committees_passed":False}
                if not len(ids):
                    summaries.append({**common,"candidate_index":np.nan,"status":"no_candidate_found"})
                    continue
                chosen = int(ids[np.lexsort((objective[ids],worst[ids]))[0]]) if policy=="joint_margin" else int(ids[np.argmin(objective[ids])])
                selected = values.iloc[[chosen]]
                key = selected.to_numpy().tobytes()
                if key not in audit_cache:
                    audit_cache[key] = {"new_audit":paired(new_audit,selected,source)[0],
                        "old_historical":paired(old_historical,selected,source)[0],
                        "fresh_original_audit":paired(fresh_models,selected,source)[0],
                        "stale_after":float(predict_1d(domain_model["pipeline"],selected)[0]),
                        "old_center_after":float(old_center["pipeline"].predict(selected)[0,stale_position])}
                cached = audit_cache[key]
                audit_q95 = {group:float(np.quantile(cached[group],.95)) for group in ["new_audit","old_historical","fresh_original_audit"]}
                informative = new_variation>1e-6 and bool(new_signal[chosen])
                old_informative = old_variation>1e-6 and bool(old_signal[chosen])
                new_pass = bool(qualified and informative and audit_q95["new_audit"]<=TOL)
                cross = bool(new_pass and old_informative and old_q95[chosen]<=TOL and audit_q95["old_historical"]<=TOL)
                fresh_cross = bool(new_pass and old_informative and old_q95[chosen]<=TOL
                                   and audit_q95["fresh_original_audit"]<=TOL)
                all_committees = bool(cross and fresh_cross)
                status = ("proxy_not_qualified" if not qualified else "proxy_uninformative" if not informative
                          else "new_audit_failed" if not new_pass else "fresh_original_audit_failed" if not fresh_cross
                          else "all_checks_passed" if all_committees else "fresh_checks_passed_historical_disagrees")
                summaries.append({**common,"candidate_index":chosen,"status":status,"objective":float(objective[chosen]),
                    "target_distance_after":float(distance[chosen]),"target_improvement":float(improvement[chosen]),
                    "target_nn_after":float(nn[chosen]),"change_latent_rms":float(change[chosen]),
                    "selection_new_q95":float(new_q95[chosen]),"selection_old_q95":float(old_q95[chosen]),
                    "worst_selection_q95":float(worst[chosen]),"new_audit_q95":audit_q95["new_audit"],
                    "old_historical_audit_q95":audit_q95["old_historical"],"fresh_original_audit_q95":audit_q95["fresh_original_audit"],
                    "all_thresholds_chemical_ok":True,"all_model_checks_passed":new_pass,
                    "cross_proxy_checks_passed":cross,"fresh_cross_proxy_checks_passed":fresh_cross,
                    "all_committees_passed":all_committees,
                    "stale_before":before_stale,"stale_after":cached["stale_after"],
                    "old_center_before":before_old,"old_center_after":cached["old_center_after"]})
                spectra.append({"transfer_id":transfer_id,"source_beer":beer,"pool":pool,"policy":policy,**selected.iloc[0].to_dict()})
                groups = {"new_selection":new_delta[chosen],"old_selection":old_delta[chosen],
                          **{group:cached[group] for group in audit_q95}}
                for group, delta in groups.items():
                    deltas.extend({"transfer_id":transfer_id,"ensemble":group,"model_index":k,"paired_delta":float(value)}
                                  for k,value in enumerate(delta))
                for scenario in scenarios:
                    for label,rule in defects.items():
                        field=rule["feature"]
                        threshold=scenario["thresholds"][field]
                        observed=float(raw_source[field].iloc[0])
                        cap=max(observed,threshold) if threshold is not None else observed
                        value=float(selected[field].iloc[0])
                        thresholds.append({"transfer_id":transfer_id,"source_beer":beer,"pool":pool,"policy":policy,
                            "scenario":scenario["scenario"],"defect":label,"feature":field,"unit":rule["unit"],
                            "threshold":threshold,"source":observed,"transferred":value,"upper_bound":cap,"passed":value<=cap+TOL})
        # Fixed previous selections are evaluated only after this round's choices.
        previous_row=previous_summary.loc[beer]
        if pd.notna(previous_row.candidate_index):
            frame=previous_spectra.loc[[previous_row.transfer_id],features]
            previous_z=reference.transform(frame)
            previous_nn=float(cdist(previous_z,target_z).min()/np.sqrt(dimension))
            previous_change=float(np.linalg.norm(previous_z-source_z)/np.sqrt(dimension))
            previous_improvement=1-float(reference.distance(frame,"Tripel")[0])/max(before,1e-8)
            geometry_ok=bool(previous_nn<=support_limit+TOL and previous_change>=.05 and previous_improvement>=.1
                             and ((frame.to_numpy()>=minimum-TOL)&(frame.to_numpy()<=maximum+TOL)).all())
            new_previous_delta=paired(new_select,frame,source)
            old_previous_delta=paired(old_select,frame,source)
            nq=float(np.quantile(new_previous_delta,.95))
            oq=float(np.quantile(old_previous_delta,.95))
            na=float(np.quantile(paired(new_audit,frame,source),.95))
            oh=float(np.quantile(paired(old_historical,frame,source),.95))
            fa=float(np.quantile(paired(fresh_models,frame,source),.95))
            chem=all(chemical_ok(frame,raw_source,s)[0] for s in scenarios)
            matched=descriptions.matched_budget.to_numpy()
            new_ok=bool(qualified and chem and geometry_ok and nq<=TOL and na<=TOL
                        and np.any(np.abs(new_previous_delta)>1e-8) and np.std(new_delta[matched].mean(axis=1))>1e-6)
            old_screen_ok=bool(oq<=TOL and np.any(np.abs(old_previous_delta)>1e-8)
                               and np.std(old_delta[matched].mean(axis=1))>1e-6)
            old_ok=bool(old_screen_ok and oh<=TOL)
            previous_checks.append({"source_beer":beer,"previous_transfer_id":previous_row.transfer_id,
                "all_thresholds_chemical_ok":chem,"geometry_ok":geometry_ok,"selection_new_q95":nq,"selection_old_q95":oq,
                "new_audit_q95":na,"old_historical_audit_q95":oh,"fresh_original_audit_q95":fa,
                "all_model_checks_passed":new_ok,"cross_proxy_checks_passed":new_ok and old_ok,
                "fresh_cross_proxy_checks_passed":new_ok and old_screen_ok and fa<=TOL,
                "all_committees_passed":new_ok and old_ok and fa<=TOL,
                "previous_target_improvement":float(previous_row.target_improvement)})
        print(f"Domain transfer {count}/11 beer={beer}: single={single_ok.sum()}, joint={joint_ok.sum()}",flush=True)
    result=pd.DataFrame(summaries)
    result.to_csv(OUT/"transfer_summary.csv",index=False, lineterminator="\n")
    pd.concat(diagnostics,ignore_index=True).to_csv(OUT/"candidate_diagnostics.csv",index=False, lineterminator="\n")
    pd.DataFrame(spectra).to_csv(OUT/"transferred_spectra.csv",index=False, lineterminator="\n")
    pd.DataFrame(deltas).to_csv(OUT/"stale_paired_deltas.csv",index=False, lineterminator="\n")
    pd.DataFrame(thresholds).to_csv(OUT/"threshold_sensitivity.csv",index=False, lineterminator="\n")
    previous_frame=pd.DataFrame(previous_checks)
    previous_frame.to_csv(OUT/"previous_same_spectra_recheck.csv",index=False, lineterminator="\n")
    policies=[{"pool":pool,"policy":policy,**policy_counts(part)} for (pool,policy),part in result.groupby(["pool","policy"])]
    pd.DataFrame(policies).to_csv(OUT/"policy_summary.csv",index=False, lineterminator="\n")
    default=result[result.is_default_policy]
    summary={"experiment":experiment,"protocol":manifest,"training_n":len(train),"sources":len(sources),"features":len(features),
        "support_nn_limit":support_limit,"proxy_qualified":qualified,"selected_domain_config":domain_model.get("selected_config",domain_model.get("config")),
        "default_policy":policy_counts(default),"policies":policies,
        "previous_original_rules":json.loads((PREVIOUS/"transfer/summary.json").read_text(encoding="utf-8"))["default_policy"],
        "previous_same_spectra_current_rules":{"sources":len(previous_frame),
            "new_model_checks_passed":int(previous_frame.all_model_checks_passed.sum()),
            "historical_cross_proxy_passed":int(previous_frame.cross_proxy_checks_passed.sum()),
            "fresh_cross_proxy_passed":int(previous_frame.fresh_cross_proxy_checks_passed.sum()),
            "all_committees_passed":int(previous_frame.all_committees_passed.sum())},
        "original_committee_protocol":"single-target 120-tree ExtraTrees; central original predictor alone was 45-output",
        "elapsed_seconds":perf_counter()-started}
    write_json(OUT/"summary.json",summary)
    print(json.dumps(summary["default_policy"],ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    with threadpool_limits(limits=1):
        main()
