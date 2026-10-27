"""Candidate contraction and predeclared proximity/committee trade-offs."""
from __future__ import annotations

from importlib import import_module
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

CORE = import_module("17_q3_transfer_domain")
OUT = SOLUTION / "outputs/q3_refined/transfer"
PRIOR = SOLUTION / "outputs/q3_domain"
ORIGINAL = SOLUTION / "outputs/q3"
OPTIMIZED = SOLUTION / "outputs/q3_optimized"
REFINED = SOLUTION / "outputs/q3_refined/stale"
POOLS = ("legacy_2004", "legacy_6004", "contracted_6004")
BOUNDS = {"absolute10":(.10,None), "absolute30":(.30,None),
          "absolute50":(.50,None), "absolute60":(.60,None),
          "retain80":(.10,.80), "retain100":(.10,1.0)}
RANKINGS = ("geometry", "margin")
CONTRACTIONS = (.2,.4,.6)
GAMMAS = (.25,.5,.75,1.0)
DEFAULT = {"arm":"refined", "pool":"contracted_6004", "bound":"retain80", "ranking":"margin"}
OLD_AUDIT_SEED = 202104123
FIXED_AUDIT_SEED = 202105123
TOL = 1e-10


def write_json(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8", newline="\n")


def target_draws(reference, legacy):
    base = reference.transform(legacy.iloc[:500])
    center = reference.style_center_["Tripel"]
    latent = np.vstack([center+c*(base-center) for c in CONTRACTIONS])
    return reference.inverse_transform(latent), latent


def candidate_values(source, legacy, contracted, median, caps):
    blocks, rows = [], []
    for family, draws in [("legacy",legacy),("contracted",contracted)]:
        for gamma in GAMMAS:
            blocks.append((1-gamma)*source.to_numpy()[0]+gamma*draws.to_numpy())
            rows.extend({"family":family,"gamma":gamma,"draw":j,
                         "contraction":CONTRACTIONS[j//500] if family=="contracted" else np.nan}
                        for j in range(len(draws)))
    for gamma in GAMMAS:
        blocks.append(((1-gamma)*source.to_numpy()[0]+gamma*median)[None])
        rows.append({"family":"median","gamma":gamma,"draw":-1,"contraction":np.nan})
    frame = pd.DataFrame(np.vstack(blocks),columns=source.columns)
    for field,cap in caps.items():
        frame[field] = frame[field].clip(lower=0,upper=cap)
    return frame,pd.DataFrame(rows)


def pool_masks(descriptions):
    median = descriptions.family.eq("median").to_numpy()
    legacy = descriptions.family.eq("legacy").to_numpy()
    return {"legacy_2004":median|(legacy&descriptions.draw.lt(500).to_numpy()),
            "legacy_6004":median|legacy,
            "contracted_6004":median|descriptions.family.eq("contracted").to_numpy()}


def fresh_trees(x, y, train, seed):
    models, indices = [], []
    for b in range(20):
        current = seed+b
        index = np.random.default_rng(current).choice(train,len(train),replace=True)
        models.append(raw_tree_pipeline(current+10000,n_estimators=120).fit(x.iloc[index],y.iloc[index]))
        indices.append(index)
    return {"models":models,"bootstrap_indices":indices,"bootstrap_seeds":list(range(seed,seed+20)),
            "train_global_indices":train,"features":x.columns.tolist(),
            "protocol":"single-target ExtraTrees 120/minleaf3/max_features.7"}


def counts(part):
    selected = part.candidate_index.notna()
    passing = part.fresh_consistency_passed
    return {"sources":len(part),"selected":int(selected.sum()),"new_passed":int(part.new_model_passed.sum()),
        "fresh_consistency_passed":int(passing.sum()),"historical_consistency_passed":int(part.historical_consistency_passed.sum()),
        "no_candidate":int((~selected).sum()),
        "median_improvement_selected":float(part.loc[selected,"target_improvement"].median()) if selected.any() else None,
        "median_improvement_passed":float(part.loc[passing,"target_improvement"].median()) if passing.any() else None,
        "median_retention_selected":float(part.loc[selected,"retained_anchor_gain"].median()) if selected.any() else None,
        "median_change_rms_selected":float(part.loc[selected,"change_latent_rms"].median()) if selected.any() else None}


def main():
    started = perf_counter()
    OUT.mkdir(parents=True,exist_ok=True)
    meta,x,y45,_ = load_q3_data()
    y = y45.stale_hops
    train = np.flatnonzero(~meta.tasting_category_fine.eq("Pils/Lager"))
    sources = np.flatnonzero(meta.tasting_category_fine.eq("Pils/Lager"))
    target = np.flatnonzero(meta.tasting_category_fine.eq("Tripel"))
    features = x.columns.tolist()
    defects = read_defect_thresholds()
    scenarios = json.loads((OPTIMIZED/"transfer/threshold_scenarios.json").read_text(encoding="utf-8"))
    manifest = {"default":DEFAULT,"pools":dict(zip(POOLS,(2004,6004,6004))),
        "bounds":{key:{"absolute_minimum":value[0],"anchor_fraction":value[1]} for key,value in BOUNDS.items()},
        "rankings":RANKINGS,"contractions":CONTRACTIONS,"gammas":GAMMAS,
        "contracted_latents":"transform frozen legacy first500 chemical rows, then mu+c*(z-mu); empirical inverse CDF",
        "anchor":"frozen q3_optimized default spectrum per source; same original geometry",
        "retention_floor":"max(.10, anchor_fraction*anchor_gain)",
        "screens":"new40 and original30 paired Q95<=0; both current centers delta<=0; nonzero paired signals",
        "fresh_audit":"new20 and new-seed original single-target120tree20; no reselection",
        "historical_audit":"original last10 reused diagnostic only, excluded from primary fresh gate",
        "old_audit_seed":OLD_AUDIT_SEED,"fixed_audit_seed":FIXED_AUDIT_SEED,
        "threshold_scenarios":scenarios,"minimum_change_rms":.05,"geometry_lambda":.2,
        "default_chosen_from_audit":False,"audit_reselection":False,
        "interpretation":"finite empirical trade-off grid; not a global Pareto optimum or tasting validation"}
    write_json(OUT/"run_manifest.json",manifest)
    geometry = joblib.load(ORIGINAL/"transfer/transfer_copula.joblib")
    np.testing.assert_array_equal(geometry["training_rows"],train)
    reference = geometry["copula"]
    legacy = pd.read_csv(PRIOR/"consensus/base_draws.csv")
    contracted,contracted_latents = target_draws(reference,legacy)
    legacy.to_csv(OUT/"legacy_base_draws.csv",index=False, lineterminator="\n")
    contracted.to_csv(OUT/"contracted_base_draws.csv",index=False, lineterminator="\n")
    np.save(OUT/"contracted_latents.npy",contracted_latents)
    pd.DataFrame({"draw":np.arange(1500),"base_draw":np.tile(np.arange(500),3),
        "contraction":np.repeat(CONTRACTIONS,500)}).to_csv(OUT/"contracted_draw_metadata.csv",index=False, lineterminator="\n")
    tz = reference.transform(x.iloc[target])
    dim = x.shape[1]
    pairwise = cdist(tz,tz)/np.sqrt(dim)
    np.fill_diagonal(pairwise,np.inf)
    support_limit = float(np.quantile(pairwise.min(axis=1),.95))
    center = tz.mean(axis=0)
    medians = x.iloc[train].median()
    minimum,maximum = x.iloc[train].min().to_numpy(),x.iloc[train].max().to_numpy()
    target_median = x.iloc[target].median().to_numpy()
    old = joblib.load(ORIGINAL/"sensory/stale_bootstrap.joblib")
    old_center = joblib.load(ORIGINAL/"sensory/transfer_proxy.joblib")
    refined_center = joblib.load(REFINED/"transfer_stale_model.joblib")
    refined_ensemble = joblib.load(REFINED/"bootstrap_ensemble.joblib")
    fixed_center = joblib.load(OPTIMIZED/"stale_fixed/transfer_stale_model.joblib")
    fixed_ensemble = joblib.load(OPTIMIZED/"stale_fixed/bootstrap_ensemble.joblib")
    old_audit = fresh_trees(x,y,train,OLD_AUDIT_SEED)
    fixed_audit = fresh_trees(x,y,train,FIXED_AUDIT_SEED)
    old_audit["excluded_global_indices"] = sources
    fixed_audit["excluded_global_indices"] = sources
    joblib.dump(old_audit,OUT/"fresh_original_audit.joblib",compress=3)
    joblib.dump(fixed_audit,OUT/"fresh_fixed_audit.joblib",compress=3)
    arms = {"refined":{"center":refined_center,"selection":refined_ensemble["models"][:40],
                       "audit":refined_ensemble["models"][40:]},
            "fixed":{"center":fixed_center,"selection":fixed_ensemble["models"][:40],"audit":fixed_audit["models"]}}
    for artifact in [old,old_center,refined_center,refined_ensemble,fixed_center,fixed_ensemble]:
        np.testing.assert_array_equal(artifact["train_global_indices"],train)
        assert artifact["features"]==features
    assert len(refined_ensemble["models"])==60
    anchor_summary = pd.read_csv(OPTIMIZED/"transfer/transfer_summary.csv")
    anchor_summary = anchor_summary[anchor_summary.is_default_policy].set_index("source_beer")
    anchor_spectra = pd.read_csv(OPTIMIZED/"transfer/transferred_spectra.csv").set_index("transfer_id")
    prior_summary = pd.read_csv(PRIOR/"consensus/transfer_summary.csv")
    prior_spectra = pd.read_csv(PRIOR/"consensus/transferred_spectra.csv").set_index("transfer_id")
    records,diags,geometry_rows,selected_rows,delta_rows,threshold_rows,rechecks,uniqueness = [],[],[],[],[],[],[],[]
    stale_position = y45.columns.get_loc("stale_hops")
    for number,index in enumerate(sources,1):
        beer = int(meta.beer.iloc[index])
        raw = x.iloc[[index]]
        source = raw.fillna(medians)
        assert all(np.isfinite(raw[r["feature"]].iloc[0]) and raw[r["feature"]].iloc[0]>=0 for r in defects.values())
        caps = {}
        for rule in defects.values():
            field = rule["feature"]
            values = [s["thresholds"][field] for s in scenarios if s["thresholds"][field] is not None]
            caps[field] = max(float(raw[field].iloc[0]),min(values)) if values else float(raw[field].iloc[0])
        candidates,descriptions = candidate_values(source,legacy,contracted,target_median,caps)
        masks = pool_masks(descriptions)
        unique,inverse = np.unique(candidates.to_numpy(),axis=0,return_inverse=True)
        unique_frame = pd.DataFrame(unique,columns=features)
        z,sz = reference.transform(candidates),reference.transform(source)
        distance = reference.distance(candidates,"Tripel")
        before = float(reference.distance(source,"Tripel")[0])
        improvement = 1-distance/max(before,1e-8)
        change = np.linalg.norm(z-sz,axis=1)/np.sqrt(dim)
        nn = cdist(z,tz).min(axis=1)/np.sqrt(dim)
        support = (nn<=support_limit+TOL)&((candidates.to_numpy()>=minimum-TOL)&(candidates.to_numpy()<=maximum+TOL)).all(axis=1)
        chemical = np.column_stack([CORE.chemical_ok(candidates,raw,s) for s in scenarios]).all(axis=1)
        geometric = support&chemical&(change>=.05)
        scale = max(float(np.linalg.norm(sz[0]-center)/np.sqrt(dim)),1e-8)
        objective = distance/max(before,1e-8)+.2*change/scale
        anchor_row = anchor_summary.loc[beer]
        anchor_frame = anchor_spectra.loc[[anchor_row.transfer_id],features]
        anchor_gain = 1-float(reference.distance(anchor_frame,"Tripel")[0])/max(before,1e-8)
        assert anchor_gain>0
        geometry_rows.append(pd.concat([pd.DataFrame({"source_beer":beer,"candidate_index":np.arange(len(candidates)),
            "target_distance":distance,"target_improvement":improvement,"change_latent_rms":change,"target_nn_rms":nn,
            "support_ok":support,"chemical_ok":chemical,"geometry_objective":objective}),descriptions],axis=1))
        for pool,mask in masks.items():
            unique_n = len(np.unique(candidates.to_numpy()[mask],axis=0))
            uniqueness.append({"source_beer":beer,"pool":pool,"nominal_candidates":int(mask.sum()),"unique_candidates":unique_n,
                "duplicate_fraction":1-unique_n/int(mask.sum()),"anchor_gain":anchor_gain})
        old_delta = CORE.paired(old["models"][:30],unique_frame,source)[inverse]
        old_q = np.quantile(old_delta,.95,axis=1)
        old_signal = np.any(np.abs(old_delta)>1e-8,axis=1)
        center_old_source = float(old_center["pipeline"].predict(source)[0,stale_position])
        center_old_delta = old_center["pipeline"].predict(unique_frame)[:,stale_position][inverse]-center_old_source
        for arm,models in arms.items():
            current = models["center"]
            qualified = bool(current["qualified_for_proxy"])
            new_delta = CORE.paired(models["selection"],unique_frame,source)[inverse]
            new_q = np.quantile(new_delta,.95,axis=1)
            new_signal = np.any(np.abs(new_delta)>1e-8,axis=1)
            center_new_source = float(predict_1d(current["pipeline"],source)[0])
            center_new_delta = predict_1d(current["pipeline"],unique_frame)[inverse]-center_new_source
            worst = np.maximum(new_q,old_q)
            center_ok = (center_new_delta<=TOL)&(center_old_delta<=TOL)
            joint = geometric&(new_q<=TOL)&(old_q<=TOL)&new_signal&old_signal&center_ok
            diags.append(pd.DataFrame({"source_beer":beer,"arm":arm,"candidate_index":np.arange(len(candidates)),
                "new_selection_q95":new_q,"old_selection_q95":old_q,"worst_selection_q95":worst,
                "center_new_delta":center_new_delta,"center_old_delta":center_old_delta,
                "new_signal":new_signal,"old_signal":old_signal,"joint_feasible":joint}))
            cache = {}
            def assess(frame):
                key=frame.to_numpy().tobytes()
                if key not in cache:
                    cache[key]={"new_audit":CORE.paired(models["audit"],frame,source)[0],
                        "old_fresh_audit":CORE.paired(old_audit["models"],frame,source)[0],
                        "old_historical":CORE.paired(old["models"][30:],frame,source)[0]}
                return cache[key]
            for pool,mask in masks.items():
                new_variation=float(np.std(new_delta[mask].mean(axis=1)))
                old_variation=float(np.std(old_delta[mask].mean(axis=1)))
                informative=new_variation>1e-6 and old_variation>1e-6
                for bound,(absolute,retention) in BOUNDS.items():
                    floor=max(absolute,retention*anchor_gain) if retention is not None else absolute
                    ids=np.flatnonzero(joint&mask&(improvement>=floor))
                    for ranking in RANKINGS:
                        uid=f"{arm}_beer{beer}_{pool}_{bound}_{ranking}"
                        common={"transfer_id":uid,"source_beer":beer,"arm":arm,"pool":pool,"bound":bound,"ranking":ranking,
                            "is_default_policy":arm==DEFAULT["arm"] and pool==DEFAULT["pool"] and bound==DEFAULT["bound"] and ranking==DEFAULT["ranking"],
                            "is_arm_default":pool==DEFAULT["pool"] and bound==DEFAULT["bound"] and ranking==DEFAULT["ranking"],
                            "anchor_gain":anchor_gain,"required_gain":floor,"qualified":qualified,"informative":informative,
                            "candidate_count":int(mask.sum()),"feasible_count":len(ids),"new_delta_sd":new_variation,"old_delta_sd":old_variation,
                            "candidate_index":np.nan,"new_model_passed":False,"fresh_consistency_passed":False,"historical_consistency_passed":False}
                        if not len(ids):
                            records.append({**common,"status":"no_candidate"});continue
                        chosen=int(ids[np.lexsort((objective[ids],worst[ids]))[0]]) if ranking=="margin" else int(ids[np.argmin(objective[ids])])
                        frame=candidates.iloc[[chosen]]
                        audited=assess(frame)
                        aq={name:float(np.quantile(delta,.95)) for name,delta in audited.items()}
                        new_pass=bool(qualified and informative and aq["new_audit"]<=TOL)
                        fresh_pass=bool(new_pass and aq["old_fresh_audit"]<=TOL)
                        history_pass=bool(new_pass and aq["old_historical"]<=TOL)
                        status="unqualified" if not qualified else "uninformative" if not informative else "new_audit_failed" if not new_pass else "old_fresh_audit_failed" if not fresh_pass else "passed"
                        records.append({**common,"candidate_index":chosen,"status":status,"target_distance_before":before,
                            "target_distance_after":float(distance[chosen]),"target_improvement":float(improvement[chosen]),
                            "retained_anchor_gain":float(improvement[chosen]/anchor_gain),"change_latent_rms":float(change[chosen]),
                            "target_nn_rms":float(nn[chosen]),"support_nn_limit":support_limit,"geometry_objective":float(objective[chosen]),
                            "new_selection_q95":float(new_q[chosen]),"old_selection_q95":float(old_q[chosen]),"worst_selection_q95":float(worst[chosen]),
                            "center_new_delta":float(center_new_delta[chosen]),"center_old_delta":float(center_old_delta[chosen]),
                            "new_audit_q95":aq["new_audit"],"old_fresh_audit_q95":aq["old_fresh_audit"],"old_historical_q95":aq["old_historical"],
                            "new_model_passed":new_pass,"fresh_consistency_passed":fresh_pass,"historical_consistency_passed":history_pass})
                        selected_rows.append({"transfer_id":uid,**frame.iloc[0].to_dict()})
                        for name,delta in {"new_selection":new_delta[chosen],"old_selection":old_delta[chosen],**audited}.items():
                            delta_rows.extend({"transfer_id":uid,"ensemble":name,"model_index":j,"delta":float(value)} for j,value in enumerate(delta))
                        for scenario in scenarios:
                            for label,rule in defects.items():
                                field=rule["feature"];threshold=scenario["thresholds"][field]
                                source_value=float(raw[field].iloc[0]);after=float(frame[field].iloc[0])
                                cap=max(source_value,threshold) if threshold is not None else source_value
                                threshold_rows.append({"transfer_id":uid,"scenario":scenario["scenario"],"defect":label,"feature":field,
                                    "source":source_value,"transferred":after,"cap":cap,"passed":after<=cap+TOL})
            references=[("original_anchor",anchor_frame)]
            for pool in ("matched_2004","expanded_6004"):
                prior_row=prior_summary[prior_summary.source_beer.eq(beer)&prior_summary.pool.eq(pool)&prior_summary.policy.eq("joint_margin")].iloc[0]
                if pd.notna(prior_row.candidate_index):
                    references.append((f"prior_{pool}",prior_spectra.loc[[prior_row.transfer_id],features]))
            for label,frame in references:
                pz=reference.transform(frame);gain=1-float(reference.distance(frame,"Tripel")[0])/max(before,1e-8)
                shift=float(np.linalg.norm(pz-sz)/np.sqrt(dim))
                support_ok=bool(cdist(pz,tz).min()/np.sqrt(dim)<=support_limit+TOL and
                    ((frame.to_numpy()>=minimum-TOL)&(frame.to_numpy()<=maximum+TOL)).all())
                chemical_ok=bool(all(CORE.chemical_ok(frame,raw,s)[0] for s in scenarios))
                nd=CORE.paired(models["selection"],frame,source)[0];od=CORE.paired(old["models"][:30],frame,source)[0]
                nq,oq=float(np.quantile(nd,.95)),float(np.quantile(od,.95))
                cdn=float(predict_1d(current["pipeline"],frame)[0]-center_new_source)
                cdo=float(old_center["pipeline"].predict(frame)[0,stale_position]-center_old_source)
                assessed=assess(frame);aq={key:float(np.quantile(value,.95)) for key,value in assessed.items()}
                evaluation_mask=masks[DEFAULT["pool"]]
                base_pass=bool(qualified and np.std(new_delta[evaluation_mask].mean(axis=1))>1e-6
                    and np.std(old_delta[evaluation_mask].mean(axis=1))>1e-6
                    and support_ok and chemical_ok and shift>=.05 and nq<=TOL and oq<=TOL and cdn<=TOL and cdo<=TOL
                    and np.any(np.abs(nd)>1e-8) and np.any(np.abs(od)>1e-8))
                for bound,(absolute,retention) in BOUNDS.items():
                    floor=max(absolute,retention*anchor_gain) if retention is not None else absolute
                    passed=bool(base_pass and gain>=floor and aq["new_audit"]<=TOL and aq["old_fresh_audit"]<=TOL)
                    rechecks.append({"source_beer":beer,"arm":arm,"reference":label,"evaluation_pool":DEFAULT["pool"],"bound":bound,"required_gain":floor,
                        "target_improvement":gain,"retained_anchor_gain":gain/anchor_gain,"change_latent_rms":shift,
                        "support_ok":support_ok,"chemical_ok":chemical_ok,"new_selection_q95":nq,"old_selection_q95":oq,
                        "center_new_delta":cdn,"center_old_delta":cdo,"new_audit_q95":aq["new_audit"],
                        "old_fresh_audit_q95":aq["old_fresh_audit"],"fresh_consistency_passed":passed})
            print(f"Balanced transfer source {number}/11 beer={beer}, arm={arm}, joint={joint.sum()}",flush=True)
    result=pd.DataFrame(records)
    result.to_csv(OUT/"transfer_summary.csv",index=False, lineterminator="\n")
    pd.concat(geometry_rows,ignore_index=True).to_csv(OUT/"candidate_geometry.csv.gz",index=False,compression="gzip", lineterminator="\n")
    pd.concat(diags,ignore_index=True).to_csv(OUT/"candidate_model_checks.csv.gz",index=False,compression="gzip", lineterminator="\n")
    pd.DataFrame(selected_rows).to_csv(OUT/"selected_spectra.csv",index=False, lineterminator="\n")
    pd.DataFrame(delta_rows).to_csv(OUT/"paired_deltas.csv",index=False, lineterminator="\n")
    pd.DataFrame(threshold_rows).to_csv(OUT/"threshold_checks.csv",index=False, lineterminator="\n")
    pd.DataFrame(rechecks).to_csv(OUT/"fixed_spectra_rechecks.csv",index=False, lineterminator="\n")
    pd.DataFrame(uniqueness).to_csv(OUT/"pool_uniqueness.csv",index=False, lineterminator="\n")
    frontier=[{"arm":arm,"pool":pool,"bound":bound,"ranking":ranking,**counts(part)}
        for (arm,pool,bound,ranking),part in result.groupby(["arm","pool","bound","ranking"])]
    pd.DataFrame(frontier).to_csv(OUT/"frontier.csv",index=False, lineterminator="\n")
    arm_defaults={arm:counts(result[result.arm.eq(arm)&result.is_arm_default]) for arm in arms}
    summary={"default":DEFAULT,"default_results":counts(result[result.is_default_policy]),"arm_defaults":arm_defaults,
        "model_qualified":{arm:bool(models["center"]["qualified_for_proxy"]) for arm,models in arms.items()},
        "nominal_candidates_per_source":12004,"sources":len(sources),"training_n":len(train),"support_limit":support_limit,
        "contracted_targets_unique":len(np.unique(contracted.to_numpy(),axis=0)),"contracted_targets_nominal":len(contracted),
        "frontier_rows":len(frontier),"policies":len(result),"selected":len(selected_rows),"threshold_rows":len(threshold_rows),
        "elapsed_seconds":perf_counter()-started}
    write_json(OUT/"summary.json",summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    with threadpool_limits(limits=1):main()
