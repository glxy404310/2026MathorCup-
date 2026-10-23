"""Verify style-held-out prediction and predeclared committee search."""
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
from threadpoolctl import threadpool_limits

from common import SOLUTION

BASE = importlib.import_module("10_q3_verify")
PREVIOUS_VERIFY = importlib.import_module("15_q3_optimized_verify")
OUT = SOLUTION / "outputs/q3_domain"
TOL = 1e-10


def score(y, prediction):
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    actual_var, predicted_var = np.var(y), np.var(prediction)
    actual_range = np.quantile(y, .95)-np.quantile(y, .05)
    predicted_range = np.quantile(prediction, .95)-np.quantile(prediction, .05)
    return {"mae": mean_absolute_error(y, prediction), "rmse": np.sqrt(mean_squared_error(y, prediction)),
            "r2": r2_score(y, prediction),
            "spearman": spearmanr(y, prediction).statistic if actual_var > 1e-20 and predicted_var > 1e-20 else np.nan,
            "prediction_variance": predicted_var, "target_variance": actual_var,
            "prediction_variance_ratio": predicted_var/actual_var if actual_var else np.nan,
            "prediction_std_ratio": np.sqrt(predicted_var/actual_var) if actual_var else np.nan,
            "prediction_range90": predicted_range, "target_range90": actual_range,
            "prediction_range90_ratio": predicted_range/actual_range if actual_range else np.nan,
            "mean_bias": np.mean(prediction-y)}


def check_frozen():
    details = []
    for folder, manifest_path in [(SOLUTION / "outputs/q3_optimized", OUT / "previous_hashes.json"),
                                  (SOLUTION / "outputs/q3", SOLUTION / "outputs/q3_optimized/baseline_hashes.json")]:
        manifest = BASE.read_json(manifest_path)
        paths = {str(path.relative_to(folder)) for path in folder.rglob("*") if path.is_file()}
        assert paths == {row["path"] for row in manifest["files"]}
        for row in manifest["files"]:
            raw = (folder/row["path"]).read_bytes()
            assert len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"], row["path"]
        details.append({"directory": str(folder.relative_to(SOLUTION)), "unchanged_files": len(paths)})
    return details


def style_weights(styles):
    labels = np.asarray(styles)
    unique, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    return len(labels)/(len(unique)*counts[inverse])


def check_local_training(model, training, target, styles, calibrated):
    if "chemical" in model.named_steps:
        PREVIOUS_VERIFY.check_pipeline_training(model, training, calibrated)
    elif "imputer" in model.named_steps:
        BASE.close(model.named_steps["imputer"].statistics_, np.nanmedian(training, axis=0), "Source-excluded raw medians")
    estimator = model.named_steps["model"]
    if hasattr(estimator, "models_"):
        for member in estimator.models_:
            check_local_training(member, training, target, styles, calibrated)
    if hasattr(estimator, "x_"):
        BASE.close(estimator.x_, model[:-1].transform(training), "Local predictor training coordinates")
        BASE.close(estimator.y_, target, "Local predictor training labels")
        BASE.close(estimator.sample_weight_, style_weights(styles), "Equal-style local training weights")
    if hasattr(estimator, "target_mean_"):
        BASE.close(estimator.target_mean_, np.average(target, weights=style_weights(styles)), "Centered weighted target mean")


def check_domain(context):
    from q3_stale_domain import make_domain_pipeline, training_fit_params
    folder = OUT / "stale"
    meta, x, y45 = context["meta"], context["x"], context["y"]
    y = y45.stale_hops
    train = np.flatnonzero(~meta.tasting_category_fine.eq("Pils/Lager"))
    sources = np.flatnonzero(meta.tasting_category_fine.eq("Pils/Lager"))
    manifest = BASE.read_json(folder/"run_manifest.json")
    summary = BASE.read_json(folder/"summary.json")
    selection = BASE.read_json(folder/"selection_result.json")
    assert manifest["pils_labels_used_in_training_selection_or_intercept"] is False
    assert manifest["features"] == x.columns.tolist()
    specs = {spec["name"]: spec for spec in manifest["candidate_specs"]}
    predictions = pd.read_csv(folder/"loso_predictions.csv")
    audit = pd.read_csv(folder/"loso_split_audit.csv")
    by_style = pd.read_csv(folder/"loso_metrics_by_style.csv")
    overall = pd.read_csv(folder/"loso_metrics_overall.csv").set_index("method")
    assert len(predictions) == len(train)*len(specs)
    assert not predictions.duplicated(["method", "global_index"]).any()
    assert set(predictions.global_index) == set(train)
    assert set(predictions.method) == set(specs)
    groups = meta.tasting_category_fine.iloc[train]
    assert set(groups.unique()) == set(manifest["heldout_groups"])
    assert len(groups.unique()) == summary["loso_styles"]
    calculated_styles = []
    selected_spec = selection["config"]
    for fold, fold_rows in audit.groupby("fold"):
        heldout = fold_rows.heldout_style.unique()
        assert len(heldout) == 1
        style = heldout[0]
        expected_test = train[groups.to_numpy() == style]
        expected_train = train[groups.to_numpy() != style]
        assert set(fold_rows.loc[fold_rows.role.eq("test"), "global_index"]) == set(expected_test)
        assert set(fold_rows.loc[fold_rows.role.eq("train"), "global_index"]) == set(expected_train)
        assert len(fold_rows) == len(train) and not fold_rows.global_index.duplicated().any()
        assert fold_rows.beer.tolist() == fold_rows.global_index.map(meta.beer).tolist()
        model = make_domain_pipeline(selected_spec, x.columns.tolist(), context["calibrated_features"],
                                     manifest["aging_features"], manifest["loso_seed"]+int(fold))
        model.fit(x.iloc[expected_train], y.iloc[expected_train],
                  **training_fit_params(selected_spec, meta.tasting_category_fine.iloc[expected_train]))
        selected_rows = predictions[predictions.method.eq(selection["selected_method"])
                                    & predictions.fold.eq(fold)].set_index("global_index").loc[expected_test]
        BASE.close(selected_rows.prediction, np.asarray(model.predict(x.iloc[expected_test])).reshape(-1),
                   "Selected LOSO prediction refitted without the held-out style")
        for method in specs:
            rows = predictions[predictions.method.eq(method) & predictions.fold.eq(fold)].set_index("global_index").loc[expected_test]
            assert rows.heldout_style.eq(style).all()
            BASE.close(rows.observed, y.iloc[expected_test], "LOSO true labels")
            calculated_styles.append({"method": method, "heldout_style": style, "fold": fold,
                                      **score(rows.observed, rows.prediction),
                                      "bias": np.mean(rows.prediction.to_numpy()-rows.observed.to_numpy())})
            if method == "mean_global":
                BASE.close(rows.prediction, y.iloc[expected_train].mean(), "LOSO global training mean")
            if method == "mean_style_balanced":
                means = y.iloc[expected_train].groupby(meta.tasting_category_fine.iloc[expected_train]).mean()
                BASE.close(rows.prediction, means.mean(), "LOSO equal-style training mean")
    metric_names = list(score(y, y))
    BASE.same_table(by_style, pd.DataFrame(calculated_styles), ["method", "heldout_style", "fold"],
                    metric_names+["bias"], "LOSO per-style metrics")
    calculated = []
    for method, spec in specs.items():
        rows = predictions[predictions.method.eq(method)].set_index("global_index").loc[train]
        styles = by_style[by_style.method.eq(method)]
        metrics = score(rows.observed, rows.prediction)
        within_prediction = within_truth = 0.
        for style in groups.unique():
            mask = groups.to_numpy() == style
            pred, truth = rows.prediction.to_numpy()[mask], rows.observed.to_numpy()[mask]
            within_prediction += np.sum((pred-pred.mean())**2)
            within_truth += np.sum((truth-truth.mean())**2)
        macro_rmse, macro_bias = styles.rmse.mean(), styles.bias.abs().mean()
        calculated.append({"method": method, **metrics,
                           "macro_style_rmse": macro_rmse, "macro_style_mae": styles.mae.mean(),
                           "macro_style_abs_bias": macro_bias, "macro_style_bias": styles.bias.mean(),
                           "selection_score": macro_rmse+manifest["bias_penalty"]*macro_bias,
                           "within_style_prediction_variance_ratio": within_prediction/within_truth})
        assert bool(metrics["prediction_variance_ratio"] >= manifest["variance_ratio_min"]) == overall.at[method,"nonconstant_qualified"]
        assert spec["selectable"] == overall.at[method,"selectable"]
    BASE.same_table(overall.reset_index(), pd.DataFrame(calculated), ["method"],
                    [key for key in calculated[0] if key != "method"], "LOSO macro and pooled metrics")
    best_mean = overall.loc[["mean_global", "mean_style_balanced"], "macro_style_rmse"].min()
    eligible = overall[overall.selectable & overall.nonconstant_qualified]
    fallback = eligible.empty
    if fallback:
        eligible = overall[overall.selectable]
    winner = eligible.reset_index().sort_values(["selection_score", "method"]).iloc[0]
    assert winner.method == selection["selected_method"] == summary["selected_method"]
    assert fallback == selection["fallback_used"]
    qualified = bool(winner.nonconstant_qualified and winner.macro_style_rmse < best_mean)
    assert qualified == selection["qualified_for_proxy"] == summary["qualified_for_proxy"]
    artifact = joblib.load(folder/"transfer_stale_model.joblib")
    ensemble = joblib.load(folder/"bootstrap_ensemble.joblib")
    for saved in [artifact, ensemble]:
        np.testing.assert_array_equal(saved["train_global_indices"], train)
        np.testing.assert_array_equal(saved["excluded_global_indices"], sources)
        assert saved["selected_config"] == selected_spec and saved["qualified_for_proxy"] == qualified
        assert saved["pils_diagnostic_used_for_selection"] is False
    check_local_training(artifact["pipeline"], x.iloc[train], y.iloc[train], groups, context["calibrated_features"])
    indices = np.asarray(ensemble["bootstrap_indices"])
    assert indices.shape == (60,len(train)) and not np.isin(indices,sources).any()
    assert ensemble["selection_model_indices"] == list(range(40)) and ensemble["audit_model_indices"] == list(range(40,60))
    with np.load(folder/"bootstrap_indices.npz") as saved:
        np.testing.assert_array_equal(saved["indices"],indices)
        np.testing.assert_array_equal(saved["bootstrap_seeds"],ensemble["bootstrap_seeds"])
    for b,(model,index,seed) in enumerate(zip(ensemble["models"],indices,ensemble["bootstrap_seeds"])):
        expected_seed = manifest["selection_bootstrap_seed"]+b if b<40 else manifest["audit_bootstrap_seed"]+b-40
        assert seed == expected_seed
        np.testing.assert_array_equal(index,np.random.default_rng(seed).choice(train,len(train),replace=True))
        check_local_training(model,x.iloc[index],y.iloc[index],meta.tasting_category_fine.iloc[index],context["calibrated_features"])
    diagnostic = pd.read_csv(folder/"pils_diagnostic_predictions.csv")
    diagnostic_metrics = pd.read_csv(folder/"pils_diagnostic_metrics.csv").set_index("method")
    assert set(diagnostic.method) == set(specs) and len(diagnostic)==len(sources)*len(specs)
    for method in specs:
        rows=diagnostic[diagnostic.method.eq(method)].set_index("global_index").loc[sources]
        BASE.close(rows.observed,y.iloc[sources],"Pils diagnostic labels")
        BASE.close(diagnostic_metrics.loc[method,metric_names],list(score(rows.observed,rows.prediction).values()),"Pils diagnostic metrics")
    rows=diagnostic[diagnostic.method.eq(selection["selected_method"])].set_index("global_index").loc[sources]
    BASE.close(rows.prediction,np.asarray(artifact["pipeline"].predict(x.iloc[sources])).reshape(-1),"Frozen selected Pils diagnostic")
    if selected_spec["family"]=="local":
        local_rows=[]
        final=artifact["pipeline"];final_estimator=final.named_steps["model"]
        k=final_estimator.n_neighbors
        final_distance=cdist(final[:-1].transform(x.iloc[sources]),final_estimator.x_)
        final_neighbors=[set(train[np.argpartition(distance,k-1)[:k]]) for distance in final_distance]
        for b,(model,index) in enumerate(zip(ensemble["models"],indices)):
            estimator=model.named_steps["model"]
            distances=cdist(model[:-1].transform(x.iloc[sources]),estimator.x_)
            predicted=np.asarray(model.predict(x.iloc[sources])).reshape(-1)
            for j,(source_index,distance) in enumerate(zip(sources,distances)):
                near=np.argpartition(distance,k-1)[:k]
                neighbors=set(index[near]);weights=estimator.sample_weight_[near]/np.maximum(distance[near],1e-12)
                _,inverse=np.unique(index[near],return_inverse=True)
                by_beer=np.bincount(inverse,weights=weights)
                unique_effective=by_beer.sum()**2/np.sum(by_beer**2)
                assert unique_effective<=len(neighbors)+1e-10
                local_rows.append({"beer":int(meta.beer.iloc[source_index]),"model_index":b,
                    "prediction":predicted[j],"nearest_unique_beers":len(neighbors),
                    "neighbor_jaccard_vs_final":len(neighbors&final_neighbors[j])/len(neighbors|final_neighbors[j]),
                    "effective_resampled_position_count":weights.sum()**2/np.sum(weights**2),
                    "effective_unique_beer_count":unique_effective,
                    "imputer_indicator_count":len(model.named_steps["imputer"].indicator_.features_)})
        expected_neighborhood=pd.DataFrame(local_rows)
        BASE.same_table(pd.read_csv(folder/"neighborhood_bootstrap_diagnostics.csv"),expected_neighborhood,
                        ["beer","model_index"],[column for column in expected_neighborhood if column not in {"beer","model_index"}],
                        "Bootstrap neighborhood and unique-beer influence diagnostics")
        expected_summary=expected_neighborhood.groupby("beer").agg(
            bootstrap_prediction_sd=("prediction","std"),median_neighbor_jaccard=("neighbor_jaccard_vs_final","median"),
            mean_unique_neighbors=("nearest_unique_beers","mean"),
            median_effective_resampled_position_count=("effective_resampled_position_count","median"),
            median_effective_unique_beer_count=("effective_unique_beer_count","median")).reset_index()
        BASE.same_table(pd.read_csv(folder/"neighborhood_bootstrap_summary.csv"),expected_summary,["beer"],
                        [column for column in expected_summary if column!="beer"],"Neighborhood diagnostic summaries")
    context.update({"domain_model":artifact,"domain_ensemble":ensemble,"train":train,"sources":sources})
    return {"non_source_rows":len(train),"heldout_styles":len(groups.unique()),"candidate_count":len(specs),
            "prediction_rows":len(predictions),"selected_method":selection["selected_method"],
            "selected_loso_predictions_refitted":len(groups.unique()),"qualified":qualified,
            "selected_macro_rmse":float(winner.macro_style_rmse),"best_mean_macro_rmse":float(best_mean),
            "selected_pils_diagnostic_rmse":float(diagnostic_metrics.at[selection["selected_method"],"rmse"]),
            "bootstrap_indices_shape":list(indices.shape),"pils_is_fresh_test":False}


def counts(frame):
    return {"sources":len(frame),"selected":int(frame.candidate_index.notna().sum()),
            "new_model_checks_passed":int(frame.all_model_checks_passed.sum()),
            "historical_cross_proxy_passed":int(frame.cross_proxy_checks_passed.sum()),
            "fresh_cross_proxy_passed":int(frame.fresh_cross_proxy_checks_passed.sum()),
            "all_committees_passed":int(frame.all_committees_passed.sum()),
            "no_candidate":int(frame.status.eq("no_candidate_found").sum())}


def check_transfer(context,experiment="transfer"):
    folder=OUT/experiment
    meta,x,y=context["meta"],context["x"],context["y"]
    train,sources=context["train"],context["sources"]
    manifest=BASE.read_json(folder/"run_manifest.json")
    summary=BASE.read_json(folder/"summary.json")
    proxy_folder=SOLUTION/manifest.get("proxy_directory","outputs/q3_domain/stale")
    current_model=joblib.load(proxy_folder/"transfer_stale_model.joblib")
    frozen_ensemble=joblib.load(proxy_folder/"bootstrap_ensemble.joblib")
    current_ensemble=joblib.load(folder/"screening_and_audit.joblib") if (folder/"screening_and_audit.joblib").exists() else frozen_ensemble
    for artifact in [current_model,current_ensemble]:
        np.testing.assert_array_equal(artifact["train_global_indices"],train)
        np.testing.assert_array_equal(artifact["excluded_global_indices"],sources)
    if manifest.get("fresh_domain_audit_seed") is not None:
        for b in range(40):
            np.testing.assert_array_equal(current_ensemble["bootstrap_indices"][b],frozen_ensemble["bootstrap_indices"][b])
            BASE.close(current_ensemble["models"][b].predict(x.iloc[train]),frozen_ensemble["models"][b].predict(x.iloc[train]),"Frozen follow-up screening models")
        for b in range(20):
            seed=manifest["fresh_domain_audit_seed"]+b
            index=current_ensemble["bootstrap_indices"][40+b]
            assert current_ensemble["bootstrap_seeds"][40+b]==seed
            np.testing.assert_array_equal(index,np.random.default_rng(seed).choice(train,len(train),replace=True))
            tree=current_ensemble["models"][40+b].named_steps["model"]
            assert tree.n_outputs_==1 and tree.n_estimators==120 and tree.min_samples_leaf==3 and tree.max_features==.7
            BASE.close(current_ensemble["models"][40+b].named_steps["imputer"].statistics_,np.nanmedian(x.iloc[index],axis=0),"Fresh current-protocol audit medians")
    assert manifest["default_pool"]=="matched_2004" and manifest["default_policy"]=="joint_margin"
    assert manifest["audit_reselection"] is False and manifest["default_selected_from_audit"] is False
    scenarios=manifest["threshold_scenarios"]
    assert scenarios==BASE.read_json(SOLUTION/"outputs/q3_optimized/transfer/threshold_scenarios.json")
    geometry=joblib.load(SOLUTION/"outputs/q3/transfer/transfer_copula.joblib")
    reference=geometry["copula"]
    np.testing.assert_array_equal(geometry["training_rows"],train)
    draws=pd.read_csv(folder/"base_draws.csv")
    previous_draws=pd.read_csv(SOLUTION/"outputs/q3_optimized/transfer/optimized_base_draws.csv")
    BASE.close(draws.iloc[:500],previous_draws,"Preserved matched base draws")
    BASE.close(draws.iloc[500:],reference.sample("Tripel",1000,manifest["additional_draw_seed"]),"Additional frozen-seed draws")
    assert len(draws)==1500 and draws.columns.tolist()==x.columns.tolist()
    original=joblib.load(SOLUTION/"outputs/q3/sensory/stale_bootstrap.joblib")
    old_center=joblib.load(SOLUTION/"outputs/q3/sensory/transfer_proxy.joblib")
    fresh=joblib.load(folder/"fresh_original_protocol_audit.joblib")
    for saved in [original,fresh]:
        np.testing.assert_array_equal(saved["train_global_indices"],train)
        assert saved["features"]==x.columns.tolist()
    assert old_center["pipeline"].named_steps["model"].n_outputs_==45
    assert len(fresh["models"])==20
    for saved in [original,fresh]:
        for model,index in zip(saved["models"],saved["bootstrap_indices"]):
            tree=model.named_steps["model"]
            assert tree.n_outputs_==1 and tree.n_estimators==120 and tree.min_samples_leaf==3 and tree.max_features==.7
            assert not np.isin(index,sources).any()
            BASE.close(model.named_steps["imputer"].statistics_,np.nanmedian(x.iloc[index],axis=0),"Original-protocol audit training medians")
    for b,(index,seed) in enumerate(zip(fresh["bootstrap_indices"],fresh["bootstrap_seeds"])):
        assert seed==manifest["fresh_audit_seed"]+b
        np.testing.assert_array_equal(index,np.random.default_rng(seed).choice(train,len(train),replace=True))
    target=x.loc[meta.tasting_category_fine.eq("Tripel")]
    tz=reference.transform(target); dimension=x.shape[1]
    pairwise=cdist(tz,tz)/np.sqrt(dimension);np.fill_diagonal(pairwise,np.inf)
    support_limit=np.quantile(pairwise.min(axis=1),.95)
    BASE.close(support_limit,summary["support_nn_limit"],"Frozen full-dimensional support")
    minimum,maximum=x.iloc[train].min().to_numpy(),x.iloc[train].max().to_numpy()
    medians=x.iloc[train].median();target_median=target.median().to_numpy()
    report=pd.read_csv(folder/"transfer_summary.csv")
    diagnostics=pd.read_csv(folder/"candidate_diagnostics.csv")
    spectra=pd.read_csv(folder/"transferred_spectra.csv").set_index("transfer_id")
    deltas_csv=pd.read_csv(folder/"stale_paired_deltas.csv")
    thresholds=pd.read_csv(folder/"threshold_sensitivity.csv")
    previous_checks=pd.read_csv(folder/"previous_same_spectra_recheck.csv").set_index("source_beer")
    previous_summary=pd.read_csv(SOLUTION/"outputs/q3_optimized/transfer/transfer_summary.csv")
    previous_summary=previous_summary[previous_summary.is_default_policy].set_index("source_beer")
    previous_spectra=pd.read_csv(SOLUTION/"outputs/q3_optimized/transfer/transferred_spectra.csv").set_index("transfer_id")
    models=current_ensemble["models"]
    selected_predictions=np.vstack([np.asarray(model.predict(spectra[x.columns])).reshape(-1) for model in models])
    original_predictions=np.vstack([np.asarray(model.predict(spectra[x.columns])).reshape(-1) for model in original["models"]])
    fresh_predictions=np.vstack([np.asarray(model.predict(spectra[x.columns])).reshape(-1) for model in fresh["models"]])
    assert len(report)==len(sources)*len(manifest["pools"])*len(manifest["policies"])
    assert set(report[report.candidate_index.notna()].transfer_id)==set(spectra.index)
    selections=scenario_count=0; maximum_quantile_error=0.
    qualified=bool(current_model["qualified_for_proxy"])
    for source_index in sources:
        beer=int(meta.beer.iloc[source_index]);raw=x.iloc[[source_index]];source=raw.fillna(medians)
        caps={}
        for field in BASE.DEFECT_FIELDS:
            levels=[scenario["thresholds"][field] for scenario in scenarios if scenario["thresholds"][field] is not None]
            caps[field]=max(float(raw[field].iloc[0]),min(levels)) if levels else float(raw[field].iloc[0])
        arrays=[];gammas=[];draw_ids=[];matched=[]
        for gamma in manifest["gammas"]:
            arrays.append((1-gamma)*source.to_numpy()[0]+gamma*draws.to_numpy())
            gammas.extend([gamma]*len(draws));draw_ids.extend(range(len(draws)));matched.extend(np.arange(len(draws))<500)
        for gamma in manifest["gammas"]:
            arrays.append(((1-gamma)*source.to_numpy()[0]+gamma*target_median)[None])
            gammas.append(gamma);draw_ids.append(-1);matched.append(True)
        values=pd.DataFrame(np.vstack(arrays),columns=x.columns)
        for field,cap in caps.items(): values[field]=values[field].clip(lower=0,upper=cap)
        z,sz=reference.transform(values),reference.transform(source)
        distance=reference.distance(values,"Tripel");before=float(reference.distance(source,"Tripel")[0])
        nn=cdist(z,tz).min(axis=1)/np.sqrt(dimension)
        change=np.linalg.norm(z-sz,axis=1)/np.sqrt(dimension);improvement=1-distance/max(before,1e-8)
        objective=distance/max(before,1e-8)+manifest["lambda"]*change/max(float(np.linalg.norm(sz[0]-tz.mean(axis=0))/np.sqrt(dimension)),1e-8)
        support=(nn<=support_limit+TOL)&(values.to_numpy()>=minimum-TOL).all(axis=1)&(values.to_numpy()<=maximum+TOL).all(axis=1)
        chemical=BASE.chemical_flags(values,caps)
        source_new=np.array([float(np.asarray(model.predict(source)).reshape(-1)[0]) for model in models])
        source_old=np.array([float(np.asarray(model.predict(source)).reshape(-1)[0]) for model in original["models"]])
        source_fresh=np.array([float(np.asarray(model.predict(source)).reshape(-1)[0]) for model in fresh["models"]])
        unique,inverse=np.unique(values.to_numpy(),axis=0,return_inverse=True);unique=pd.DataFrame(unique,columns=x.columns)
        new_delta=np.column_stack([np.asarray(model.predict(unique)).reshape(-1)-source_new[b] for b,model in enumerate(models[:40])])[inverse]
        old_delta=np.column_stack([np.asarray(model.predict(unique)).reshape(-1)-source_old[b] for b,model in enumerate(original["models"][:30])])[inverse]
        nq,oq=np.quantile(new_delta,.95,axis=1),np.quantile(old_delta,.95,axis=1);worst=np.maximum(nq,oq)
        new_signal=np.any(np.abs(new_delta)>1e-8,axis=1);old_signal=np.any(np.abs(old_delta)>1e-8,axis=1)
        base_ok=support&chemical&(improvement>=.1)&(change>=.05)
        single=base_ok&(nq<=TOL)&new_signal;joint=single&(oq<=TOL)&old_signal
        diag=diagnostics[diagnostics.source_beer.eq(beer)].sort_values("candidate_index")
        assert len(diag)==len(values)==manifest["pools"]["expanded_6004"]
        np.testing.assert_array_equal(diag.candidate_index,np.arange(len(values)))
        np.testing.assert_array_equal(diag.draw,draw_ids);BASE.close(diag.gamma,gammas,"Interpolation gamma")
        np.testing.assert_array_equal(diag.matched_budget,matched)
        BASE.close(diag[["target_distance","target_improvement","target_nn_rms","change_latent_rms","selection_new_q95","selection_old_q95","worst_selection_q95","geometry_objective"]],
                   np.column_stack([distance,improvement,nn,change,nq,oq,worst,objective]),"Reconstructed screening-only candidate statistics")
        for column,expected in [("support_ok",support),("all_thresholds_chemical_ok",chemical),("new_signal",new_signal),
                                ("old_signal",old_signal),("single_feasible",single),("joint_feasible",joint)]:
            np.testing.assert_array_equal(diag[column],expected)
        for _,row in report[report.source_beer.eq(beer)].iterrows():
            mask=np.asarray(matched) if row.pool=="matched_2004" else np.ones(len(values),bool)
            eligible=(single if row.policy=="single_geometry" else joint)&mask;ids=np.flatnonzero(eligible)
            assert row.n_candidates==mask.sum() and row.n_selection_feasible==len(ids)
            nv,ov=np.std(new_delta[mask].mean(axis=1)),np.std(old_delta[mask].mean(axis=1))
            BASE.close([row.new_candidate_delta_sd,row.old_candidate_delta_sd],[nv,ov],"Per-pool proxy response")
            if not len(ids): assert pd.isna(row.candidate_index) and row.status=="no_candidate_found";continue
            chosen=int(ids[np.lexsort((objective[ids],worst[ids]))[0]]) if row.policy=="joint_margin" else int(ids[np.argmin(objective[ids])])
            assert chosen==int(row.candidate_index),"Audit entered candidate selection"
            selections+=1;final=spectra.loc[[row.transfer_id],x.columns];BASE.close(final,values.iloc[[chosen]],"Rebuilt selected spectrum")
            position=spectra.index.get_loc(row.transfer_id)
            nd=selected_predictions[:,position]-source_new;od=original_predictions[:,position]-source_old;fd=fresh_predictions[:,position]-source_fresh
            groups_delta={"new_selection":nd[:40],"old_selection":od[:30],"new_audit":nd[40:],"old_historical":od[30:],"fresh_original_audit":fd}
            actual=[row.selection_new_q95,row.selection_old_q95,row.new_audit_q95,row.old_historical_audit_q95,row.fresh_original_audit_q95]
            expected=[np.quantile(difference,.95) for difference in groups_delta.values()]
            BASE.close(actual,expected,"Selected paired screening and audit quantiles")
            maximum_quantile_error=max(maximum_quantile_error,float(np.max(np.abs(np.asarray(actual)-expected))))
            new_pass=bool(qualified and nv>1e-6 and new_signal[chosen] and expected[2]<=TOL)
            historical=bool(new_pass and ov>1e-6 and old_signal[chosen] and expected[1]<=TOL and expected[3]<=TOL)
            fresh_pass=bool(new_pass and ov>1e-6 and old_signal[chosen] and expected[1]<=TOL and expected[4]<=TOL)
            assert [new_pass,historical,fresh_pass,historical and fresh_pass]==[row.all_model_checks_passed,row.cross_proxy_checks_passed,row.fresh_cross_proxy_checks_passed,row.all_committees_passed]
            for label,difference in groups_delta.items():
                stored=deltas_csv[deltas_csv.transfer_id.eq(row.transfer_id)&deltas_csv.ensemble.eq(label)].sort_values("model_index")
                np.testing.assert_array_equal(stored.model_index,np.arange(len(difference)));BASE.close(stored.paired_delta,difference,"Per-model paired difference")
            checks=thresholds[thresholds.transfer_id.eq(row.transfer_id)]
            assert len(checks)==len(scenarios)*len(BASE.DEFECT_FIELDS)
            for check in checks.itertuples(index=False):
                scenario=next(s for s in scenarios if s["scenario"]==check.scenario);threshold=scenario["thresholds"][check.feature]
                observed=float(raw[check.feature].iloc[0]);cap=max(observed,threshold) if threshold is not None else observed
                value=float(final[check.feature].iloc[0]);BASE.close([check.source,check.transferred,check.upper_bound],[observed,value,cap],"Scenario concentration")
                assert bool(0<=value<=cap+TOL)==check.passed;scenario_count+=1
        previous=previous_spectra.loc[[previous_summary.at[beer,"transfer_id"]],x.columns]
        nd=np.array([float(np.asarray(model.predict(previous)).reshape(-1)[0])-source_new[b] for b,model in enumerate(models)])
        od=np.array([float(np.asarray(model.predict(previous)).reshape(-1)[0])-source_old[b] for b,model in enumerate(original["models"])])
        fd=np.array([float(np.asarray(model.predict(previous)).reshape(-1)[0])-source_fresh[b] for b,model in enumerate(fresh["models"])])
        pq=[np.quantile(nd[:40],.95),np.quantile(od[:30],.95),np.quantile(nd[40:],.95),np.quantile(od[30:],.95),np.quantile(fd,.95)]
        recorded=previous_checks.loc[beer]
        BASE.close(recorded[["selection_new_q95","selection_old_q95","new_audit_q95","old_historical_audit_q95","fresh_original_audit_q95"]],pq,"Previous frozen-spectrum current-rule quantiles")
        pz=reference.transform(previous);pi=1-float(reference.distance(previous,"Tripel")[0])/max(before,1e-8)
        pg=bool(pi>=.1 and np.linalg.norm(pz-sz)/np.sqrt(dimension)>=.05 and cdist(pz,tz).min()/np.sqrt(dimension)<=support_limit+TOL
                and (previous.to_numpy()>=minimum-TOL).all() and (previous.to_numpy()<=maximum+TOL).all())
        pc=bool(BASE.chemical_flags(previous,caps)[0]);assert pc==recorded.all_thresholds_chemical_ok and pg==recorded.geometry_ok
        matched_mask=np.asarray(matched)
        pn=bool(qualified and pg and pc and pq[0]<=TOL and pq[2]<=TOL and np.any(np.abs(nd[:40])>1e-8) and np.std(new_delta[matched_mask].mean(axis=1))>1e-6)
        po=bool(pq[1]<=TOL and np.any(np.abs(od[:30])>1e-8) and np.std(old_delta[matched_mask].mean(axis=1))>1e-6)
        assert [pn,pn and po and pq[3]<=TOL,pn and po and pq[4]<=TOL,pn and po and pq[3]<=TOL and pq[4]<=TOL]==list(recorded[["all_model_checks_passed","cross_proxy_checks_passed","fresh_cross_proxy_checks_passed","all_committees_passed"]])
        print(f"Verified domain candidates: source {beer}",flush=True)
    policies=[]
    for (pool,policy),group in report.groupby(["pool","policy"]):policies.append({"pool":pool,"policy":policy,**counts(group)})
    BASE.same_table(pd.read_csv(folder/"policy_summary.csv"),pd.DataFrame(policies),["pool","policy"],list(counts(report)),"Domain policy summaries")
    default=report[report.is_default_policy];assert default.pool.eq(manifest["default_pool"]).all() and default.policy.eq(manifest["default_policy"]).all()
    assert counts(default)==summary["default_policy"]
    for name,column in [("new_model_checks_passed","all_model_checks_passed"),("historical_cross_proxy_passed","cross_proxy_checks_passed"),
                        ("fresh_cross_proxy_passed","fresh_cross_proxy_checks_passed"),("all_committees_passed","all_committees_passed")]:
        assert int(previous_checks[column].sum())==summary["previous_same_spectra_current_rules"][name]
    context.setdefault("transfer_summaries",{})[experiment]=summary
    return {"experiment":experiment,"reconstructed_candidates":len(diagnostics),"selection_only_argmin_checks":selections,"scenario_concentration_checks":scenario_count,
            "maximum_paired_quantile_error":maximum_quantile_error,"default_policy":counts(default),
            "original_committee_outputs":1,"original_center_outputs":45,"fresh_original_protocol_models":20}


def check_reports(context):
    tables=OUT/"report_tables"
    provenance=BASE.read_json(OUT/"report_source_manifest.json")
    for source in provenance:
        path=SOLUTION/source["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest()==source["sha256"],source["path"]
        if source.get("rows") is not None:assert len(pd.read_csv(path))==source["rows"]
    overall=pd.read_csv(OUT/"stale/loso_metrics_overall.csv")
    pils=pd.read_csv(OUT/"stale/pils_diagnostic_metrics.csv")
    selected=context["domain_model"]["selected_method"]
    expected=overall.merge(pils[["method","rmse","mae","r2"]].rename(
        columns={"rmse":"pils_rmse","mae":"pils_mae","r2":"pils_r2"}),on="method",validate="one_to_one")
    comparison=pd.read_csv(tables/"prediction_method_comparison.csv")
    BASE.same_table(comparison,expected,["method"],["macro_style_rmse","pils_rmse","pils_mae","pils_r2"],"Report prediction comparison")
    assert comparison.is_selected.equals(comparison.method.eq(selected))
    by_style=pd.read_csv(OUT/"stale/loso_metrics_by_style.csv")
    BASE.same_table(pd.read_csv(tables/"selected_loso_by_style.csv"),by_style[by_style.method.eq(selected)],
                    ["heldout_style"],["rmse","mae","bias"],"Report selected LOSO styles")
    predictions=pd.read_csv(OUT/"stale/pils_diagnostic_predictions.csv")
    influence=pd.read_csv(tables/"pils_source_error_influence.csv").set_index("beer")
    assert set(influence.index)==set(context["meta"].beer.iloc[context["sources"]])
    method_map={"domain_prediction":selected,"matched_single_tree_prediction":"fixed_single_tree",
                "previous_single_tree_prediction":"fixed_single_tree","old_multioutput_prediction":"old_multioutput_tree",
                "training_mean_prediction":"mean_global"}
    for column,method in method_map.items():
        if column not in influence:continue
        rows=predictions[predictions.method.eq(method)].set_index("beer").loc[influence.index]
        BASE.close(influence[column],rows.prediction,"Report source prediction influence")
        BASE.close(influence.observed,rows.observed,"Report source observed values")
    for column in influence:
        if column.endswith("_sse_share"):
            prefix=column.removesuffix("_sse_share")
            prediction_column="domain_prediction" if prefix=="domain_selected" else prefix+"_prediction"
            if prediction_column in influence:
                squared=(influence[prediction_column]-influence.observed)**2
                BASE.close(influence[column],squared/squared.sum(),"Report per-source squared-error shares")
    check_names={"new":"all_model_checks_passed","historical":"cross_proxy_checks_passed",
                 "fresh":"fresh_cross_proxy_checks_passed","all":"all_committees_passed"}
    for experiment,folder_name in [("domain","transfer"),("consensus","consensus")]:
        folder=OUT/folder_name
        transfer=pd.read_csv(folder/"transfer_summary.csv")
        manifest=BASE.read_json(folder/"run_manifest.json")
        previous=pd.read_csv(folder/"previous_same_spectra_recheck.csv")
        thresholds=pd.read_csv(folder/"threshold_sensitivity.csv")
        expected_policies=[]
        for (pool,policy),group in transfer.groupby(["pool","policy"]):
            valid=group.candidate_index.notna()
            expected_policies.append({"pool":pool,"policy":policy,"sources":len(group),"selected":int(valid.sum()),
                **{name+"_passed":int(group[column].sum()) for name,column in check_names.items()},
                "median_target_improvement_selected":group.loc[valid,"target_improvement"].median(),
                "median_change_rms_selected":group.loc[valid,"change_latent_rms"].median(),
                "median_target_improvement_fresh_pass":group.loc[group.fresh_cross_proxy_checks_passed,"target_improvement"].median()})
        expected_policies=pd.DataFrame(expected_policies)
        BASE.same_table(pd.read_csv(tables/f"{experiment}_policy_comparison.csv"),expected_policies,["pool","policy"],
                        [column for column in expected_policies if column not in {"pool","policy"}],"Report policy counts and tradeoffs")
        default=transfer[transfer.is_default_policy]
        passed=thresholds.groupby(["transfer_id","scenario"]).passed.all()
        expected_scenarios=[]
        for scenario in manifest["threshold_scenarios"]:
            chemical=np.array([bool(passed.get((transfer_id,scenario["scenario"]),False)) for transfer_id in default.transfer_id])
            expected_scenarios.append({"scenario":scenario["scenario"],"sources":len(default),"chemical_only":int(chemical.sum()),
                                      **{name+"_passed":int((chemical&default[column].to_numpy(bool)).sum()) for name,column in check_names.items()}})
        expected_scenarios=pd.DataFrame(expected_scenarios)
        BASE.same_table(pd.read_csv(tables/f"{experiment}_default_threshold_pass_counts.csv"),expected_scenarios,["scenario"],
                        [column for column in expected_scenarios if column!="scenario"],"Report default threshold counts")
        source_comparison=pd.read_csv(tables/f"{experiment}_default_source_comparison.csv").set_index("source_beer")
        BASE.close(source_comparison.loc[default.source_beer,"target_improvement"],default.target_improvement,"Report default source geometry")
        for column in check_names.values():
            np.testing.assert_array_equal(source_comparison.loc[default.source_beer,column],default[column])
            np.testing.assert_array_equal(source_comparison.loc[previous.source_beer,"previous_"+column],previous[column])
    target_comparison=pd.read_csv(tables/"all_default_target_comparison.csv").set_index("source_beer")
    for prefix,folder_name,pool in [("domain","transfer","matched_2004"),("consensus","consensus","matched_2004"),
                                    ("expanded","consensus","expanded_6004")]:
        rows=pd.read_csv(OUT/folder_name/"transfer_summary.csv")
        rows=rows[rows.pool.eq(pool)&rows.policy.eq("joint_margin")]
        BASE.close(target_comparison.loc[rows.source_beer,prefix+"_target_improvement"],rows.target_improvement,"Report target comparison")
        BASE.close(target_comparison.loc[rows.source_beer,prefix+"_change_latent_rms"],rows.change_latent_rms,"Report chemical-change comparison")
    report_path=SOLUTION/"docs/q3_domain_results.md"
    text=report_path.read_text(encoding="utf-8")
    assert len(text)>100 and selected in text
    assert f"{pils.set_index('method').at[selected,'rmse']:.4f}" in text
    assert "不是" in text or "不能" in text
    figures=["prediction_domain_validation","transfer_policy_and_thresholds","target_distance_and_consistency"]
    for name in figures:
        for extension in ["png","svg"]:
            path=OUT/"figures"/f"{name}.{extension}"
            assert path.is_file() and path.stat().st_size>100
    report_summary=BASE.read_json(OUT/"report_summary.json")
    assert report_summary["selected_method"]==selected
    BASE.close(report_summary["selected_pils_diagnostic_rmse"],pils.set_index("method").at[selected,"rmse"],"Report diagnostic summary")
    comparison_by_method=comparison.set_index("method")
    baseline_min=comparison_by_method.loc[["fixed_single_tree","old_multioutput_tree","mean_global"],"pils_rmse"].min()
    assert report_summary["pils_prediction_improved"]==bool(comparison_by_method.at[selected,"pils_rmse"]<baseline_min)
    for experiment,folder_name in [("domain","transfer"),("consensus","consensus")]:
        transfer=pd.read_csv(OUT/folder_name/"transfer_summary.csv")
        default=transfer[transfer.is_default_policy]
        previous=pd.read_csv(OUT/folder_name/"previous_same_spectra_recheck.csv")
        expected_default={"sources":len(default),"selected":int(default.candidate_index.notna().sum()),
                          **{name:int(default[column].sum()) for name,column in check_names.items()}}
        expected_previous={"sources":len(previous),**{name:int(previous[column].sum()) for name,column in check_names.items()}}
        assert report_summary["experiments"][experiment]["default"]==expected_default
        assert report_summary["experiments"][experiment]["previous_same_spectra"]==expected_previous
    consensus=pd.read_csv(OUT/"consensus/transfer_summary.csv")
    expanded=consensus[consensus.pool.eq("expanded_6004")&consensus.policy.eq("joint_margin")]
    expanded_summary=report_summary["consensus_expanded_same_policy"]
    assert expanded_summary["budget"]==6004 and expanded_summary["sources"]==len(expanded)
    assert expanded_summary["selected"]==expanded.candidate_index.notna().sum()
    assert expanded_summary["fresh"]==expanded.fresh_cross_proxy_checks_passed.sum()
    BASE.close([expanded_summary["median_target_improvement_selected"],expanded_summary["median_change_rms_selected"]],
               [expanded.target_improvement.median(),expanded.change_latent_rms.median()],"Report expanded-budget tradeoffs")
    return {"hashed_report_sources":len(provenance),"prediction_comparisons":len(comparison),
            "experiments_with_policy_and_scenario_tables":2,"source_diagnostics":len(influence),
            "figure_pairs":len(figures),"results_report_exists":True}


def main():
    start=perf_counter();result={"status":"running","checks":[]};active="frozen_previous_results_before"
    try:
        result["checks"].append({"name":active,"details":check_frozen()})
        context,contract=BASE.load_contract();provenance=BASE.check_hashes()
        result["checks"].append({"name":"original_inputs","details":{"contract":contract,"provenance":provenance}})
        for active,operation in [("non_source_loso_selection",check_domain),("predeclared_transfer_search",check_transfer),
                                 ("fixed_protocol_consensus_followup",lambda ctx:check_transfer(ctx,"consensus")),("final_artifacts",check_reports)]:
            details=operation(context);result["checks"].append({"name":active,"details":details});print(f"Verified: {active}",flush=True)
        active="frozen_previous_results_after";result["checks"].append({"name":active,"details":check_frozen()})
        result["status"]="passed";result["limitations"]=["Pils results are repeatedly viewed diagnostics, not fresh external tests.",
            "LOSO-selected scores are optimistic after candidate selection; bootstrap fixes the chosen configuration.",
            "Committee consistency and audit quantiles do not establish true taste or causal defect preservation."]
    except Exception as error:
        result.update({"status":"failed","failed_check":active,"error":f"{type(error).__name__}: {error}"});raise
    finally:
        result["elapsed_seconds"]=perf_counter()-start
        (OUT/"verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),flush=True)


if __name__=="__main__":
    with threadpool_limits(limits=1):main()
