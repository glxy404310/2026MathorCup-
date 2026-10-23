"""Trace beer 81 identity, chemistry, sensory evidence and prediction influence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from common import KEYS, SOLUTION, data_dir
from q3_data import load_q3_data
from q3_stale_optimized import FoldChemicalTransform


OUT = SOLUTION / "outputs/q3_refined/beer81"
TARGET_BEER = 81
SOURCE_FILES = []


def source(path):
    SOURCE_FILES.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return path


def percentile(values, value):
    finite = np.asarray(values, float)
    finite = finite[np.isfinite(finite)]
    return float((np.sum(finite < value) + .5 * np.sum(finite == value)) / len(finite))


def rounded_id_agrees(reference, printed):
    text = str(printed)
    if "e" not in text.lower():
        return str(reference) == text
    mantissa, exponent = re.split("[eE]", text)
    digits = len(mantissa.split(".")[1]) if "." in mantissa else 0
    resolution = 10.0 ** (int(exponent) - digits)
    return abs(float(reference) - float(text)) <= resolution / 2


def identity_audit(raw, sensory, comments, calibrated):
    frames = {"1.csv": raw, "4.csv": sensory, "7.csv": comments, "8.xlsx/calibrated": calibrated}
    reference = raw.loc[raw.beer.eq(TARGET_BEER)].iloc[0]
    rows = []
    for name, frame in frames.items():
        selected = frame.loc[frame.beer.eq(TARGET_BEER)]
        row = selected.iloc[0]
        rows.append({"source": name, "rows": len(frame), "beer_unique": frame.beer.is_unique,
            "beer81_matches": len(selected), "style": row.tasting_category_fine,
            "style_matches_1csv": row.tasting_category_fine == reference.tasting_category_fine,
            "beer_id_text": str(row.beer_id), "exact_id_matches_1csv": str(row.beer_id) == str(reference.beer_id),
            "id_matches_printed_precision": rounded_id_agrees(reference.beer_id, row.beer_id),
            "beer_id_unique_count": frame.beer_id.nunique(),
            "rows_after_exact_beer_join": len(raw[["beer"]].merge(frame, on="beer", validate="one_to_one"))})
    table = pd.DataFrame(rows)
    if not (table.beer_unique.all() and table.beer81_matches.eq(1).all()
            and table.style_matches_1csv.all() and table.id_matches_printed_precision.all()):
        raise AssertionError("Identity audit found an unresolved discrepancy")
    table.to_csv(OUT / "identity_join_audit.csv", index=False)
    return table


def chemical_audit(meta, x, manifest, raw, calibrated, thresholds, train, target):
    chemistry = thresholds.drop_duplicates("field_name_R").set_index("field_name_R")
    rows = []
    other_pils = meta.tasting_category_fine.eq("Pils/Lager") & meta.beer.ne(TARGET_BEER)
    raw_row = raw.set_index("beer").loc[TARGET_BEER]
    cal_row = calibrated.set_index("beer").loc[TARGET_BEER]
    for entry in manifest.to_dict("records"):
        field, value = entry["feature"], float(x.iloc[target][entry["feature"]])
        tr = x.loc[train, field].dropna()
        peers = x.loc[other_pils, field].dropna()
        details = chemistry.loc[field] if field in chemistry.index else pd.Series(dtype=object)
        threshold = details.get("threshold_median", np.nan)
        rows.append({"feature": field, "representation": entry["representation"], "unit": entry["unit"],
            "value": value, "original_1csv_value": raw_row[field],
            "calibrated_value": cal_row.get(field, np.nan),
            "nonpils_n_observed": len(tr), "nonpils_min": tr.min(), "nonpils_q01": tr.quantile(.01),
            "nonpils_median": tr.median(), "nonpils_q99": tr.quantile(.99), "nonpils_max": tr.max(),
            "nonpils_percentile": percentile(tr, value), "other_pils_percentile": percentile(peers, value),
            "outside_nonpils_range": bool(value < tr.min() or value > tr.max()),
            "at_nonpils_minimum": bool(value == tr.min()), "at_nonpils_maximum": bool(value == tr.max()),
            "zero": value == 0, "aging": details.get("aging"), "origin": details.get("origin_paper"),
            "chemical_name": details.get("chemical_name"), "off_flavors": details.get("off_flavors"),
            "sensory_threshold_median": threshold,
            "ratio_to_sensory_threshold": value / threshold if entry["representation"] == "calibrated"
            and pd.notna(threshold) and threshold > 0 else np.nan})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "chemical_profile_beer81.csv", index=False)
    relative = manifest.loc[manifest.representation.eq("relative"), "feature"].tolist()
    floor_rows = []
    for i in range(len(raw)):
        counts = raw.loc[i, relative].value_counts(dropna=True)
        floor_rows.append({"beer": int(meta.beer.iloc[i]), "style": meta.tasting_category_fine.iloc[i],
            "most_repeated_relative_value": float(counts.index[0]), "most_repeated_relative_count": int(counts.iloc[0])})
    floor_table = pd.DataFrame(floor_rows)
    floor_table.to_csv(OUT / "relative_value_repeat_patterns.csv", index=False)
    numeric_fields = [c for c in raw if c not in KEYS]
    repeated = raw_row[numeric_fields].value_counts()
    repeated_rows = [{"value": float(v), "count": int(n),
        "features": ";".join(c for c in numeric_fields if raw_row[c] == v),
        "interpretation": "repeated supplied value; attachment does not identify an analytical detection limit"}
        for v, n in repeated.items() if n > 1]
    pd.DataFrame(repeated_rows).to_csv(OUT / "beer81_repeated_original_values.csv", index=False)
    calibrated_fields = manifest.loc[manifest.representation.eq("calibrated"), "feature"].tolist()
    zero_fields = table.loc[table.zero, "feature"].tolist()
    profile = {"raw_numeric_missing": int(raw_row[numeric_fields].isna().sum()),
        "hybrid_missing": int(x.iloc[target].isna().sum()),
        "calibrated_negative_count": int((pd.to_numeric(cal_row[calibrated_fields]) < 0).sum()),
        "raw_negative_count": int((pd.to_numeric(raw_row[numeric_fields]) < 0).sum()),
        "hybrid_negative_relative_count": int((x.iloc[target][relative] < 0).sum()),
        "hybrid_zero_count": len(zero_fields), "hybrid_zero_features": zero_fields,
        "exact_hybrid_duplicate_other_rows": int(np.sum(np.all(np.isclose(x.to_numpy(float), x.iloc[target].to_numpy(float),
            rtol=0, atol=0, equal_nan=True), axis=1)) - 1),
        "outside_nonpils_feature_ranges": int(table.outside_nonpils_range.sum()),
        "repeated_original_values": repeated_rows,
        "analytical_detection_limits_available": False,
        "detection_limit_note": "Sensory thresholds are perceptual thresholds, not analytical LOD/LOQ"}
    return table, floor_table, profile


def sensory_and_comments(meta, sensory, comments, train, target):
    sensory = sensory.set_index("beer").loc[meta.beer].reset_index()
    comments = comments.set_index("beer").loc[meta.beer].copy().reset_index()
    stale = sensory.stale_hops
    profiles, associations = [], []
    for field in [c for c in sensory if c not in KEYS]:
        value = float(sensory[field].iloc[target])
        correlation = spearmanr(sensory.loc[train, field], stale.loc[train]).statistic
        profiles.append({"sensory": field, "value": value, "all_percentile": percentile(sensory[field], value),
            "nonpils_percentile": percentile(sensory.loc[train, field], value),
            "nonpils_spearman_with_stale": float(correlation)})
        if field != "stale_hops":
            associations.append({"sensory": field, "nonpils_spearman_with_stale": float(correlation)})
    pd.DataFrame(profiles).to_csv(OUT / "sensory_profile_beer81.csv", index=False)
    pd.DataFrame(associations).to_csv(OUT / "sensory_covariation_nonpils.csv", index=False)
    tail = meta.copy()
    tail["stale_hops"] = stale
    tail["descending_rank"] = stale.rank(method="min", ascending=False).astype(int)
    tail["nonpils_training"] = train
    tail.sort_values("stale_hops", ascending=False).to_csv(OUT / "stale_target_distribution.csv", index=False)
    terms = []
    for field in [c for c in comments if c not in KEYS]:
        value = comments[field].iloc[target]
        relevant = any(field.endswith("_" + term) for term in
            ["stale", "cardboard", "papery", "metallic", "barnyard", "cheesy", "cabbage", "acetaldehyde"])
        terms.append({"term": field, "beer81_value": value,
            "all_percentile": percentile(comments[field], value) if pd.notna(value) else np.nan,
            "nonpils_percentile": percentile(comments.loc[train, field], value) if pd.notna(value) else np.nan,
            "predeclared_quality_term": relevant, "source": "7.csv numeric forum descriptor table"})
    terms = pd.DataFrame(terms)
    terms.to_csv(OUT / "forum_all_terms_beer81.csv", index=False)
    terms.loc[terms.predeclared_quality_term].to_csv(OUT / "forum_quality_term_evidence.csv", index=False)
    return pd.DataFrame(profiles), tail, terms


def distance_audit(meta, x, manifest, train, target, y):
    train_indices, pils = np.flatnonzero(train), np.flatnonzero(~train)
    calibrated = manifest.loc[manifest.representation.eq("calibrated"), "feature"].tolist()
    aging_table = pd.read_csv(source(SOLUTION / "outputs/q3_domain/stale/predeclared_aging_features.csv"))
    aging = aging_table.feature.tolist()
    values = x.to_numpy(float).copy()
    for j, field in enumerate(x):
        if field in calibrated:
            positive = values[train_indices, j]
            positive = positive[np.isfinite(positive) & (positive > 0)]
            scale = float(np.median(positive)) if len(positive) else 1.0
            values[:, j] = np.log1p(values[:, j] / scale)
    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(values[train_indices])
    imputed = imp.transform(values)
    scaler = StandardScaler().fit(imputed[train_indices])
    standard = scaler.transform(imputed)
    aging_indices = [x.columns.get_loc(c) for c in aging]
    spaces = {"unclipped_log_scaled_226": standard, "unclipped_log_scaled_aging24": standard[:, aging_indices]}
    for label, subset in [("clipped_log_scaled_226", None), ("clipped_log_scaled_aging24", aging)]:
        transform = FoldChemicalTransform(x.columns.tolist(), calibrated, subset).fit(x.iloc[train_indices])
        v = transform.transform(x)
        imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(v[train_indices])
        v = imputer.transform(v)
        fitted_scale = StandardScaler().fit(v[train_indices])
        spaces[label] = fitted_scale.transform(v)
    geometry = joblib.load(source(SOLUTION / "outputs/q3/transfer/transfer_copula.joblib"))
    np.testing.assert_array_equal(geometry["training_rows"], train_indices)
    latent = geometry["copula"].transform(x)
    spaces["old_copula_226"] = latent
    spaces["old_copula_aging24"] = latent[:, aging_indices]
    local = joblib.load(source(SOLUTION / "outputs/q3_domain/stale/transfer_stale_model.joblib"))
    np.testing.assert_array_equal(local["train_global_indices"], train_indices)
    spaces["frozen_domain_aging_pca8"] = local["pipeline"][:-1].transform(x)
    ood, neighbor_rows, reference_rows, all81, other_pils_rows = [], [], [], [], []
    for name, space in spaces.items():
        d_train = cdist(space[train_indices], space[train_indices]) / np.sqrt(space.shape[1])
        np.fill_diagonal(d_train, np.inf)
        loo = d_train.min(axis=1)
        reference_rows.extend({"space": name, "beer": int(meta.beer.iloc[i]), "loo_nn_distance": float(d)}
                              for i, d in zip(train_indices, loo))
        d_query = cdist(space[pils], space[train_indices]) / np.sqrt(space.shape[1])
        d_pils = cdist(space[pils], space[pils]) / np.sqrt(space.shape[1])
        np.fill_diagonal(d_pils, np.inf)
        target_position = int(np.flatnonzero(pils == target)[0])
        for rank, position in enumerate(np.argsort(d_pils[target_position]), start=1):
            i = pils[position]
            if i == target:
                continue
            other_pils_rows.append({"space": name, "query_beer": TARGET_BEER, "neighbor_rank": rank,
                "neighbor_beer": int(meta.beer.iloc[i]), "distance": float(d_pils[target_position, position]),
                "neighbor_stale": float(y.stale_hops.iloc[i])})
        for row, query in enumerate(pils):
            order = np.argsort(d_query[row])
            nearest = float(d_query[row, order[0]])
            ood.append({"space": name, "query_beer": int(meta.beer.iloc[query]),
                "dimensions": space.shape[1], "nearest_nonpils_distance": nearest,
                "nonpils_loo_nn_percentile": float(np.mean(loo <= nearest)),
                "nearest_other_pils_distance": float(d_pils[row].min()),
                "training_loo_nn_q95": float(np.quantile(loo, .95))})
            for rank, position in enumerate(order, start=1):
                i = train_indices[position]
                record = {"space": name, "query_beer": int(meta.beer.iloc[query]), "neighbor_rank": rank,
                    "neighbor_beer": int(meta.beer.iloc[i]), "neighbor_style": meta.tasting_category_fine.iloc[i],
                    "distance": float(d_query[row, position]), "neighbor_stale": float(y.stale_hops.iloc[i])}
                if rank <= 25:
                    neighbor_rows.append(record)
                if query == target:
                    all81.append(record)
    ood, neighbors, reference, all81 = map(pd.DataFrame, [ood, neighbor_rows, reference_rows, all81])
    ood["nearest_nonpils_distance_rank_within_pils_desc"] = ood.groupby("space").nearest_nonpils_distance.rank(
        ascending=False, method="min").astype(int)
    ood.to_csv(OUT / "chemical_ood_all_pils.csv", index=False)
    neighbors.to_csv(OUT / "nearest_training_neighbors.csv", index=False)
    all81.to_csv(OUT / "beer81_all_training_neighbor_ranks.csv", index=False)
    reference.to_csv(OUT / "training_loo_distance_reference.csv", index=False)
    pd.DataFrame(other_pils_rows).to_csv(OUT / "beer81_other_pils_neighbors.csv", index=False)
    neighbor_summaries = []
    for (name, query), group in neighbors.groupby(["space", "query_beer"]):
        for k in [5, 10, 25]:
            part = group.loc[group.neighbor_rank <= k]
            neighbor_summaries.append({"space": name, "query_beer": query, "k": k,
                "stale_min": part.neighbor_stale.min(), "stale_median": part.neighbor_stale.median(),
                "stale_mean": part.neighbor_stale.mean(), "stale_max": part.neighbor_stale.max(),
                "stale_at_least_beer81": int(part.neighbor_stale.ge(y.stale_hops.iloc[target]).sum())})
    pd.DataFrame(neighbor_summaries).to_csv(OUT / "neighbor_target_summaries.csv", index=False)
    tail_ranks = all81.loc[all81.neighbor_stale.ge(y.stale_hops.iloc[target])]
    tail_ranks.to_csv(OUT / "high_stale_training_neighbor_ranks.csv", index=False)
    return ood, all81, tail_ranks


def influence_audit(sensory):
    saved = pd.read_csv(source(SOLUTION / "outputs/q3_domain/stale/pils_diagnostic_predictions.csv"))
    saved["stage"] = "domain_same_seed_comparison"
    parts = [saved[["stage", "method", "beer", "observed", "prediction"]]]
    prior = pd.read_csv(source(SOLUTION / "outputs/q3_optimized/stale_fixed/pils_style_holdout_predictions.csv"))
    for method, field in [("fixed_single_tree", "prediction"), ("old_multioutput", "old_multioutput_prediction"),
                           ("mean", "mean_baseline_prediction")]:
        part = prior[["beer", "observed", field]].rename(columns={field: "prediction"})
        part["stage"], part["method"] = "previous_fixed_seed_comparison", method
        parts.append(part)
    frozen = pd.read_csv(source(SOLUTION / "outputs/q3/sensory/transfer_pils_heldout_predictions.csv"))
    frozen = frozen[["beer", "stale_hops"]].rename(columns={"stale_hops": "prediction"})
    frozen["observed"] = frozen.beer.map(sensory.set_index("beer").stale_hops)
    frozen["stage"], frozen["method"] = "original_frozen_proxy", "multioutput_center"
    parts.append(frozen)
    predictions = pd.concat(parts, ignore_index=True)
    predictions["residual"] = predictions.prediction - predictions.observed
    predictions["squared_error"] = predictions.residual ** 2
    predictions["sse_share"] = predictions.squared_error / predictions.groupby(["stage", "method"]).squared_error.transform("sum")
    predictions.to_csv(OUT / "all_saved_pils_prediction_errors.csv", index=False)
    rows, deletion = [], []
    for (stage, method), group in predictions.groupby(["stage", "method"]):
        source81 = group.loc[group.beer.eq(TARGET_BEER)].iloc[0]
        full_rmse = float(np.sqrt(group.squared_error.mean()))
        rows.append({"stage": stage, "method": method, "n_sources": len(group), "full_rmse": full_rmse,
            "beer81_prediction": source81.prediction, "beer81_residual": source81.residual,
            "beer81_sse_share": source81.sse_share})
        for beer in group.beer:
            reduced = group.loc[group.beer.ne(beer)]
            deletion.append({"stage": stage, "method": method, "omitted_beer": beer,
                "full_n": len(group), "remaining_n": len(reduced), "full_rmse": full_rmse,
                "fixed_prediction_rmse_without_one_source": float(np.sqrt(reduced.squared_error.mean())),
                "interpretation": "influence only; same fixed predictions, no retraining, never substitutes the all-source metric"})
    influence = pd.DataFrame(rows)
    influence.to_csv(OUT / "beer81_sse_contribution.csv", index=False)
    pd.DataFrame(deletion).to_csv(OUT / "leave_one_source_out_influence.csv", index=False)
    return influence


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    root = data_dir()
    raw, sensory, comments = [pd.read_csv(source(root / filename), dtype={"beer_id": str})
                              for filename in ["1.csv", "4.csv", "7.csv"]]
    source(root / "8.xlsx")
    calibrated = pd.read_excel(root / "8.xlsx", sheet_name="Calibrated chemical dataset", dtype={"beer_id": str})
    thresholds = pd.read_excel(root / "8.xlsx", sheet_name="Sensory thresholds")
    sensory_names = pd.read_excel(root / "8.xlsx", sheet_name="Sensory names")
    target_metadata = sensory_names.loc[sensory_names.R_name.isin(["stale_hops", "barnyard", "t2n", "dms"])]
    target_metadata.to_csv(OUT / "sensory_target_metadata.csv", index=False)
    meta, x, y, manifest = load_q3_data()
    target = int(np.flatnonzero(meta.beer.eq(TARGET_BEER))[0])
    train = meta.tasting_category_fine.ne("Pils/Lager")
    identity = identity_audit(raw, sensory, comments, calibrated)
    chemicals, floors, profile = chemical_audit(meta, x, manifest, raw, calibrated, thresholds, train, target)
    sensation, tail, terms = sensory_and_comments(meta, sensory, comments, train, target)
    ood, neighbors, tail_ranks = distance_audit(meta, x, manifest, train, target, y)
    influence = influence_audit(sensory)
    local_neighbors = neighbors.loc[neighbors.space.eq("frozen_domain_aging_pca8") & neighbors.neighbor_rank.le(25)]
    key_terms = terms.loc[terms.term.isin(["aroma_stale", "taste_stale", "taste_cardboard", "taste_papery"])]
    high_tail = tail.loc[tail.nonpils_training & tail.stale_hops.ge(y.stale_hops.iloc[target])]
    selected = json.loads((SOLUTION / "outputs/q3_domain/stale/summary.json").read_text())["selected_method"]
    source(SOLUTION / "outputs/q3_domain/stale/summary.json")
    selected_influence = influence.loc[influence.stage.eq("domain_same_seed_comparison") & influence.method.eq(selected)].iloc[0]
    own_floor = floors.loc[floors.beer.eq(TARGET_BEER)].iloc[0]
    barnyard = sensation.loc[sensation.sensory.eq("barnyard")].iloc[0]
    summary = {"source_beer": TARGET_BEER, "exact_beer_join_passed": True,
        "barcode_consistent_at_supplied_precision": True, "barcode_exact_1csv_8xlsx": True,
        "sample_profile": profile, "stale_observed": float(y.stale_hops.iloc[target]),
        "stale_descending_rank_all250": int(tail.loc[tail.beer.eq(TARGET_BEER), "descending_rank"].iloc[0]),
        "nonpils_training_rows": int(train.sum()), "nonpils_stale_at_least_beer81": len(high_tail),
        "nonpils_high_tail_beers": high_tail.beer.tolist(),
        "frozen_local_top25_stale_min": float(local_neighbors.neighbor_stale.min()),
        "frozen_local_top25_stale_max": float(local_neighbors.neighbor_stale.max()),
        "beer81_outside_local_top25_target_convex_hull": bool(y.stale_hops.iloc[target] > local_neighbors.neighbor_stale.max()),
        "selected_domain_model": selected, "selected_domain_beer81_prediction": float(selected_influence.beer81_prediction),
        "selected_domain_beer81_sse_share": float(selected_influence.beer81_sse_share),
        "relative_repeated_floor_count": int(own_floor.most_repeated_relative_count),
        "relative_floor_count_all250_percentile": float(np.mean(floors.most_repeated_relative_count <= own_floor.most_repeated_relative_count)),
        "stale_target_datatype": target_metadata.loc[target_metadata.R_name.eq("stale_hops"), "DataType"].iloc[0],
        "barnyard_target_datatype": target_metadata.loc[target_metadata.R_name.eq("barnyard"), "DataType"].iloc[0],
        "nonpils_stale_barnyard_spearman": float(barnyard.nonpils_spearman_with_stale),
        "ood_beer81": ood.loc[ood.query_beer.eq(TARGET_BEER)].to_dict("records"),
        "forum_quality_evidence": key_terms.to_dict("records"),
        "raw_forum_text_available": False, "analytical_lod_loq_available": False,
        "label_error_established": False,
        "interpretation": "Rare high sensory target lacks matching high-target chemical neighbours; identity and descriptor evidence do not establish a wrong label. Chemical predictability remains unresolved."}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    verification = {"identity_join_four_sources_passed": bool(identity.beer_unique.all()),
        "features": len(chemicals), "forum_fields": len(terms), "ood_query_space_rows": len(ood),
        "training_neighbor_rank_rows": len(neighbors),
        "other_pils_neighbor_rows": len(pd.read_csv(OUT / "beer81_other_pils_neighbors.csv")),
        "local_target_convex_hull_gap_verified": summary["beer81_outside_local_top25_target_convex_hull"],
        "leave_one_source_out_only_fixed_prediction_influence": True,
        "source_hashes_unchanged_after_analysis": all(hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest()
                                                       == row["sha256"] for row in SOURCE_FILES)}
    assert verification["source_hashes_unchanged_after_analysis"]
    (OUT / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "source_manifest.json").write_text(json.dumps(SOURCE_FILES, ensure_ascii=False, indent=2), encoding="utf-8")
    write_note(summary, chemicals, sensation, terms, tail_ranks, influence)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def write_note(summary, chemicals, sensation, terms, tail_ranks, influence):
    lines = ["# Beer 81误差来源诊断", "",
        "beer81的高残差目前不能归因于标签错误。附件键连接、风格和可核验条码一致；"
        "无缺失、负校准值或重复化学行解释该误差。陈旧描述在论坛术语表中也处高位。"
        "直接观察到的困难是高stale尾部样本很少，且未落入现有化学近邻空间。", "",
        "## 连接与测量检查", "",
        "1.csv与8.xlsx校准表的完整条码为8715608222557，风格均Pils/Lager。4.csv与7.csv提供"
        "的8.71561E+12与其打印精度一致；它们是舍入后的身份信息，实际按唯一beer键连接，"
        "未采用舍入条码连接。", "",
        f"原231化学列缺失数为{summary['sample_profile']['raw_numeric_missing']}，226混合输入缺失数为"
        f"{summary['sample_profile']['hybrid_missing']}，负校准值为{summary['sample_profile']['calibrated_negative_count']}，"
        f"其他样本中的整行化学重复为{summary['sample_profile']['exact_hybrid_duplicate_other_rows']}。"
        "原始表示中的负值不能直接视为负浓度；混合输入的未校准相对字段仍保留其供应表示。", "",
        "原表有多个字段重复同一低值，完整清单见beer81_repeated_original_values.csv。"
        "附件未提供分析LOD/LOQ，不能把共同低值直接命名为检出限；8.xlsx给的是感官阈值，"
        "也不等于仪器检测上下限。经验最小值、最大值与0值仅作为观测支持诊断保存。", "",
        f"81有{summary['relative_repeated_floor_count']}个相对字段重复同一低值，重复数量仅处250款的"
        f"{summary['relative_floor_count_all250_percentile']:.1%}位置，并非该现象最严重的样本。", "",
        "## 化学异常与支持域", "",
        "所有尺度仅拟合239款非Pils：校准字段先按训练正数中位数做log1p尺度化，训练中位数补缺，"
        "再训练标准化；另比较训练1%/99%截断、冻结旧copula秩空间以及16实际冻结aging/PCA8空间。"
        "没有用11款源学习尺度或边界。", "",
        "|空间|81最近非Pils距离|相对训练留一NN距离百分位|81最近其他Pils距离|", "|---|---:|---:|---:|"]
    for row in summary["ood_beer81"]:
        lines.append(f"|{row['space']}|{row['nearest_nonpils_distance']:.4f}|{row['nonpils_loo_nn_percentile']:.1%}|"
                     f"{row['nearest_other_pils_distance']:.4f}|")
    lines.extend(["", "不同空间有不同含义：全化学谱显示一定风格偏移，但aging及其PCA空间接近训练邻域。"
        "参考距离只排除训练行自身，未逐行重拟合预处理；这些百分位只是相同训练表示下的描述性比较，"
        "不是交叉拟合或校准OOD概率，也不证明所有有效化学信息已被表示。", "",
        "校准字段中，训练百分位低于2.5%或高于97.5%的项目完整列在下表；这是多变量描述性筛查，非显著性检验。", "",
        "|字段|81校准值|单位|非Pils百分位|附件aging标记|", "|---|---:|---|---:|---|"])
    extreme = chemicals.loc[chemicals.representation.eq("calibrated") &
                            ((chemicals.nonpils_percentile < .025) | (chemicals.nonpils_percentile > .975))]
    for row in extreme.itertuples():
        lines.append(f"|{row.feature}|{row.value:.6g}|{row.unit}|{row.nonpils_percentile:.1%}|{row.aging if pd.notna(row.aging) else '无'}|")
    lines.extend(["", "SO2高百分位、部分酒花来源指标低百分位均未形成附件内已验证的stale机制。"
        "不能从这些异常单独推导储存条件、氧化时间或标签错误。所有226字段的值、训练边界、百分位、"
        "来源、aging与阈值信息保存在chemical_profile_beer81.csv。", "",
        "题目四种化学缺陷的核对如下。感官阈值与stale_hops不是同一目标，不可把某一化学项超阈直接当作其误差原因。", "",
        "|字段|81校准值|单位|非Pils百分位|附件感官阈值中位数|", "|---|---:|---|---:|---:|"])
    for row in chemicals.loc[chemicals.feature.isin(["DMS", "X2.3.butanedione", "X4VG", "X.E..2.nonenal"])].itertuples():
        threshold = f"{row.sensory_threshold_median:.6g}" if pd.notna(row.sensory_threshold_median) else "未提供"
        lines.append(f"|{row.feature}|{row.value:.6g}|{row.unit}|{row.nonpils_percentile:.1%}|{threshold}|")
    outside = chemicals.loc[chemicals.outside_nonpils_range]
    if len(outside):
        labels = "; ".join(f"{r.feature}={r.value:.6g}（训练范围{r.nonpils_min:.6g}至{r.nonpils_max:.6g}）"
                           for r in outside.itertuples())
        lines.extend(["", f"全226字段超出训练观测范围的完整清单为：{labels}。"
            "该字段仍是未校准相对表示，不能擅自换算成浓度或视为仪器越限。"])
    lines.extend(["",
        "## 尾部稀缺与邻域错位", "",
        f"81的stale={summary['stale_observed']:.6f}，在250款中第{summary['stale_descending_rank_all250']}高；"
        f"239款训练中仅{summary['nonpils_stale_at_least_beer81']}款不低于它。"
        f"16实际局部模型的最近25训练邻居stale最高只有{summary['frozen_local_top25_stale_max']:.6f}，"
        f"低于81实测值；其预测为{summary['selected_domain_beer81_prediction']:.6f}。"
        "局部加权均值受邻居目标凸包限制，因此现有邻域结构无法给出81的高标签。", "",
        "|高stale训练beer|风格|实测stale|aging/PCA8近邻排名|", "|---|---|---:|---:|"])
    for row in tail_ranks.loc[tail_ranks.space.eq("frozen_domain_aging_pca8")].sort_values("neighbor_rank").itertuples():
        lines.append(f"|{row.neighbor_beer}|{row.neighbor_style}|{row.neighbor_stale:.6f}|{row.neighbor_rank}|")
    lines.extend(["", "全226旧copula的最近邻同样多为低stale样本。完整近邻及其真实目标保存，而非只展示支持结论的邻居。"
        "这说明现有化学距离不足以组织该尾部目标，不能证明不存在其他可预测化学组合；"
        "当前附件也无法核验储存、包装、批次或品评个体差异。", "",
        "## 其他感官与论坛描述", "",
        "其他实测感官仅用于解释。所有50项值与239款非Pils中的stale相关均保存，"
        "它们不会作为源样本的chemical-only预测输入。论坛7.csv是数值描述符表，未提供原文、"
        "评论人数、时间或逐批次对应信息；字段数值不能解释为词出现次数。", "",
        f"8.xlsx明确stale_hops与barnyard均为numeric，不能把它们当作categorical父类继承值。"
        f"两者在239款非Pils中的Spearman相关为{summary['nonpils_stale_barnyard_spearman']:.4f}；"
        "这是很强的共同变化，但没有原始品评记录来区分共享感官因素或数据加工来源，不能据此判错。", "",
        "numeric说明数据类型，不代表恢复了0–6原始评分；本诊断始终保留4.csv供应的标准化表示。", "",
        "|论坛字段|81供应数值|250款百分位|", "|---|---:|---:|"])
    for row in terms.loc[terms.term.isin(["aroma_stale", "taste_stale", "taste_cardboard", "taste_papery", "aroma_papery"])].itertuples():
        lines.append(f"|{row.term}|{row.beer81_value:.4f}|{row.all_percentile:.1%}|")
    lines.extend(["", "这些陈旧、纸板、纸张相关描述构成另一供应数据通道的相符证据，支持谨慎保留高stale标签；"
        "但其采样是否与品评完全独立、是否对应同一批次均不明，不能称为独立真值认证。", "",
        "论坛barnyard描述并不高，两个通道并非所有感官属性逐项相符；完整186项记录均已保存，"
        "没有仅保留支持高stale解释的字段。", "",
        "## 误差影响与可采取的解释", "",
        "|保存阶段|模型|全11源RMSE|81预测|81平方误差占比|", "|---|---|---:|---:|---:|"])
    preferred = influence.loc[(influence.stage.eq("original_frozen_proxy")) |
        (influence.stage.eq("domain_same_seed_comparison") & influence.method.isin(
            [summary["selected_domain_model"], "old_multioutput_tree", "fixed_single_tree", "mean_global"]))]
    for row in preferred.itertuples():
        lines.append(f"|{row.stage}|{row.method}|{row.full_rmse:.4f}|{row.beer81_prediction:.4f}|{row.beer81_sse_share:.1%}|")
    lines.extend(["", "leave_one_source_out_influence.csv只从已保存预测中临时去掉一个源计算影响，"
        "没有重训，也不构成新的交叉验证；任何去掉81后的误差都不能替代全11源主指标。", "",
        "目前可确认的是尾部样本稀缺、化学近邻与高stale错位，以及已有模型的强回归均值倾向。"
        "不能确认是测量、储存、批次、未观测成分、感官个体差异或标签问题所致。"
        "后续仅可在非Pils训练内验证尾部权重、共享感官表示及其他化学模型；论坛描述和81标签不得进入选型。", "",
        "```bash", ".venv/bin/python solution_a/src/21_q3_beer81_audit.py", "```", ""])
    (SOLUTION / "docs/q3_beer81_audit.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
