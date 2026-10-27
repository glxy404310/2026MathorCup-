"""Summarize saved evidence for the reference review without fitting models.

Run: python solution_a/supplement/reference_tables.py
Read from a paper builder: load_reference_tables(solution_root)
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd


SOLUTION = Path(__file__).resolve().parents[1]
DESTINATION = SOLUTION / "supplement/results/reference_tables"
TOLERANCE = 1e-10
POLICY = {"arm": "fixed", "pool": "contracted_6004", "bound": "retain80", "ranking": "margin"}
DEFECT_ORDER = ["DMS", "Diacetyl", "4VG", "trans_2_nonenal"]
# Identity checked before this revision. These are provenance, not live inputs.
REVIEW_BASELINE = {
    "scope": "Pre-revision 353-page delivery checked on 2026-10-25; current submission hashes may change.",
    "pdf_pages": 353,
    "pdf_sha256": "03287ed5830de2d60d91cf1c2cd27223a16324e67c101cafe4167bc91fbdcdd1",
    "archive_sha256": "5545c955b303d4b362df54e7173ef436240f0bf21564ce8c2305c553f4fc3a75",
    "comparison_report_sha256": "d1e76bcb19b713dd8adcbd0dba3fae48142714128e434554b0627360f5b1fbc3",
    "packaged_report_matched_working_report": True,
    "packaged_transfer_matched_saved_transfer": True,
}
TABLE_NAMES = [
    "transfer_defect_concentrations", "transfer_defect_activity_summary",
    "beer96_chemical", "beer96_committees", "ethyl_acetate_module_by_fold",
    "ethyl_acetate_F_esters_ethac_by_fold", "rebuild_grid_field_counts",
    "rebuild_changed_policies", "rebuild_changed_frontier",
]


def digest(path):
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def newline_digest(path):
    return sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def resolve_input(root, relative):
    """Resolve workspace or lean-archive saved-result locations."""
    direct = root / relative
    if direct.is_file():
        return direct
    if relative.startswith("outputs/"):
        return root.parent / "paper_results" / relative.removeprefix("outputs/")
    if relative.startswith("review/verification/"):
        return root.parent / "verification" / relative.removeprefix("review/verification/")
    if relative.startswith("supplement/results/"):
        return root.parent / "supplement_results" / relative.removeprefix("supplement/results/")
    return direct


def equal_value(left, right):
    if pd.isna(left) and pd.isna(right):
        return True
    if isinstance(left, (float, np.floating)) and isinstance(right, (float, int, np.floating)):
        return bool(np.isclose(left, right, rtol=0, atol=TOLERANCE))
    return bool(left == right)


def load_reference_tables(solution_root=SOLUTION):
    """Return named saved DataFrames plus summary; this read interface writes nothing."""
    root = Path(solution_root)
    folder = resolve_input(root, "supplement/results/reference_tables/summary.json").parent
    tables = {name: pd.read_csv(folder / f"{name}.csv") for name in TABLE_NAMES}
    tables["summary"] = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    return tables


def build_reference_tables(solution_root=SOLUTION):
    root = Path(solution_root).resolve()
    output = root / "supplement/results/reference_tables"
    tracked = {}

    def source(relative):
        path = resolve_input(root, relative)
        tracked[relative] = {"path": relative, "bytes": path.stat().st_size, "sha256": digest(path),
                             "newline_normalized_sha256": newline_digest(path)}
        return path

    def csv(relative):
        return pd.read_csv(source(relative))

    def js(relative):
        return json.loads(source(relative).read_text(encoding="utf-8"))

    transfer = csv("outputs/q3_refined/transfer/transfer_summary.csv")
    selected = transfer.copy()
    for field, value in POLICY.items():
        selected = selected[selected[field].eq(value)]
    assert len(selected) == 11 and selected.source_beer.is_unique
    assert selected.candidate_index.notna().all()
    assert selected.fresh_consistency_passed.sum() == 10
    assert selected.loc[~selected.fresh_consistency_passed, "source_beer"].tolist() == [96]
    scenarios = js("outputs/q3_optimized/transfer/threshold_scenarios.json")
    metadata = js("outputs/q3/sensory/defect_thresholds.json")
    threshold = csv("outputs/q3_refined/transfer/threshold_checks.csv")
    threshold = threshold[threshold.transfer_id.isin(selected.transfer_id)].copy()
    assert len(scenarios) == 10 and len(threshold) == 11 * 4 * 10
    for _, group in threshold.groupby(["transfer_id", "defect"]):
        assert group.scenario.nunique() == 10
        assert group.source.nunique() == group.transferred.nunique() == group.feature.nunique() == 1
    concentration = threshold.groupby(["transfer_id", "defect", "feature"], as_index=False).agg(
        source=("source", "first"), transferred=("transferred", "first"),
        cap=("cap", "min"), scenario_count=("scenario", "nunique"), passed=("passed", "all"))
    concentration = concentration.merge(selected[["transfer_id", "source_beer"]],
                                          on="transfer_id", validate="many_to_one")
    concentration["unit"] = concentration.defect.map(lambda name: metadata[name]["unit"])
    concentration["unit_display"] = concentration.unit.replace({"ug/L": "μg/L"})
    concentration["delta"] = concentration.transferred - concentration.source
    concentration["slack"] = concentration.cap - concentration.transferred
    concentration["at_upper"] = np.isclose(concentration.transferred, concentration.cap,
                                            rtol=0, atol=TOLERANCE)
    concentration["at_zero"] = np.isclose(concentration.transferred, 0, rtol=0, atol=TOLERANCE)
    assert concentration.passed.all() and (concentration.slack >= -TOLERANCE).all()
    assert (concentration.transferred >= -TOLERANCE).all()
    for row in concentration.itertuples():
        values = [s["thresholds"][row.feature] for s in scenarios if s["thresholds"][row.feature] is not None]
        expected = max(row.source, min(values)) if values else row.source
        assert np.isclose(row.cap, expected, rtol=0, atol=TOLERANCE)
    source_order = {beer: number for number, beer in enumerate(selected.source_beer)}
    concentration = concentration.sort_values(
        ["source_beer", "defect"], key=lambda column: column.map(
            source_order if column.name == "source_beer" else {name: i for i, name in enumerate(DEFECT_ORDER)}))
    concentration = concentration[["source_beer", "transfer_id", "defect", "feature", "unit", "unit_display",
                                   "source", "transferred", "delta", "cap", "slack", "scenario_count",
                                   "passed", "at_upper", "at_zero"]].reset_index(drop=True)
    activity = concentration.groupby("defect").agg(
        sources=("source_beer", "size"), all_scenarios_passed=("passed", "sum"),
        at_upper_count=("at_upper", "sum"), at_zero_count=("at_zero", "sum"),
        minimum_slack=("slack", "min"), maximum_slack=("slack", "max")).reindex(DEFECT_ORDER).reset_index()
    upper_counts = activity.set_index("defect").at_upper_count.to_dict()
    zero_counts = activity.set_index("defect").at_zero_count.to_dict()
    assert upper_counts == {"DMS": 11, "Diacetyl": 7, "4VG": 0, "trans_2_nonenal": 10}
    assert zero_counts["4VG"] == 11
    chemical96 = concentration[concentration.source_beer.eq(96)].reset_index(drop=True)
    beer96 = selected[selected.source_beer.eq(96)].iloc[0]
    deltas = csv("outputs/q3_refined/transfer/paired_deltas.csv")
    deltas = deltas[deltas.transfer_id.eq(beer96.transfer_id)]
    stages = [
        ("new_selection", "current_selection", 40, True, "new_selection_q95"),
        ("old_selection", "old_selection", 30, True, "old_selection_q95"),
        ("new_audit", "current_fresh_audit", 20, True, "new_audit_q95"),
        ("old_fresh_audit", "old_fresh_audit", 20, True, "old_fresh_audit_q95"),
        ("old_historical", "old_historical_diagnostic", 10, False, "old_historical_q95"),
    ]
    committees = []
    for ensemble, stage, count, gate, field in stages:
        group = deltas[deltas.ensemble.eq(ensemble)].sort_values("model_index")
        assert len(group) == count and group.model_index.is_unique
        q95 = float(group.delta.quantile(0.95))
        assert np.isclose(q95, beer96[field], rtol=0, atol=TOLERANCE)
        committees.append({"source_beer": 96, "transfer_id": beer96.transfer_id, "stage": stage,
                           "ensemble": ensemble, "model_count": count,
                           "positive_count": int(group.delta.gt(0).sum()), "q95": q95,
                           "median_delta": float(group.delta.median()), "minimum_delta": float(group.delta.min()),
                           "maximum_delta": float(group.delta.max()), "q95_passed": q95 <= TOLERANCE,
                           "used_in_primary_gate": gate})
    for field, stage in [("center_new_delta", "current_center"), ("center_old_delta", "old_center")]:
        value = float(beer96[field])
        committees.append({"source_beer": 96, "transfer_id": beer96.transfer_id, "stage": stage,
                           "ensemble": "center_point", "model_count": 1, "positive_count": int(value > 0),
                           "q95": value, "median_delta": value, "minimum_delta": value,
                           "maximum_delta": value, "q95_passed": value <= TOLERANCE,
                           "used_in_primary_gate": True})
    committees = pd.DataFrame(committees)
    fresh96 = committees[committees.stage.eq("current_fresh_audit")].iloc[0]
    assert fresh96.positive_count == 2 and fresh96.q95 > 0
    assert int(beer96.feasible_count) == 13 and beer96.status == "new_audit_failed"

    membership = csv("supplement/results/q2_permutation/module_membership.csv")
    module_metrics = csv("supplement/results/q2_permutation/module_importance_by_fold.csv")
    individual = csv("supplement/results/q2_permutation/importance_by_fold.csv")
    membership["tokens"] = membership.members.map(lambda text: text.split(" | "))
    member = membership[membership.tokens.map(lambda tokens: "ethyl_acetate" in tokens)].copy()
    assert member.fold.tolist() == [1, 2, 3, 4, 5]
    assert member["size"].tolist() == [11, 12, 9, 11, 27]
    assert all(len(row.tokens) == row.size and row.tokens.count("ethyl_acetate") == 1
               for row in member.itertuples())
    member["members_json"] = member.tokens.map(lambda value: json.dumps(value, ensure_ascii=False))
    member = member.rename(columns={"members": "memberlist"}).drop(columns="tokens")
    module_metrics = module_metrics.rename(columns={"delta_mse": "module_delta_mse",
                                                    "repeat_sd": "module_repeat_sd"})
    module = member.merge(module_metrics.drop(columns="size"), on=["fold", "module"], validate="one_to_many")
    single = individual[individual.feature.eq("ethyl_acetate")].rename(
        columns={"sensory_target": "target", "delta_mse": "individual_delta_mse",
                 "repeat_sd": "individual_repeat_sd"}).drop(columns="feature")
    module = module.merge(single, on=["fold", "target"], validate="one_to_one")
    module = module[["fold", "module", "size", "target", "memberlist", "members_json",
                     "module_delta_mse", "individual_delta_mse", "module_repeat_sd", "individual_repeat_sd"]]
    module = module.sort_values(["target", "fold"]).reset_index(drop=True)
    module_case = module[module.target.eq("F_esters_ethac")].reset_index(drop=True)
    assert len(module) == 225 and len(module_case) == 5

    # The report was byte-matched to the reviewed 353-page delivery. Do not make
    # later PDF/ZIP regeneration a prerequisite for these numerical summaries.
    comparison_path = source("review/verification/rebuilt_numeric_comparison.json")
    comparison_bytes = comparison_path.read_bytes()
    comparison = json.loads(comparison_bytes)
    assert sha256(comparison_bytes.replace(b"\r\n", b"\n")).hexdigest() == REVIEW_BASELINE["comparison_report_sha256"]
    evidence = comparison["q3_transfer_full_grid"]
    changes = evidence["changed_policies"]
    fields = ["candidate_index", "status", "new_model_passed", "fresh_consistency_passed", "historical_consistency_passed"]
    assert len({record["transfer_id"] for record in changes}) == len(changes)
    recounted = {field: sum(not equal_value(record["original"][field], record["rebuilt"][field])
                            for record in changes) for field in fields}
    assert all(recounted[field] == evidence["changed_policy_fields"][field] for field in fields)
    assert recounted == {"candidate_index": 16, "status": 7, "new_model_passed": 5,
                         "fresh_consistency_passed": 4, "historical_consistency_passed": 5}
    assert len(changes) == evidence["changed_policies_count"] == 16
    assert evidence["policies"] == 792 and evidence["frontier_rows"] == 72
    frontier_changes = evidence["frontier_count_changed_rows"]
    assert len(frontier_changes) == 9
    for record in frontier_changes:
        assert set(record["original"]) == set(record["rebuilt"])
        assert any(record["original"][field] != record["rebuilt"][field] for field in record["original"])
    for headline in comparison["q3_transfer_headlines"].values():
        assert all(abs(value) <= TOLERANCE for value in headline["difference"].values())
    assert all(comparison["q3_transfer_selected_candidates_and_statuses_match"].values())
    field_counts = pd.DataFrame([{"field": field, "changed_rows": count, "total_rows": 792}
                                for field, count in recounted.items()])
    changed_policies = pd.json_normalize(changes, sep="_")
    changed_frontier = pd.json_normalize(frontier_changes, sep="_")
    summary = {
        "status": "PASS", "scope": "Saved-data summaries only; no fitting or new reconstruction run.",
        "policy": POLICY, "sources": 11, "concentration_rows": 44, "chemical_scenarios": 10,
        "chemical_all_passed": True, "upper_counts": upper_counts, "zero_counts": zero_counts,
        "interpretation": "Touching a bound is descriptive; it does not identify a binding causal bottleneck or a KKT multiplier.",
        "beer96": {"candidate_count": int(beer96.candidate_count), "feasible_count": 13,
                   "chemical_all_passed": True, "status": beer96.status,
                   "current_fresh_q95": float(fresh96.q95), "current_fresh_positive_count": 2,
                   "old_fresh_q95": float(beer96.old_fresh_audit_q95)},
        "ethyl_acetate_module": {"feature": "ethyl_acetate", "token_match": "exact pipe-delimited member token",
                                 "fold_sizes": member["size"].tolist(), "all_target_rows": 225,
                                 "case_target": "F_esters_ethac", "case_rows": 5,
                                 "scope": "Q1 231-variable fold graphs; not a Q3 226-variable soft constraint or causal importance."},
        "reconstruction": {"evidence_scope": "Recount of detailed saved historical reconstruction evidence packaged in the current verified delivery.",
                           "new_rebuild_performed": False, "raw_rebuilt_folder_available": (root.parent / comparison["rebuilt_outputs"]).exists(),
                           "historical_original_equals_current_outputs_required": False,
                           "historical_scope": "The report original/rebuilt pair is recounted internally; current transfer CSVs independently supply the chemistry and beer96 tables.",
                           "comparison_report": "review/verification/rebuilt_numeric_comparison.json",
                           "review_baseline_provenance": REVIEW_BASELINE,
                           "policies": 792, "changed_policies": 16, "changed_fields": recounted,
                           "frontier_rows": 72, "changed_frontier_count_rows": 9,
                           "principal_transfer_headlines": "PASS", "principal_transfer_rows": comparison["q3_transfer_comparison_rows"],
                           "full_grid_exact_reproduction": comparison["exact_full_grid_reproduction"]},
        "positive_count_definition": "Number of saved paired deltas strictly greater than zero; Q95 pass tolerance is 1e-10.",
    }
    tables = dict(zip(TABLE_NAMES, [concentration, activity, chemical96, committees, module, module_case,
                                   field_counts, changed_policies, changed_frontier]))
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False, float_format="%.17g")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for relative, record in tracked.items():
        assert digest(resolve_input(root, relative)) == record["sha256"], f"Input changed during summary: {relative}"
    manifest = {"script": {"path": "supplement/reference_tables.py", "sha256": digest(Path(__file__))},
                "inputs": list(tracked.values()), "outputs": [], "no_model_fitting": True,
                "text_hash_rule": "Exact bytes, or CRLF/LF-only equivalence verified by newline_normalized_sha256.",
                "read_interface": "from supplement.reference_tables import load_reference_tables; tables = load_reference_tables(solution_root)"}
    for path in sorted(output.iterdir()):
        if path.name == "input_hash_manifest.json":
            continue
        manifest["outputs"].append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                                    "sha256": digest(path), "newline_normalized_sha256": newline_digest(path)})
    (output / "input_hash_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def verify_reference_tables(solution_root=SOLUTION):
    """Verify saved summaries and their numerical inputs; no PDF/ZIP dependency."""
    root = Path(solution_root).resolve()
    output = resolve_input(root, "supplement/results/reference_tables/summary.json").parent
    manifest = json.loads((output / "input_hash_manifest.json").read_text(encoding="utf-8"))
    assert digest(Path(__file__)) == manifest["script"]["sha256"]
    assert all(not item["path"].startswith("submission/") for item in manifest["inputs"])
    for item in manifest["inputs"]:
        path = resolve_input(root, item["path"])
        exact = path.stat().st_size == item["bytes"] and digest(path) == item["sha256"]
        assert exact or newline_digest(path) == item["newline_normalized_sha256"], item["path"]
    for item in manifest["outputs"]:
        path = resolve_input(root, item["path"])
        exact = path.stat().st_size == item["bytes"] and digest(path) == item["sha256"]
        assert exact or newline_digest(path) == item["newline_normalized_sha256"], item["path"]
    tables = load_reference_tables(root)
    concentration = tables["transfer_defect_concentrations"]
    assert len(concentration) == 44 and concentration.source_beer.nunique() == 11
    assert concentration.passed.all() and concentration.scenario_count.eq(10).all()
    assert concentration.groupby("defect").at_upper.sum().to_dict() == {
        "4VG": 0, "DMS": 11, "Diacetyl": 7, "trans_2_nonenal": 10}
    assert concentration[concentration.defect.eq("4VG")].at_zero.all()
    current_audit = tables["beer96_committees"].query("stage == 'current_fresh_audit'").iloc[0]
    assert current_audit.model_count == 20 and current_audit.positive_count == 2
    assert np.isclose(current_audit.q95, 0.0210027907557986, rtol=0, atol=TOLERANCE)
    case = tables["ethyl_acetate_F_esters_ethac_by_fold"]
    assert case.fold.tolist() == [1, 2, 3, 4, 5] and case["size"].tolist() == [11, 12, 9, 11, 27]
    assert all(json.loads(row.members_json).count("ethyl_acetate") == 1 for row in case.itertuples())
    counts = tables["rebuild_grid_field_counts"].set_index("field").changed_rows.to_dict()
    assert counts["candidate_index"] == 16 and counts["status"] == 7 and counts["fresh_consistency_passed"] == 4
    assert len(tables["rebuild_changed_frontier"]) == 9
    return {"status": "PASS", "numerical_inputs_checked": len(manifest["inputs"]),
            "outputs_checked": len(manifest["outputs"]), "concentration_rows": 44,
            "committee_rows": 7, "module_case_rows": 5, "new_model_fitting": False,
            "current_submission_files_required": False}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Read and verify saved summaries without writing")
    if parser.parse_args().verify_only:
        result = verify_reference_tables()
    else:
        result = build_reference_tables()
        result = {"status": result["status"], "output": str(DESTINATION),
                  "rows": result["concentration_rows"], "module_case_rows": 5,
                  "grid_changed_policies": result["reconstruction"]["changed_policies"]}
    print(json.dumps(result, ensure_ascii=False))
