"""Exploratory Q2 bootstrap with the observed style counts held fixed.

Writes supplementary results in results/q2_stratified. Existing Q2 rankings and frozen outputs remain
the reference. Run with the solution's requirements-lock.txt environment.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SOLUTION = HERE.parent
OUT = HERE / "results/q2_stratified"
REPOSITORY = SOLUTION.parent
SEED = 20261023
BOOTSTRAPS = 200


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv(frame, name):
    frame.to_csv(OUT / name, index=False, encoding="utf-8-sig",
                 float_format="%.10g", lineterminator="\n")


def set_score(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    helper_path = SOLUTION / "src/03_q2_network.py"
    spec = importlib.util.spec_from_file_location("q2_existing_helper", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    files = [helper_path, helper.DATA / "1.csv", helper.DATA / "4.csv",
             SOLUTION / "outputs/q2/all_candidate_stability.csv",
             SOLUTION / "outputs/q2/top3_compounds.csv",
             SOLUTION / "outputs/q2/summary.json"]
    before_hashes = {p.relative_to(REPOSITORY).as_posix(): sha256(p) for p in files}
    chemical = pd.read_csv(helper.DATA / "1.csv")
    sensory = pd.read_csv(helper.DATA / "4.csv")
    candidates = [c for c in chemical.columns[3:] if c not in helper.EXCLUDE_X]
    targets = [c for c in sensory.columns[3:] if c not in helper.EXCLUDE_Y]
    merged = chemical.merge(sensory[["beer", *targets]], on="beer",
                            validate="one_to_one", how="inner")
    assert merged.shape[0] == 250 and len(candidates) == 218 and len(targets) == 45
    x = merged[candidates].to_numpy(float)
    y = merged[targets].to_numpy(float)
    rho = helper.cross_correlation(x, y, "spearman")
    initial_top = helper.top3(rho)
    original = pd.read_csv(SOLUTION / "outputs/q2/all_candidate_stability.csv")
    original_top = pd.read_csv(SOLUTION / "outputs/q2/top3_compounds.csv")
    for t, target in enumerate(targets):
        expected = original_top[original_top.sensory_target.eq(target)].sort_values("rank")
        assert expected.compound.tolist() == [candidates[j] for j in initial_top[:, t]]
        for j in range(len(candidates)):
            existing = original[(original.sensory_target == target)
                                & (original.compound == candidates[j])]
            assert len(existing) == 1
            assert abs(float(existing.spearman_rho.iloc[0]) - rho[j, t]) < 1e-9

    labels = merged.tasting_category_fine.astype(str).to_numpy()
    # First appearance order is explicit, rather than relying on groupby sorting.
    styles = list(pd.unique(labels))
    style_indices = [np.flatnonzero(labels == style) for style in styles]
    style_sizes = [len(indices) for indices in style_indices]
    counts = np.zeros((len(candidates), len(targets)), dtype=np.int64)
    sampled_indices = np.empty((BOOTSTRAPS, len(merged)), dtype=np.int16)
    winners = np.empty((BOOTSTRAPS, 3, len(targets)), dtype=np.int16)
    rng = np.random.default_rng(SEED)
    for b in range(BOOTSTRAPS):
        rows = np.concatenate([rng.choice(indices, size=len(indices), replace=True)
                               for indices in style_indices])
        assert all(int(np.sum(labels[rows] == style)) == size
                   for style, size in zip(styles, style_sizes))
        selected = helper.top3(helper.cross_correlation(x[rows], y[rows], "spearman"))
        counts[selected, np.arange(len(targets))] += 1
        sampled_indices[b] = rows
        winners[b] = selected
    assert np.all(counts.sum(axis=0) == 3 * BOOTSTRAPS)
    frequency = counts / BOOTSTRAPS
    extra = []
    for t, target in enumerate(targets):
        for j, candidate in enumerate(candidates):
            extra.append({"sensory_target": target, "compound": candidate,
                          "stratified_top3_count": int(counts[j, t]),
                          "stratified_top3_frequency": float(frequency[j, t])})
    comparison = original.merge(pd.DataFrame(extra), on=["sensory_target", "compound"],
                                validate="one_to_one")
    comparison["frequency_difference_stratified_minus_global"] = (
        comparison.stratified_top3_frequency - comparison.bootstrap_top3_frequency)
    csv(comparison, "all_candidates_comparison.csv")
    top = comparison[comparison["rank"].le(3)].copy()
    csv(top, "original_top3_comparison.csv")
    set_rows = []
    for t, target in enumerate(targets):
        base_set = [candidates[j] for j in initial_top[:, t]]
        global_freq = original[original.sensory_target.eq(target)].set_index("compound")
        global_order = global_freq.reindex(candidates).bootstrap_top3_frequency.to_numpy()
        global_set = [candidates[j] for j in np.argsort(-global_order, kind="stable")[:3]]
        stratified_set = [candidates[j] for j in np.argsort(-frequency[:, t], kind="stable")[:3]]
        per_draw = [helper.jaccard(initial_top[:, t], winners[b, :, t])
                    for b in range(BOOTSTRAPS)]
        set_rows.append({
            "sensory_target": target,
            "original_full_sample_spearman_top3": " | ".join(base_set),
            "global_frequency_ranked_top3": " | ".join(global_set),
            "stratified_frequency_ranked_top3": " | ".join(stratified_set),
            "jaccard_original_vs_stratified_frequency": set_score(base_set, stratified_set),
            "jaccard_global_frequency_vs_stratified_frequency": set_score(global_set, stratified_set),
            "original_full_sample_set_changed": False,
            "frequency_ranked_set_changed_global_vs_stratified": set(global_set) != set(stratified_set),
            "stratified_draw_mean_jaccard_with_original": float(np.mean(per_draw)),
            "stratified_draw_exact_original_top3_fraction": float(np.mean(np.array(per_draw) == 1)),
            "original_top3_stratified_frequency_ge_0p8": int(np.sum(frequency[initial_top[:, t], t] >= .8)),
        })
    sets = pd.DataFrame(set_rows)
    csv(sets, "target_set_comparison.csv")
    cases = comparison[((comparison.sensory_target == "X4vg") & (comparison.compound == "X4VG"))
                       | ((comparison.sensory_target == "F_esters_ethac") & (comparison.compound == "ethyl_acetate"))
                       | ((comparison.sensory_target == "F_malt_all") & (comparison.compound == "ethanol..v.v."))].copy()
    assert len(cases) == 3
    csv(cases, "case_studies.csv")
    np.savez_compressed(OUT / "draw_indices_and_selections.npz", row_indices=sampled_indices,
                        selected_feature_indices=winners,
                        candidate_order=np.asarray(candidates), target_order=np.asarray(targets),
                        beer_order=merged.beer.to_numpy(), style_order=np.asarray(styles))
    after_hashes = {p.relative_to(REPOSITORY).as_posix(): sha256(p) for p in files}
    assert before_hashes == after_hashes, "An input or frozen Q2 file changed during the run."
    summary = {
        "scope": "Exploratory follow-up after the original Q2 results were examined; no original ranking or frozen result changed.",
        "seed": SEED, "bootstraps": BOOTSTRAPS,
        "resampling": "Independently resample each style with replacement, retaining its observed count; concatenate styles in their first-appearance order.",
        "n_beers": 250, "n_candidate_compounds": 218, "n_targets": 45,
        "style_counts": dict(zip(styles, style_sizes)),
        "n_styles": len(styles), "minimum_style_n": min(style_sizes),
        "singleton_styles": [style for style, size in zip(styles, style_sizes) if size == 1],
        "same_helper": "solution_a/src/03_q2_network.py::cross_correlation, top3, jaccard",
        "original_full_sample_top3_exactly_matched": True,
        "original_full_sample_ranking_changed": False,
        "original_top3_positions": len(top),
        "original_top3_global_frequency_ge_0p8": int(top.bootstrap_top3_frequency.ge(.8).sum()),
        "original_top3_stratified_frequency_ge_0p8": int(top.stratified_top3_frequency.ge(.8).sum()),
        "all_candidates_stratified_frequency_ge_0p8": int(comparison.stratified_top3_frequency.ge(.8).sum()),
        "original_top3_mean_frequency_global": float(top.bootstrap_top3_frequency.mean()),
        "original_top3_mean_frequency_stratified": float(top.stratified_top3_frequency.mean()),
        "original_top3_median_frequency_stratified": float(top.stratified_top3_frequency.median()),
        "original_top3_mean_abs_frequency_change": float(top.frequency_difference_stratified_minus_global.abs().mean()),
        "original_top3_max_abs_frequency_change": float(top.frequency_difference_stratified_minus_global.abs().max()),
        "frequency_ranked_top3_sets_changed": int(sets.frequency_ranked_set_changed_global_vs_stratified.sum()),
        "frequency_ranked_top3_mean_jaccard_global_vs_stratified": float(sets.jaccard_global_frequency_vs_stratified_frequency.mean()),
        "targets_whose_stratified_frequency_ranked_set_differs_from_original": int(sets.jaccard_original_vs_stratified_frequency.lt(1).sum()),
        "original_vs_stratified_frequency_ranked_mean_jaccard": float(sets.jaccard_original_vs_stratified_frequency.mean()),
        "stratified_draw_mean_jaccard_with_original": float(sets.stratified_draw_mean_jaccard_with_original.mean()),
        "max_single_frequency_monte_carlo_standard_error": float(np.sqrt(.25 / BOOTSTRAPS)),
        "frequency_set_definition": "Top three ranked by bootstrap inclusion frequency, input-column-order tie break. This sensitivity summary is not a replacement for the original full-sample Spearman Top-3.",
        "cases": cases[["sensory_target", "compound", "rank", "spearman_rho", "bootstrap_top3_frequency", "stratified_top3_frequency", "frequency_difference_stratified_minus_global", "within_style_spearman_rho"]].to_dict("records"),
        "input_sha256_before": before_hashes, "input_sha256_after": after_hashes,
        "inputs_and_frozen_q2_unchanged": before_hashes == after_hashes,
        "python": platform.python_version(),
        "package_versions": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn"]},
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "Preserving the observed style mixture does not remove between-style confounding.",
            "The real sample count remains 250; bootstrap draws are not new beers.",
            "Rare styles carry little information about within-style uncertainty; a singleton style would always resample its sole row and cannot estimate within-style variation. There are no singleton styles in these data (minimum n=3).",
            "The identical seed does not make these draws paired with the old global-row bootstrap, because their sampling algorithms differ.",
            "Differences of a few percentage points are subject to Monte Carlo variation, and no significance claim is made for a frequency difference.",
            "No causal inference, false-discovery guarantee, or held-out predictive validation is provided by this experiment.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ["input_sha256_before", "input_sha256_after", "style_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
