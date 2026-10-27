"""Check provenance, sample alignment, saved metrics and graph invariants.

These checks assess numerical consistency and saved-output integrity.
"""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from common import SOLUTION, data_dir, load_xy


def main(output_path=None):
    checks = []
    meta, x, y = load_xy()
    sources = pd.read_csv(SOLUTION / "outputs/audit/source_files.csv")
    for row in sources.itertuples():
        actual = hashlib.sha256((data_dir() / row.source).read_bytes()).hexdigest()
        assert actual == row.sha256, f"Source changed: {row.source}"
    checks.append(f"Original source SHA-256 unchanged: {len(sources)} files")

    reference_fold = None
    models_checked = 0
    for directory in ["q1", "q1_robust"]:
        folder = SOLUTION / "outputs" / directory
        metrics = pd.read_csv(folder / "metrics_overall.csv")
        for row in metrics.itertuples():
            oof = pd.read_csv(folder / f"oof_{row.model}.csv")
            assert oof.beer.is_unique and oof.beer.tolist() == meta.beer.tolist()
            assert (oof.tasting_category_fine == meta.tasting_category_fine).all()
            assert set(oof.fold) == set(range(1, 6))
            if reference_fold is None:
                reference_fold = oof.fold.to_numpy()
            np.testing.assert_array_equal(oof.fold, reference_fold)
            prediction = oof[y.columns].to_numpy()
            assert np.isfinite(prediction).all()
            np.testing.assert_allclose(
                [row.mae, row.rmse, row.macro_r2],
                [mean_absolute_error(y, prediction),
                 np.sqrt(mean_squared_error(y, prediction)),
                 r2_score(y, prediction, multioutput="uniform_average")],
                rtol=1e-10, atol=1e-12,
            )
            models_checked += 1
    checks.append(f"Sample alignment, folds and independently recomputed metrics: {models_checked} OOF tables")

    naive = pd.read_csv(SOLUTION / "outputs/q1/oof_mean.csv")
    for fold in range(1, 6):
        test = reference_fold == fold
        expected = np.tile(y.loc[~test].mean().to_numpy(), (test.sum(), 1))
        np.testing.assert_allclose(naive.loc[test, y.columns], expected, atol=1e-12)
    checks.append("Mean baseline equals the training-fold target mean for every held-out beer")

    bounds = pd.read_csv(SOLUTION / "outputs/q1_robust/clipping_bounds.csv")
    for fold in range(1, 6):
        stored = bounds[bounds.fold == fold].set_index("feature").loc[x.columns]
        actual = np.nanquantile(x.loc[reference_fold != fold], [.01, .99], axis=0)
        np.testing.assert_allclose(stored[["train_lower_01", "train_upper_99"]].to_numpy().T,
                                   actual, rtol=1e-10, atol=1e-12)
    checks.append("Saved outer-fold clipping bounds match only the corresponding training rows")

    for filename in ["q1/ridge_model.joblib", "q1/extra_trees_model.joblib",
                     "q1_robust/ridge_winsor_model.joblib"]:
        saved = joblib.load(SOLUTION / "outputs" / filename)
        assert saved["features"] == x.columns.tolist() and saved["targets"] == y.columns.tolist()
        prediction = saved["pipeline"].predict(x.iloc[:2])
        assert prediction.shape == (2, 45) and np.isfinite(prediction).all()
    checks.append("All three saved models reload and predict using the recorded feature/target order")

    q2 = SOLUTION / "outputs/q2"
    all_scores = pd.read_csv(q2 / "all_candidate_stability.csv")
    top = pd.read_csv(q2 / "top3_compounds.csv")
    nodes = pd.read_csv(q2 / "network_nodes.csv")
    summary = json.loads((q2 / "summary.json").read_text(encoding="utf-8"))
    assert len(all_scores) == 218 * 45 and len(top) == 3 * 45 and len(nodes) == 231
    assert not all_scores.duplicated(["sensory_target", "compound"]).any()
    assert (top.groupby("sensory_target").size() == 3).all()
    np.testing.assert_allclose(all_scores.groupby("sensory_target").bootstrap_top3_frequency.sum(), 3)
    assert all_scores.bootstrap_top3_frequency.between(0, 1).all()
    assert (all_scores.bootstrap_valid_draws == summary["bootstrap_draws"]).all()
    for threshold in [.6, .7, .8]:
        edges = pd.read_csv(q2 / f"network_edges_rho_{threshold:.1f}.csv")
        assert not (edges.source == edges.target).any()
        assert not edges.duplicated(["source", "target"]).any()
        assert (edges.absolute_weight >= threshold - 1e-10).all()
        assert int(nodes[f"degree_{threshold:.1f}"].sum()) == 2 * len(edges)
    checks.append("Q2 candidate counts, bootstrap frequencies and undirected graph degree sums reconcile")

    for path in ["docs/stage1_results.md", "docs/data_audit.md", "docs/sensory_mapping_review.md",
                 "docs/q1_validation_review.md", "docs/q2_q3_design_review.md",
                 "outputs/figures/q1_baseline_and_missingness.png", "README.md"]:
        assert (SOLUTION / path).is_file(), f"Missing deliverable: {path}"
    checks.append("Stage notes, independent review and chart files exist")
    result = {"status": "passed", "checks": checks,
              "scope": "Saved-output integrity and numerical consistency; not proof of model adequacy"}
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional report path; existing results are read-only by default.")
    main(parser.parse_args().output)
