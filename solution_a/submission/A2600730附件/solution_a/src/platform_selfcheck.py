"""Check portable paths, source hashes and frozen-history tamper detection."""
import hashlib
import importlib
import io
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys
import tempfile
import unittest
from unittest.mock import patch

from common import SOLUTION, resolve_recorded_path
import common

sys.path.insert(0, str(SOLUTION))
OPTIMIZED = importlib.import_module("run_q3_optimized")
REFINED = importlib.import_module("run_q3_refined")
RUNNER = importlib.import_module("run_all")


class PlatformChecks(unittest.TestCase):
    def test_working_directory_cannot_shadow_project_sources(self):
        with tempfile.TemporaryDirectory(prefix="mathorcup-paths-") as temporary:
            project, shadow = Path(temporary) / "project", Path(temporary) / "shadow"
            for folder, data in [(project, b"project source"), (shadow, b"unrelated source")]:
                (folder / "solution_a").mkdir(parents=True)
                (folder / "solution_a/example.txt").write_bytes(data)
            previous = Path.cwd()
            try:
                os.chdir(shadow)
                with patch.object(common, "ROOT", project), patch.object(common, "SOLUTION", project / "solution_a"):
                    expected = project / "solution_a/example.txt"
                    self.assertEqual(resolve_recorded_path("solution_a/example.txt"), expected)
                    self.assertEqual(resolve_recorded_path("/Users/old/repo/solution_a/example.txt"), expected)
            finally:
                os.chdir(previous)

    def test_rebuild_refuses_existing_results(self):
        with tempfile.TemporaryDirectory(prefix="mathorcup-runner-") as temporary:
            root = Path(temporary)
            (root / "outputs").mkdir()
            saved = root / "outputs/saved.txt"
            saved.write_bytes(b"preserve this result")
            with patch.object(RUNNER, "HERE", root), patch.object(RUNNER, "run") as execute:
                with patch.object(sys, "argv", ["run_all.py", "--rebuild"]), patch.object(sys, "stderr", io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        RUNNER.main()
                self.assertEqual(error.exception.code, 2)
                execute.assert_not_called()
            self.assertEqual(saved.read_bytes(), b"preserve this result")

    def test_clean_rebuild_orders_every_phase(self):
        with tempfile.TemporaryDirectory(prefix="mathorcup-runner-") as temporary:
            with patch.object(RUNNER, "HERE", Path(temporary)), patch.object(RUNNER, "run") as execute:
                with patch.object(sys, "argv", ["run_all.py", "--rebuild"]):
                    RUNNER.main()
                phases = [call.args[0] for call in execute.call_args_list]
            self.assertEqual(phases, ["01_audit.py", "02_q1_baseline.py", "02b_q1_robust.py",
                                      "03_q2_network.py", "04_stage_summary.py", "05_verify_outputs.py",
                                      "run_q3.py", "run_q3_optimized.py", "run_q3_domain.py", "run_q3_refined.py", "run_supplement.py"])

    def test_rebuild_refuses_existing_supplement(self):
        with tempfile.TemporaryDirectory(prefix="mathorcup-supplement-") as temporary:
            root = Path(temporary)
            saved = root / "supplement/results/sentinel.csv"
            saved.parent.mkdir(parents=True)
            saved.write_bytes(b"preserve supplementary results")
            with patch.object(RUNNER, "HERE", root), patch.object(RUNNER, "run") as execute:
                with patch.object(sys, "argv", ["run_all.py", "--rebuild"]), patch.object(sys, "stderr", io.StringIO()):
                    with self.assertRaises(SystemExit):
                        RUNNER.main()
                execute.assert_not_called()
            self.assertEqual(saved.read_bytes(), b"preserve supplementary results")

    def test_portable_relative_paths(self):
        self.assertEqual(PureWindowsPath(r"stale\summary.json").as_posix(),
                         PurePosixPath("stale/summary.json").as_posix())

    def test_all_three_frozen_histories(self):
        snapshots = [SOLUTION / "outputs" / name for name in
                     ["q3_optimized/baseline_hashes.json", "q3_domain/previous_hashes.json",
                      "q3_refined/previous_hashes.json"]]
        before = [path.read_bytes() for path in snapshots]
        REFINED.preserve_refined()
        self.assertEqual(before, [path.read_bytes() for path in snapshots])

    def test_recorded_sources_resolve_and_match(self):
        manifest = SOLUTION / "outputs/q3_refined/beer81/source_manifest.json"
        for row in json.loads(manifest.read_text(encoding="utf-8")):
            path = resolve_recorded_path(row["path"])
            self.assertTrue(path.is_file(), row["path"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row["sha256"])

    def test_changed_file_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="mathorcup-freeze-") as temporary:
            root = Path(temporary)
            baseline, snapshot = root / "q3", root / "snapshot.json"
            target = baseline / "stale/summary.json"
            target.parent.mkdir(parents=True)
            target.write_text('{"label":"检验"}\n', encoding="utf-8", newline="\n")
            raw = target.read_bytes()
            record = {"files": [{"path": "stale/summary.json", "bytes": len(raw),
                                 "sha256": hashlib.sha256(raw).hexdigest()}]}
            (baseline / "verification.json").write_text("{}\n", encoding="utf-8", newline="\n")
            raw_verification = (baseline / "verification.json").read_bytes()
            record["files"].append({"path": "verification.json", "bytes": len(raw_verification),
                                    "sha256": hashlib.sha256(raw_verification).hexdigest()})
            snapshot.write_text(json.dumps(record), encoding="utf-8", newline="\n")
            with patch.object(OPTIMIZED, "BASELINE", baseline), patch.object(OPTIMIZED, "SNAPSHOT", snapshot):
                OPTIMIZED.preserve_baseline()
                target.write_bytes(raw + b" ")
                with self.assertRaisesRegex(RuntimeError, "frozen baseline has changed"):
                    OPTIMIZED.preserve_baseline()


if __name__ == "__main__":
    unittest.main(verbosity=2)
