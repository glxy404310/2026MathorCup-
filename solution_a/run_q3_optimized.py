"""Reproduce the optimization while preserving the frozen first-round baseline."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BASELINE = HERE / "outputs/q3"
SNAPSHOT = HERE / "outputs/q3_optimized/baseline_hashes.json"


def preserve_baseline():
    paths = sorted(path for path in BASELINE.rglob("*") if path.is_file())
    if not (BASELINE / "verification.json").exists():
        raise RuntimeError("Run the original third-question experiment before the optimization.")
    rows = [{"path": str(path.relative_to(BASELINE)), "bytes": path.stat().st_size,
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]
    if SNAPSHOT.exists():
        expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
        if rows != expected["files"]:
            raise RuntimeError("The frozen baseline has changed. Restore it or run a separate experiment copy.")
    else:
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        record = {"baseline_directory": "solution_a/outputs/q3", "hash_algorithm": "sha256",
                  "file_count": len(rows), "files": rows}
        SNAPSHOT.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    preserve_baseline()
    for script in ["11_q3_generation_optimized.py", "12_q3_stale_optimized.py",
                   "12b_q3_stale_fixed.py", "13_q3_transfer_optimized.py",
                   "14_q3_optimization_report.py", "15_q3_optimized_verify.py"]:
        print(f"Running {script}", flush=True)
        subprocess.run([sys.executable, str(HERE / "src" / script)], check=True)
