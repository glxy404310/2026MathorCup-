"""Reproduce style-held-out prediction and joint committee transfer checks."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from run_q3_optimized import preserve_baseline

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE / "outputs/q3_optimized"
SNAPSHOT = HERE / "outputs/q3_domain/previous_hashes.json"


def preserve_previous():
    preserve_baseline()
    if not (PREVIOUS / "verification.json").exists():
        raise RuntimeError("Run run_q3_optimized.py before the style-held-out experiment.")
    files = [{"path":path.relative_to(PREVIOUS).as_posix(),"bytes":path.stat().st_size,
              "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
             for path in sorted(PREVIOUS.rglob("*")) if path.is_file()]
    files.sort(key=lambda row: row["path"])
    if SNAPSHOT.exists():
        if files != json.loads(SNAPSHOT.read_text(encoding="utf-8"))["files"]:
            raise RuntimeError("The frozen optimization has changed; restore it or use a separate experiment copy.")
    else:
        SNAPSHOT.parent.mkdir(parents=True,exist_ok=True)
        SNAPSHOT.write_text(json.dumps({"baseline_directory":"solution_a/outputs/q3_optimized",
            "hash_algorithm":"sha256","file_count":len(files),"files":files},
            ensure_ascii=False,indent=2)+"\n",encoding="utf-8", newline="\n")


if __name__ == "__main__":
    preserve_previous()
    for script in ["16_q3_stale_domain.py","17_q3_transfer_domain.py","20_q3_transfer_consensus.py",
                   "18_q3_domain_report.py","19_q3_domain_verify.py"]:
        print(f"Running {script}",flush=True)
        subprocess.run([sys.executable, "-X", "utf8", str(HERE/"src"/script)],check=True)
