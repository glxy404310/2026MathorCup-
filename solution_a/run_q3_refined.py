"""Reproduce beer81 diagnosis and balanced transfer without changing prior runs."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from run_q3_domain import preserve_previous

HERE=Path(__file__).resolve().parent
PRIOR=HERE/"outputs/q3_domain"
SNAPSHOT=HERE/"outputs/q3_refined/previous_hashes.json"


def preserve_refined():
    preserve_previous()
    if not (PRIOR/"verification.json").exists():
        raise RuntimeError("Run run_q3_domain.py before the refinement experiment.")
    files=[{"path":path.relative_to(PRIOR).as_posix(),"bytes":path.stat().st_size,
            "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
           for path in sorted(PRIOR.rglob("*")) if path.is_file()]
    files.sort(key=lambda row: row["path"])
    if SNAPSHOT.exists():
        if files!=json.loads(SNAPSHOT.read_text(encoding="utf-8"))["files"]:
            raise RuntimeError("The frozen domain experiment changed; restore it or use a separate copy.")
    else:
        SNAPSHOT.parent.mkdir(parents=True,exist_ok=True)
        SNAPSHOT.write_text(json.dumps({"baseline_directory":"solution_a/outputs/q3_domain",
            "hash_algorithm":"sha256","file_count":len(files),"files":files},ensure_ascii=False,indent=2)+"\n",encoding="utf-8", newline="\n")


if __name__=="__main__":
    preserve_refined()
    for name in ["21_q3_beer81_audit.py","22_q3_stale_refined.py","23_q3_transfer_balanced.py",
                 "24_q3_refinement_report.py","25_q3_refined_verify.py"]:
        print(f"Running {name}",flush=True)
        command = [sys.executable, "-X", "utf8", str(HERE/"src"/name)]
        if name == "25_q3_refined_verify.py":
            command += ["--output", str(HERE/"outputs/q3_refined/verification.json")]
        subprocess.run(command, check=True)
