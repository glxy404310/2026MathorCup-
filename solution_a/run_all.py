"""Check saved experiments, or explicitly rebuild them in a clean copy."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def run(script, *arguments):
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    path = HERE / (script if script.startswith("run_") else "src/" + script)
    print(f"Running {script}", flush=True)
    subprocess.run([sys.executable, "-X", "utf8", str(path), *map(str, arguments)],
                   check=True, env=env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true",
                        help="Train all experiments; requires an absent or empty outputs directory.")
    parser.add_argument("--report-dir", type=Path, default=HERE / "review/verification",
                        help="Reports from saved-result checks (outside frozen experiment folders).")
    args = parser.parse_args()
    outputs = HERE / "outputs"
    if args.rebuild:
        supplement = HERE / "supplement/results"
        if (outputs.exists() and any(outputs.iterdir())) or (supplement.exists() and any(supplement.iterdir())):
            parser.error("--rebuild requires a clean copy with no existing outputs; saved results were left unchanged.")
        for script in ["01_audit.py", "02_q1_baseline.py", "02b_q1_robust.py",
                       "03_q2_network.py", "04_stage_summary.py"]:
            run(script)
        run("05_verify_outputs.py", "--output", outputs / "verification.json")
        for script in ["run_q3.py", "run_q3_optimized.py", "run_q3_domain.py", "run_q3_refined.py"]:
            run(script)
        run("run_supplement.py", "--rebuild")
    else:
        if not (outputs / "q3_refined/verification.json").is_file():
            parser.error("Saved experiments are incomplete; use --rebuild in a clean copy with the original attachments.")
        if args.report_dir.resolve().is_relative_to(outputs.resolve()):
            parser.error("--report-dir must be outside outputs to preserve the cited experiments.")
        run("platform_selfcheck.py")
        run("05_verify_outputs.py", "--output", args.report_dir / "q1_q2.json")
        run("25_q3_refined_verify.py", "--output", args.report_dir / "q3_refined.json")
        run("run_supplement.py")


if __name__ == "__main__":
    main()
