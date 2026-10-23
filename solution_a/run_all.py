"""Reproduce all implemented experiments and saved-output checks."""
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
if (HERE / "outputs/q3_optimized/baseline_hashes.json").exists():
    # Legacy verifiers write elapsed-time metadata inside their result folders;
    # the current verifier reads those frozen experiments without rewriting.
    scripts = ["01_audit.py", "05_verify_outputs.py"]
else:
    scripts = ["01_audit.py", "02_q1_baseline.py", "02b_q1_robust.py",
               "03_q2_network.py", "04_stage_summary.py", "05_verify_outputs.py",
               "06_q3_sensory.py", "07_q3_generation.py", "08_q3_transfer.py",
               "09_q3_report.py", "10_q3_verify.py"]
for script in scripts:
    print(f"Running {script}", flush=True)
    subprocess.run([sys.executable, str(HERE / "src" / script)], check=True)
if (HERE / "outputs/q3_refined/previous_hashes.json").exists():
    from run_q3_refined import preserve_refined
    preserve_refined()
elif (HERE / "outputs/q3_domain/previous_hashes.json").exists():
    from run_q3_domain import preserve_previous
    preserve_previous()
    subprocess.run([sys.executable, str(HERE / "run_q3_domain.py")], check=True)
else:
    subprocess.run([sys.executable, str(HERE / "run_q3_optimized.py")], check=True)
    subprocess.run([sys.executable, str(HERE / "run_q3_domain.py")], check=True)
subprocess.run([sys.executable, str(HERE / "run_q3_refined.py")], check=True)
