"""Reproduce the conditional generation and constrained transfer experiments."""
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
for script in ["06_q3_sensory.py", "07_q3_generation.py", "08_q3_transfer.py",
               "09_q3_report.py", "10_q3_verify.py"]:
    print(f"Running {script}", flush=True)
    subprocess.run([sys.executable, "-X", "utf8", str(HERE / "src" / script)], check=True)
