"""Build or check the supplementary post-hoc analyses in their own directory."""
import argparse
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def run(name):
    subprocess.run([sys.executable, "-X", "utf8", str(HERE / "supplement" / name)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    folder = HERE / "supplement/results"
    if args.rebuild:
        if folder.exists() and any(folder.iterdir()):
            parser.error("Supplementary results already exist; rebuild in a clean copy.")
        for name in ["q1_comparison.py", "q1_additional_checks.py", "q2_stratified_bootstrap.py", "q2_permutation.py"]:
            run(name)
    elif not (folder / "q2_permutation/summary.json").exists():
        parser.error("Supplementary results are missing; run --rebuild after the main experiments.")
    run("verify_supplement.py")


if __name__ == "__main__":
    main()
