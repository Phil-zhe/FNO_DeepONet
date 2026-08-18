from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


ROOT = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    py = sys.executable
    run([py, "scripts/generate_data.py", "--config", args.config])
    run([py, "scripts/plot_dataset.py", "--config", args.config])
    run([py, "scripts/train_deeponet.py", "--config", args.config])
    run([py, "scripts/evaluate.py", "--config", args.config])


if __name__ == "__main__":
    main()
