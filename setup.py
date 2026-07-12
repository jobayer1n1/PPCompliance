#!/usr/bin/env python3
"""Bootstrap the PPCompliance development environment.

This script installs the Python packages used across the repo and the Node.js
modules required by the practice analyzer.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PRACTICE_ANALYZER_DIR = ROOT / "PP_analyzer" / "practice_analyzer"

PYTHON_PACKAGES = [
    "pandas",
    "numpy",
    "scikit-learn",
    "nltk",
    "torch",
    "torchvision",
    "torchaudio",
    "pytorch-lightning<2",
    "torchmetrics",
    "transformers",
    "beautifulsoup4",
    "tqdm",
    "xlwt",
]

OPTIONAL_PYTHON_PACKAGES = [
    "tensorflow",
]

NODE_PACKAGES = [
    "esprima",
    "estraverse",
]

NLTK_RESOURCES = [
    "punkt",
    "averaged_perceptron_tagger",
    "stopwords",
    "wordnet",
    "omw-1.4",
]


def run(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def install_python_packages() -> None:
    print("Installing Python packages...")
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    run([sys.executable, "-m", "pip", "install", *PYTHON_PACKAGES])

    for package in OPTIONAL_PYTHON_PACKAGES:
        if package == "tensorflow" and sys.version_info >= (3, 12):
            print(
                "Skipping tensorflow: it is not published for "
                f"Python {sys.version_info.major}.{sys.version_info.minor}."
            )
            continue

        print(f"Installing optional package: {package}")
        try:
            run([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as exc:
            print(f"Warning: failed to install {package} ({exc}); continuing.")


def install_nltk_data() -> None:
    print("Downloading NLTK resources...")
    code = (
        "import nltk\n"
        f"resources = {NLTK_RESOURCES!r}\n"
        "for resource in resources:\n"
        "    nltk.download(resource)\n"
    )
    run([sys.executable, "-c", code])


def install_node_packages() -> None:
    print("Installing Node.js packages for the practice analyzer...")
    if not PRACTICE_ANALYZER_DIR.exists():
        raise FileNotFoundError(f"Missing directory: {PRACTICE_ANALYZER_DIR}")

    npm_executable = shutil.which("npm") or shutil.which("npm.cmd")
    if npm_executable is None:
        print("Skipping Node.js packages: npm was not found on PATH.")
        print("Install Node.js first, then rerun this script or use --skip-node.")
        return

    run([npm_executable, "install", "--no-save", *NODE_PACKAGES], cwd=PRACTICE_ANALYZER_DIR)


def main() -> int:
    parser = argparse.ArgumentParser(description="Set up the PPCompliance environment.")
    parser.add_argument(
        "--skip-nltk",
        action="store_true",
        help="Skip downloading NLTK corpora.",
    )
    parser.add_argument(
        "--skip-node",
        action="store_true",
        help="Skip installing Node.js packages.",
    )
    args = parser.parse_args()

    install_python_packages()

    if not args.skip_nltk:
        install_nltk_data()

    if not args.skip_node:
        install_node_packages()

    print("Setup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
