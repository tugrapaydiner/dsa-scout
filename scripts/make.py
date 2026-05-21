from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TARGETS = {
    "install": [sys.executable, "-m", "pip", "install", "-e", "."],
    "test": [
        sys.executable,
        "-m",
        "pytest",
        "--cov=dsa_scout",
        "--cov-report=term",
        "--cov-fail-under=80",
    ],
    "lint": [sys.executable, "-m", "ruff", "check", "dsa_scout/", "tests/", "scripts/"],
    "type": [sys.executable, "-m", "mypy", "--strict", "dsa_scout/", "scripts/"],
    "smoke": [sys.executable, "-m", "dsa_scout.cli", "smoke"],
    "reproduce": [sys.executable, "-m", "dsa_scout.cli", "reproduce"],
    "release-check": [sys.executable, "scripts/verify_release.py"],
    "clean": [],
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in TARGETS:
        print(f"usage: make [{'|'.join(TARGETS)}]", file=sys.stderr)
        return 2
    target = sys.argv[1]
    if target == "clean":
        for directory in [".pytest_cache", ".mypy_cache", ".ruff_cache", "build", "dist"]:
            path = Path(directory)
            if path.exists():
                import shutil

                shutil.rmtree(path, ignore_errors=True)
        for pattern in ["plots/*.png", "plots/*.svg", "results/*.json", "results/*.pt"]:
            for path in Path(".").glob(pattern):
                path.unlink()
        return 0
    commands = [TARGETS[target]]
    if target == "lint":
        commands.append(
            [
                sys.executable,
                "-m",
                "ruff",
                "format",
                "--check",
                "dsa_scout/",
                "tests/",
                "scripts/",
            ]
        )
    for command in commands:
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
