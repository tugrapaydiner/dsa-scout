"""Typer CLI for DSA-Scout."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

app = typer.Typer(help="DSA-Scout experiments and analysis.")
console = Console()

BaseDirOption = Annotated[Path, typer.Option(help="Repository/output root.")]


@app.command()
def reproduce(
    base_dir: BaseDirOption = Path("."),
    steps: Annotated[int, typer.Option(help="Training steps.")] = 2000,
    max_length: Annotated[int, typer.Option(help="Maximum GPT-2 tokens per text.")] = 1024,
    skip_training: Annotated[
        bool,
        typer.Option("--skip-training", help="Reuse an existing trained checkpoint."),
    ] = False,
) -> None:
    """Run the full DSA-Scout study, optionally reusing an existing checkpoint."""
    from dsa_scout.experiment import run_full_study

    summary = run_full_study(
        base_dir=base_dir,
        force_train=not skip_training,
        training_steps=steps,
        max_length=max_length,
    )
    console.print_json(json.dumps(summary["summary_stats"]))


@app.command()
def smoke(base_dir: BaseDirOption = Path(".")) -> None:
    """Run a quick GPT-2 and plotting smoke test."""
    from dsa_scout.experiment import run_smoke

    result = run_smoke(base_dir=base_dir)
    console.print_json(json.dumps(result))


if __name__ == "__main__":
    app()
