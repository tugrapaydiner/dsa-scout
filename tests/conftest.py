from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture(scope="session")
def small_hidden(device: torch.device) -> torch.Tensor:
    torch.manual_seed(int(os.environ.get("DSA_SCOUT_TEST_SEED", "0")))
    return torch.randn(32, 768, device=device)


@pytest.fixture()
def tmp_png(tmp_path: Path) -> Path:
    return tmp_path / "plot.png"
