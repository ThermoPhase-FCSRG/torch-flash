from __future__ import annotations

from pathlib import Path

import pytest
import torch

from torch_flash import ChemicalState, component_set, peng_robinson_1978


@pytest.fixture(scope="session")
def not_cleared_data() -> Path:
    """Return local research inputs, skipping consumers in public checkouts."""
    directory = Path(__file__).parent / "data" / "not-cleared"
    if not directory.is_dir():
        pytest.skip("not-cleared research data are not present in this checkout")
    return directory


@pytest.fixture
def binary_components():
    return component_set(("methane", "n_butane"))


@pytest.fixture
def binary_model(binary_components):
    return peng_robinson_1978(binary_components)


@pytest.fixture
def two_phase_state():
    return ChemicalState(
        torch.tensor(270.0, dtype=torch.float64),
        torch.tensor(3.0e6, dtype=torch.float64),
        torch.tensor([0.5, 0.5], dtype=torch.float64),
    )
