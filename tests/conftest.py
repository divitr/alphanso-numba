"""
Pytest configuration and fixtures for ALPHANSO integration tests.
"""

import json
import pytest
from pathlib import Path


# Ground truth directory
GROUND_TRUTH_DIR = Path(__file__).parent / "ground_truth" / "results"


@pytest.fixture(scope="session", autouse=True)
def ensure_nuclear_data():
    """Ensure nuclear data is available before any tests run."""
    from alphanso.data_manager import ensure_data
    ensure_data()


@pytest.fixture(scope="session")
def beam_ground_truth():
    """Load beam ground truth data."""
    path = GROUND_TRUTH_DIR / "beam.json"
    if not path.exists():
        pytest.skip("Beam ground truth not found. Run generate_ground_truth.py first.")
    with open(path) as f:
        data = json.load(f)
    return data['results']


@pytest.fixture(scope="session")
def homogeneous_ground_truth():
    """Load homogeneous ground truth data."""
    path = GROUND_TRUTH_DIR / "homogeneous.json"
    if not path.exists():
        pytest.skip("Homogeneous ground truth not found. Run generate_ground_truth.py first.")
    with open(path) as f:
        data = json.load(f)
    return data['results']


@pytest.fixture(scope="session")
def interface_ground_truth():
    """Load interface ground truth data."""
    path = GROUND_TRUTH_DIR / "interface.json"
    if not path.exists():
        pytest.skip("Interface ground truth not found. Run generate_ground_truth.py first.")
    with open(path) as f:
        data = json.load(f)
    return data['results']


@pytest.fixture(scope="session")
def sandwich_ground_truth():
    """Load sandwich ground truth data."""
    path = GROUND_TRUTH_DIR / "sandwich.json"
    if not path.exists():
        pytest.skip("Sandwich ground truth not found. Run generate_ground_truth.py first.")
    with open(path) as f:
        data = json.load(f)
    return data['results']
