"""
Integration tests for data_dir optional parameters.

Tests that an_xs_data_dir, stopping_power_data_dir, and decay_data_dir
overrides in Transport.calculate() work correctly.

Strategy:
- Equivalence tests: override with the same library that defaults resolve to,
  verify identical results.
- Smoke tests: override with a different library (SRIM vs ASTAR), verify
  the calculation runs and produces valid output.
- Combined tests: multiple overrides at once.
"""

import copy
import os

import pytest

from alphanso.transport import Transport
from alphanso.data_manager import get_data_dir
from tests.integration.configs import (
    BEAM_CONFIGS, HOMOGENEOUS_CONFIGS, INTERFACE_CONFIGS, SANDWICH_CONFIGS,
)
from tests.utils.comparison import compare_results

# ---------------------------------------------------------------------------
# Data directory paths
# ---------------------------------------------------------------------------
DATA_ROOT = str(get_data_dir())
TENDL_DIR = os.path.join(DATA_ROOT, "an_xs", "TENDL")
SRIM_DIR = os.path.join(DATA_ROOT, "stopping", "SRIM")
ENDFBVIII_DIR = os.path.join(DATA_ROOT, "decay", "ENDFBVIII")


def _get_config(name, config_list):
    """Return a deep copy of the named config from a list."""
    for cfg in config_list:
        if cfg["name"] == name:
            return copy.deepcopy(cfg)
    raise ValueError(f"Config '{name}' not found")


# ===================================================================
# decay_data_dir equivalence tests (ENDFBVIII is the default)
# ===================================================================

class TestDecayDataDir:

    @pytest.mark.integration
    def test_decay_data_dir_homogeneous(self):
        config = _get_config("homog_po210_be", HOMOGENEOUS_CONFIGS)
        result_default = Transport.calculate(config)

        config_override = _get_config("homog_po210_be", HOMOGENEOUS_CONFIGS)
        config_override["decay_data_dir"] = ENDFBVIII_DIR
        result_override = Transport.calculate(config_override)

        passed, messages = compare_results(result_override, result_default, "homogeneous")
        assert passed, "\n".join(messages)

    @pytest.mark.integration
    def test_decay_data_dir_sandwich(self):
        config = _get_config("sandwich_1layer_thick", SANDWICH_CONFIGS)
        result_default = Transport.calculate(config)

        config_override = _get_config("sandwich_1layer_thick", SANDWICH_CONFIGS)
        config_override["decay_data_dir"] = ENDFBVIII_DIR
        result_override = Transport.calculate(config_override)

        passed, messages = compare_results(result_override, result_default, "sandwich")
        assert passed, "\n".join(messages)


# ===================================================================
# an_xs_data_dir equivalence tests (TENDL-only targets)
# ===================================================================

class TestAnXsDataDir:

    @pytest.mark.integration
    def test_an_xs_data_dir_homogeneous_mg(self):
        config = _get_config("homog_po_am_mg", HOMOGENEOUS_CONFIGS)
        result_default = Transport.calculate(config)

        config_override = _get_config("homog_po_am_mg", HOMOGENEOUS_CONFIGS)
        config_override["an_xs_data_dir"] = TENDL_DIR
        result_override = Transport.calculate(config_override)

        passed, messages = compare_results(result_override, result_default, "homogeneous")
        assert passed, "\n".join(messages)

    @pytest.mark.integration
    def test_an_xs_data_dir_homogeneous_p_cl(self):
        config = _get_config("homog_leu_p_cl", HOMOGENEOUS_CONFIGS)
        result_default = Transport.calculate(config)

        config_override = _get_config("homog_leu_p_cl", HOMOGENEOUS_CONFIGS)
        config_override["an_xs_data_dir"] = TENDL_DIR
        result_override = Transport.calculate(config_override)

        passed, messages = compare_results(result_override, result_default, "homogeneous")
        assert passed, "\n".join(messages)

    @pytest.mark.integration
    def test_an_xs_data_dir_homogeneous_ne(self):
        config = _get_config("homog_wgpu_au_ne", HOMOGENEOUS_CONFIGS)
        result_default = Transport.calculate(config)

        config_override = _get_config("homog_wgpu_au_ne", HOMOGENEOUS_CONFIGS)
        config_override["an_xs_data_dir"] = TENDL_DIR
        result_override = Transport.calculate(config_override)

        passed, messages = compare_results(result_override, result_default, "homogeneous")
        assert passed, "\n".join(messages)


# ===================================================================
# stopping_power_data_dir smoke tests (SRIM != ASTAR, so results differ)
# ===================================================================

def _assert_valid_result(result, calc_type):
    """Assert that a result dict has valid, non-trivial output."""
    assert isinstance(result["an_yield"], float)
    assert result["an_yield"] > 0
    assert result["an_spectrum"] is not None
    assert len(result["an_spectrum"]) > 0

    if calc_type == "homogeneous":
        assert isinstance(result["combined_yield"], float)
        assert result["combined_yield"] >= 0

    if calc_type == "sandwich":
        assert "yield_layers" in result


class TestStoppingPowerDataDir:

    @pytest.mark.integration
    def test_stopping_power_data_dir_beam(self):
        config = _get_config("beam_be9_3p5MeV", BEAM_CONFIGS)
        config["stopping_power_data_dir"] = SRIM_DIR
        result = Transport.calculate(config)
        _assert_valid_result(result, "beam")

    @pytest.mark.integration
    def test_stopping_power_data_dir_homogeneous(self):
        config = _get_config("homog_po210_be", HOMOGENEOUS_CONFIGS)
        config["stopping_power_data_dir"] = SRIM_DIR
        result = Transport.calculate(config)
        _assert_valid_result(result, "homogeneous")

    @pytest.mark.integration
    def test_stopping_power_data_dir_interface(self):
        config = _get_config("interface_po210_li", INTERFACE_CONFIGS)
        config["stopping_power_data_dir"] = SRIM_DIR
        result = Transport.calculate(config)
        _assert_valid_result(result, "interface")

    @pytest.mark.integration
    def test_stopping_power_data_dir_sandwich(self):
        config = _get_config("sandwich_3layer", SANDWICH_CONFIGS)
        config["stopping_power_data_dir"] = SRIM_DIR
        result = Transport.calculate(config)
        _assert_valid_result(result, "sandwich")


# ===================================================================
# Combined tests — multiple overrides at once
# ===================================================================

class TestAllDataDirs:

    @pytest.mark.integration
    def test_all_data_dirs_beam(self):
        config = _get_config("beam_be9_3p5MeV", BEAM_CONFIGS)
        config["an_xs_data_dir"] = TENDL_DIR
        config["stopping_power_data_dir"] = SRIM_DIR
        result = Transport.calculate(config)
        _assert_valid_result(result, "beam")

    @pytest.mark.integration
    def test_all_data_dirs_homogeneous(self):
        config = _get_config("homog_po210_be", HOMOGENEOUS_CONFIGS)
        config["an_xs_data_dir"] = TENDL_DIR
        config["stopping_power_data_dir"] = SRIM_DIR
        config["decay_data_dir"] = ENDFBVIII_DIR
        result = Transport.calculate(config)
        _assert_valid_result(result, "homogeneous")

    @pytest.mark.integration
    def test_all_data_dirs_interface(self):
        config = _get_config("interface_po210_li", INTERFACE_CONFIGS)
        config["an_xs_data_dir"] = TENDL_DIR
        config["stopping_power_data_dir"] = SRIM_DIR
        result = Transport.calculate(config)
        _assert_valid_result(result, "interface")

    @pytest.mark.integration
    def test_all_data_dirs_sandwich(self):
        config = _get_config("sandwich_3layer", SANDWICH_CONFIGS)
        config["an_xs_data_dir"] = TENDL_DIR
        config["stopping_power_data_dir"] = SRIM_DIR
        result = Transport.calculate(config)
        _assert_valid_result(result, "sandwich")
