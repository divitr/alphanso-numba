"""
Cross-validation tests: sandwich configs that should reproduce interface results.

These tests verify that sandwich calculations with negligible intermediate layers
produce results equivalent to the corresponding interface calculation.

Reference: interface_wgpu_c (WG-Pu source on C target)
"""

import numpy as np
import pytest
from alphanso.transport import Transport
from tests.utils.comparison import compare_scalar


# Reference interface config (same as interface_wgpu_c in configs.py)
INTERFACE_REF = {
    "name": "interface_wgpu_c",
    "calc_type": "interface",
    "source_matdef": {"Pu-239": 0.93, "Pu-240": 0.06, "Pu-241": 0.01},
    "source_density": 19.8,
    "target_matdef": {"C-12": 0.989, "C-13": 0.011},
}

# Sandwich with target-material interstitial (C layer between Pu and C target).
# Since the interstitial is the same material as the target, total neutron
# production (layers + target) should match the interface result.
SANDWICH_C_INTERSTITIAL = {
    "name": "sandwich_wgpu_c_interstitial_c",
    "calc_type": "sandwich",
    "source_matdef": {"Pu-239": 0.93, "Pu-240": 0.06, "Pu-241": 0.01},
    "source_density": 19.8,
    "target_matdef": {"C-12": 0.989, "C-13": 0.011},
    "intermediate_layers": [
        {"matdef": {"C-12": 0.989, "C-13": 0.011}, "density": 2.26, "thickness": 0.0001}
    ],
}

# Sandwich with Au interstitial at negligible thickness and density.
# The areal density (1e-14 g/cm^2) is effectively vacuum, so alphas
# pass through unaffected and the result should match the interface.
SANDWICH_AU_NEGLIGIBLE = {
    "name": "sandwich_wgpu_c_interstitial_au_negligible",
    "calc_type": "sandwich",
    "source_matdef": {"Pu-239": 0.93, "Pu-240": 0.06, "Pu-241": 0.01},
    "source_density": 19.8,
    "target_matdef": {"C-12": 0.989, "C-13": 0.011},
    "intermediate_layers": [
        {"matdef": {"Au": 1.0}, "density": 1e-6, "thickness": 1e-8}
    ],
}


@pytest.fixture(scope="module")
def interface_reference():
    """Run the interface_wgpu_c calculation once for all tests."""
    return Transport.calculate(INTERFACE_REF)


class TestSandwichInterfaceEquivalence:
    """Verify sandwich configs with negligible layers match interface results."""

    @pytest.mark.integration
    def test_sandwich_c_interstitial_matches_interface(
        self, interface_reference
    ):
        """
        Sandwich with C interstitial (same material as target) should give
        the same total yield and roughly the same spectrum as the interface.
        """
        ref = interface_reference
        actual = Transport.calculate(SANDWICH_C_INTERSTITIAL)

        # Total sandwich yield = yield from layers + yield from target
        sandwich_total = actual["an_yield"]
        interface_yield = ref["an_yield"]

        # Yields should match within 1%
        passed, msg = compare_scalar(
            sandwich_total, interface_yield,
            rel_tol=0.01, name="an_yield (sandwich vs interface)"
        )
        assert passed, msg

        # Spectra should be roughly similar — use L2 norm (element-wise
        # relative error blows up on near-zero bins where the interface
        # is exactly 0 but the sandwich has a tiny nonzero value)
        actual_spec = np.array(actual["an_spectrum"])
        ref_spec = np.array(ref["an_spectrum"])
        l2_rel_err = np.linalg.norm(actual_spec - ref_spec) / np.linalg.norm(ref_spec)
        assert l2_rel_err < 0.01, (
            f"an_spectrum L2 relative error {l2_rel_err:.4e} exceeds 1% tolerance"
        )

    @pytest.mark.integration
    def test_sandwich_au_negligible_matches_interface(
        self, interface_reference
    ):
        """
        Sandwich with Au interstitial at negligible thickness/density should
        give the same yield and spectrum as the interface.
        """
        ref = interface_reference
        actual = Transport.calculate(SANDWICH_AU_NEGLIGIBLE)

        sandwich_total = actual["an_yield"]
        interface_yield = ref["an_yield"]

        # With effectively zero interstitial, yields should match very closely
        passed, msg = compare_scalar(
            sandwich_total, interface_yield,
            rel_tol=0.01, name="an_yield (sandwich vs interface)"
        )
        assert passed, msg

        # Spectrum L2 norm check (see comment in test above)
        actual_spec = np.array(actual["an_spectrum"])
        ref_spec = np.array(ref["an_spectrum"])
        l2_rel_err = np.linalg.norm(actual_spec - ref_spec) / np.linalg.norm(ref_spec)
        assert l2_rel_err < 0.01, (
            f"an_spectrum L2 relative error {l2_rel_err:.4e} exceeds 1% tolerance"
        )
