"""
Integration tests for ALPHANSO sandwich calculations.

These tests compare current calculation results against validated
ground truth to ensure no regressions have occurred.
"""

import pytest
from alphanso.transport import Transport
from tests.integration.configs import SANDWICH_CONFIGS
from tests.utils.comparison import compare_results


class TestSandwichCalculations:
    """Integration tests for sandwich problem calculations."""

    @pytest.mark.integration
    @pytest.mark.parametrize("config", SANDWICH_CONFIGS, ids=lambda c: c['name'])
    def test_sandwich_calculation(self, config, sandwich_ground_truth):
        """
        Test that sandwich calculations match ground truth within tolerance.

        Compares:
        - an_yield: Total neutron yield (n/s/cm^2)
        - yield_target: Neutrons from target region
        - yield_layers: Per-layer yield breakdown
        - an_spectrum: Normalized neutron energy spectrum
        """
        name = config['name']

        # Skip if no ground truth for this config
        if name not in sandwich_ground_truth:
            pytest.skip(f"No ground truth for {name}")

        expected = sandwich_ground_truth[name]

        # Check for error in ground truth
        if 'error' in expected:
            pytest.skip(f"Ground truth recorded error: {expected['error']}")

        # Run calculation
        actual = Transport.calculate(config)

        # Compare results
        passed, messages = compare_results(actual, expected, 'sandwich')

        # Build assertion message
        full_message = f"\n{name} comparison results:\n" + "\n".join(f"  {m}" for m in messages)

        assert passed, full_message
