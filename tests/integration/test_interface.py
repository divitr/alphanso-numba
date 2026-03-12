"""
Integration tests for ALPHANSO interface calculations.

These tests compare current calculation results against validated
ground truth to ensure no regressions have occurred.
"""

import pytest
from alphanso.transport import Transport
from tests.integration.configs import INTERFACE_CONFIGS
from tests.utils.comparison import compare_results


class TestInterfaceCalculations:
    """Integration tests for interface problem calculations."""

    @pytest.mark.integration
    @pytest.mark.parametrize("config", INTERFACE_CONFIGS, ids=lambda c: c['name'])
    def test_interface_calculation(self, config, interface_ground_truth):
        """
        Test that interface calculations match ground truth within tolerance.

        Compares:
        - an_yield: Neutron yield per cm^2 per second
        - an_spectrum: Normalized neutron energy spectrum
        """
        name = config['name']

        # Skip if no ground truth for this config
        if name not in interface_ground_truth:
            pytest.skip(f"No ground truth for {name}")

        expected = interface_ground_truth[name]

        # Check for error in ground truth
        if 'error' in expected:
            pytest.skip(f"Ground truth recorded error: {expected['error']}")

        # Run calculation
        actual = Transport.calculate(config)

        # Compare results
        passed, messages = compare_results(actual, expected, 'interface')

        # Build assertion message
        full_message = f"\n{name} comparison results:\n" + "\n".join(f"  {m}" for m in messages)

        assert passed, full_message
