# Example Python script to run an ALPHANSO calculation directly

from alphanso.transport import Transport
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# --- Beam Calculation ---
# This example demonstrates a simple beam calculation.

print("Running Beam Calculation...")
beam_config = {
    "name": "Example Beam Calculation from Script",
    "calc_type": "beam",
    "matdef": {
        "Be-9": 1.0
    },
    "beam_energy": 5.0,
    "output_dir": "./beam_example_output",
    # "num_alpha_groups": 15000,
    # "min_alpha_energy": 1e-11,
    # "max_alpha_energy": 15,
}

beam_results = Transport.calculate(beam_config)
print(beam_results)

# --- Homogeneous Calculation ---
# This example demonstrates a homogeneous calculation with a UO2-like material.

print("\nRunning Homogeneous Calculation...")
homogeneous_config = {
    "name": "Example Homogeneous Calculation from Script",
    "calc_type": "homogeneous",
    "matdef": {
        92235: 0.5,   # U-235 (ZAID format: ZZZAAA)
        92238: 0.35,  # U-238
        8000: 0.15    # Natural oxygen (AAA=000 for natural abundances)
    },
    "output_dir": "./homogeneous_example_output"
}

homogeneous_results = Transport.calculate(homogeneous_config)
print(f"Alpha-n yield:        {homogeneous_results['an_yield']:.3e} n/s/g")
print(f"SF prompt yield:      {homogeneous_results['sf_yield']:.3e} n/s/g")
print(f"SF delayed yield:     {homogeneous_results['delayedn_strength']:.3e} n/s/g")
print(f"Combined yield:       {homogeneous_results['combined_yield']:.3e} n/s/g")

# --- Interface Calculation ---
# This example demonstrates an interface calculation with Pu-238 source and Be-9 target.

print("\nRunning Interface Calculation...")
interface_config = {
    "name": "Example Interface Calculation from Script",
    "calc_type": "interface",
    "source_matdef": {
        "Pu-238": 1.0
    },
    "source_density": 19.8,  # g/cm^3
    "target_matdef": {
        "Be-9": 1.0
    },
    "output_dir": "./interface_example_output"
}

interface_results = Transport.calculate(interface_config)
print(f"Interface yield: {interface_results['an_yield']:.3e} n/s/cm^2")

# --- Sandwich Calculation ---
# This example demonstrates a multi-layer sandwich calculation.

print("\nRunning Sandwich Calculation...")
sandwich_config = {
    "name": "Example Multi-Layer Sandwich from Script",
    "calc_type": "sandwich",
    "source_matdef": {
        "Pu-238": 1.0
    },
    "source_density": 19.8,  # g/cm^3
    "target_matdef": {
        "Be-9": 1.0
    },
    "intermediate_layers": [
        {
            "matdef": {"C": 1.0}, # Natural abundances can be input by omitting atomic mass
            "density": 2.26,
            "thickness": 1.0e-4  # 1 micron
        },
        {
            "matdef": {"Al-27": 1.0},
            "density": 2.70,
            "thickness": 1.0e-4  # 1 micron
        }
    ],
    "output_dir": "./sandwich_example_output",
    # "num_alpha_groups": 15000,
    # "min_alpha_energy": 1e-11,
    # "max_alpha_energy": 15,
}

sandwich_results = Transport.calculate(sandwich_config)
print(f"Sandwich total yield: {sandwich_results['an_yield']:.3e} n/s/cm^2")
print(f"Target yield: {sandwich_results['yield_target']:.3e} n/s/cm^2")
print(f"Per-layer yields: {[f'{y:.3e}' for y in sandwich_results['yield_layers']]}")
