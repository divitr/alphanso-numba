"""
Test configurations for ALPHANSO integration tests.

30 total configurations:
- 5 beam problems
- 10 homogeneous problems
- 5 interface problems
- 10 sandwich problems

Materials covered:
- Targets: Be, C, O, Al, Ne, Si, Mg, P, Cl, Li
- Sources: LEU, HEU, WG-Pu, Po-210, Am-241, Ra-223
- Inerts: Pb, Au
"""

# =============================================================================
# BEAM PROBLEMS (5)
# =============================================================================

BEAM_CONFIGS = [
    # Test 1: Be-9 target with single beam_energy (3.5 MeV)
    {
        "name": "beam_be9_3p5MeV",
        "calc_type": "beam",
        "matdef": {"Be-9": 1.0},
        "beam_energy": 3.5,
    },

    # Test 2: Carbon target mixture with beam_intensities (multi-energy)
    {
        "name": "beam_c_multi",
        "calc_type": "beam",
        "matdef": {"C-12": 0.989, "C-13": 0.011},
        "beam_intensities": [[3.0, 0.3], [4.0, 0.4], [5.0, 0.3]],
    },

    # Test 3: Oxygen target with low energy beam (0.5 MeV)
    {
        "name": "beam_o_low",
        "calc_type": "beam",
        "matdef": {"O-17": 0.5, "O-18": 0.5},
        "beam_energy": 0.5,
    },

    # Test 4: Mixed Li/Be target with beam_intensities (4-5 MeV range)
    {
        "name": "beam_li_be_mix",
        "calc_type": "beam",
        "matdef": {"Li-7": 0.4, "Be-9": 0.6},
        "beam_intensities": [[4.0, 0.25], [4.5, 0.5], [5.0, 0.25]],
    },

    # Test 5: Al target with high energy beam (8 MeV)
    {
        "name": "beam_al27_high",
        "calc_type": "beam",
        "matdef": {"Al-27": 1.0},
        "beam_energy": 8.0,
    }
]


# =============================================================================
# HOMOGENEOUS PROBLEMS (10)
# =============================================================================

HOMOGENEOUS_CONFIGS = [
    # Test 1: Po-210 in Be matrix (classic PoBe source)
    {
        "name": "homog_po210_be",
        "calc_type": "homogeneous",
        "matdef": {"Po-210": 0.01, "Be-9": 0.99},
    },

    # Test 2: Am-241 in Be/O matrix
    {
        "name": "homog_am241_be_o",
        "calc_type": "homogeneous",
        "matdef": {"Am-241": 0.02, "Be-9": 0.48, "O-17": 0.5},
    },

    # Test 3: Low enriched uranium (LEU) in Li target
    {
        "name": "homog_leu_li",
        "calc_type": "homogeneous",
        "matdef": {"U-235": 0.015, "U-238": 0.385, "Li-7": 0.6},
    },

    # Test 4: Highly enriched uranium (HEU) in C matrix
    {
        "name": "homog_heu_c",
        "calc_type": "homogeneous",
        "matdef": {"U-235": 0.093, "U-238": 0.007, "C-12": 0.9},
    },

    # Test 5: Weapons-grade Pu in Al matrix
    {
        "name": "homog_wgpu_al",
        "calc_type": "homogeneous",
        "matdef": {"Pu-239": 0.093, "Pu-240": 0.006, "Pu-241": 0.001, "Al-27": 0.9},
    },

    # Test 6: Ra-223 in Si matrix
    {
        "name": "homog_ra223_si",
        "calc_type": "homogeneous",
        "matdef": {"Ra-223": 0.01, "Si-28": 0.99},
    },

    # Test 7: Mixed source (Po-210 + Am-241) in Mg matrix
    {
        "name": "homog_po_am_mg",
        "calc_type": "homogeneous",
        "matdef": {"Po-210": 0.005, "Am-241": 0.005, "Mg-24": 0.99},
    },

    # Test 8: HEU with inert Pb in Be
    {
        "name": "homog_heu_pb_be",
        "calc_type": "homogeneous",
        "matdef": {"U-235": 0.05, "U-238": 0.004, "Pb": 0.446, "Be-9": 0.5},
    },

    # Test 9: LEU in P/Cl matrix
    {
        "name": "homog_leu_p_cl",
        "calc_type": "homogeneous",
        "matdef": {"U-235": 0.014, "U-238": 0.386, "P-31": 0.3, "Cl-35": 0.3},
    },

    # Test 10: WG-Pu with Au in Ne matrix
    {
        "name": "homog_wgpu_au_ne",
        "calc_type": "homogeneous",
        "matdef": {"Pu-239": 0.047, "Pu-240": 0.003, "Au": 0.15, "Ne-20": 0.8},
    }
]


# =============================================================================
# INTERFACE PROBLEMS (5)
# =============================================================================

INTERFACE_CONFIGS = [
    # Test 1: Am-241 source on Be target
    {
        "name": "interface_am241_be",
        "calc_type": "interface",
        "source_matdef": {"Am-241": 1.0},
        "source_density": 13.67,
        "target_matdef": {"Be-9": 1.0},
    },

    # Test 2: Po-210 source on Li target
    {
        "name": "interface_po210_li",
        "calc_type": "interface",
        "source_matdef": {"Po-210": 1.0},
        "source_density": 9.196,
        "target_matdef": {"Li-7": 1.0},
    },

    # Test 3: WG-Pu source on C target
    {
        "name": "interface_wgpu_c",
        "calc_type": "interface",
        "source_matdef": {"Pu-239": 0.93, "Pu-240": 0.06, "Pu-241": 0.01},
        "source_density": 19.8,
        "target_matdef": {"C-12": 0.989, "C-13": 0.011},
    },

    # Test 4: HEU source (with Pb) on Al target
    {
        "name": "interface_heu_pb_al",
        "calc_type": "interface",
        "source_matdef": {"U-235": 0.8, "U-238": 0.06, "Pb": 0.14},
        "source_density": 18.5,
        "target_matdef": {"Al-27": 1.0},
    },

    # Test 5: Ra-223 source on Si/Mg mixed target
    {
        "name": "interface_ra223_si_mg",
        "calc_type": "interface",
        "source_matdef": {"Ra-223": 1.0},
        "source_density": 5.5,
        "target_matdef": {"Si-28": 0.7, "Mg-24": 0.3},
    }
]


# =============================================================================
# SANDWICH PROBLEMS (10)
# =============================================================================

SANDWICH_CONFIGS = [
    # Test 1: Single thin layer (Am-241 | Al | Be)
    {
        "name": "sandwich_1layer_thin",
        "calc_type": "sandwich",
        "source_matdef": {"Am-241": 1.0},
        "source_density": 13.67,
        "target_matdef": {"Be-9": 1.0},
        "intermediate_layers": [
            {"matdef": {"Al-27": 1.0}, "density": 2.70, "thickness": 0.001}
        ],
    },

    # Test 2: Single thick layer (Po-210 | Pb | Li)
    {
        "name": "sandwich_1layer_thick",
        "calc_type": "sandwich",
        "source_matdef": {"Po-210": 1.0},
        "source_density": 9.196,
        "target_matdef": {"Li-7": 1.0},
        "intermediate_layers": [
            {"matdef": {"Pb": 1.0}, "density": 11.34, "thickness": 0.005}
        ],
    },

    # Test 3: Two thin layers (WG-Pu | C | Al | Be)
    {
        "name": "sandwich_2layer",
        "calc_type": "sandwich",
        "source_matdef": {"Pu-239": 0.93, "Pu-240": 0.06, "Pu-241": 0.01},
        "source_density": 19.8,
        "target_matdef": {"Be-9": 1.0},
        "intermediate_layers": [
            {"matdef": {"C-12": 1.0}, "density": 2.26, "thickness": 0.0002},
            {"matdef": {"Al-27": 1.0}, "density": 2.70, "thickness": 0.0002}
        ],
    },

    # Test 4: Three ultra-thin layers (HEU | Au | Pb | C | O)
    {
        "name": "sandwich_3layer",
        "calc_type": "sandwich",
        "source_matdef": {"U-235": 0.93, "U-238": 0.07},
        "source_density": 19.0,
        "target_matdef": {"O-17": 0.5, "O-18": 0.5},
        "intermediate_layers": [
            {"matdef": {"Au": 1.0}, "density": 19.3, "thickness": 0.0001},
            {"matdef": {"Pb": 1.0}, "density": 11.34, "thickness": 0.0001},
            {"matdef": {"C-12": 1.0}, "density": 2.26, "thickness": 0.0001}
        ],
    },

    # Test 5: Single layer with mixed intermediate (LEU | Si/Mg | Al)
    {
        "name": "sandwich_mixed_layer",
        "calc_type": "sandwich",
        "source_matdef": {"U-235": 0.035, "U-238": 0.965},
        "source_density": 18.95,
        "target_matdef": {"Al-27": 1.0},
        "intermediate_layers": [
            {"matdef": {"Si-28": 0.6, "Mg-24": 0.4}, "density": 2.5, "thickness": 0.002}
        ],
    },

    # Test 6: Two layers with Au (Ra-223 | Au | Al | Li)
    {
        "name": "sandwich_au_al",
        "calc_type": "sandwich",
        "source_matdef": {"Ra-223": 1.0},
        "source_density": 5.5,
        "target_matdef": {"Li-7": 1.0},
        "intermediate_layers": [
            {"matdef": {"Au": 1.0}, "density": 19.3, "thickness": 0.0003},
            {"matdef": {"Al-27": 1.0}, "density": 2.70, "thickness": 0.0003}
        ],
    },

    # Test 7: Mixed source with single layer (Po + Am | C | Be)
    {
        "name": "sandwich_mixed_source",
        "calc_type": "sandwich",
        "source_matdef": {"Po-210": 0.5, "Am-241": 0.5},
        "source_density": 11.0,
        "target_matdef": {"Be-9": 1.0},
        "intermediate_layers": [
            {"matdef": {"C-13": 1.0}, "density": 2.26, "thickness": 0.001}
        ],
    },

    # Test 8: Four ultra-thin layers (WG-Pu | Al | C | Si | Pb | Be)
    {
        "name": "sandwich_4layer",
        "calc_type": "sandwich",
        "source_matdef": {"Pu-239": 0.93, "Pu-240": 0.06, "Pu-241": 0.01},
        "source_density": 19.8,
        "target_matdef": {"Be-9": 1.0},
        "intermediate_layers": [
            {"matdef": {"Al-27": 1.0}, "density": 2.70, "thickness": 0.00008},
            {"matdef": {"C-12": 1.0}, "density": 2.26, "thickness": 0.00008},
            {"matdef": {"Si-28": 1.0}, "density": 2.33, "thickness": 0.00008},
            {"matdef": {"Pb": 1.0}, "density": 11.34, "thickness": 0.00008}
        ],
    },

    # Test 9: Single layer, minimal thickness (HEU | Al | C)
    {
        "name": "sandwich_minimal",
        "calc_type": "sandwich",
        "source_matdef": {"U-235": 0.93, "U-238": 0.07},
        "source_density": 19.0,
        "target_matdef": {"C-12": 0.989, "C-13": 0.011},
        "intermediate_layers": [
            {"matdef": {"Al-27": 1.0}, "density": 2.70, "thickness": 0.0001}
        ],
    },

    # Test 10: Two layers with P/Cl target (Am-241 | Mg | Si | P/Cl)
    {
        "name": "sandwich_p_cl_target",
        "calc_type": "sandwich",
        "source_matdef": {"Am-241": 1.0},
        "source_density": 13.67,
        "target_matdef": {"P-31": 0.6, "Cl-35": 0.4},
        "intermediate_layers": [
            {"matdef": {"Mg-24": 1.0}, "density": 1.74, "thickness": 0.0005},
            {"matdef": {"Si-28": 1.0}, "density": 2.33, "thickness": 0.0005}
        ],
    }
]


# =============================================================================
# ALL CONFIGS
# =============================================================================

ALL_CONFIGS = {
    "beam": BEAM_CONFIGS,
    "homogeneous": HOMOGENEOUS_CONFIGS,
    "interface": INTERFACE_CONFIGS,
    "sandwich": SANDWICH_CONFIGS
}
