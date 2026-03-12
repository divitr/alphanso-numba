"""
Miscellaneous utility functions for ALPHANSO.
"""

import numpy as np
from typing import List, Tuple
from scipy.interpolate import interp1d

from .parsers import get_stopping_power
from .atomic_data_loader import (
    get_atomic_mass,
    get_natural_abundance,
    get_atomic_number,
    get_natural_isotopes
)


def matdef_to_zaids(matdef_input):
    """
    Convert material definition to ZAID format and calculate atom fractions.

    Args:
        matdef_input: Dict with isotope names/ZAIDs as keys and mass fractions as values
                     Examples: {'Al-27': 1.0}, {13027: 1.0}, {'C': 1.0} (natural element)
                     Natural elements can be specified as 'C', 'O', or as ZAIDs like 6000, 8000

    Returns:
        tuple: (mass_fractions_dict, atom_fractions_dict) with ZAIDs as keys
    """
    zaid_fractions = {}
    for key, value in matdef_input.items():
        if isinstance(key, str):
            parts = key.split('-')
            z = get_atomic_number(parts[0])
            if z is not None:
                if len(parts) == 2:
                    zaid = z * 1000 + int(parts[1])
                else:
                    zaid = z * 1000
                zaid_fractions[zaid] = float(value)
        else:
            zaid_fractions[int(key)] = float(value)

    mass_fractions = {}
    for zaid, value in zaid_fractions.items():
        a = zaid % 1000
        if a == 0:
            z = zaid // 1000
            natural_isos = get_natural_isotopes(z)
            if natural_isos:
                total_abundance = sum(
                    get_natural_abundance(iso) or 0 for iso in natural_isos
                )
                if total_abundance > 0:
                    for iso_zaid in natural_isos:
                        abundance = get_natural_abundance(iso_zaid)
                        if abundance and abundance > 0:
                            mass_fractions[iso_zaid] = value * (abundance / total_abundance)
        else:
            mass_fractions[zaid] = value

    total_mass = sum(mass_fractions.values())
    if total_mass > 0:
        for zaid in mass_fractions:
            mass_fractions[zaid] /= total_mass

    atom_fractions = {}
    total_atoms = 0.0
    for zaid, mass_frac in mass_fractions.items():
        atomic_mass = get_atomic_mass(zaid)
        if atomic_mass is not None and atomic_mass > 0:
            atom_frac = mass_frac / atomic_mass
            atom_fractions[zaid] = atom_frac
            total_atoms += atom_frac

    if total_atoms > 0:
        for zaid in atom_fractions:
            atom_fractions[zaid] /= total_atoms

    return mass_fractions, atom_fractions


def rebin_xs(xs_dict, ebins, extrapolate=False):
    """Rebin a cross section dictionary to a new energy grid via linear interpolation."""
    energies = np.array(sorted(xs_dict.keys()))
    cross_sections = np.array([xs_dict[e] for e in energies])

    if extrapolate:
        fill_left = cross_sections[0]
        fill_right = cross_sections[-1]
    else:
        fill_left = 0
        fill_right = 0

    interpolated_cross_sections = np.interp(
        ebins, energies, cross_sections, left=fill_left, right=fill_right)

    new_xs_dict = dict(zip(ebins, interpolated_cross_sections))

    return new_xs_dict


def get_composite_stopping(mass_fractions, data_dir=None):
    """Calculate composite stopping power via Bragg-Kleeman weighting.

    Args:
        mass_fractions: Dict with ZAIDs as keys and mass fractions as values
        data_dir: Optional data directory path

    Returns:
        Dict with energies as keys and composite stopping powers as values
    """
    atom_fractions = {}
    total_atoms = 0.0

    for zaid, mass_frac in mass_fractions.items():
        atomic_mass = get_atomic_mass(zaid)
        if atomic_mass is not None and atomic_mass > 0:
            atom_frac = mass_frac / atomic_mass
            atom_fractions[zaid] = atom_frac
            total_atoms += atom_frac

    if total_atoms > 0:
        for zaid in atom_fractions:
            atom_fractions[zaid] /= total_atoms

    stopping_data_dict = {}
    all_energies = set()

    for zaid, afrac in atom_fractions.items():
        stopping_data = get_stopping_power(zaid, data_dir)
        if stopping_data is not None:
            stopping_data_dict[zaid] = (stopping_data, afrac)
            all_energies.update(stopping_data.keys())

    if not all_energies:
        return {}

    sorted_energies = sorted(all_energies)

    composite_stopping = {}

    for energy in sorted_energies:
        total_stopping = 0.0

        for zaid, (stopping_data, afrac) in stopping_data_dict.items():
            if energy in stopping_data:
                sp_value = stopping_data[energy]
            else:
                energies_list = sorted(stopping_data.keys())
                sp_values = [stopping_data[e] for e in energies_list]
                sp_value = np.interp(energy, energies_list, sp_values)

            total_stopping += sp_value * afrac

        composite_stopping[energy] = total_stopping

    return composite_stopping


def rebin_endf_spectrum(
    endf_spectrum: List[Tuple[float, float]],
    neutron_energy_bins: np.ndarray
) -> np.ndarray:
    """
    Rebin ENDF group integrals spectrum to neutron energy bins.

    Args:
        endf_spectrum: List of (energy [MeV], intensity [fraction]) tuples
        neutron_energy_bins: Energy bin edges [MeV]

    Returns:
        Spectrum values for each bin [len(neutron_energy_bins)-1], normalized
    """
    nng = len(neutron_energy_bins) - 1
    spectrum = np.zeros(nng)

    if not endf_spectrum:
        return spectrum

    endf_energies = np.array([e for e, _ in endf_spectrum])
    endf_intensities = np.array([i for _, i in endf_spectrum])

    total_intensity = np.sum(endf_intensities)
    if total_intensity > 0:
        endf_intensities = endf_intensities / total_intensity
    else:
        return spectrum

    interp_func = interp1d(
        endf_energies,
        endf_intensities,
        kind='linear',
        bounds_error=False,
        fill_value=0.0
    )

    for n in range(nng):
        e_low = min(neutron_energy_bins[n], neutron_energy_bins[n + 1])
        e_high = max(neutron_energy_bins[n], neutron_energy_bins[n + 1])

        if e_low < 0 or e_high <= e_low:
            continue

        bin_center = (e_low + e_high) / 2.0
        bin_width = e_high - e_low

        intensity_at_center = float(interp_func(bin_center))
        spectrum[n] = intensity_at_center

    total = np.sum(spectrum)
    if total > 0:
        spectrum = spectrum / total

    return spectrum
