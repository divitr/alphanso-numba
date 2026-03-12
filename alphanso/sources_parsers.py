import re
import os
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from .constants import ALPH_MASS, ANEUT_MASS, AMU_TO_MEV, ZALP, ALPH
from .atomic_data_loader import get_atomic_mass as get_atomic_mass_from_db

logger = logging.getLogger(__name__)


def _calculate_q_value(target_zaid: int, product_zaid: int) -> Optional[float]:
    """
    Calculate Q-value for (alpha,n) reaction from nuclear masses.

    Q = [M(target) + M(alpha)] - [M(product) + M(neutron)] (in amu)
      = (mass of reactants) - (mass of products)
      = net energy released (if positive) or required (if negative)
    Q (MeV) = (Q in amu) * 931.494

    Args:
        target_zaid: ZAID of target nucleus
        product_zaid: ZAID of product nucleus

    Returns:
        Q-value in MeV or None if a mass is missing.
    """

    m_target = get_atomic_mass_from_db(target_zaid)
    m_product = get_atomic_mass_from_db(product_zaid)

    if m_target is None or m_product is None:
        return None

    q_amu = (m_target + ALPH_MASS) - (m_product + ANEUT_MASS)
    q_value = q_amu * AMU_TO_MEV

    return q_value


def get_sources_an_xs(z: int,
                      a: int,
                      symbol: str,
                      data_dir: Optional[os.PathLike] = None) -> Dict[float,
                                                                      float]:
    """
    Get the (a,n) reaction cross section data for a given ZAID from SOURCES data.

    Args:
        z: Atomic number,
        a: Atomic mass number,
        symbol: Element symbol,
        data_dir: Directory containing the data.

    Returns:
        {energy (MeV): cross_section (barns)}: (a,n) reaction cross section,
        None: If the ZAID is not found in the tape3 file.

    Raises:
        FileNotFoundError: If the tape3 file is not found.
    """

    if data_dir is None:
        raise ValueError(
            "data_dir must be provided to read SOURCES tape3 data")

    filepath = os.path.join(str(data_dir), "tape3")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    zaid_int = z * 10000 + a * 10

    header_re = re.compile(r'^\s*(\d{6,10})\b')

    energy_cs_dict = {}
    found_zaid = False

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            m_header = header_re.match(line)
            if m_header:
                full_number = m_header.group(1)
                current_zaid_int = int(full_number)

                if not found_zaid and str(zaid_int) in str(current_zaid_int):
                    found_zaid = True
                    continue
                elif found_zaid:
                    break

            if found_zaid and m_header is None:
                numbers = _extract_fortran_floats(line)

                for i in range(0, len(numbers), 2):
                    if i + 1 < len(numbers):
                        energy = numbers[i]
                        cross_section = numbers[i + 1]
                        cross_section = cross_section / 1e3
                        energy_cs_dict[energy] = cross_section

    if not found_zaid:
        return None

    return energy_cs_dict


def get_sources_stopping_power(zaid: int,
                               data_dir: Optional[os.PathLike] = None,
                               energies: np.ndarray = None,
                               atomic_mass: float = None) -> Dict[float,
                                                                  float]:
    """
    Get the stopping power for a given ZAID from SOURCES data.

    This function reads the tape2 file to extract Ziegler coefficients and calculates stopping power values
    using the exact same formula as the sources Fortran code, including both nuclear and electronic stopping.

    The calculation follows the original sources Fortran implementation:
    1. Nuclear stopping: Uses Ziegler-Biersack-Littmark formula with empirical parameters
    2. Electronic stopping: Uses Ziegler coefficients for low energy (E ≤ 30 MeV) and
       exponential fit for high energy (E > 30 MeV)

    Args:
        zaid: ZAID identifier (format: Z*1000 + A)
        data_dir: Directory containing the data (must be "sources")
        energies: Array of energies in MeV to calculate stopping power for.
                 If None, uses a default range from 0.1 to 15 MeV
        atomic_mass: Atomic mass in amu. If None, will be extracted from the data.

    Returns:
        Dictionary {energy (MeV): stopping_power (MeV cm^2)}

    Raises:
        FileNotFoundError: If the tape2 file is not found
        ValueError: If the ZAID is not found or has insufficient coefficients
    """

    if data_dir is None:
        raise ValueError(
            "data_dir must be provided to read SOURCES tape2 data.")

    target_z = zaid // 1000
    filepath = os.path.join(str(data_dir), "tape2")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}.")

    zaid_re = re.compile(r'^\s*(\d+)\s+')
    number_re = re.compile(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+')

    header_line = None
    buffer = []
    found_zaid = False
    in_data_section = False

    try:
        with open(filepath, 'r') as f:
            for line in f:
                if not in_data_section:
                    if line.strip().startswith('1 0.9661'):
                        in_data_section = True
                    else:
                        continue

                m = zaid_re.match(line)
                if m:
                    if found_zaid:
                        break

                    current_zaid = m.group(1)
                    if current_zaid == str(target_z):
                        header_line = line
                        found_zaid = True
                        buffer = [line]
                    continue

                if found_zaid:
                    if number_re.search(line):
                        buffer.append(line)

        if not found_zaid:
            raise ValueError(
                f"ZAID {zaid} (Z={target_z}) not found in tape2 file")

        data_string = ' '.join(buffer)
        data_strings = number_re.findall(data_string)

        try:
            coeffs = np.array([float(ds) for ds in data_strings])
        except ValueError as e:
            raise ValueError(
                f"Error parsing coefficient values for ZAID {zaid}: {e}.")

        if len(coeffs) < 9:
            raise ValueError(
                f"Insufficient coefficients for ZAID {zaid}: found {len(coeffs)}, need at least 9.")

        if energies is None:
            energies = np.arange(0.1, 15.1, 0.1)

        energy_list = []
        stopping_power_list = []

        for energy_mev in energies:
            try:
                if energy_mev <= 0:
                    stopping_power_list.append(0.0)
                    energy_list.append(energy_mev)
                    continue

                dcx = 0.0

                zmat = target_z
                amat = atomic_mass if atomic_mass is not None else coeffs[0]

                term = ZALP**0.66667 + zmat**0.66667
                sterm = np.sqrt(term)
                bot = ZALP * zmat * (ALPH + amat) * sterm
                rep = 32530.0 * amat * energy_mev / bot

                if rep < 0.001:
                    dcx = 1.593 * np.sqrt(rep)
                elif rep < 10.0:
                    bot = 1.0 + 6.8 * rep + 3.4 * rep**1.5
                    dcx = 1.7 * np.sqrt(rep) * np.log(rep + 2.71828) / bot
                else:
                    dcx = np.log(0.47 * rep) / (2.0 * rep)

                if energy_mev <= 30.0:
                    slow = coeffs[1] * (1e3 * energy_mev) ** coeffs[2]
                    shigh = (coeffs[3] / energy_mev) * np.log(1.0 + \
                             coeffs[4] / energy_mev + coeffs[5] * energy_mev)
                    denom = slow + shigh

                    if denom > 0:
                        dcx += slow * shigh / denom
                else:
                    eil = np.log(1.0 / energy_mev)
                    arg = coeffs[6] + eil * \
                        (coeffs[7] + coeffs[8] * eil + coeffs[9] * eil * eil)
                    dcx += np.exp(arg)

                # Convert from S4C units to physical units
                # S4C dcx is in eV/(10^15 atoms/cm^2), convert to MeV*cm^2/atom
                # Factor: (1/10^15) converts S4C units, (1/10^6) converts eV to
                # MeV
                stopping_power = dcx / 1e15 / 1e6

                stopping_power_list.append(stopping_power)
                energy_list.append(energy_mev)

            except Exception as e:
                stopping_power_list.append(0.0)
                energy_list.append(energy_mev)

        return dict(zip(energy_list, stopping_power_list))

    except Exception as e:
        raise ValueError(
            f"Error processing tape2 file for ZAID {zaid}: {str(e)}.")


def get_sources_branching_info(zaid: str,
                               data_dir: Optional[os.PathLike] = None) -> Tuple[float,
                                                                                List[float],
                                                                                Dict[float,
                                                                                     List[float]]]:
    """
    Get the branching information for alpha decay from SOURCES data.

    Args:
        zaid: ZAID identifier (format: "00080170" for O-17)
        data_dir: Directory containing the data (must be "sources")

    Returns:
        A tuple containing:
        - q_value: Q-value for the alpha decay (MeV)
        - level_energies: List of level energies (MeV)
        - branching_data: Dictionary of branching fractions {energy: branching_fractions}

    Raises:
        ValueError: If data_dir is not "sources" or contains "sources"
        FileNotFoundError: If the tape4 file is not found
        ValueError: If the ZAID is not found or data cannot be parsed
    """
    if not (data_dir == "sources" or "sources" in str(data_dir)):
        raise ValueError(
            f"data_dir must be 'sources' or contain 'sources', got: {data_dir}")

    filepath = os.path.join(data_dir, "tape4")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    zaid_re = re.compile(r'^\s*(\d{8})\b')
    number_re = re.compile(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+\.?')

    all_entries = []
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        current_entry_lines = []
        for line in lines:
            match = zaid_re.match(line)
            if match:
                if current_entry_lines:
                    all_entries.append(current_entry_lines)
                current_entry_lines = [line]
            elif current_entry_lines:
                current_entry_lines.append(line)
        if current_entry_lines:
            all_entries.append(current_entry_lines)

    except Exception as e:
        raise ValueError(f"Error reading tape4 file: {str(e)}")

    target_entry = None
    matching_entries = [
        entry for entry in all_entries if str(zaid) in entry[0]]

    if not matching_entries:
        raise ValueError(f"ZAID {zaid} not found in tape4 file")

    target_entry = matching_entries[0]

    header_line = target_entry[0].strip()
    level_definitions = target_entry[1].strip()

    level_tokens = re.findall(
        r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+\.?', level_definitions)
    numeric_vals: List[float] = []
    for tok in level_tokens:
        try:
            numeric_vals.append(float(tok))
        except ValueError:
            continue

    q_value_from_line: Optional[float] = None
    level_energies: List[float] = []
    if len(numeric_vals) >= 2:
        q_value_from_line = numeric_vals[0]
        candidate_levels = numeric_vals[1:]
        if len(candidate_levels) >= 2:
            last_val = candidate_levels[-1]
            if abs(last_val - round(last_val)
                   ) < 1e-6 and 0 < round(last_val) <= 20:
                candidate_levels = candidate_levels[:-1]
        level_energies = candidate_levels

    target_zaid = int(zaid)
    z = target_zaid // 10000
    a = (target_zaid % 10000) // 10
    target_zaid_std = z * 1000 + a
    product_zaid_std = (z + 2) * 1000 + (a + 3)

    q_value_calc = _calculate_q_value(target_zaid_std, product_zaid_std)
    if q_value_from_line is not None:
        q_value = q_value_from_line
    else:
        q_value = q_value_calc if q_value_calc is not None else 0.0

    table_data = []
    for line in target_entry[2:]:
        line_content = line.split('*')[0].strip()
        if not line_content:
            continue
        row_strings = number_re.findall(line_content)
        if row_strings:
            table_data.append([float(ds) for ds in row_strings])

    if not table_data:
        return q_value, level_energies, {}

    branching_data = {}
    for row in table_data:
        if len(row) >= 2:
            alpha_energy = row[0]
            raw_fractions = row[1:]

            if len(raw_fractions) < len(level_energies):
                branching_fractions = raw_fractions + \
                    [0.0] * (len(level_energies) - len(raw_fractions))
            elif len(raw_fractions) > len(level_energies):
                branching_fractions = raw_fractions[:len(level_energies)]
            else:
                branching_fractions = raw_fractions

            row_sum = sum(branching_fractions)
            if row_sum > 0:
                branching_fractions = [
                    f / row_sum for f in branching_fractions]
            else:
                branching_fractions = [1.0] + [0.0] * (len(level_energies) - 1)

            branching_data[alpha_energy] = branching_fractions

    return q_value, level_energies, branching_data


def get_sources_decay_data(
    zaid: int,
    data_dir: Optional[os.PathLike] = None,
    decay_mode: str = 'alpha',
    return_params: bool = False
) -> Union[
    Tuple[float, List[Tuple[float, float]]],
    Tuple[float, List[Tuple[float, float]], Dict[str, float]]
]:
    """
    Get decay data for a given ZAID from SOURCES-4C tape5 file.

    Args:
        zaid: ZAID identifier (format: Z*1000 + A for standard ZZZAAA format)
        data_dir: Directory containing the data (must be "sources" or contain "sources")
        decay_mode: Decay mode to extract - 'alpha' or 'sf' (spontaneous fission)
        return_params: If True, return additional parameters dict (SF: nubar, watt_a, watt_b)

    Returns:
        If return_params=False:
            - decay_strength: Emission rate per atom [particles/s/atom or neutrons/s/atom]
            - spectrum: List of (energy [MeV], intensity [fraction]) tuples
        If return_params=True:
            - decay_strength: Emission rate per atom
            - spectrum: List of (energy [MeV], intensity [fraction]) tuples
            - params: Dict with keys {'decay_constant', 'sf_branching', 'nubar', 'watt_a', 'watt_b'}

    Raises:
        ValueError: If data_dir is not "sources" or does not contain "sources"
        FileNotFoundError: If the tape5 file is not found
        ValueError: If the ZAID is not found in the file
        ValueError: If decay_mode is not 'alpha' or 'sf'
    """

    if not (data_dir == "sources" or "sources" in str(data_dir)):
        raise ValueError(
            f"data_dir must be 'sources' or contain 'sources', got: {data_dir}")

    filepath = os.path.join(data_dir, "tape5")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    # tape5 uses ZZZAAA0 format (e.g., Pu-238 = 94238 -> 942380)
    # Convert from standard ZZAAA (5-digit) to tape5's 6-digit format
    z = zaid // 1000
    a = zaid % 1000
    s4c_zaid = z * 10000 + a * 10  # e.g., 94*10000 + 238*10 = 942380

    zaid_re = re.compile(r'^\s*(\d{6})\s+(\d+)\s*,')

    found_zaid = False
    half_life = None
    branching_ratio = None
    num_bins = None
    alpha_intensities = []

    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i]

            match = zaid_re.match(line)
            if match:
                current_zaid = int(match.group(1))

                if current_zaid == s4c_zaid:
                    found_zaid = True
                    num_bins = int(match.group(2))

                    i += 1
                    if i >= len(lines):
                        raise ValueError(f"Incomplete data for ZAID {zaid}")

                    params_line = lines[i]
                    params = _extract_fortran_floats(params_line)

                    decay_constant = 0.0
                    sf_branching = 0.0
                    nubar = 0.0
                    watt_a = 0.0
                    watt_b = 0.0
                    dn_branching = 0.0
                    branching_ratio = 0.0

                    if num_bins < 50:
                        if len(params) >= 6:
                            decay_constant = params[0]
                            sf_branching = params[1]
                            nubar = params[2]
                            watt_a = params[3]
                            watt_b = params[4]
                            dn_branching = params[5]
                            branching_ratio = 1.0 - sf_branching - dn_branching
                        elif len(params) >= 5:
                            decay_constant = params[0]
                            sf_branching = params[1]
                            nubar = params[2]
                            watt_a = params[3]
                            watt_b = params[4]
                            dn_branching = 0.0
                            branching_ratio = 1.0 - sf_branching - dn_branching
                        elif len(params) >= 2:
                            decay_constant = params[0]
                            sf_branching = params[1]
                            branching_ratio = 1.0 - sf_branching
                        else:
                            raise ValueError(
                                f"Expected 2, 5, or 6 parameters, got {len(params)} for ZAID {zaid}")

                        half_life = np.log(
                            2) / decay_constant if decay_constant > 0 else 0.0
                    else:
                        if len(params) < 2:
                            raise ValueError(
                                f"Could not parse decay parameters for ZAID {zaid}")
                        half_life = params[0]
                        branching_ratio = params[1]
                        sf_branching = 0.0

                    i += 1

                    all_values = []
                    while i < len(lines):
                        if zaid_re.match(lines[i]):
                            break
                        values = _extract_fortran_floats(lines[i])
                        all_values.extend(values)
                        i += 1

                    if len(all_values) >= 2 * num_bins - 2:
                        alpha_spectrum_discrete = []
                        for j in range(0, len(all_values), 2):
                            if j + 1 < len(all_values):
                                energy = all_values[j]
                                intensity = all_values[j + 1]
                                alpha_spectrum_discrete.append(
                                    (energy, intensity))
                        alpha_intensities = alpha_spectrum_discrete
                    else:
                        alpha_intensities = all_values[:num_bins]

                    break
                elif found_zaid:
                    break

            i += 1

        if not found_zaid:
            if return_params:
                return 0.0, [], {}
            else:
                return 0.0, []

        if half_life > 0 and decay_constant == 0.0:
            decay_constant = np.log(2) / half_life

        if isinstance(
            alpha_intensities,
            list) and len(alpha_intensities) > 0 and isinstance(
            alpha_intensities[0],
                tuple):
            alpha_spectrum = alpha_intensities
        else:
            if len(alpha_intensities) != num_bins:
                raise ValueError(
                    f"Expected {num_bins} intensity values but found {len(alpha_intensities)} for ZAID {zaid}")

            energy_max = 15.0
            energy_step = energy_max / num_bins

            alpha_spectrum = []
            for i, intensity in enumerate(alpha_intensities):
                energy = (i + 0.5) * energy_step
                alpha_spectrum.append((energy, intensity))

        params_dict = {
            'decay_constant': decay_constant,
            'alpha_branching': branching_ratio,
            'sf_branching': sf_branching,
            'nubar': nubar,
            'watt_a': watt_a,
            'watt_b': watt_b,
            'dn_branching': dn_branching,
            'half_life': half_life
        }

        if decay_mode == 'alpha':
            alpha_decay_strength = decay_constant * branching_ratio
            if return_params:
                return alpha_decay_strength, alpha_spectrum, params_dict
            else:
                return alpha_decay_strength, alpha_spectrum

        elif decay_mode == 'sf':
            if sf_branching == 0.0 or nubar == 0.0:
                if return_params:
                    return 0.0, [], params_dict
                else:
                    return 0.0, []

            if watt_a <= 0.0 or watt_b <= 0.0:
                logger.warning(
                    f"ZAID {zaid} has SF branching but invalid Watt parameters (a={watt_a}, b={watt_b}).")
                if return_params:
                    return 0.0, [], params_dict
                else:
                    return 0.0, []

            sf_decay_strength = decay_constant * sf_branching * nubar

            if return_params:
                return sf_decay_strength, [], params_dict
            else:
                return sf_decay_strength, []

        else:
            raise ValueError(
                f"Unknown decay_mode: {decay_mode}. Must be 'alpha' or 'sf'")

    except Exception as e:
        raise ValueError(
            f"Error processing tape5 file for ZAID {zaid}: {str(e)}.")


def _extract_fortran_floats(text: str) -> List[float]:
    """
    Extract floating-point numbers from SOURCES-4C formatted text.

    Supports both standard scientific notation (e.g., 1.23e+04) and FORTRAN-style
    notation without the 'e' (e.g., 1.2345+00).
    Also handles concatenated numbers like '4.5442e+008.8811e+00' with no spaces.
    """
    split_pattern = re.compile(r'(?<=[+-]\d{2})(?=\d+\.)')
    text = split_pattern.sub(' ', text)

    std_pattern = re.compile(r'[-+]?\d+\.\d*[eE][+-]?\d+')

    fortran_pattern = re.compile(r'[-+]?\d+\.\d*[+-]\d{1,3}')

    simple_pattern = re.compile(r'[-+]?\d+\.\d*')

    all_matches = []

    for match in std_pattern.finditer(text):
        all_matches.append((match.start(), match.group()))

    for match in fortran_pattern.finditer(text):
        all_matches.append((match.start(), match.group()))

    for match in simple_pattern.finditer(text):
        covered = False
        for start, _ in all_matches:
            if abs(match.start() - start) < 2:
                covered = True
                break
        if not covered:
            all_matches.append((match.start(), match.group()))

    all_matches.sort(key=lambda x: x[0])
    tokens = [match[1] for match in all_matches]

    numbers: List[float] = []
    for tok in tokens:
        if ('e' not in tok and 'E' not in tok) and re.search(
                r"\d\.\d+[+-]\d+$", tok):
            tok = re.sub(r"([0-9]\.[0-9]+)([+-]\d+)$", r"\1e\2", tok)
        try:
            numbers.append(float(tok))
        except ValueError:
            continue
    return numbers
