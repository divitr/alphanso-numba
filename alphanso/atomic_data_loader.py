"""
Provides efficient loading and caching of atomic data.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
import threading
import time


class AtomicDataLoader:
    """Thread-safe atomic data loader with caching."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern for global data cache."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(AtomicDataLoader, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the atomic data loader."""
        if hasattr(self, '_initialized'):
            return

        self._initialized = True
        self._data_dir = Path(__file__).parent / "data" / "atomic_data"
        self._cache = {}
        self._cache_lock = threading.RLock()
        self._load_timestamp = None

        self._metadata_file = self._data_dir / "metadata.json"
        self._atomic_data_file = self._data_dir / "atomic_data.json"
        self._abundances_file = self._data_dir / "natural_abundances.json"

        self._atomic_data_loaded = False
        self._abundances_loaded = False

        self._element_symbols_file = self._data_dir / "element_symbols.json"
        self._element_symbols = {}
        self._symbol_to_z = {}
        self._load_element_symbols()

    def _load_element_symbols(self):
        """Load element symbols from JSON file."""
        if not self._element_symbols_file.exists():
            raise FileNotFoundError(
                f"Element symbols file not found: {self._element_symbols_file}")

        with open(self._element_symbols_file, 'r') as f:
            self._element_symbols = json.load(f)

        self._symbol_to_z = {v: int(k)
                             for k, v in self._element_symbols.items()}

    def _ensure_atomic_data_loaded(self):
        """Ensure atomic data is loaded into cache."""
        if self._atomic_data_loaded:
            return

        with self._cache_lock:
            if self._atomic_data_loaded:
                return

            if not self._atomic_data_file.exists():
                raise FileNotFoundError(
                    f"Atomic data file not found: {self._atomic_data_file}")

            with open(self._atomic_data_file, 'r') as f:
                atomic_data = json.load(f)

            self._cache['atomic_data'] = atomic_data
            self._atomic_data_loaded = True
            self._load_timestamp = time.time()

    def _ensure_abundances_loaded(self):
        """Ensure abundance data is loaded into cache."""
        if self._abundances_loaded:
            return

        with self._cache_lock:
            if self._abundances_loaded:
                return

            if not self._abundances_file.exists():
                raise FileNotFoundError(
                    f"Abundance data file not found: {self._abundances_file}")

            with open(self._abundances_file, 'r') as f:
                abundance_data = json.load(f)

            self._cache['abundances'] = abundance_data
            self._abundances_loaded = True

    def get_isotope_data(self, zaid: int) -> Optional[Dict[str, Any]]:
        """Get isotope data for a specific ZAID.

        Args:
            zaid: ZAID (ZZAAA format)

        Returns:
            Dict with isotope data or None if not found
        """
        self._ensure_atomic_data_loaded()

        isotopes = self._cache['atomic_data']['isotopes']
        return isotopes.get(str(zaid))

    def get_element_data(self, z: int) -> Optional[Dict[str, Any]]:
        """Get element data for a specific atomic number.

        Args:
            z: Atomic number

        Returns:
            Dict with element data or None if not found
        """
        self._ensure_atomic_data_loaded()

        elements = self._cache['atomic_data']['elements']
        return elements.get(str(z))

    def get_natural_abundances(self, z: int) -> Optional[Dict[str, Any]]:
        """Get natural abundance data for an element.

        Args:
            z: Atomic number

        Returns:
            Dict with natural abundance data or None if not found
        """
        self._ensure_abundances_loaded()

        abundances = self._cache['abundances']['abundances']
        return abundances.get(str(z))

    def get_atomic_mass(self, zaid: int) -> Optional[float]:
        """Get atomic mass for a specific ZAID.

        Args:
            zaid: ZAID (ZZAAA format)

        Returns:
            Atomic mass in u or None if not found
        """
        z, a = self.zaid_to_z_a(zaid)
        if a == 0:
            natural_isotopes = self.get_natural_isotopes(z)

            if not natural_isotopes:
                return None

            total_weighted_mass = 0.0
            total_abundance = 0.0

            for isotope_zaid in natural_isotopes:
                isotope_data = self.get_isotope_data(isotope_zaid)
                if isotope_data and isotope_data.get('abundance', 0) > 0:
                    mass = isotope_data['mass']
                    abundance = isotope_data['abundance']
                    total_weighted_mass += mass * abundance
                    total_abundance += abundance

            if total_abundance > 0:
                return total_weighted_mass / total_abundance
            else:
                return None
        else:
            isotope_data = self.get_isotope_data(zaid)
            return isotope_data['mass'] if isotope_data else None

    def get_natural_abundance(self, zaid: int) -> Optional[float]:
        """Get natural abundance for a specific ZAID.

        Args:
            zaid: ZAID (ZZAAA format)

        Returns:
            Natural abundance (0-1) or None if not naturally occurring
        """
        isotope_data = self.get_isotope_data(zaid)
        return isotope_data['abundance'] if isotope_data else None

    def get_standard_atomic_weight(self, z: int) -> Optional[float]:
        """Get standard atomic weight for an element.

        Args:
            z: Atomic number

        Returns:
            Standard atomic weight in u or None if not available
        """
        element_data = self.get_element_data(z)
        return element_data['standard_atomic_weight'] if element_data else None

    def zaid_to_z_a(self, zaid: int) -> Tuple[int, int]:
        """Convert ZAID to (Z, A) format.

        Args:
            zaid: ZAID (ZZAAA format)_

        Returns:
            Tuple of (atomic_number, mass_number)
        """
        z = zaid // 1000
        a = zaid % 1000
        return (z, a)

    def z_a_to_zaid(self, z: int, a: int) -> int:
        """Convert (Z, A) to ZAID format.

        Args:
            z: Atomic number
            a: Mass number

        Returns:
            ZAID (ZZAAA format)
        """
        if a == 0:
            return z * 1000
        else:
            return z * 1000 + a

    def get_element_symbol(self, z: int) -> Optional[str]:
        """Get element symbol from atomic number.

        Args:
            z: Atomic number

        Returns:
            Element symbol or None if invalid Z
        """
        return self._element_symbols.get(str(z))

    def get_atomic_number(self, symbol: str) -> Optional[int]:
        """Get atomic number from element symbol.

        Args:
            symbol: Element symbol (case-insensitive)

        Returns:
            Atomic number or None if invalid symbol
        """
        return self._symbol_to_z.get(symbol.capitalize())

    def is_naturally_occurring(self, zaid: int) -> bool:
        """Check if an isotope is naturally occurring.

        Args:
            zaid: ZAID (ZZAAA format)

        Returns:
            True if naturally occurring with abundance > 0
        """
        abundance = self.get_natural_abundance(zaid)
        return abundance is not None and abundance > 0

    def get_natural_isotopes(self, z: int) -> List[int]:
        """Get list of naturally occurring isotopes for an element.

        Args:
            z: Atomic number

        Returns:
            List of ZAIDs for naturally occurring isotopes
        """
        abundance_data = self.get_natural_abundances(z)
        if not abundance_data:
            return []

        natural_zaids = []
        for a_str in abundance_data['isotopes']:
            a = int(a_str)
            zaid = z * 1000 + a
            natural_zaids.append(zaid)

        return natural_zaids

    def get_all_isotopes(self, z: int) -> List[int]:
        """Get list of all known isotopes for an element.

        Args:
            z: Atomic number

        Returns:
            List of ZAIDs for all known isotopes
        """
        element_data = self.get_element_data(z)
        if not element_data:
            return []

        all_zaids = []
        for a_str in element_data['isotopes']:
            a = int(a_str)
            zaid = z * 1000 + a
            all_zaids.append(zaid)

        return all_zaids

    def get_metadata(self) -> Dict[str, Any]:
        """Get atomic data metadata.

        Returns:
            Metadata dictionary
        """
        if not self._metadata_file.exists():
            return {}

        with open(self._metadata_file, 'r') as f:
            return json.load(f)

    def clear_cache(self):
        """Clear all cached data to force reload."""
        with self._cache_lock:
            self._cache.clear()
            self._atomic_data_loaded = False
            self._abundances_loaded = False
            self._load_timestamp = None

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached data.

        Returns:
            Dictionary with cache information
        """
        return {
            'atomic_data_loaded': self._atomic_data_loaded,
            'abundances_loaded': self._abundances_loaded,
            'load_timestamp': self._load_timestamp,
            'cache_size': len(self._cache),
            'data_files': {
                'atomic_data': str(self._atomic_data_file),
                'abundances': str(self._abundances_file),
                'metadata': str(self._metadata_file)
            }
        }

    def __getstate__(self):
        """Custom pickle method to handle threading locks."""
        state = self.__dict__.copy()
        state['_cache_lock'] = None
        return state

    def __setstate__(self, state):
        """Custom unpickle method to recreate threading locks."""
        self.__dict__.update(state)
        self._cache_lock = threading.RLock()


atomic_data = AtomicDataLoader()


def get_atomic_mass(zaid: int) -> Optional[float]:
    """Get atomic mass for a ZAID."""
    return atomic_data.get_atomic_mass(zaid)


def get_natural_abundance(zaid: int) -> Optional[float]:
    """Get natural abundance for a ZAID."""
    return atomic_data.get_natural_abundance(zaid)


def get_element_symbol(z: int) -> Optional[str]:
    """Get element symbol from atomic number."""
    return atomic_data.get_element_symbol(z)


def get_atomic_number(symbol: str) -> Optional[int]:
    """Get atomic number from element symbol."""
    return atomic_data.get_atomic_number(symbol)


def zaid_to_z_a(zaid: int) -> Tuple[int, int]:
    """Convert ZAID to (Z, A)."""
    return atomic_data.zaid_to_z_a(zaid)


def z_a_to_zaid(z: int, a: int) -> int:
    """Convert (Z, A) to ZAID."""
    return atomic_data.z_a_to_zaid(z, a)


def get_natural_isotopes(z: int) -> List[int]:
    """Get naturally occurring isotopes for an element."""
    return atomic_data.get_natural_isotopes(z)
