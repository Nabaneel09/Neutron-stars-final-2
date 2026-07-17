"""! @file units.py
@brief Unit systems and conversion functions (SI <-> geometrized <-> cgs/natural).

The integrator works in **geometrized units** (G = c = 1) with the
kilometre as the base length. In this system:

- length      : km
- mass        : km            (M_kg * G/c^2 / 1000)
- energy density and pressure : km^-2   (X_SI * G/c^4 * 1e6)

so that the TOV equations take their dimensionless textbook form and
all dynamical variables are O(1e-4 .. 1e2) for neutron stars, which is
numerically well conditioned.

Every function is a pure, vectorized (NumPy-aware) conversion.
"""

from __future__ import annotations

from typing import Union

import numpy as np

from . import constants as k

ArrayLike = Union[float, np.ndarray]

# Conversion factors (module-level, computed once)
_PRESSURE_SI_TO_GEOM = k.G / k.C**4 * k.KM**2  # Pa -> km^-2
_DENSITY_SI_TO_GEOM = k.G / k.C**2 * k.KM**2  # kg/m^3 -> km^-2
_MASS_SI_TO_GEOM = k.G / k.C**2 / k.KM  # kg -> km


# ----------------------------------------------------------------------
# Pressure / energy density
# ----------------------------------------------------------------------
def pressure_si_to_geom(p_si: ArrayLike) -> ArrayLike:
    """!Convert pressure from SI [Pa] to geometrized [km^-2].

    @param p_si  Pressure in pascal.
    @return      Pressure in km^-2 (multiplied by G/c^4).
    """
    return np.asarray(p_si, dtype=float) * _PRESSURE_SI_TO_GEOM


def pressure_geom_to_si(p_geom: ArrayLike) -> ArrayLike:
    """!Convert pressure from geometrized [km^-2] to SI [Pa]."""
    return np.asarray(p_geom, dtype=float) / _PRESSURE_SI_TO_GEOM


def energy_density_si_to_geom(eps_si: ArrayLike) -> ArrayLike:
    """!Convert an energy density from SI [J m^-3] to geometrized [km^-2]."""
    return np.asarray(eps_si, dtype=float) * _PRESSURE_SI_TO_GEOM


def energy_density_geom_to_si(eps_geom: ArrayLike) -> ArrayLike:
    """!Convert an energy density from geometrized [km^-2] to SI [J m^-3]."""
    return np.asarray(eps_geom, dtype=float) / _PRESSURE_SI_TO_GEOM


# ----------------------------------------------------------------------
# Mass density
# ----------------------------------------------------------------------
def density_si_to_geom(rho_si: ArrayLike) -> ArrayLike:
    """!Convert a mass density from SI [kg m^-3] to geometrized [km^-2].

    The mass density is interpreted as an energy density rho*c^2.
    """
    return np.asarray(rho_si, dtype=float) * _DENSITY_SI_TO_GEOM


def density_geom_to_si(rho_geom: ArrayLike) -> ArrayLike:
    """!Convert a mass density from geometrized [km^-2] to SI [kg m^-3]."""
    return np.asarray(rho_geom, dtype=float) / _DENSITY_SI_TO_GEOM


def density_cgs_to_geom(rho_cgs: ArrayLike) -> ArrayLike:
    """!Convert a mass density from cgs [g cm^-3] to geometrized [km^-2]."""
    return density_si_to_geom(np.asarray(rho_cgs, dtype=float) * k.GCM3_TO_KGM3)


def density_geom_to_cgs(rho_geom: ArrayLike) -> ArrayLike:
    """!Convert a mass density from geometrized [km^-2] to cgs [g cm^-3]."""
    return density_geom_to_si(rho_geom) / k.GCM3_TO_KGM3


def pressure_cgs_to_geom(p_cgs: ArrayLike) -> ArrayLike:
    """!Convert pressure from cgs [dyn cm^-2] to geometrized [km^-2]."""
    return pressure_si_to_geom(np.asarray(p_cgs, dtype=float) * k.DYNCM2_TO_PA)


def pressure_geom_to_cgs(p_geom: ArrayLike) -> ArrayLike:
    """!Convert pressure from geometrized [km^-2] to cgs [dyn cm^-2]."""
    return pressure_geom_to_si(p_geom) / k.DYNCM2_TO_PA


# ----------------------------------------------------------------------
# Mass
# ----------------------------------------------------------------------
def mass_si_to_geom(m_kg: ArrayLike) -> ArrayLike:
    """!Convert mass from SI [kg] to geometrized [km] (GM/c^2)."""
    return np.asarray(m_kg, dtype=float) * _MASS_SI_TO_GEOM


def mass_geom_to_si(m_km: ArrayLike) -> ArrayLike:
    """!Convert mass from geometrized [km] to SI [kg]."""
    return np.asarray(m_km, dtype=float) / _MASS_SI_TO_GEOM


def mass_geom_to_msun(m_km: ArrayLike) -> ArrayLike:
    """!Convert mass from geometrized [km] to solar masses."""
    return np.asarray(m_km, dtype=float) / k.M_SUN_GEOM_KM


def mass_msun_to_geom(m_msun: ArrayLike) -> ArrayLike:
    """!Convert mass from solar masses to geometrized [km]."""
    return np.asarray(m_msun, dtype=float) * k.M_SUN_GEOM_KM


# ----------------------------------------------------------------------
# Natural (nuclear) units
# ----------------------------------------------------------------------
def energy_density_mevfm3_to_geom(eps: ArrayLike) -> ArrayLike:
    """!Convert an energy density from nuclear units [MeV fm^-3] to km^-2."""
    return energy_density_si_to_geom(np.asarray(eps, dtype=float) * k.MEV / k.FM**3)


def energy_density_geom_to_mevfm3(eps_geom: ArrayLike) -> ArrayLike:
    """!Convert an energy density from km^-2 to nuclear units [MeV fm^-3]."""
    return energy_density_geom_to_si(eps_geom) / (k.MEV / k.FM**3)


def number_density_si_to_fm3(n_si: ArrayLike) -> ArrayLike:
    """!Convert a number density from [m^-3] to [fm^-3]."""
    return np.asarray(n_si, dtype=float) * k.FM**3
