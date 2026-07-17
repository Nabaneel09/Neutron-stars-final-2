"""! @file constants.py
@brief Physical and astrophysical constants (CODATA 2018 / IAU nominal values).

All constants are given in SI units. Geometrized-unit conversion factors
derived from them live in :mod:`neutronstar.units`.

@section refs References
- CODATA 2018 recommended values, Rev. Mod. Phys. 93, 025010 (2021).
- IAU 2015 Resolution B3 (nominal solar values).
"""

from __future__ import annotations

from typing import Final

# ----------------------------------------------------------------------
# Fundamental constants (SI, CODATA 2018)
# ----------------------------------------------------------------------

#: Newtonian gravitational constant [m^3 kg^-1 s^-2]
G: Final[float] = 6.674_30e-11

#: Speed of light in vacuum [m s^-1] (exact)
C: Final[float] = 2.997_924_58e8

#: Reduced Planck constant [J s]
HBAR: Final[float] = 1.054_571_817e-34

#: Boltzmann constant [J K^-1] (exact)
K_B: Final[float] = 1.380_649e-23

#: Neutron rest mass [kg]
M_NEUTRON: Final[float] = 1.674_927_498_04e-27

#: Proton rest mass [kg]
M_PROTON: Final[float] = 1.672_621_923_69e-27

#: Electron rest mass [kg]
M_ELECTRON: Final[float] = 9.109_383_7015e-31

#: Atomic mass unit [kg]
M_U: Final[float] = 1.660_539_066_60e-27

#: Baryon mass adopted for baryon (rest-mass) density bookkeeping [kg].
#: We follow the pure-neutron-matter convention and use the neutron mass.
M_BARYON: Final[float] = M_NEUTRON

# ----------------------------------------------------------------------
# Astrophysical constants
# ----------------------------------------------------------------------

#: Solar mass [kg] (IAU nominal)
M_SUN: Final[float] = 1.988_47e30

#: Nuclear saturation (baryon) number density [m^-3] (n_0 = 0.16 fm^-3)
N_SAT: Final[float] = 0.16e45

#: Nuclear saturation mass density [kg m^-3] (rho_0 ~ 2.7e17 kg/m^3)
RHO_SAT: Final[float] = N_SAT * M_BARYON

# ----------------------------------------------------------------------
# Derived geometrized scales
# ----------------------------------------------------------------------

#: Half Schwarzschild radius of the Sun, GM_sun/c^2 [m] (~1476.6 m)
M_SUN_GEOM_M: Final[float] = G * M_SUN / C**2

#: GM_sun/c^2 expressed in km (~1.4766 km)
M_SUN_GEOM_KM: Final[float] = M_SUN_GEOM_M / 1.0e3

# ----------------------------------------------------------------------
# Unit multipliers
# ----------------------------------------------------------------------

KM: Final[float] = 1.0e3  #: one kilometre in metres
MEV: Final[float] = 1.602_176_634e-13  #: one MeV in joules
FM: Final[float] = 1.0e-15  #: one femtometre in metres

#: dyn/cm^2 -> Pa
DYNCM2_TO_PA: Final[float] = 0.1
#: g/cm^3 -> kg/m^3
GCM3_TO_KGM3: Final[float] = 1.0e3
