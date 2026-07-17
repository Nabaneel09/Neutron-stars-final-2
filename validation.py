"""! @file validation.py
@brief Quantitative validation against analytic solutions and literature.

Four independent benchmarks:

1. **Interior Schwarzschild solution** (constant density): M(R) and the
   full P(r) profile are analytic; the numerical solution must agree to
   the integrator tolerance.
2. **Lane-Emden equation** (Newtonian n = 3/2 polytrope): the Newtonian
   branch of the code is compared against an *independent* integration
   of the dimensionless Lane-Emden equation.
3. **Oppenheimer-Volkoff (1939)**: the ideal neutron Fermi gas must give
   the historical maximum mass ~ 0.71 M_sun.
4. **Modern EOS**: SLy and APR global properties (M_max, R_1.4,
   Lambda_1.4, I_1.4) against published values
   (Douchin & Haensel 2001; Read et al. 2009; Hinderer et al. 2010).

Additional physical-consistency checks: Buchdahl bound C < 4/9 and the
causality condition c_s <= c along every profile.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.integrate import solve_ivp

from . import units as u
from .eos import ConstantDensityEOS, EquationOfState, FermiGasEOS, PolytropicEOS
from .tov import Sequence, compute_sequence, maximum_mass, solve_at_mass, solve_star
from .utils import get_logger

log = get_logger(__name__)


# ----------------------------------------------------------------------
# 1. Interior Schwarzschild (constant density) -- fully analytic
# ----------------------------------------------------------------------
def schwarzschild_interior_pressure(eps0: float, r_over_R: np.ndarray,
                                    compactness: float) -> np.ndarray:
    r"""!Analytic P(r) of the interior Schwarzschild solution.

    \f[
      P(r) = \varepsilon_0\,
      \frac{\sqrt{1-2Cx^2}-\sqrt{1-2C}}{3\sqrt{1-2C}-\sqrt{1-2Cx^2}},
      \qquad x = r/R .
    \f]

    @param eps0         Constant energy density [km^-2].
    @param r_over_R     Radial coordinate normalized to the radius.
    @param compactness  C = M/R < 4/9.
    @return             Pressure [km^-2].
    """
    x = np.asarray(r_over_R, dtype=float)
    s_r = np.sqrt(1.0 - 2.0 * compactness * x**2)
    s_1 = np.sqrt(1.0 - 2.0 * compactness)
    return eps0 * (s_r - s_1) / (3.0 * s_1 - s_r)


def validate_constant_density(p_c_over_eps0: float = 0.1,
                              rho_cgs: float = 1.0e15,
                              rtol: float = 1.0e-10) -> Dict[str, float]:
    """!Compare the numerical model with the interior Schwarzschild solution.

    Given P_c/eps_0 = beta, the analytic compactness follows from
    sqrt(1-2C) = (1+beta)/(1+3beta), hence R = sqrt(3C/(4 pi eps_0)) and
    M = C R.

    @param p_c_over_eps0  Central pressure in units of eps_0.
    @param rho_cgs        Density of the incompressible fluid [g/cm^3].
    @param rtol           Integrator tolerance.
    @return dict with analytic/numeric M, R and maximum errors.
    """
    eos = ConstantDensityEOS.from_cgs(rho_cgs)
    beta = p_c_over_eps0
    s1 = (1.0 + beta) / (1.0 + 3.0 * beta)
    c_an = 0.5 * (1.0 - s1**2)
    r_an = float(np.sqrt(3.0 * c_an / (4.0 * np.pi * eos.eps0)))
    m_an = c_an * r_an  # km

    star = solve_star(eos, p_c=beta * eos.eps0, rtol=rtol,
                      tidal=True, inertia=True,
                      p_surface=beta * eos.eps0 * 1.0e-12)
    p_analytic = schwarzschild_interior_pressure(
        eos.eps0, star.r / r_an, c_an)
    err_profile = float(np.max(np.abs(star.p - p_analytic))
                        / (beta * eos.eps0))
    out = {
        "R_numeric_km": star.R, "R_analytic_km": r_an,
        "M_numeric_Msun": star.M,
        "M_analytic_Msun": float(u.mass_geom_to_msun(m_an)),
        "err_R": abs(star.R - r_an) / r_an,
        "err_M": abs(star.M_km - m_an) / m_an,
        "err_P_profile_max": err_profile,
    }
    log.info("constant density: err_R=%.2e err_M=%.2e err_P=%.2e",
             out["err_R"], out["err_M"], out["err_P_profile_max"])
    return out


# ----------------------------------------------------------------------
# 2. Lane-Emden n = 3/2 (independent Newtonian benchmark)
# ----------------------------------------------------------------------
def lane_emden(n: float = 1.5, rtol: float = 1.0e-12) -> Dict[str, float]:
    r"""!Integrate the Lane-Emden equation and return (xi_1, -xi^2 theta').

    \f[ \frac{1}{\xi^2}\frac{d}{d\xi}\Big(\xi^2\frac{d\theta}{d\xi}\Big)
        = -\theta^n \f]
    with theta(0) = 1, theta'(0) = 0. Solved as a first-order system with
    the regular series start theta = 1 - xi^2/6 + n xi^4/120.

    @param n     Polytropic index.
    @param rtol  Integrator tolerance.
    @return dict with xi1 and mass integral xi1^2 |theta'(xi1)|.
    """
    x0 = 1.0e-6

    def rhs(x: float, y: np.ndarray) -> list:
        theta, dtheta = y
        return [dtheta, -max(theta, 0.0) ** n - 2.0 * dtheta / x]

    def surf(x: float, y: np.ndarray) -> float:
        return y[0]
    surf.terminal = True  # type: ignore[attr-defined]
    surf.direction = -1  # type: ignore[attr-defined]

    y0 = [1.0 - x0**2 / 6.0 + n * x0**4 / 120.0, -x0 / 3.0 + n * x0**3 / 30.0]
    sol = solve_ivp(rhs, (x0, 50.0), y0, method="DOP853", rtol=rtol,
                    atol=1e-14, events=surf)
    xi1 = float(sol.t_events[0][0])
    dtheta1 = float(sol.y_events[0][0][1])
    return {"xi1": xi1, "mass_integral": -xi1**2 * dtheta1}


def validate_newtonian_polytrope(rho_c_cgs: float = 5.0e14,
                                 rtol: float = 1.0e-10) -> Dict[str, float]:
    """!Newtonian n = 3/2 polytrope vs the Lane-Emden solution.

    R = a xi_1 and M = 4 pi a^3 rho_c xi_1^2 |theta'(xi_1)| with
    a^2 = (n+1) K rho_c^{1/n - 1} / (4 pi).

    @param rho_c_cgs  Central rest-mass density [g/cm^3].
    @param rtol       Integrator tolerance.
    @return dict comparing radii and masses.
    """
    eos = PolytropicEOS.neutron_nonrel()
    n = 1.0 / (eos.gamma - 1.0)  # = 3/2
    rho_c = float(u.density_cgs_to_geom(rho_c_cgs))
    le = lane_emden(n, rtol=1e-13)

    a = np.sqrt((n + 1.0) * eos.k_geom * rho_c ** (1.0 / n - 1.0) / (4.0 * np.pi))
    r_an = a * le["xi1"]
    m_an = 4.0 * np.pi * a**3 * rho_c * le["mass_integral"]

    # very deep surface cut: for Gamma = 5/3, P ~ (R-r)^{5/2} near the
    # surface, so a cut at 1e-18 P_c biases R only at the 1e-7 level
    star = solve_star(eos, rho_c=rho_c, newtonian=True, rtol=rtol,
                      atol_p_frac=1.0e-20,
                      p_surface=float(eos.p_of_rho(rho_c)) * 1.0e-18)
    out = {
        "R_numeric_km": star.R, "R_laneemden_km": float(r_an),
        "M_numeric_Msun": star.M,
        "M_laneemden_Msun": float(u.mass_geom_to_msun(m_an)),
        "err_R": abs(star.R - r_an) / r_an,
        "err_M": abs(star.M_km - m_an) / m_an,
    }
    log.info("Lane-Emden n=3/2: err_R=%.2e err_M=%.2e", out["err_R"], out["err_M"])
    return out


# ----------------------------------------------------------------------
# 3/4. Literature benchmarks
# ----------------------------------------------------------------------
#: published reference values used in the comparison table
LITERATURE = {
    "fermi-gas": {"M_max": 0.7102, "source": "Oppenheimer & Volkoff (1939)"},
    "SLy": {"M_max": 2.05, "R_1.4": 11.72, "Lambda_1.4": 297.0,
            "source": "Douchin & Haensel (2001); Hinderer et al. (2010)"},
    "APR(AP4)": {"M_max": 2.21, "R_1.4": 11.40, "Lambda_1.4": 260.0,
                 "source": "Akmal et al. (1998); Read et al. (2009)"},
}


def validate_fermi_gas() -> Dict[str, float]:
    """!Reproduce the Oppenheimer-Volkoff maximum mass (~0.71 M_sun)."""
    eos = FermiGasEOS()
    eps_grid = np.geomspace(5.0e14, 1.0e17, 40)
    seq = compute_sequence(eos, eps_grid, tidal=False, inertia=False)
    mm = maximum_mass(eos, seq)
    ref = LITERATURE["fermi-gas"]["M_max"]
    out = {"M_max_numeric": mm["M_max"], "M_max_literature": ref,
           "R_at_Mmax_km": mm["R"],
           "eps_c_at_Mmax_cgs": mm["eps_c_cgs"],
           "err_M_max": abs(mm["M_max"] - ref) / ref}
    log.info("Fermi gas: M_max = %.4f M_sun (OV 1939: %.4f)",
             mm["M_max"], ref)
    return out


def validate_modern_eos(eos: EquationOfState, seq: Sequence) -> Dict[str, float]:
    """!Compare a modern EOS sequence with published global properties.

    @param eos  EOS instance ("SLy" or "APR(AP4)").
    @param seq  Sequence computed with tidal terms enabled.
    @return dict with numeric vs literature M_max, R_1.4, Lambda_1.4.
    """
    ref = LITERATURE[eos.name]
    mm = maximum_mass(eos, seq)
    star14 = solve_at_mass(eos, seq, 1.4)
    out = {
        "M_max_numeric": mm["M_max"], "M_max_literature": ref["M_max"],
        "err_M_max": abs(mm["M_max"] - ref["M_max"]) / ref["M_max"],
        "R_1.4_numeric": star14.R, "R_1.4_literature": ref["R_1.4"],
        "err_R_1.4": abs(star14.R - ref["R_1.4"]) / ref["R_1.4"],
        "Lambda_1.4_numeric": float(star14.Lambda or np.nan),
        "Lambda_1.4_literature": ref["Lambda_1.4"],
        "I_1.4_1e45_gcm2": float(star14.I_45 or np.nan),
        "eps_c_Mmax_cgs": mm["eps_c_cgs"],
    }
    log.info("%s: M_max=%.3f (lit %.2f)  R_1.4=%.2f km (lit %.2f)  "
             "Lambda_1.4=%.0f (lit %.0f)", eos.name, mm["M_max"],
             ref["M_max"], star14.R, ref["R_1.4"],
             out["Lambda_1.4_numeric"], ref["Lambda_1.4"])
    return out


# ----------------------------------------------------------------------
# Physical-consistency checks
# ----------------------------------------------------------------------
def buchdahl_check(seq: Sequence) -> Dict[str, float]:
    """!Verify the Buchdahl bound C = GM/Rc^2 < 4/9 along a sequence.

    @return dict with the maximum compactness and its margin to 4/9.
    """
    c = np.asarray(u.mass_msun_to_geom(seq.M)) / seq.R
    return {"C_max": float(np.max(c)),
            "buchdahl_bound": 4.0 / 9.0,
            "margin": float(4.0 / 9.0 - np.max(c)),
            "satisfied": bool(np.all(c < 4.0 / 9.0))}


def causality_check(eos: EquationOfState, eps_c_max_cgs: float) -> Dict[str, float]:
    """!Check c_s <= c up to the central pressure of the maximum-mass star.

    @param eos            Equation of state.
    @param eps_c_max_cgs  Central energy density of the M_max model [g/cm^3].
    @return dict with the maximum c_s^2/c^2 on the physically realized range.
    """
    p_max = float(eos.p_of_eps(float(u.density_cgs_to_geom(eps_c_max_cgs))))
    p = np.geomspace(eos.p_surface, p_max, 400)
    cs2 = np.asarray(eos.cs2_of_p(p))
    finite = cs2[np.isfinite(cs2)]
    return {"cs2_max": float(np.max(finite)),
            "causal": bool(np.all(finite <= 1.0 + 1e-10))}


def run_all_validations(seq_sly: Sequence, seq_apr: Sequence,
                        eos_sly: EquationOfState,
                        eos_apr: EquationOfState) -> List[Dict[str, object]]:
    """!Run every benchmark and return rows for the validation CSV table."""
    rows: List[Dict[str, object]] = []

    cd = validate_constant_density()
    rows.append({"benchmark": "interior Schwarzschild (M)", "numeric":
                 cd["M_numeric_Msun"], "reference": cd["M_analytic_Msun"],
                 "rel_error": cd["err_M"], "source": "analytic"})
    rows.append({"benchmark": "interior Schwarzschild (P profile)", "numeric":
                 cd["err_P_profile_max"], "reference": 0.0,
                 "rel_error": cd["err_P_profile_max"], "source": "analytic"})

    le = validate_newtonian_polytrope()
    rows.append({"benchmark": "Lane-Emden n=3/2 (R)", "numeric":
                 le["R_numeric_km"], "reference": le["R_laneemden_km"],
                 "rel_error": le["err_R"], "source": "Lane-Emden"})
    rows.append({"benchmark": "Lane-Emden n=3/2 (M)", "numeric":
                 le["M_numeric_Msun"], "reference": le["M_laneemden_Msun"],
                 "rel_error": le["err_M"], "source": "Lane-Emden"})

    fg = validate_fermi_gas()
    rows.append({"benchmark": "OV maximum mass (Fermi gas)", "numeric":
                 fg["M_max_numeric"], "reference": fg["M_max_literature"],
                 "rel_error": fg["err_M_max"],
                 "source": LITERATURE["fermi-gas"]["source"]})

    for eos, seq in ((eos_sly, seq_sly), (eos_apr, seq_apr)):
        mv = validate_modern_eos(eos, seq)
        src = LITERATURE[eos.name]["source"]
        rows.append({"benchmark": f"{eos.name} M_max", "numeric":
                     mv["M_max_numeric"], "reference": mv["M_max_literature"],
                     "rel_error": mv["err_M_max"], "source": src})
        rows.append({"benchmark": f"{eos.name} R(1.4 M_sun)", "numeric":
                     mv["R_1.4_numeric"], "reference": mv["R_1.4_literature"],
                     "rel_error": mv["err_R_1.4"], "source": src})
        rows.append({"benchmark": f"{eos.name} Lambda(1.4 M_sun)", "numeric":
                     mv["Lambda_1.4_numeric"],
                     "reference": mv["Lambda_1.4_literature"],
                     "rel_error": abs(mv["Lambda_1.4_numeric"]
                                      - mv["Lambda_1.4_literature"])
                     / mv["Lambda_1.4_literature"], "source": src})
    return rows
