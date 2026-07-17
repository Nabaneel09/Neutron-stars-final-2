r"""! @file eos.py
@brief Equations of state (EOS) for cold, catalysed neutron-star matter.

Six models are implemented, all exposing the same interface in
**geometrized units** (G = c = 1, lengths in km, so P and eps in km^-2):

1. `ConstantDensityEOS`     -- incompressible fluid (analytic benchmark);
2. `PolytropicEOS`          -- P = K rho^Gamma (includes the non-relativistic
                               degenerate-neutron-gas polytrope, Gamma = 5/3,
                               with K derived from first principles);
3. `FermiGasEOS`            -- full ideal relativistic neutron Fermi gas
                               (the EOS used by Oppenheimer & Volkoff 1939);
4. `PiecewisePolytropeEOS`  -- three-piece core parametrization of
                               Read, Lackey, Owen & Friedman (2009), matched
                               to the SLy crust;
5. `sly()`                  -- SLy4 unified EOS via the analytic fit of
                               Haensel & Potekhin (2004);
6. `apr()`                  -- APR (AP4) EOS as a Read et al. piecewise
                               polytrope over the SLy crust.

The thermodynamics is kept consistent: for tabulated/fitted models the
rest-mass (baryon) density is reconstructed from the first law of
thermodynamics at T = 0,
\f[
    d\rho_b/\rho_b = d\varepsilon / (\varepsilon + P),
\f]
and the sound speed is obtained as \f$ c_s^2 = dP/d\varepsilon \f$.

@section refs References
- S. L. Shapiro & S. A. Teukolsky, *Black Holes, White Dwarfs and Neutron
  Stars*, Wiley (1983), ch. 2 (Fermi gas).
- R. R. Silbar & S. Reddy, Am. J. Phys. 72, 892 (2004).
- P. Haensel & A. Y. Potekhin, A&A 428, 191 (2004) (SLy analytic fit).
- J. S. Read, B. D. Lackey, B. J. Owen, J. L. Friedman, PRD 79, 124032 (2009).
- F. Douchin & P. Haensel, A&A 380, 151 (2001) (SLy tables).
"""

from __future__ import annotations

import abc
from typing import Union

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

from . import constants as k
from . import units as u
from .utils import get_logger

log = get_logger(__name__)

ArrayLike = Union[float, np.ndarray]


class EOSError(ValueError):
    """!Raised when an EOS is queried outside its domain of validity."""


class EquationOfState(abc.ABC):
    """!Abstract base class for barotropic zero-temperature EOS.

    All quantities are in geometrized units (km^-2 for P, eps, rho).
    `rho` denotes the rest-mass (baryon) density, `eps` the total
    mass-energy density.
    """

    #: human-readable model name
    name: str = "abstract"

    @abc.abstractmethod
    def eps_of_p(self, p: ArrayLike) -> ArrayLike:
        """!Total energy density eps(P) [km^-2]."""

    @abc.abstractmethod
    def rho_of_p(self, p: ArrayLike) -> ArrayLike:
        """!Rest-mass density rho(P) [km^-2]."""

    @abc.abstractmethod
    def p_of_rho(self, rho: ArrayLike) -> ArrayLike:
        """!Pressure P(rho) [km^-2] from rest-mass density."""

    @abc.abstractmethod
    def p_of_eps(self, eps: ArrayLike) -> ArrayLike:
        """!Pressure P(eps) [km^-2] from total energy density."""

    @abc.abstractmethod
    def cs2_of_p(self, p: ArrayLike) -> ArrayLike:
        """!Squared adiabatic sound speed c_s^2 = dP/deps (units of c^2)."""

    #: smallest pressure at which the model is trusted (surface cut) [km^-2]
    p_surface: float = 0.0
    #: largest tabulated/valid pressure [km^-2] (np.inf if analytic)
    p_max: float = np.inf

    # ------------------------------------------------------------------
    def gamma_of_p(self, p: ArrayLike) -> ArrayLike:
        """!Relativistic adiabatic index Gamma_1 = (eps+P)/P * dP/deps.

        @param p  Pressure [km^-2].
        @return   Dimensionless adiabatic index.
        """
        p = np.asarray(p, dtype=float)
        eps = self.eps_of_p(p)
        return (eps + p) / np.maximum(p, 1e-300) * self.cs2_of_p(p)

    def validate(self) -> None:
        """!Cheap self-consistency checks (monotonicity, causal core range).

        @throws EOSError if P(eps) is non-monotonic on the sampled range.
        """
        p = np.geomspace(max(self.p_surface, 1e-15), min(self.p_max, 1e2), 200)
        eps = np.asarray(self.eps_of_p(p))
        if np.any(np.diff(eps) <= 0.0):
            raise EOSError(f"{self.name}: eps(P) is not strictly increasing")
        if np.any(eps <= 0.0):
            raise EOSError(f"{self.name}: non-positive energy density")


# ======================================================================
# 1. Constant density (incompressible)
# ======================================================================
class ConstantDensityEOS(EquationOfState):
    """!Incompressible fluid, eps(P) = eps_0 = const.

    The corresponding stellar model is the interior Schwarzschild
    solution, for which M(R) and P(r) are analytic -- used as the primary
    code validation benchmark. The sound speed is formally infinite
    (returned as `np.inf`); the model is acausal by construction.
    """

    def __init__(self, eps0: float) -> None:
        """!@param eps0  Constant energy density [km^-2] (> 0)."""
        if eps0 <= 0.0:
            raise EOSError("eps0 must be positive")
        self.eps0 = float(eps0)
        self.name = "constant-density"

    @classmethod
    def from_cgs(cls, rho_cgs: float) -> "ConstantDensityEOS":
        """!Build from a density in g/cm^3."""
        return cls(float(u.density_cgs_to_geom(rho_cgs)))

    def eps_of_p(self, p: ArrayLike) -> ArrayLike:
        return np.full_like(np.asarray(p, dtype=float), self.eps0)

    def rho_of_p(self, p: ArrayLike) -> ArrayLike:
        return np.full_like(np.asarray(p, dtype=float), self.eps0)

    def p_of_rho(self, rho: ArrayLike) -> ArrayLike:
        raise EOSError("P(rho) is undefined for an incompressible fluid; "
                       "specify the central pressure directly")

    def p_of_eps(self, eps: ArrayLike) -> ArrayLike:
        raise EOSError("P(eps) is undefined for an incompressible fluid; "
                       "specify the central pressure directly")

    def cs2_of_p(self, p: ArrayLike) -> ArrayLike:
        return np.full_like(np.asarray(p, dtype=float), np.inf)


# ======================================================================
# 2. Polytrope
# ======================================================================
class PolytropicEOS(EquationOfState):
    """!Polytropic EOS P = K rho^Gamma with eps = rho + P/(Gamma-1).

    The internal-energy term follows from the first law of thermodynamics
    for an adiabatic fluid, so that c_s^2 = Gamma P / (eps + P) exactly.
    """

    def __init__(self, k_geom: float, gamma: float, name: str = "polytrope") -> None:
        """!
        @param k_geom  Polytropic constant in geometrized units
                       ([km^-2]^(1-Gamma)).
        @param gamma   Adiabatic exponent Gamma > 1.
        @param name    Model label.
        """
        if gamma <= 1.0:
            raise EOSError("Gamma must exceed 1")
        if k_geom <= 0.0:
            raise EOSError("K must be positive")
        self.k_geom = float(k_geom)
        self.gamma = float(gamma)
        self.name = name

    # -- constructors ---------------------------------------------------
    @classmethod
    def from_si(cls, k_si: float, gamma: float, name: str = "polytrope") -> "PolytropicEOS":
        """!Build from K in SI units (P[Pa] = K rho[kg/m^3]^Gamma).

        The geometrized constant follows from
        K_geom = A * K_SI / B^Gamma with A = G/c^4 * 1e6, B = G/c^2 * 1e6.
        """
        a = k.G / k.C**4 * k.KM**2
        b = k.G / k.C**2 * k.KM**2
        return cls(a * k_si / b**gamma, gamma, name)

    @classmethod
    def neutron_nonrel(cls) -> "PolytropicEOS":
        r"""!Non-relativistic degenerate neutron gas, Gamma = 5/3.

        From first principles (Pauli blocking of an ideal Fermi gas),
        \f[
           P = \frac{(3\pi^2)^{2/3}}{5}\frac{\hbar^2}{m_n^{8/3}}\rho^{5/3},
        \f]
        the leading term of the Fermi-gas expansion at x = p_F/m_n c << 1.
        """
        k_si = (3.0 * np.pi**2) ** (2.0 / 3.0) * k.HBAR**2 / (
            5.0 * k.M_NEUTRON ** (8.0 / 3.0)
        )
        return cls.from_si(k_si, 5.0 / 3.0, name="polytrope-n3/2")

    # -- interface -------------------------------------------------------
    def p_of_rho(self, rho: ArrayLike) -> ArrayLike:
        rho = np.asarray(rho, dtype=float)
        return self.k_geom * rho**self.gamma

    def rho_of_p(self, p: ArrayLike) -> ArrayLike:
        p = np.maximum(np.asarray(p, dtype=float), 0.0)
        return (p / self.k_geom) ** (1.0 / self.gamma)

    def eps_of_p(self, p: ArrayLike) -> ArrayLike:
        p = np.maximum(np.asarray(p, dtype=float), 0.0)
        return self.rho_of_p(p) + p / (self.gamma - 1.0)

    def p_of_eps(self, eps: ArrayLike) -> ArrayLike:
        """!Invert eps(P) with a bracketed root find (vectorized via loop)."""
        eps_arr = np.atleast_1d(np.asarray(eps, dtype=float))
        out = np.empty_like(eps_arr)
        for i, e in enumerate(eps_arr):
            if e <= 0.0:
                raise EOSError("eps must be positive")
            out[i] = brentq(lambda p, e=e: self.eps_of_p(p) - e, 0.0, 10.0 * e,
                            xtol=1e-300, rtol=1e-14)
        return out[0] if np.isscalar(eps) or np.ndim(eps) == 0 else out

    def cs2_of_p(self, p: ArrayLike) -> ArrayLike:
        p = np.maximum(np.asarray(p, dtype=float), 0.0)
        return self.gamma * p / (self.eps_of_p(p) + p)


# ======================================================================
# Tabulated backbone (used by Fermi gas, SLy fit and piecewise polytropes)
# ======================================================================
class TabulatedEOS(EquationOfState):
    """!Monotone log-log PCHIP interpolation of a (rho, P, eps) table.

    PCHIP (monotone cubic Hermite) interpolation is used because it
    preserves the monotonicity of the thermodynamic relations -- a cubic
    spline can overshoot near piecewise-polytrope kinks and produce a
    locally negative dP/deps, which breaks the TOV right-hand side.
    """

    def __init__(self, rho: np.ndarray, p: np.ndarray, eps: np.ndarray,
                 name: str, surface_rho_cgs: float = 1.0e7) -> None:
        """!
        @param rho  Rest-mass density table [km^-2], strictly increasing.
        @param p    Pressure table [km^-2], strictly increasing.
        @param eps  Energy density table [km^-2], strictly increasing.
        @param name Model label.
        @param surface_rho_cgs  Rest-mass density [g/cm^3] defining the
               numerical stellar surface; the residual envelope below this
               density is geometrically negligible (< a few metres).
        """
        rho, p, eps = (np.asarray(x, dtype=float) for x in (rho, p, eps))
        if np.any(np.diff(p) <= 0) or np.any(np.diff(rho) <= 0) or np.any(np.diff(eps) <= 0):
            raise EOSError(f"{name}: table columns must be strictly increasing")
        self.name = name
        self._lp, self._lrho, self._leps = np.log(p), np.log(rho), np.log(eps)
        self._leps_of_lp = PchipInterpolator(self._lp, self._leps, extrapolate=True)
        self._lrho_of_lp = PchipInterpolator(self._lp, self._lrho, extrapolate=True)
        self._lp_of_lrho = PchipInterpolator(self._lrho, self._lp, extrapolate=True)
        self._lp_of_leps = PchipInterpolator(self._leps, self._lp, extrapolate=True)
        self._dleps_dlp = self._leps_of_lp.derivative()
        self.p_max = float(p[-1])
        self.p_min = float(p[0])
        rho_s = float(u.density_cgs_to_geom(surface_rho_cgs))
        self.p_surface = float(self.p_of_rho(max(rho_s, float(rho[0]) * 1.001)))

    def _clip_lp(self, p: ArrayLike) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        if np.any(p <= 0.0):
            raise EOSError(f"{self.name}: non-positive pressure query")
        return np.clip(np.log(p), self._lp[0], self._lp[-1])

    def eps_of_p(self, p: ArrayLike) -> ArrayLike:
        return np.exp(self._leps_of_lp(self._clip_lp(p)))

    def rho_of_p(self, p: ArrayLike) -> ArrayLike:
        return np.exp(self._lrho_of_lp(self._clip_lp(p)))

    def p_of_rho(self, rho: ArrayLike) -> ArrayLike:
        rho = np.asarray(rho, dtype=float)
        if np.any(rho <= 0.0):
            raise EOSError(f"{self.name}: non-positive density query")
        lr = np.clip(np.log(rho), self._lrho[0], self._lrho[-1])
        return np.exp(self._lp_of_lrho(lr))

    def p_of_eps(self, eps: ArrayLike) -> ArrayLike:
        eps = np.asarray(eps, dtype=float)
        if np.any(eps <= 0.0):
            raise EOSError(f"{self.name}: non-positive energy density query")
        le = np.clip(np.log(eps), self._leps[0], self._leps[-1])
        return np.exp(self._lp_of_leps(le))

    def cs2_of_p(self, p: ArrayLike) -> ArrayLike:
        """!c_s^2 = dP/deps = (P/eps) / (d ln eps / d ln P)."""
        lp = self._clip_lp(p)
        slope = np.maximum(self._dleps_dlp(lp), 1e-12)
        return np.exp(lp - self._leps_of_lp(lp)) / slope


# ======================================================================
# 3. Ideal relativistic neutron Fermi gas (Oppenheimer-Volkoff 1939)
# ======================================================================
class FermiGasEOS(TabulatedEOS):
    r"""!Ideal, fully degenerate relativistic neutron gas.

    Parametric form in the relativity parameter x = p_F / (m_n c):
    \f{eqnarray*}{
      \varepsilon(x) &=& K\,[x\sqrt{1+x^2}(1+2x^2)-\sinh^{-1}x],\\
      P(x) &=& \tfrac{K}{3}[x\sqrt{1+x^2}(2x^2-3)+3\sinh^{-1}x],\\
      n(x) &=& x^3 (m_n c/\hbar)^3/(3\pi^2),
    \f}
    with K = m_n^4 c^5 / (8 pi^2 hbar^3). This is the EOS with which
    Oppenheimer & Volkoff obtained M_max ~= 0.71 M_sun in 1939.
    """

    def __init__(self, x_min: float = 1.0e-3, x_max: float = 2.0e2,
                 n_points: int = 4000) -> None:
        """!
        @param x_min     Lowest relativity parameter tabulated.
        @param x_max     Highest relativity parameter tabulated.
        @param n_points  Log-spaced table size.
        """
        x = np.geomspace(x_min, x_max, n_points)
        big_k = k.M_NEUTRON**4 * k.C**5 / (8.0 * np.pi**2 * k.HBAR**3)  # J/m^3
        sq = np.sqrt(1.0 + x**2)
        eps_si = big_k * (x * sq * (1.0 + 2.0 * x**2) - np.arcsinh(x))
        p_si = big_k / 3.0 * (x * sq * (2.0 * x**2 - 3.0) + 3.0 * np.arcsinh(x))
        n_si = x**3 * (k.M_NEUTRON * k.C / k.HBAR) ** 3 / (3.0 * np.pi**2)
        rho_si = n_si * k.M_NEUTRON
        super().__init__(
            rho=u.density_si_to_geom(rho_si),
            p=u.pressure_si_to_geom(p_si),
            eps=u.energy_density_si_to_geom(eps_si),
            name="fermi-gas",
            surface_rho_cgs=1.0,
        )
        # true surface: integrate down to a tiny fraction of the table
        self.p_surface = self.p_min * 10.0


# ======================================================================
# SLy analytic fit of Haensel & Potekhin (2004)
# ======================================================================
#: fit coefficients a_1..a_18 of Haensel & Potekhin (2004), eq. (14)
_HP_SLY_COEFFS = np.array([
    6.22, 6.121, 0.005925, 0.16326, 6.48, 11.4971, 19.105, 0.8938,
    6.54, 11.4950, -22.775, 1.5707, 4.3, 14.08, 27.80, -1.653, 1.50, 14.67,
])


def hp_sly_pressure_cgs(rho_cgs: ArrayLike) -> np.ndarray:
    """!SLy pressure from the Haensel-Potekhin (2004) analytic fit.

    @param rho_cgs  Total mass(-energy) density [g/cm^3], valid for
                    approximately 1e5 < rho < 4e15.
    @return         Pressure [dyn/cm^2].
    """
    a = _HP_SLY_COEFFS
    xi = np.log10(np.asarray(rho_cgs, dtype=float))

    def f0(t: np.ndarray) -> np.ndarray:
        # logistic switching function; clip to avoid overflow in exp
        return 1.0 / (np.exp(np.clip(t, -60.0, 60.0)) + 1.0)

    zeta = ((a[0] + a[1] * xi + a[2] * xi**3) / (1.0 + a[3] * xi)
            * f0(a[4] * (xi - a[5]))
            + (a[6] + a[7] * xi) * f0(a[8] * (a[9] - xi))
            + (a[10] + a[11] * xi) * f0(a[12] * (a[13] - xi))
            + (a[14] + a[15] * xi) * f0(a[16] * (a[17] - xi)))
    return 10.0**zeta


def _first_law_rest_mass(eps: np.ndarray, p: np.ndarray) -> np.ndarray:
    """!Reconstruct the rest-mass density from the T = 0 first law.

    Integrates d ln(rho_b) = d(eps) / (eps + P) on the given grid, with
    the boundary condition rho_b -> eps at the lowest (crust) point where
    P << eps and internal energy is negligible.

    @param eps  Energy density grid [any consistent units], increasing.
    @param p    Pressure on the same grid.
    @return     Rest-mass density on the grid (same units as eps).
    """
    integrand = 1.0 / (eps + p)
    lnrho = np.concatenate(([0.0], cumulative_trapezoid(integrand, eps)))
    return eps[0] * np.exp(lnrho)


def sly(n_points: int = 3000, rho_min_cgs: float = 1.0e6,
        rho_max_cgs: float = 4.0e15) -> TabulatedEOS:
    """!SLy4 unified EOS from the Haensel-Potekhin analytic representation.

    The fit variable is the total mass-energy density rho = eps/c^2; the
    rest-mass density is reconstructed from the first law
    (`_first_law_rest_mass`), which keeps c_s^2 = dP/deps and the baryon
    mass thermodynamically consistent.

    @param n_points      Table resolution (log-spaced).
    @param rho_min_cgs   Lower table edge [g/cm^3].
    @param rho_max_cgs   Upper table edge [g/cm^3].
    @return              `TabulatedEOS` instance named "SLy".
    """
    rho_cgs = np.geomspace(rho_min_cgs, rho_max_cgs, n_points)
    p_geom = np.asarray(u.pressure_cgs_to_geom(hp_sly_pressure_cgs(rho_cgs)))
    eps_geom = np.asarray(u.density_cgs_to_geom(rho_cgs))
    rho_b = _first_law_rest_mass(eps_geom, p_geom)
    return TabulatedEOS(rho_b, p_geom, eps_geom, name="SLy")


# ======================================================================
# 4/6. Piecewise polytropes (Read et al. 2009) over the SLy crust
# ======================================================================
#: dividing rest-mass densities of the Read et al. core parametrization
_RHO1_CGS = 10.0**14.7
_RHO2_CGS = 10.0**15.0


class PiecewisePolytropeEOS(TabulatedEOS):
    r"""!Three-piece polytropic core (Read et al. 2009) + SLy crust.

    The core is parametrized by (log10 p1, Gamma1, Gamma2, Gamma3) where
    p1 = P(rho1) in dyn/cm^2 at rho1 = 10^14.7 g/cm^3, with piece
    boundaries at rho1 and rho2 = 10^15 g/cm^3. Energy-density continuity
    constants a_i enforce a continuous eps(rho) across each boundary:
    \f[
      \varepsilon_i(\rho) = (1+a_i)\rho c^2 + \frac{K_i \rho^{\Gamma_i}}{\Gamma_i-1}.
    \f]
    Below the crust-core junction (where the Gamma1 polytrope pressure
    equals the SLy-fit pressure) the Haensel-Potekhin SLy crust is used.
    """

    def __init__(self, logp1: float, g1: float, g2: float, g3: float,
                 name: str, n_points: int = 4000) -> None:
        """!
        @param logp1  log10 of the pressure [dyn/cm^2] at rho = 10^14.7 g/cm^3.
        @param g1     Adiabatic index of the first core piece.
        @param g2     Adiabatic index of the second core piece.
        @param g3     Adiabatic index of the third core piece.
        @param name   Model label (e.g. "AP4-pp").
        @param n_points  Total table resolution.
        """
        c2 = k.C**2 * 1.0e4  # (cm/s)^2, converts g/cm^3 -> erg/cm^3
        p1 = 10.0**logp1
        kk = [p1 / _RHO1_CGS**g1, p1 / _RHO1_CGS**g2, 0.0]
        kk[2] = kk[1] * _RHO2_CGS ** (g2 - g3)
        gammas, ks = (g1, g2, g3), tuple(kk)

        # --- crust: HP fit up to the top of the inner crust, then the
        #     last crust polytrope extended until it meets the core piece
        #     (the matching prescription of Read et al. 2009, Sec. III.A) --
        rho_c0 = 2.62780e12  # g/cm^3, top of the Read et al. crust fit
        dlr = 0.01
        g_c = (np.log(float(hp_sly_pressure_cgs(rho_c0 * (1 + dlr))))
               - np.log(float(hp_sly_pressure_cgs(rho_c0 * (1 - dlr))))) \
            / (np.log(1 + dlr) - np.log(1 - dlr))
        k_c = float(hp_sly_pressure_cgs(rho_c0)) / rho_c0**g_c
        if g_c >= g1:  # pragma: no cover - defensive
            raise EOSError(f"{name}: crust slope {g_c:.2f} >= Gamma1 {g1}")
        rho_j = (k_c / kk[0]) ** (1.0 / (g1 - g_c))
        if not rho_c0 < rho_j < _RHO1_CGS:
            raise EOSError(f"{name}: junction {rho_j:.3e} outside "
                           f"({rho_c0:.2e}, {_RHO1_CGS:.2e}) g/cm^3")
        log.info("%s: crust-core junction at rho = %.3e g/cm^3 "
                 "(crust Gamma = %.3f)", name, rho_j, g_c)

        # --- energy-density continuity constants -------------------------
        # eps = rho c^2 at the crust anchor (P << eps there); each further
        # piece i carries a_i so that eps(rho) is continuous
        a_c = -k_c * rho_c0 ** (g_c - 1.0) / ((g_c - 1.0) * c2)

        def eps_ext(rho: ArrayLike) -> np.ndarray:
            rho = np.asarray(rho, dtype=float)
            return (1.0 + a_c) * rho * c2 + k_c * rho**g_c / (g_c - 1.0)

        a = np.zeros(3)
        a[0] = float(eps_ext(rho_j)) / (rho_j * c2) - 1.0 \
            - kk[0] * rho_j ** (g1 - 1.0) / ((g1 - 1.0) * c2)

        def eps_piece(i: int, rho: ArrayLike) -> np.ndarray:
            rho = np.asarray(rho, dtype=float)
            return (1.0 + a[i]) * rho * c2 + ks[i] * rho ** gammas[i] / (gammas[i] - 1.0)

        a[1] = eps_piece(0, _RHO1_CGS) / (_RHO1_CGS * c2) - 1.0 \
            - kk[1] * _RHO1_CGS ** (g2 - 1.0) / ((g2 - 1.0) * c2)
        a[2] = eps_piece(1, _RHO2_CGS) / (_RHO2_CGS * c2) - 1.0 \
            - kk[2] * _RHO2_CGS ** (g3 - 1.0) / ((g3 - 1.0) * c2)

        # --- assemble the full table (crust + extension + 3 core pieces) --
        rho_crust = np.geomspace(1.0e6, rho_c0, n_points // 3, endpoint=False)
        p_crust = hp_sly_pressure_cgs(rho_crust)
        eps_crust = rho_crust * c2  # erg/cm^3 (internal energy negligible)

        rho_ext = np.geomspace(rho_c0, rho_j, n_points // 6, endpoint=False)
        p_ext = k_c * rho_ext**g_c
        eps_e = eps_ext(rho_ext)

        rho_core = np.geomspace(rho_j, 4.0e15, n_points // 2)
        p_core = np.empty_like(rho_core)
        eps_core = np.empty_like(rho_core)
        for j, r in enumerate(rho_core):
            i = 0 if r < _RHO1_CGS else (1 if r < _RHO2_CGS else 2)
            p_core[j] = ks[i] * r ** gammas[i]
            eps_core[j] = eps_piece(i, r)

        rho_all = np.concatenate([rho_crust, rho_ext, rho_core])
        p_all = np.concatenate([p_crust, p_ext, p_core])
        eps_all = np.concatenate([eps_crust, eps_e, eps_core])

        super().__init__(
            rho=u.density_cgs_to_geom(rho_all),
            p=u.pressure_cgs_to_geom(p_all),
            eps=u.pressure_cgs_to_geom(eps_all),  # erg/cm^3 == dyn/cm^2
            name=name,
        )
        self.core_params = {"logp1": logp1, "Gamma": gammas, "rho_junction_cgs": rho_j}


def apr(n_points: int = 4000) -> PiecewisePolytropeEOS:
    """!APR (AP4) EOS: Read et al. (2009) fit, Table III.

    AP4 is the piecewise-polytrope representation of the
    Akmal-Pandharipande-Ravenhall (1998) A18+delta v+UIX* EOS.
    """
    return PiecewisePolytropeEOS(34.269, 2.830, 3.445, 3.348, "APR(AP4)", n_points)


def sly_pp(n_points: int = 4000) -> PiecewisePolytropeEOS:
    """!SLy as a Read et al. piecewise polytrope (cross-check of `sly`)."""
    return PiecewisePolytropeEOS(34.384, 3.005, 2.988, 2.851, "SLy-pp", n_points)


# ======================================================================
# Factory
# ======================================================================
def get_eos(name: str) -> EquationOfState:
    """!Return an EOS instance by name.

    @param name  One of: "constant" (1e15 g/cm^3), "poly-nr" (neutron
                 Gamma=5/3 polytrope), "fermi", "sly", "sly-pp", "apr".
    @throws EOSError for unknown names.
    """
    key = name.lower()
    if key in ("constant", "incompressible"):
        return ConstantDensityEOS.from_cgs(1.0e15)
    if key in ("poly-nr", "polytrope"):
        return PolytropicEOS.neutron_nonrel()
    if key in ("fermi", "fermi-gas", "ov"):
        return FermiGasEOS()
    if key == "sly":
        return sly()
    if key == "sly-pp":
        return sly_pp()
    if key in ("apr", "ap4"):
        return apr()
    raise EOSError(f"unknown EOS '{name}'")
