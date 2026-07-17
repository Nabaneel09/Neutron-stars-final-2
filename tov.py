r"""! @file tov.py
@brief Tolman-Oppenheimer-Volkoff structure equations and stellar models.

Integrates, in a single pass from the centre to the surface,

- the TOV pressure and mass equations,
- the metric potential Phi(r) (g_tt = -e^{2 Phi}),
- the baryon (rest) mass  dm_b/dr = 4 pi r^2 rho (1-2m/r)^{-1/2},
- the quadrupole tidal response y(r) (Hinderer 2008; Postnikov et al. 2010),
- the Hartle slow-rotation frame-dragging equation for the moment of inertia,

in geometrized units (G = c = 1, lengths in km). The Newtonian structure
equations are available with `newtonian=True` for direct comparison.

@section eqs Equations solved
\f{eqnarray*}{
 dm/dr   &=& 4\pi r^2 \varepsilon \\
 dP/dr   &=& -(\varepsilon+P)\frac{m+4\pi r^3 P}{r(r-2m)} \\
 d\Phi/dr&=& \frac{m+4\pi r^3 P}{r(r-2m)}
\f}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.integrate import solve_ivp

from . import constants as k
from . import units as u
from .eos import EquationOfState
from .solver import integrate_fixed
from .utils import get_logger

log = get_logger(__name__)

FOUR_PI = 4.0 * np.pi

#: SciPy methods accepted by solve_star
ADAPTIVE_METHODS = ("RK45", "DOP853", "Radau", "BDF", "LSODA")
#: hand-written fixed-step methods (see solver.py)
FIXED_METHODS = ("euler", "heun", "rk4")


class IntegrationError(RuntimeError):
    """!Raised when the surface P = P_surf is not reached before r_max."""


# ----------------------------------------------------------------------
@dataclass
class StarModel:
    """!Result container for a single stellar model.

    Radii in km, masses in solar masses unless suffixed otherwise.
    Profile arrays run from the centre (index 0) to the surface.
    """

    eos_name: str
    method: str
    newtonian: bool
    # central values (geometrized)
    p_c: float
    eps_c: float
    rho_c: float
    # global quantities
    R: float  #: circumferential radius [km]
    M: float  #: gravitational mass [M_sun]
    M_km: float  #: gravitational mass [km]
    Mb: float  #: baryon mass [M_sun]
    # profiles (geometrized; r in km)
    r: np.ndarray
    p: np.ndarray
    m: np.ndarray
    eps: np.ndarray
    rho: np.ndarray
    phi: np.ndarray  #: normalized so e^{2 phi(R)} = 1 - 2M/R
    # optional extras
    y_R: Optional[float] = None  #: tidal response at the surface
    k2: Optional[float] = None  #: quadrupole tidal Love number
    Lambda: Optional[float] = None  #: dimensionless tidal deformability
    I_kgm2: Optional[float] = None  #: moment of inertia [kg m^2]
    n_rhs: int = 0  #: number of RHS evaluations used
    meta: dict = field(default_factory=dict)

    # -- derived global quantities ------------------------------------
    @property
    def compactness(self) -> float:
        """!Dimensionless compactness C = GM/(Rc^2)."""
        return self.M_km / self.R

    @property
    def z_surf(self) -> float:
        """!Gravitational redshift of a photon emitted at the surface."""
        return 1.0 / np.sqrt(1.0 - 2.0 * self.compactness) - 1.0

    @property
    def binding_energy(self) -> float:
        """!Gravitational binding energy E_b = (M_b - M) [M_sun c^2]."""
        return self.Mb - self.M

    @property
    def v_escape(self) -> float:
        """!Newtonian escape velocity at the surface, in units of c."""
        return float(np.sqrt(2.0 * self.compactness))

    @property
    def I_45(self) -> Optional[float]:
        """!Moment of inertia in units of 10^45 g cm^2."""
        return None if self.I_kgm2 is None else self.I_kgm2 * 1.0e7 / 1.0e45

    # -- derived profiles ----------------------------------------------
    def metric_gtt(self) -> np.ndarray:
        """!Metric coefficient g_tt = -e^{2 Phi(r)}."""
        return -np.exp(2.0 * self.phi)

    def metric_grr(self) -> np.ndarray:
        """!Metric coefficient g_rr = (1 - 2m/r)^{-1}."""
        with np.errstate(divide="ignore", invalid="ignore"):
            grr = 1.0 / (1.0 - 2.0 * self.m / self.r)
        grr[self.r == 0.0] = 1.0
        return grr

    def local_gravity(self) -> np.ndarray:
        """!Proper acceleration of a static observer, g(r) [m/s^2].

        g = (m + 4 pi r^3 P) / (r^2 sqrt(1-2m/r)) in geometrized km^-1,
        converted to SI.
        """
        r = np.where(self.r > 0.0, self.r, np.inf)
        geo = (self.m + FOUR_PI * r**3 * self.p) / (
            r**2 * np.sqrt(1.0 - 2.0 * self.m / r)
        )
        return geo * k.C**2 / k.KM  # km^-1 -> m/s^2


# ----------------------------------------------------------------------
def _series_start(eos: EquationOfState, p_c: float, r0: float,
                  tidal: bool, inertia: bool, newtonian: bool) -> np.ndarray:
    """!Initial state at r = r0 from the regular series expansion.

    Uses m ~ (4 pi/3) eps_c r^3 and
    P ~ P_c - (2 pi/3)(eps_c+P_c)(eps_c+3P_c) r^2 (relativistic) to avoid
    the coordinate singularity of the TOV RHS at r = 0.
    """
    eps_c = float(eos.eps_of_p(p_c))
    rho_c = float(eos.rho_of_p(p_c))
    if newtonian:
        p0 = p_c - (2.0 * np.pi / 3.0) * rho_c**2 * r0**2
        return np.array([p0, FOUR_PI / 3.0 * rho_c * r0**3])
    p0 = p_c - (2.0 * np.pi / 3.0) * (eps_c + p_c) * (eps_c + 3.0 * p_c) * r0**2
    m0 = FOUR_PI / 3.0 * eps_c * r0**3
    mb0 = FOUR_PI / 3.0 * rho_c * r0**3
    y = [p0, m0, 0.0, mb0]
    if tidal:
        y.append(2.0)  # y(0) = 2 for the l = 2 perturbation
    if inertia:
        w2 = (8.0 * np.pi / 5.0) * (eps_c + p_c)
        y.append(1.0 + w2 * r0**2)  # omega-bar (arbitrary scale)
        y.append(2.0 * w2 * r0**5)  # v = r^4 j d(omega-bar)/dr, j ~= 1
    return np.array(y)


def _series_start_regularized(eos: EquationOfState, p_c: float, r0: float,
                              newtonian: bool) -> np.ndarray:
    """!Series start for the regularized state [P, u, Phi, m_b], u = m/r^3.

    Includes the O(r^2) corrections
    P = P_c + p_2 r^2, u = (4 pi/3) eps_c + (4 pi/5) eps_2 r^2 with
    eps_2 = p_2 / c_s^2(P_c), so the start error is O(r0^4) -- required
    to preserve the fourth order of RK4 when starting at r0 = h.
    """
    eps_c = float(eos.eps_of_p(p_c))
    rho_c = float(eos.rho_of_p(p_c))
    cs2_c = float(eos.cs2_of_p(p_c))
    inv_cs2 = 0.0 if not np.isfinite(cs2_c) else 1.0 / cs2_c
    if newtonian:
        p2 = -(2.0 * np.pi / 3.0) * rho_c**2
        rho2 = p2 * rho_c * inv_cs2 / (eps_c + p_c)
        u0 = FOUR_PI / 3.0 * rho_c + (FOUR_PI / 5.0) * rho2 * r0**2
        return np.array([p_c + p2 * r0**2, u0, 0.0, 0.0])
    p2 = -(2.0 * np.pi / 3.0) * (eps_c + p_c) * (eps_c + 3.0 * p_c)
    eps2 = p2 * inv_cs2
    rho2 = p2 * rho_c * inv_cs2 / (eps_c + p_c)
    u0 = FOUR_PI / 3.0 * eps_c + (FOUR_PI / 5.0) * eps2 * r0**2
    phi0 = (2.0 * np.pi / 3.0) * (eps_c + 3.0 * p_c) * r0**2
    mb0 = FOUR_PI / 3.0 * rho_c * r0**3 + (FOUR_PI / 5.0) * rho2 * r0**5
    return np.array([p_c + p2 * r0**2, u0, phi0, mb0])


def _make_rhs_regularized(eos: EquationOfState, p_floor: float,
                          newtonian: bool,
                          counter: list) -> Callable[[float, np.ndarray], np.ndarray]:
    r"""!Regularized TOV RHS with u = m/r^3 (smooth at the origin).

    The plain system has Jacobian entries df/dm ~ 1/r^2 near the centre,
    which amplify the O(h^5) local error of a fixed-step method to
    O(h^2) globally. In terms of u = m/r^3,
    \f{eqnarray*}{
      dP/dr &=& -(\varepsilon+P)\, r\,(u+4\pi P)/(1-2ur^2),\\
      du/dr &=& (4\pi\varepsilon - 3u)/r,
    \f}
    every coefficient is bounded and smooth at r -> 0 (the numerator of
    du/dr vanishes at the centre), so RK4 retains its nominal order.
    State: [P, u, Phi, m_b].
    """

    def rhs(r: float, z: np.ndarray) -> np.ndarray:
        counter[0] += 1
        p, uvar = z[0], z[1]
        out = np.zeros_like(z)
        if r <= 0.0:
            return out
        p_q = max(p, p_floor)
        rho = float(eos.rho_of_p(p_q))
        if newtonian:
            out[0] = -rho * uvar * r
            out[1] = (FOUR_PI * rho - 3.0 * uvar) / r
            return out
        eps = float(eos.eps_of_p(p_q))
        e_fac = 1.0 - 2.0 * uvar * r**2
        if e_fac <= 0.0:
            return out
        dphi = r * (uvar + FOUR_PI * p) / e_fac
        out[0] = -(eps + p) * dphi
        out[1] = (FOUR_PI * eps - 3.0 * uvar) / r
        out[2] = dphi
        out[3] = FOUR_PI * r**2 * rho / np.sqrt(e_fac)
        return out

    return rhs


def _make_rhs(eos: EquationOfState, p_floor: float,
              tidal: bool, inertia: bool, newtonian: bool,
              counter: list) -> Callable[[float, np.ndarray], np.ndarray]:
    """!Build the TOV (or Newtonian) right-hand side closure.

    State layout: [P, m, Phi, m_b, (y_tidal), (omega-bar, v)]
    (Newtonian: [P, m]).
    """

    def rhs(r: float, y: np.ndarray) -> np.ndarray:
        counter[0] += 1
        p = y[0]
        out = np.zeros_like(y)
        if r <= 0.0:
            return out
        # Clamp only the EOS *lookups* at a positive floor. Steps and
        # internal RK stages may legitimately probe P <= 0 just outside
        # the surface; returning a zero RHS there would make f
        # discontinuous and destroy both the event location and the
        # convergence order. With the clamp, f is a smooth continuation.
        p_q = max(p, p_floor)
        if newtonian:
            rho = float(eos.rho_of_p(p_q))
            out[0] = -y[1] * rho / r**2
            out[1] = FOUR_PI * r**2 * rho
            return out

        m = y[1]
        eps = float(eos.eps_of_p(p_q))
        rho = float(eos.rho_of_p(p_q))
        e_fac = 1.0 - 2.0 * m / r
        if e_fac <= 0.0:  # inside the Schwarzschild bound: unphysical
            return out
        dphi = (m + FOUR_PI * r**3 * p) / (r**2 * e_fac)
        out[0] = -(eps + p) * dphi  # TOV pressure equation
        out[1] = FOUR_PI * r**2 * eps  # mass continuity
        out[2] = dphi  # metric potential
        out[3] = FOUR_PI * r**2 * rho / np.sqrt(e_fac)  # baryon mass

        idx = 4
        if tidal:
            yt = y[idx]
            cs2 = float(eos.cs2_of_p(p_q))
            inv_cs2 = 0.0 if not np.isfinite(cs2) else 1.0 / max(cs2, 1e-12)
            f_r = (1.0 - FOUR_PI * r**2 * (eps - p)) / e_fac
            q_r = (FOUR_PI * (5.0 * eps + 9.0 * p + (eps + p) * inv_cs2)
                   - 6.0 / r**2) / e_fac - 4.0 * dphi**2
            out[idx] = -(yt**2 + yt * f_r + r**2 * q_r) / r
            idx += 1
        if inertia:
            w, v = y[idx], y[idx + 1]
            j = np.exp(-y[2]) * np.sqrt(e_fac)
            dlnj = -dphi + (m / r**2 - out[1] / r) / e_fac
            out[idx] = v / (r**4 * j)
            out[idx + 1] = -4.0 * r**3 * j * dlnj * w
        return out

    return rhs


# ----------------------------------------------------------------------
def solve_star(
    eos: EquationOfState,
    eps_c: Optional[float] = None,
    p_c: Optional[float] = None,
    rho_c: Optional[float] = None,
    *,
    method: str = "DOP853",
    rtol: float = 1.0e-9,
    atol_p_frac: float = 1.0e-14,
    r0: float = 1.0e-6,
    r_max: float = 250.0,
    h: Optional[float] = None,
    tidal: bool = True,
    inertia: bool = True,
    newtonian: bool = False,
    p_surface: Optional[float] = None,
) -> StarModel:
    """!Integrate one stellar model from the centre to the surface.

    Exactly one of `eps_c` (central energy density), `rho_c` (central
    rest-mass density) or `p_c` (central pressure), all in geometrized
    km^-2, must be given.

    @param eos         Equation of state instance.
    @param eps_c       Central energy density [km^-2].
    @param p_c         Central pressure [km^-2] (mandatory for the
                       incompressible EOS, where eps does not fix P).
    @param rho_c       Central rest-mass density [km^-2].
    @param method      "RK45" | "DOP853" | "Radau" | "BDF" | "LSODA"
                       (adaptive, SciPy) or "euler" | "heun" | "rk4"
                       (fixed step, requires `h`).
    @param rtol        Relative tolerance of the adaptive integrator.
    @param atol_p_frac Absolute pressure tolerance as a fraction of P_c.
    @param r0          Series-expansion starting radius [km].
    @param r_max       Abort radius [km].
    @param h           Fixed step size [km] for euler/heun/rk4.
    @param tidal       Integrate the tidal response y(r) and compute k2.
    @param inertia     Integrate the Hartle frame-dragging equation.
    @param newtonian   Solve Newtonian hydrostatics instead of TOV.
    @param p_surface   Override the EOS surface-pressure cut [km^-2].
    @return            Fully populated `StarModel`.
    @throws ValueError        for inconsistent central conditions/methods.
    @throws IntegrationError  if the surface is not reached.

    Complexity: O(N_steps) RHS calls; adaptive N_steps ~ 1e2-1e4.
    """
    n_given = sum(x is not None for x in (eps_c, p_c, rho_c))
    if n_given != 1:
        raise ValueError("give exactly one of eps_c, p_c, rho_c")
    if p_c is None:
        p_c = float(eos.p_of_eps(eps_c)) if eps_c is not None \
            else float(eos.p_of_rho(rho_c))
    p_c = float(p_c)
    if p_c <= 0.0:
        raise ValueError("central pressure must be positive")

    if newtonian:
        tidal = inertia = False
    p_surf = float(p_surface if p_surface is not None
                   else max(eos.p_surface, p_c * 1.0e-12))
    if p_surf >= p_c:
        raise ValueError("surface pressure exceeds central pressure")

    counter = [0]
    rhs = _make_rhs(eos, 0.5 * p_surf, tidal, inertia, newtonian, counter)
    y0 = _series_start(eos, p_c, r0, tidal, inertia, newtonian)

    if method in ADAPTIVE_METHODS:
        def surface(r: float, y: np.ndarray) -> float:
            return y[0] - p_surf
        surface.terminal = True  # type: ignore[attr-defined]
        surface.direction = -1  # type: ignore[attr-defined]
        atol = np.full_like(y0, 1.0e-12)
        atol[0] = p_c * atol_p_frac
        sol = solve_ivp(rhs, (r0, r_max), y0, method=method, rtol=rtol,
                        atol=atol, events=surface, dense_output=False,
                        max_step=1.0)
        if sol.status != 1:
            raise IntegrationError(
                f"{eos.name}: no surface found below r = {r_max} km "
                f"(status={sol.status}, {sol.message})")
        r_arr = np.append(sol.t, sol.t_events[0][0])
        y_arr = np.vstack([sol.y.T, sol.y_events[0][0]])
    elif method in FIXED_METHODS:
        if h is None:
            raise ValueError("fixed-step methods require a step size h")
        # Fixed-step path: use the regularized u = m/r^3 formulation and
        # start at r = h. Both are needed for the nominal convergence
        # order: the plain system has df/dm ~ 1/r^2 near the origin,
        # which degrades any fixed-step method to O(h^2). Tidal and
        # rotational perturbations are not propagated on this path.
        tidal = inertia = False
        counter = [0]
        rhs_reg = _make_rhs_regularized(eos, 0.5 * p_surf, newtonian, counter)
        r_start = max(r0, h)
        z0 = _series_start_regularized(eos, p_c, r_start, newtonian)
        r_arr, z_arr = integrate_fixed(
            rhs_reg, r_start, z0, h, stop=lambda r, z: z[0] - p_surf,
            method=method)
        # transform back: m = u r^3 (Newtonian path stores m in col 1 too)
        y_arr = z_arr.copy()
        y_arr[:, 1] = z_arr[:, 1] * r_arr**3
        if newtonian:
            y_arr = y_arr[:, :2]
    else:
        raise ValueError(f"unknown method '{method}'")

    return _postprocess(eos, method, newtonian, p_c, p_surf,
                        r_arr, y_arr, tidal, inertia, counter[0])


def _postprocess(eos: EquationOfState, method: str, newtonian: bool,
                 p_c: float, p_surf: float, r_arr: np.ndarray,
                 y_arr: np.ndarray, tidal: bool, inertia: bool,
                 n_rhs: int) -> StarModel:
    """!Assemble a `StarModel` from raw integration output."""
    r_s = float(r_arr[-1])
    p_prof = np.maximum(y_arr[:, 0], p_surf)
    m_prof = y_arr[:, 1]
    m_s = float(m_prof[-1])

    eps_prof = np.asarray(eos.eps_of_p(p_prof))
    rho_prof = np.asarray(eos.rho_of_p(p_prof))

    if newtonian:
        phi = np.zeros_like(r_arr)
        mb_s = m_s
    else:
        # normalize Phi to the exterior Schwarzschild solution
        phi = y_arr[:, 2]
        phi = phi + 0.5 * np.log(1.0 - 2.0 * m_s / r_s) - phi[-1]
        mb_s = float(y_arr[:, 3][-1])

    star = StarModel(
        eos_name=eos.name, method=method, newtonian=newtonian,
        p_c=p_c, eps_c=float(eos.eps_of_p(p_c)), rho_c=float(eos.rho_of_p(p_c)),
        R=r_s, M=float(u.mass_geom_to_msun(m_s)), M_km=m_s,
        Mb=float(u.mass_geom_to_msun(mb_s)),
        r=r_arr, p=p_prof, m=m_prof, eps=eps_prof, rho=rho_prof, phi=phi,
        n_rhs=n_rhs,
    )
    if newtonian:
        return star

    idx = 4
    if tidal:
        y_r = float(y_arr[:, idx][-1])
        # correction for a finite surface energy density (density jump)
        eps_s = float(eos.eps_of_p(p_surf))
        y_r -= FOUR_PI * r_s**3 * eps_s / m_s
        star.y_R = y_r
        star.k2 = love_number_k2(star.compactness, y_r)
        star.Lambda = (2.0 / 3.0) * star.k2 / star.compactness**5
        idx += 1
    if inertia:
        w_r = float(y_arr[:, idx][-1])
        v_r = float(y_arr[:, idx + 1][-1])
        j_r = float(np.exp(-y_arr[:, 2][-1]) * np.sqrt(1.0 - 2.0 * m_s / r_s))
        big_j = v_r / (6.0 * j_r)  # J in units of the central omega-bar
        omega = w_r + 2.0 * big_j / r_s**3
        i_geom = big_j / omega  # km^3
        star.I_kgm2 = i_geom * 1.0e9 * k.C**2 / k.G  # m^3 -> kg m^2
    return star


def love_number_k2(c: float, y: float) -> float:
    """!Quadrupole tidal Love number k2(C, y_R) (Hinderer 2008).

    @param c  Compactness C = M/R.
    @param y  Logarithmic derivative of the metric perturbation at the
              surface (after any surface-density correction).
    @return   k2 (dimensionless).
    """
    if not 0.0 < c < 0.5:
        raise ValueError("compactness must lie in (0, 1/2)")
    num = (8.0 / 5.0) * c**5 * (1.0 - 2.0 * c) ** 2 * (2.0 + 2.0 * c * (y - 1.0) - y)
    den = (2.0 * c * (6.0 - 3.0 * y + 3.0 * c * (5.0 * y - 8.0))
           + 4.0 * c**3 * (13.0 - 11.0 * y + c * (3.0 * y - 2.0)
                           + 2.0 * c**2 * (1.0 + y))
           + 3.0 * (1.0 - 2.0 * c) ** 2 * (2.0 - y + 2.0 * c * (y - 1.0))
           * np.log(1.0 - 2.0 * c))
    return float(num / den)


# ----------------------------------------------------------------------
@dataclass
class Sequence:
    """!A one-parameter family of models scanned in central energy density."""

    eos_name: str
    eps_c: np.ndarray  #: central energy densities [km^-2]
    R: np.ndarray  #: radii [km]
    M: np.ndarray  #: gravitational masses [M_sun]
    Mb: np.ndarray  #: baryon masses [M_sun]
    Lambda: np.ndarray  #: tidal deformabilities
    I_45: np.ndarray  #: moments of inertia [1e45 g cm^2]
    stable: np.ndarray  #: stability flag dM/d eps_c > 0

    @property
    def i_max(self) -> int:
        """!Index of the maximum-mass model."""
        return int(np.argmax(self.M))

    @property
    def M_max(self) -> float:
        """!Maximum gravitational mass [M_sun]."""
        return float(self.M[self.i_max])

    def star_at_mass(self, m_target: float) -> int:
        """!Index of the stable-branch model closest to a target mass.

        @throws ValueError if the target exceeds the maximum mass.
        """
        if m_target > self.M_max:
            raise ValueError(f"target {m_target} M_sun exceeds M_max = {self.M_max:.3f}")
        branch = np.where(self.stable)[0]
        return int(branch[np.argmin(np.abs(self.M[branch] - m_target))])


def solve_at_mass(eos: EquationOfState, seq: Sequence, m_target: float,
                  **kwargs) -> StarModel:
    """!Solve the stable-branch model with exactly the target mass.

    Brackets the target between neighbouring sequence grid points and
    root-finds M(log eps_c) = m_target with Brent's method.

    @param eos       Equation of state used for `seq`.
    @param seq       Previously computed sequence.
    @param m_target  Desired gravitational mass [M_sun].
    @param kwargs    Forwarded to `solve_star` for the final solve.
    @return          `StarModel` with M = m_target to ~1e-10.
    @throws ValueError if the target exceeds the maximum mass.
    """
    from scipy.optimize import brentq

    i = seq.star_at_mass(m_target)
    branch = np.where(seq.stable)[0]
    pos = int(np.where(branch == i)[0][0])
    lo = seq.eps_c[branch[max(pos - 1, 0)]]
    hi = seq.eps_c[branch[min(pos + 1, len(branch) - 1)]]

    def f(log_eps: float) -> float:
        s = solve_star(eos, eps_c=float(np.exp(log_eps)),
                       tidal=False, inertia=False)
        return s.M - m_target

    if f(np.log(lo)) * f(np.log(hi)) > 0:  # target on a grid node
        eps_best = seq.eps_c[i]
    else:
        eps_best = float(np.exp(brentq(f, np.log(lo), np.log(hi),
                                       xtol=1e-12)))
    return solve_star(eos, eps_c=float(eps_best), **kwargs)


def compute_sequence(
    eos: EquationOfState,
    eps_c_cgs: np.ndarray,
    *,
    method: str = "DOP853",
    rtol: float = 1.0e-9,
    tidal: bool = True,
    inertia: bool = True,
) -> Sequence:
    """!Solve a family of models over a grid of central densities.

    @param eos        Equation of state.
    @param eps_c_cgs  Central *energy* densities [g/cm^3] (i.e. eps/c^2).
    @param method     Integration method for each model.
    @param rtol       Relative tolerance for each model.
    @param tidal      Compute tidal deformability along the sequence.
    @param inertia    Compute the moment of inertia along the sequence.
    @return           `Sequence` with one entry per central density.

    Models that fail to integrate are dropped with a warning.
    """
    rows = []
    for e_cgs in np.asarray(eps_c_cgs, dtype=float):
        try:
            s = solve_star(eos, eps_c=float(u.density_cgs_to_geom(e_cgs)),
                           method=method, rtol=rtol, tidal=tidal,
                           inertia=inertia)
            rows.append((float(u.density_cgs_to_geom(e_cgs)), s))
        except (IntegrationError, ValueError) as exc:
            log.warning("%s: dropped eps_c=%.3e g/cm^3 (%s)", eos.name, e_cgs, exc)
    if len(rows) < 3:
        raise IntegrationError(f"{eos.name}: sequence has fewer than 3 models")
    eps = np.array([r[0] for r in rows])
    mass = np.array([r[1].M for r in rows])
    stable = np.gradient(mass, eps) > 0.0
    return Sequence(
        eos_name=eos.name,
        eps_c=eps,
        R=np.array([r[1].R for r in rows]),
        M=mass,
        Mb=np.array([r[1].Mb for r in rows]),
        Lambda=np.array([np.nan if r[1].Lambda is None else r[1].Lambda for r in rows]),
        I_45=np.array([np.nan if r[1].I_45 is None else r[1].I_45 for r in rows]),
        stable=stable,
    )


def maximum_mass(eos: EquationOfState, seq: Sequence) -> dict:
    """!Refine the maximum mass by golden-section search around the grid peak.

    @param eos  Equation of state used for `seq`.
    @param seq  A previously computed sequence (provides the bracket).
    @return     dict with M_max [M_sun], R [km], eps_c [g/cm^3], C.
    """
    from scipy.optimize import minimize_scalar

    i = seq.i_max
    lo = seq.eps_c[max(i - 1, 0)]
    hi = seq.eps_c[min(i + 1, len(seq.eps_c) - 1)]

    def neg_mass(log_eps: float) -> float:
        s = solve_star(eos, eps_c=float(np.exp(log_eps)),
                       tidal=False, inertia=False)
        return -s.M

    res = minimize_scalar(neg_mass, bounds=(np.log(lo), np.log(hi)),
                          method="bounded", options={"xatol": 1e-6})
    eps_best = float(np.exp(res.x))
    star = solve_star(eos, eps_c=eps_best, tidal=False, inertia=False)
    return {
        "M_max": star.M,
        "R": star.R,
        "eps_c_cgs": float(u.density_geom_to_cgs(eps_best)),
        "compactness": star.compactness,
    }
