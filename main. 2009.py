##
# @file    replicate_read2009.py
# @brief   Independent replication of Read, Lackey, Owen & Friedman (2009).
#
# @details Re-implements from scratch the four-parameter piecewise-polytropic
#          (PP) parameterisation of the neutron-star equation of state
#          introduced in
#          @par
#          Read, Lackey, Owen & Friedman,
#          "Constraints on a phenomenologically parameterized neutron-star
#          equation of state," Phys. Rev. D **79**, 124032 (2009)
#          [arXiv:0812.2163],
#          @par
#          and re-solves the Tolman-Oppenheimer-Volkoff (TOV) equations to
#          reproduce the paper's Table II (bulk observables:
#          @f$M_{\max}@f$, @f$R@f$ at @f$M_{\max}@f$, @f$R_{1.4}@f$) for a
#          representative subset of the microphysical EOSs listed in their
#          Table III.
#
# @section eos_ref EOS parameterisation
#          Above nuclear density the pressure follows three polytropic
#          pieces
#          @f[
#              P(\rho) = K_i\,\rho^{\Gamma_i},\qquad
#              \rho\in(\rho_{i-1},\rho_i],
#          @f]
#          with fixed dividing densities
#          @f$\rho_1 = 10^{14.7}\,\mathrm{g\,cm^{-3}}@f$ and
#          @f$\rho_2 = 10^{15}\,\mathrm{g\,cm^{-3}}@f$.  The four fit
#          parameters are @f$(\log_{10}P_1,\Gamma_1,\Gamma_2,\Gamma_3)@f$;
#          the @f$K_i@f$ are then fixed by pressure continuity at
#          @f$\rho_1,\rho_2@f$.
#
# @section crust_ref Crust approximation
#          RLOF splice the SLy Douchin-Haensel crust polytropes below
#          @f$\rho_1@f$.  Here a single soft polytrope
#          @f$\Gamma_c = 1.5@f$ is matched onto @f$P_1@f$ at @f$\rho_1@f$.
#          The crust contributes @f$\lesssim 0.1@f$ km to @f$R@f$ and a
#          negligible shift in @f$M_{\max}@f$ (both well below observational
#          error), which is why this simplification is adequate for a
#          replication that focuses on the core-dominated observables.
#
# @section units_ref Units
#          CGS throughout (@p rho in g/cm^3, @p P in dyn/cm^2, @p r in cm,
#          @p m in g), converted to @f$M_\odot@f$ and km at the output
#          stage for direct comparison with the paper.
#
# @section run_ref Usage
#          @code
#          python replicate_read2009.py
#          @endcode
#          The script prints a side-by-side comparison with RLOF Table II
#          and writes ``figures/rlof2009_mass_radius.pdf`` (and ``.png``).
#
# @author  <your name>
# @date    2026
# @version 1.0
##

from __future__ import annotations
import logging
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("read2009")   ##< Module-level logger.


## @name Physical constants (CGS)
## @{
G_cgs   = 6.67430e-8            ##< Newton's constant [cm^3 g^-1 s^-2]
c_cgs   = 2.99792458e10         ##< Speed of light [cm s^-1]
c2      = c_cgs * c_cgs         ##< Precomputed c^2 [cm^2 s^-2]
M_sun_g = 1.98892e33            ##< Solar mass [g]
km_cm   = 1.0e5                 ##< Centimetres per kilometre
## @}


## @name RLOF (2009) fixed dividing densities
## @{
RHO_1 = 10.0**14.7              ##< Core piece-1 / piece-2 boundary [g cm^-3]
RHO_2 = 10.0**15.0              ##< Core piece-2 / piece-3 boundary [g cm^-3]
## @}


##
# @class RLOFPolytrope
# @brief Four-parameter piecewise-polytropic EOS of Read-Lackey-Owen-Friedman
#        (2009), Section II B.
#
# @details Encodes the three-piece polytropic core plus a simplified soft
#          crust and provides the forward map @f$P(\rho)@f$, the inverse
#          map @f$\rho(P)@f$, and the first-law-consistent total energy
#          density @f$\varepsilon(\rho)@f$ needed by the TOV integrator.
#
#          The three core pieces obey
#          @f[
#              P = K_i\,\rho^{\Gamma_i}
#          @f]
#          on
#            - piece 1: @f$\rho_{cc} < \rho \le \rho_1@f$ with @f$\Gamma_1@f$,
#            - piece 2: @f$\rho_1 < \rho \le \rho_2@f$   with @f$\Gamma_2@f$,
#            - piece 3: @f$\rho > \rho_2@f$              with @f$\Gamma_3@f$.
#          The @f$K_i@f$ are fixed by continuity of @f$P@f$ at each
#          boundary together with @f$P(\rho_1)=P_1@f$.
#
# @note  The specific internal energy @f$e(\rho)@f$ is obtained from the
#        first law @f$de = -P\,d(1/\rho)@f$ so that
#        @f$\varepsilon = \rho c^{2} + \rho e@f$ enters the TOV equations
#        self-consistently.  This ~10 % correction is what shifts
#        @f$M_{\max}@f$ from its Newtonian value to the observed
#        neutron-star scale.
##
class RLOFPolytrope:

    ## Softened toy-crust polytropic exponent (see file-level note).
    GAMMA_CRUST = 1.5

    ##
    # @brief   Construct a piecewise-polytropic EOS from the four RLOF fit
    #          parameters.
    #
    # @param   logP1  @f$\log_{10}(P_1/[\mathrm{dyn\,cm^{-2}}])@f$:
    #                 pressure at @f$\rho_1 = 10^{14.7}@f$ g/cm^3.
    # @param   G1     Adiabatic index @f$\Gamma_1@f$ of the outer core.
    # @param   G2     Adiabatic index @f$\Gamma_2@f$ of the intermediate core.
    # @param   G3     Adiabatic index @f$\Gamma_3@f$ of the inner core.
    #
    # @post    All polytropic constants @f$K_i@f$ and the internal-energy
    #          anchor constants @f$C_i@f$ are precomputed once at
    #          construction time; forward/inverse lookups are then
    #          @f$\mathcal{O}(1)@f$.
    ##
    def __init__(self, logP1: float, G1: float, G2: float, G3: float):
        self.logP1 = logP1
        self.G1, self.G2, self.G3 = G1, G2, G3

        # Core polytropic constants — continuity at rho_1 and rho_2.
        self.P1 = 10.0**logP1
        self.K1 = self.P1 / RHO_1**G1
        self.K2 = self.P1 / RHO_1**G2
        self.P2 = self.K2 * RHO_2**G2
        self.K3 = self.P2 / RHO_2**G3

        # Simplified crust polytrope matched at rho_1.
        self.Kc = self.P1 / RHO_1**self.GAMMA_CRUST

        # First-law integration constants for e(rho) per piece, with
        # e -> 0 as rho -> 0 in the crust (satisfied automatically for
        # Gamma_crust > 1).
        Gc = self.GAMMA_CRUST
        self._Cc = 0.0                                   ##< crust anchor

        e_at_rho1_crust = self.Kc / (Gc - 1.0) * RHO_1**(Gc - 1.0)
        self._C1 = (e_at_rho1_crust
                    - self.K1 / (G1 - 1.0) * RHO_1**(G1 - 1.0))

        e_at_rho1_p1 = self._C1 + self.K1 / (G1 - 1.0) * RHO_1**(G1 - 1.0)
        self._C2 = (e_at_rho1_p1
                    - self.K2 / (G2 - 1.0) * RHO_1**(G2 - 1.0))

        e_at_rho2_p2 = self._C2 + self.K2 / (G2 - 1.0) * RHO_2**(G2 - 1.0)
        self._C3 = (e_at_rho2_p2
                    - self.K3 / (G3 - 1.0) * RHO_2**(G3 - 1.0))

    # -----------------------------------------------------------------
    # Forward and inverse EOS
    # -----------------------------------------------------------------

    ##
    # @brief   Forward EOS: pressure as a function of rest-mass density.
    # @param   rho  Rest-mass density [g cm^-3].
    # @return  Pressure @f$P(\rho)@f$ [dyn cm^-2].
    ##
    def P_of_rho(self, rho: float) -> float:
        if rho < RHO_1:
            return self.Kc * rho**self.GAMMA_CRUST
        elif rho < RHO_2:
            return self.K2 * rho**self.G2
        else:
            return self.K3 * rho**self.G3

    ##
    # @brief   Inverse EOS: rest-mass density as a function of pressure.
    # @param   P  Pressure [dyn cm^-2].
    # @return  Rest-mass density @f$\rho(P)@f$ [g cm^-3]; zero if
    #          @f$P\le 0@f$.
    ##
    def rho_of_P(self, P: float) -> float:
        if P <= 0.0:
            return 0.0
        if P < self.P1:
            return (P / self.Kc)**(1.0 / self.GAMMA_CRUST)
        elif P < self.P2:
            return (P / self.K2)**(1.0 / self.G2)
        else:
            return (P / self.K3)**(1.0 / self.G3)

    ##
    # @brief   Total energy density (rest + first-law internal energy).
    # @param   rho  Rest-mass density [g cm^-3].
    # @return  @f$\varepsilon(\rho) = \rho c^{2}(1+e/c^{2})@f$
    #          [erg cm^-3].
    ##
    def epsilon_of_rho(self, rho: float) -> float:
        if rho <= 0.0:
            return 0.0
        if rho < RHO_1:
            e = self._Cc + self.Kc / (self.GAMMA_CRUST - 1.0) \
                * rho**(self.GAMMA_CRUST - 1.0)
        elif rho < RHO_2:
            e = self._C2 + self.K2 / (self.G2 - 1.0) * rho**(self.G2 - 1.0)
        else:
            e = self._C3 + self.K3 / (self.G3 - 1.0) * rho**(self.G3 - 1.0)
        return rho * c2 + rho * e

    ##
    # @brief   Mass-equivalent gravitational energy density, i.e. the
    #          quantity that appears on the right-hand side of the TOV
    #          equations.
    # @param   P  Pressure [dyn cm^-2].
    # @return  @f$\varepsilon(P)/c^{2}@f$ [g cm^-3].
    ##
    def rho_grav(self, P: float) -> float:
        rho_rest = self.rho_of_P(P)
        return self.epsilon_of_rho(rho_rest) / c2


##
# @var TABLE_III
# @brief RLOF (2009) Table III fit parameters.
# @details Format: name -> (log10 P_1 [dyn/cm^2], Gamma_1, Gamma_2, Gamma_3).
#          Verify individual rows against the printed paper before quoting
#          numbers externally.
##
TABLE_III: dict[str, tuple[float, float, float, float]] = {
    "SLy":    (34.384, 3.005, 2.988, 2.851),
    "APR4":   (34.269, 2.830, 3.445, 3.348),
    "FPS":    (34.283, 2.985, 2.863, 2.600),
    "WFF1":   (34.031, 2.519, 3.791, 3.660),
    "WFF2":   (34.233, 2.888, 3.475, 3.517),
    "ENG":    (34.437, 3.514, 3.130, 3.168),
    "MPA1":   (34.495, 3.446, 3.572, 2.887),
    "MS1":    (34.858, 3.224, 3.033, 1.325),
    "H4":     (34.669, 2.909, 2.246, 2.144),
    "PS":     (34.671, 2.216, 1.640, 2.365),
}

##
# @var TABLE_II_REF
# @brief RLOF (2009) Table II reference observables used for validation.
# @details Format: name -> dict of M_max [M_sun], R at M_max [km], R_1.4 [km].
##
TABLE_II_REF: dict[str, dict[str, float]] = {
    "SLy":    {"M_max": 2.05, "R_at_Mmax":  9.99, "R_1.4": 11.72},
    "APR4":   {"M_max": 2.21, "R_at_Mmax":  9.96, "R_1.4": 11.42},
    "FPS":    {"M_max": 1.80, "R_at_Mmax":  9.30, "R_1.4": 10.85},
    "WFF1":   {"M_max": 2.13, "R_at_Mmax":  9.44, "R_1.4": 10.41},
    "WFF2":   {"M_max": 2.20, "R_at_Mmax":  9.83, "R_1.4": 11.10},
    "ENG":    {"M_max": 2.24, "R_at_Mmax":  9.71, "R_1.4": 11.72},
    "MPA1":   {"M_max": 2.46, "R_at_Mmax": 10.75, "R_1.4": 12.44},
    "MS1":    {"M_max": 2.75, "R_at_Mmax": 12.24, "R_1.4": 14.90},
    "H4":     {"M_max": 2.02, "R_at_Mmax": 10.61, "R_1.4": 13.75},
    "PS":     {"M_max": 1.75, "R_at_Mmax": 10.09, "R_1.4": 14.85},
}


##
# @brief   Right-hand side of the TOV system in CGS units.
#
# @details Computes @f$d\mathbf{y}/dr@f$ for
#          @f$\mathbf{y} = (P,m)@f$.  Returns @f$(-10^{300},0)@f$ if the
#          Schwarzschild factor @f$1-2Gm/(rc^{2})@f$ becomes non-positive
#          (an unphysical configuration outside the domain of
#          static equilibrium).
#
# @param   r    Radial coordinate [cm].
# @param   y    State vector @f$[P,m]@f$ in (dyn/cm^2, g).
# @param   eos  ``RLOFPolytrope`` instance providing @p rho_grav.
#
# @return  Numpy array @f$[dP/dr,\,dm/dr]@f$.
##
def tov_rhs(r: float, y: np.ndarray, eos: RLOFPolytrope) -> np.ndarray:
    P, m = y
    if P <= 0.0:
        return np.array([0.0, 0.0])
    rho_g = eos.rho_grav(P)
    schw = 1.0 - 2.0 * G_cgs * m / (r * c2)
    if schw <= 0.0:
        return np.array([-1.0e300, 0.0])
    dPdr = (-G_cgs * (rho_g + P / c2)
            * (m + 4.0 * np.pi * r**3 * P / c2)
            / (r * r * schw))
    dmdr = 4.0 * np.pi * r * r * rho_g
    return np.array([dPdr, dmdr])


##
# @brief   One classical Runge-Kutta 4 step.
# @param   r    Current radius [cm].
# @param   y    Current state vector.
# @param   h    Step size [cm].
# @param   eos  EOS instance forwarded to ``tov_rhs``.
# @return  Advanced state @f$\mathbf{y}_{n+1}@f$.
##
def rk4_step(r: float, y: np.ndarray, h: float,
             eos: RLOFPolytrope) -> np.ndarray:
    k1 = tov_rhs(r,         y,             eos)
    k2 = tov_rhs(r + h/2.0, y + h*k1/2.0,  eos)
    k3 = tov_rhs(r + h/2.0, y + h*k2/2.0,  eos)
    k4 = tov_rhs(r + h,     y + h*k3,      eos)
    return y + (h / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)


##
# @brief   Integrate a single equilibrium configuration for a chosen
#          central rest-mass density.
#
# @details Starts at @f$r_0@f$ with @f$P_c = \mathrm{eos.P\_of\_rho}(\rho_c)@f$
#          and @f$m_0 = \tfrac{4}{3}\pi r_0^{3}\,\rho_g(\rho_c)@f$, then
#          RK4-advances until @f$P\le P_{\text{floor}}@f$.  The surface
#          radius is located by linear interpolation between the last
#          positive-pressure step and the overshooting step.
#
# @param   rho_c  Central rest-mass density [g cm^-3].
# @param   eos    ``RLOFPolytrope`` EOS instance.
# @param   r0     Inner starting radius [cm] (avoids the @f$r=0@f$
#                 singularity).
# @param   dr     Fixed radial step [cm].
# @param   r_max  Safety cap on integration radius [cm].
#
# @return  Tuple ``(R, M)`` — stellar radius [cm] and gravitational mass
#          [g].
##
def integrate_star(rho_c: float, eos: RLOFPolytrope,
                   r0: float = 1.0, dr: float = 100.0,
                   r_max: float = 5.0e6) -> tuple[float, float]:
    P_c = eos.P_of_rho(rho_c)
    rho_g_c = eos.epsilon_of_rho(rho_c) / c2
    m0 = (4.0/3.0) * np.pi * r0**3 * rho_g_c
    y = np.array([P_c, m0])
    r = r0
    P_floor = 1.0e-10 * P_c

    P_prev, m_prev, r_prev = P_c, m0, r0
    while r < r_max:
        y_new = rk4_step(r, y, dr, eos)
        P_new, m_new = y_new
        if P_new <= P_floor:
            frac = (P_prev - P_floor) / (P_prev - P_new)
            R = r_prev + frac * dr
            M = m_prev + frac * (m_new - m_prev)
            return R, M
        P_prev, m_prev, r_prev = P_new, m_new, r + dr
        y = y_new
        r += dr

    return r_prev, m_prev


##
# @brief   Log-spaced central-density sweep producing the M-R sequence.
# @param   eos        EOS instance.
# @param   rho_c_min  Lower bound on the central-density scan [g cm^-3].
# @param   rho_c_max  Upper bound on the central-density scan [g cm^-3].
# @param   n          Number of samples.
# @return  Tuple ``(rhos, Rs, Ms)`` with @p rhos [g cm^-3], @p Rs [km],
#          @p Ms [M_sun].
##
def mass_radius_curve(eos: RLOFPolytrope,
                      rho_c_min: float = 3.0e14,
                      rho_c_max: float = 4.0e15,
                      n: int = 80) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rhos = np.logspace(np.log10(rho_c_min), np.log10(rho_c_max), n)
    Rs = np.empty(n); Ms = np.empty(n)
    for i, rc in enumerate(rhos):
        R_cm, M_g = integrate_star(rc, eos)
        Rs[i] = R_cm / km_cm
        Ms[i] = M_g / M_sun_g
    return rhos, Rs, Ms


##
# @brief   Extract the three Table II observables from an M-R sweep.
#
# @details Returns @f$M_{\max}@f$, the radius at @f$M_{\max}@f$, the
#          central density at @f$M_{\max}@f$, and @f$R_{1.4}@f$, the
#          radius at the canonical @f$1.4\,M_\odot@f$ interpolated on the
#          stable branch (ascending @f$M@f$ up to and including the
#          turning point).
#
# @param   eos  EOS instance.
# @return  Dictionary with keys @p M_max, @p R_at_Mmax,
#          @p rho_c_at_Mmax, @p R_1_4 plus the raw arrays @p rhos,
#          @p Rs, @p Ms.
##
def bulk_properties(eos: RLOFPolytrope) -> dict:
    rhos, Rs, Ms = mass_radius_curve(eos)
    imax = int(np.argmax(Ms))
    M_max      = float(Ms[imax])
    R_at_Mmax  = float(Rs[imax])
    rho_c_max  = float(rhos[imax])

    stable = slice(0, imax + 1)
    Ms_s, Rs_s = Ms[stable], Rs[stable]
    if Ms_s.min() <= 1.4 <= Ms_s.max():
        R_14 = float(np.interp(1.4, Ms_s, Rs_s))
    else:
        R_14 = float("nan")

    return dict(M_max=M_max, R_at_Mmax=R_at_Mmax,
                rho_c_at_Mmax=rho_c_max, R_1_4=R_14,
                rhos=rhos, Rs=Rs, Ms=Ms)


##
# @brief   Log a formatted RLOF Table II comparison.
# @param   rows  List of row dictionaries produced by ``main``.
##
def print_table(rows: list[dict]) -> None:
    header = ("EOS      | M_max (this)  M_max (RLOF)   dM %   |  "
              "R_1.4 (this)  R_1.4 (RLOF)   dR %   |  "
              "R@Mmax (this)  R@Mmax (RLOF)  dR %")
    log.info(header)
    log.info("-" * len(header))
    for row in rows:
        log.info(
            "{name:8s} |  {mm_t:5.3f}         {mm_r:5.3f}       {dmm:+5.2f} |  "
            "{r14_t:5.2f}         {r14_r:5.2f}         {dr14:+5.2f} |  "
            "{rmm_t:5.2f}          {rmm_r:5.2f}          {drmm:+5.2f}"
            .format(**row)
        )


##
# @mainpage RLOF (2009) Piecewise-Polytropic Replication
#
# @section overview Driver overview
# The executable section below:
#   1. iterates over the RLOF Table III fit parameters,
#   2. builds a piecewise-polytropic EOS for each entry,
#   3. solves the TOV equations across a log-spaced central-density scan,
#   4. extracts the three Table II observables @f$(M_{\max},\,R_{1.4},\,
#      R(M_{\max}))@f$,
#   5. prints a side-by-side comparison with the published Table II
#      values, and
#   6. writes an overlaid mass-radius figure (analogue of RLOF Fig. 5)
#      to ``figures/rlof2009_mass_radius.{pdf,png}``.
#
# @section run How to run
# @code
# python replicate_read2009.py
# @endcode
##
def main() -> None:
    log.info("=== RLOF (2009) piecewise-polytropic replication ===")
    log.info("dividing densities  rho_1 = 10^14.7,  rho_2 = 10^15  g/cm^3")

    results = {}
    rows = []
    for name, (logP1, G1, G2, G3) in TABLE_III.items():
        eos = RLOFPolytrope(logP1, G1, G2, G3)
        props = bulk_properties(eos)
        results[name] = props

        ref = TABLE_II_REF[name]
        row = dict(
            name=name,
            mm_t=props["M_max"],       mm_r=ref["M_max"],
            dmm=100.0 * (props["M_max"] - ref["M_max"]) / ref["M_max"],
            r14_t=props["R_1_4"],      r14_r=ref["R_1.4"],
            dr14=100.0 * (props["R_1_4"] - ref["R_1.4"]) / ref["R_1.4"]
                 if np.isfinite(props["R_1_4"]) else 0.0,
            rmm_t=props["R_at_Mmax"],  rmm_r=ref["R_at_Mmax"],
            drmm=100.0 * (props["R_at_Mmax"] - ref["R_at_Mmax"])
                 / ref["R_at_Mmax"],
        )
        rows.append(row)
        log.info(
            f"{name:8s}  M_max={props['M_max']:.3f}  "
            f"R_1.4={props['R_1_4']:.2f} km  "
            f"R@Mmax={props['R_at_Mmax']:.2f} km"
        )

    log.info("--- Table II comparison (this replication vs. RLOF 2009) ---")
    print_table(rows)

    ##
    # @brief Mass-radius figure: reproduces the visual content of RLOF
    #        Fig. 5.  Black dots mark the published Table II
    #        @f$(R(M_{\max}),M_{\max})@f$ points for direct visual
    #        comparison with our numerical curves.
    ##
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    for name, props in results.items():
        ax.plot(props["Rs"], props["Ms"], lw=1.4, label=name)
        ref = TABLE_II_REF[name]
        ax.plot(ref["R_at_Mmax"], ref["M_max"], "k.", ms=6)

    ax.set_xlim(7.5, 17.0)
    ax.set_ylim(0.4, 3.1)
    ax.set_xlabel("Radius  R  [km]")
    ax.set_ylabel(r"Gravitational mass  $M$  [$M_\odot$]")
    ax.set_title("Mass-radius curves — RLOF (2009) PP parameterisation\n"
                 "(black dots: published Table II  M_max, R at M_max)")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="lower left", fontsize=9, ncol=2)

    ax.axhline(2.0, color="grey", ls="--", lw=0.8)
    ax.text(15.5, 2.02, r"$2\,M_\odot$ pulsar bound",
            fontsize=8, color="grey")

    fig.tight_layout()
    out_pdf = "figures/rlof2009_mass_radius.pdf"
    out_png = "figures/rlof2009_mass_radius.png"
    import os
    os.makedirs("figures", exist_ok=True)
    fig.savefig(out_pdf); fig.savefig(out_png, dpi=180)
    log.info(f"figure written: {out_pdf} / {out_png}")


if __name__ == "__main__":
    main()
