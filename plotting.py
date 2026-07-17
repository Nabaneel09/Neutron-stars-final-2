"""! @file plotting.py
@brief Publication-quality figures (matplotlib), saved as PDF and PNG.

Every public function takes precomputed data and a target directory and
returns the list of files written. A consistent serif style close to the
journal default of APS/A&A is applied by `setup_style()`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
from typing import Sequence as Seq

import matplotlib

matplotlib.use("Agg")  # headless backend; must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import units as u  # noqa: E402
from .eos import EquationOfState  # noqa: E402
from .tov import Sequence, StarModel  # noqa: E402
from .utils import get_logger  # noqa: E402

log = get_logger(__name__)

#: colour cycle used for the EOS models (colour-blind friendly)
EOS_COLORS = {"SLy": "#0072B2", "APR(AP4)": "#D55E00", "SLy-pp": "#56B4E9",
              "fermi-gas": "#009E73", "polytrope-n3/2": "#CC79A7",
              "constant-density": "#999999"}


def setup_style() -> None:
    """!Apply the project-wide matplotlib style (serif, thin frames)."""
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.6,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
    })


def save_fig(fig: plt.Figure, name: str, figdir: Path | str) -> List[Path]:
    """!Save a figure as both PDF (vector) and PNG (300 dpi raster).

    @param fig     Matplotlib figure.
    @param name    Base file name without extension.
    @param figdir  Target directory (created if missing).
    @return        Paths of the two files written.
    """
    figdir = Path(figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        p = figdir / f"{name}.{ext}"
        fig.savefig(p, **kw)
        paths.append(p)
    plt.close(fig)
    log.info("figure written: %s (.pdf/.png)", figdir / name)
    return paths


def _color(name: str) -> str:
    return EOS_COLORS.get(name, "k")


# ----------------------------------------------------------------------
def plot_eos_comparison(models: Seq[EquationOfState], figdir: Path) -> None:
    """!P(rho) for all EOS models, log-log, with the saturation density."""
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for eos in models:
        rho_cgs = np.geomspace(1e13, 3.5e15, 400)
        rho = np.asarray(u.density_cgs_to_geom(rho_cgs))
        try:
            p = np.asarray(eos.p_of_rho(rho))
        except Exception:  # incompressible has no P(rho)
            continue
        p_cgs = np.asarray(u.pressure_geom_to_cgs(p))
        ax.loglog(rho_cgs, p_cgs, label=eos.name, color=_color(eos.name))
    ax.axvline(2.7e14, color="0.6", ls=":", lw=1)
    ax.text(2.9e14, 2e31, r"$\rho_0$", color="0.4")
    ax.set_xlabel(r"$\rho\ \mathrm{[g\,cm^{-3}]}$")
    ax.set_ylabel(r"$P\ \mathrm{[dyn\,cm^{-2}]}$")
    ax.legend(loc="lower right")
    save_fig(fig, "eos_comparison", figdir)


def plot_sound_speed(models: Seq[EquationOfState], figdir: Path) -> None:
    """!c_s^2/c^2 vs density, with the causal bound highlighted."""
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for eos in models:
        rho_cgs = np.geomspace(2e13, 3.5e15, 500)
        try:
            p = np.asarray(eos.p_of_rho(np.asarray(u.density_cgs_to_geom(rho_cgs))))
        except Exception:
            continue
        cs2 = np.asarray(eos.cs2_of_p(p))
        ax.semilogx(rho_cgs, cs2, label=eos.name, color=_color(eos.name))
    ax.axhline(1.0, color="crimson", ls="--", lw=1)
    ax.text(0.97, 1.015, "causal limit $c_s = c$", color="crimson",
            fontsize=9, ha="right", va="bottom",
            transform=ax.get_yaxis_transform())
    ax.axhline(1.0 / 3.0, color="0.6", ls=":", lw=1)
    ax.text(0.97, 0.345, r"conformal $c_s^2 = c^2/3$", color="0.4",
            fontsize=9, ha="right", va="bottom",
            transform=ax.get_yaxis_transform())
    ax.set_xlabel(r"$\rho\ \mathrm{[g\,cm^{-3}]}$")
    ax.set_ylabel(r"$c_s^2/c^2$")
    ax.set_ylim(0, 1.25)
    ax.legend(loc="upper left")
    save_fig(fig, "sound_speed", figdir)


def plot_adiabatic_index(models: Seq[EquationOfState], figdir: Path) -> None:
    """!Relativistic adiabatic index Gamma_1 vs density."""
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for eos in models:
        rho_cgs = np.geomspace(1e10, 3.5e15, 600)
        try:
            p = np.asarray(eos.p_of_rho(np.asarray(u.density_cgs_to_geom(rho_cgs))))
        except Exception:
            continue
        ax.semilogx(rho_cgs, np.asarray(eos.gamma_of_p(p)),
                    label=eos.name, color=_color(eos.name))
    ax.axhline(4.0 / 3.0, color="0.6", ls=":", lw=1)
    ax.text(2e10, 1.37, r"$\Gamma = 4/3$", color="0.4", fontsize=9)
    ax.set_xlabel(r"$\rho\ \mathrm{[g\,cm^{-3}]}$")
    ax.set_ylabel(r"adiabatic index $\Gamma_1$")
    ax.set_ylim(0.4, 4.2)
    ax.legend(loc="upper left")
    save_fig(fig, "adiabatic_index", figdir)


# ----------------------------------------------------------------------
def plot_mass_radius(seqs: Seq[Sequence], figdir: Path) -> None:
    """!Mass-radius diagram with exclusion regions and observations.

    Shaded wedges: Schwarzschild radius (R < 2GM/c^2), Buchdahl bound
    (R < 9GM/4c^2) and the causal bound R < 2.82 GM/c^2 (Lattimer &
    Prakash 2007). Horizontal bands: PSR J0740+6620 and the canonical
    1.4 M_sun.
    """
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    r_line = np.linspace(4.0, 18.0, 100)
    msun_km = float(u.mass_msun_to_geom(1.0))
    for coeff, col in ((2.0, "0.25"), (9.0 / 4.0, "0.45"), (2.82, "0.65")):
        ax.fill_between(r_line, r_line / (coeff * msun_km), 3.2,
                        color=col, alpha=0.35, lw=0)
        ax.plot(r_line, r_line / (coeff * msun_km), color=col, lw=0.8)
    ax.axhspan(2.08 - 0.07, 2.08 + 0.07, color="goldenrod", alpha=0.30, lw=0)
    ax.text(16.9, 2.10, "PSR J0740+6620", fontsize=8, ha="right")
    ax.axhline(1.4, color="0.5", ls=":", lw=1)
    for seq in seqs:
        st = seq.stable
        ax.plot(seq.R[st], seq.M[st], label=seq.eos_name,
                color=_color(seq.eos_name))
        ax.plot(seq.R[~st], seq.M[~st], color=_color(seq.eos_name),
                ls="--", lw=1.0, alpha=0.7)
        i = seq.i_max
        ax.plot(seq.R[i], seq.M[i], "o", ms=4, color=_color(seq.eos_name))
    ax.set_xlabel(r"$R$ [km]")
    ax.set_ylabel(r"$M\ [M_\odot]$")
    ax.set_xlim(6, 18)
    ax.set_ylim(0, 3.0)
    ax.legend(loc="upper left", fontsize=8.5)
    save_fig(fig, "mass_radius", figdir)


def plot_sequences_vs_density(seqs: Seq[Sequence], figdir: Path) -> None:
    """!M(eps_c) and R(eps_c) with the stability turning point marked."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharex=True)
    for seq in seqs:
        eps_cgs = np.asarray(u.density_geom_to_cgs(seq.eps_c))
        for ax, val in ((axes[0], seq.M), (axes[1], seq.R)):
            ax.semilogx(eps_cgs[seq.stable], val[seq.stable],
                        color=_color(seq.eos_name), label=seq.eos_name)
            ax.semilogx(eps_cgs[~seq.stable], val[~seq.stable],
                        color=_color(seq.eos_name), ls="--", lw=1, alpha=0.7)
        axes[0].plot(eps_cgs[seq.i_max], seq.M[seq.i_max], "o", ms=4,
                     color=_color(seq.eos_name))
    axes[0].set_ylabel(r"$M\ [M_\odot]$")
    axes[1].set_ylabel(r"$R$ [km]")
    for ax in axes:
        ax.set_xlabel(r"$\varepsilon_c/c^2\ \mathrm{[g\,cm^{-3}]}$")
    axes[0].legend(fontsize=8.5)
    axes[0].text(0.05, 0.9, r"$dM/d\varepsilon_c > 0$: stable",
                 transform=axes[0].transAxes, fontsize=8.5)
    fig.tight_layout()
    save_fig(fig, "sequences_vs_density", figdir)


# ----------------------------------------------------------------------
def plot_profiles(star: StarModel, figdir: Path,
                  name: str = "profiles_canonical") -> None:
    """!Four-panel interior structure of a single model.

    Panels: P(r), eps(r), m(r) and the metric functions
    -g_tt = e^{2 Phi}, g_rr = (1-2m/r)^{-1}.
    """
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0), sharex=True)
    r = star.r
    axes[0, 0].plot(r, u.pressure_geom_to_cgs(star.p), color="#0072B2")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel(r"$P\ \mathrm{[dyn\,cm^{-2}]}$")
    axes[0, 1].plot(r, u.density_geom_to_cgs(star.eps), color="#D55E00")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylabel(r"$\varepsilon/c^2\ \mathrm{[g\,cm^{-3}]}$")
    axes[1, 0].plot(r, u.mass_geom_to_msun(star.m), color="#009E73")
    axes[1, 0].set_ylabel(r"$m(r)\ [M_\odot]$")
    axes[1, 1].plot(r, np.exp(2 * star.phi), label=r"$-g_{tt}=e^{2\Phi}$",
                    color="#0072B2")
    axes[1, 1].plot(r, star.metric_grr(), label=r"$g_{rr}$", color="#D55E00")
    axes[1, 1].axhline(1.0, color="0.7", lw=0.8, ls=":")
    axes[1, 1].set_ylabel("metric coefficients")
    axes[1, 1].legend()
    for ax in axes[1]:
        ax.set_xlabel(r"$r$ [km]")
    title = (f"{star.eos_name}:  $M={star.M:.3f}\\,M_\\odot$, "
             f"$R={star.R:.2f}$ km, $C={star.compactness:.3f}$")
    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_fig(fig, name, figdir)


def plot_gravity_redshift(star: StarModel, figdir: Path) -> None:
    """!Local gravity g(r) and cumulative escape velocity/compactness."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    axes[0].plot(star.r, star.local_gravity() / 1.0e12, color="#0072B2")
    axes[0].set_xlabel(r"$r$ [km]")
    axes[0].set_ylabel(r"$g(r)\ [10^{12}\,\mathrm{m\,s^{-2}}]$")
    with np.errstate(divide="ignore", invalid="ignore"):
        vesc = np.sqrt(2.0 * star.m / np.where(star.r > 0, star.r, np.inf))
    axes[1].plot(star.r, vesc, color="#D55E00",
                 label=r"$v_{esc}(r)/c=\sqrt{2m/r}$")
    axes[1].axhline(star.v_escape, color="0.5", ls=":", lw=1)
    axes[1].text(0.4, star.v_escape + 0.02,
                 f"surface: $v_{{esc}}={star.v_escape:.2f}c$, "
                 f"$z_{{surf}}={star.z_surf:.3f}$", fontsize=9)
    axes[1].set_xlabel(r"$r$ [km]")
    axes[1].set_ylabel(r"$v_{esc}/c$")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    save_fig(fig, "gravity_escape", figdir)


def plot_newton_vs_tov(seq_tov: Sequence, seq_newt: Sequence,
                       figdir: Path) -> None:
    """!Relativistic vs Newtonian M(rho_c) for the same polytropic EOS."""
    fig, ax = plt.subplots(figsize=(5.4, 4.1))
    for seq, lab, ls in ((seq_tov, "TOV", "-"), (seq_newt, "Newtonian", "--")):
        eps_cgs = np.asarray(u.density_geom_to_cgs(seq.eps_c))
        ax.loglog(eps_cgs, seq.M, ls, label=lab,
                  color="#0072B2" if lab == "TOV" else "#D55E00")
    ax.set_xlabel(r"$\rho_c\ \mathrm{[g\,cm^{-3}]}$")
    ax.set_ylabel(r"$M\ [M_\odot]$")
    ax.set_title(r"$\Gamma=5/3$ neutron polytrope", fontsize=11)
    ax.legend()
    save_fig(fig, "newton_vs_tov", figdir)


# ----------------------------------------------------------------------
def plot_tidal(seqs: Seq[Sequence], figdir: Path) -> None:
    """!Tidal deformability Lambda(M) with the GW170817 constraint."""
    fig, ax = plt.subplots(figsize=(5.4, 4.1))
    for seq in seqs:
        st = seq.stable & np.isfinite(seq.Lambda)
        ax.semilogy(seq.M[st], seq.Lambda[st], label=seq.eos_name,
                    color=_color(seq.eos_name))
    ax.errorbar([1.4], [190.0], yerr=[[120.0], [390.0]], fmt="s", ms=5,
                color="k", capsize=3, label=r"GW170817 $\Lambda_{1.4}$")
    ax.set_xlabel(r"$M\ [M_\odot]$")
    ax.set_ylabel(r"$\Lambda$")
    ax.set_xlim(1.0, 2.3)
    ax.legend()
    save_fig(fig, "tidal_deformability", figdir)


def plot_inertia_binding(seqs: Seq[Sequence], figdir: Path) -> None:
    """!Moment of inertia I(M) and binding energy E_b(M)."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    for seq in seqs:
        st = seq.stable & np.isfinite(seq.I_45)
        axes[0].plot(seq.M[st], seq.I_45[st], label=seq.eos_name,
                     color=_color(seq.eos_name))
        be = seq.Mb - seq.M
        axes[1].plot(seq.M[seq.stable], be[seq.stable],
                     color=_color(seq.eos_name), label=seq.eos_name)
    axes[0].set_xlabel(r"$M\ [M_\odot]$")
    axes[0].set_ylabel(r"$I\ [10^{45}\,\mathrm{g\,cm^2}]$")
    axes[1].set_xlabel(r"$M\ [M_\odot]$")
    axes[1].set_ylabel(r"$E_b = (M_b - M)\ [M_\odot c^2]$")
    for ax in axes:
        ax.legend(fontsize=8.5)
    fig.tight_layout()
    save_fig(fig, "inertia_binding", figdir)


# ----------------------------------------------------------------------
def plot_convergence(h_values: np.ndarray, errors: Dict[str, np.ndarray],
                     figdir: Path) -> None:
    """!Global-error convergence of the fixed-step methods, log-log.

    Reference slopes h^1, h^2, h^4 are drawn for comparison.
    """
    fig, ax = plt.subplots(figsize=(5.4, 4.1))
    marks = {"euler": "o", "heun": "s", "rk4": "^"}
    for name, err in errors.items():
        ax.loglog(h_values, err, marks.get(name, "o") + "-", label=name)
    for order in (1, 2, 4):
        ref = errors.get("euler" if order == 1 else "heun" if order == 2
                         else "rk4")
        if ref is not None and np.isfinite(ref[-1]):
            c0 = ref[-1] / h_values[-1] ** order
            ax.loglog(h_values, c0 * h_values**order, "k:", lw=0.8)
            ax.text(h_values[0] * 1.1, c0 * h_values[0] ** order * 1.4,
                    f"$h^{order}$", fontsize=9)
    ax.set_xlabel(r"step size $h$ [km]")
    ax.set_ylabel(r"$|M(h) - M_{ref}| / M_{ref}$")
    ax.legend()
    save_fig(fig, "convergence_fixed_step", figdir)


def plot_work_precision(data: Dict[str, Dict[str, np.ndarray]],
                        figdir: Path) -> None:
    """!Work-precision diagram: achieved error vs number of RHS calls.

    @param data  {method: {"nrhs": array, "err": array}}.
    """
    fig, ax = plt.subplots(figsize=(5.4, 4.1))
    for name, d in data.items():
        good = np.asarray(d["err"]) > 0
        ax.loglog(np.asarray(d["err"])[good], np.asarray(d["nrhs"])[good],
                  "o-", ms=4, label=name)
    ax.set_xlabel(r"relative error in $M$")
    ax.set_ylabel("RHS evaluations")
    ax.invert_xaxis()
    ax.legend()
    save_fig(fig, "work_precision", figdir)


def plot_sensitivity(res: Dict[str, np.ndarray], figdir: Path) -> None:
    """!Sensitivity of (M, R) to the solver tolerance for one model.

    @param res  {"rtol", "dM", "dR"} arrays from the sensitivity study.
    """
    fig, ax = plt.subplots(figsize=(5.4, 4.1))
    ax.loglog(res["rtol"], np.maximum(res["dM"], 1e-16), "o-",
              label=r"$|\Delta M|/M$")
    ax.loglog(res["rtol"], np.maximum(res["dR"], 1e-16), "s-",
              label=r"$|\Delta R|/R$")
    ax.plot(res["rtol"], res["rtol"], "k:", lw=0.8, label=r"$\propto$ rtol")
    ax.set_xlabel("solver relative tolerance")
    ax.set_ylabel("relative change vs reference")
    ax.legend()
    save_fig(fig, "sensitivity_tolerance", figdir)
