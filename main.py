"""! @file main.py
@brief End-to-end pipeline: sequences, validation, sensitivity, figures.

Run from the project root:
@code{.sh}
python main.py            # full run (~2-4 min)
python main.py --quick    # coarse grids, for smoke testing
@endcode

Outputs:
- figures/  : all plots as PDF + PNG
- results/  : CSV tables (sequences, canonical stars, validation,
              convergence, sensitivity, summary)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from neutronstar import eos as eos_mod  # noqa: E402
from neutronstar import plotting as plots  # noqa: E402
from neutronstar import units as u  # noqa: E402
from neutronstar import validation as val  # noqa: E402
from neutronstar.tov import (compute_sequence, maximum_mass,  # noqa: E402
                             solve_at_mass, solve_star)
from neutronstar.utils import get_logger, write_csv, write_kv_csv  # noqa: E402

log = get_logger("neutronstar.main")

ROOT = Path(__file__).parent
FIGDIR = ROOT / "figures"
RESDIR = ROOT / "results"


def build_sequences(n_pts: int) -> dict:
    """!Compute M-R sequences for the physical EOS models."""
    seqs = {}
    grids = {
        "sly": np.geomspace(2.2e14, 4.0e15, n_pts),
        "apr": np.geomspace(2.2e14, 4.0e15, n_pts),
        "fermi": np.geomspace(1.0e14, 1.0e17, n_pts),
        "poly-nr": np.geomspace(5.0e13, 8.0e15, n_pts),
    }
    for name, grid in grids.items():
        t0 = time.perf_counter()
        e = eos_mod.get_eos(name)
        e.validate()
        seqs[name] = (e, compute_sequence(e, grid))
        log.info("sequence %-8s: %d models in %.1f s", name,
                 len(seqs[name][1].M), time.perf_counter() - t0)
    return seqs


def export_sequences(seqs: dict) -> None:
    """!Write one CSV per EOS sequence."""
    for name, (_, s) in seqs.items():
        write_csv(
            RESDIR / f"sequence_{name}.csv",
            ["eps_c_gcm3", "R_km", "M_Msun", "Mb_Msun", "compactness",
             "z_surf", "Lambda", "I_1e45_gcm2", "binding_Msun", "stable"],
            [np.asarray(u.density_geom_to_cgs(s.eps_c)), s.R, s.M, s.Mb,
             np.asarray(u.mass_msun_to_geom(s.M)) / s.R,
             1.0 / np.sqrt(1.0 - 2.0 * np.asarray(u.mass_msun_to_geom(s.M)) / s.R) - 1.0,
             s.Lambda, s.I_45, s.Mb - s.M, s.stable.astype(int)],
        )


def canonical_star(seqs: dict) -> "object":
    """!Solve, export and plot the canonical 1.4 M_sun SLy model."""
    e_sly, s_sly = seqs["sly"]
    star = solve_at_mass(e_sly, s_sly, 1.4, rtol=1e-10)
    write_csv(
        RESDIR / "profile_sly_1.4Msun.csv",
        ["r_km", "P_dyncm2", "eps_gcm3", "rho_gcm3", "m_Msun",
         "exp2phi", "grr", "g_ms2"],
        [star.r, np.asarray(u.pressure_geom_to_cgs(star.p)),
         np.asarray(u.density_geom_to_cgs(star.eps)),
         np.asarray(u.density_geom_to_cgs(star.rho)),
         np.asarray(u.mass_geom_to_msun(star.m)),
         np.exp(2 * star.phi), star.metric_grr(), star.local_gravity()],
    )
    plots.plot_profiles(star, FIGDIR)
    plots.plot_gravity_redshift(star, FIGDIR)
    log.info("canonical star: M=%.4f R=%.3f C=%.4f z=%.4f Lambda=%.1f "
             "I45=%.3f Eb=%.4f", star.M, star.R, star.compactness,
             star.z_surf, star.Lambda, star.I_45, star.binding_energy)
    return star


def newton_comparison(n_pts: int) -> None:
    """!Newtonian vs TOV sequences for the Gamma = 5/3 polytrope."""
    e = eos_mod.get_eos("poly-nr")
    grid = np.geomspace(5.0e13, 8.0e15, n_pts)
    seq_tov = compute_sequence(e, grid, tidal=False, inertia=False)
    rows = []
    for e_cgs in grid:
        try:
            s = solve_star(e, rho_c=float(u.density_cgs_to_geom(e_cgs)),
                           newtonian=True)
            rows.append((float(u.density_cgs_to_geom(e_cgs)), s.R, s.M, s.Mb))
        except Exception as exc:  # pragma: no cover
            log.warning("newtonian model dropped: %s", exc)
    from neutronstar.tov import Sequence
    arr = np.array(rows)
    seq_newt = Sequence("newtonian", arr[:, 0], arr[:, 1], arr[:, 2],
                        arr[:, 3], np.full(len(arr), np.nan),
                        np.full(len(arr), np.nan),
                        np.ones(len(arr), dtype=bool))
    plots.plot_newton_vs_tov(seq_tov, seq_newt, FIGDIR)


def convergence_study(quick: bool) -> None:
    """!Fixed-step convergence (order verification) + work-precision.

    Order verification uses the constant-density star (smooth RHS
    through the surface: nominal orders 1/2/4 are recovered). A second
    RK4 curve on the Gamma = 5/3 polytrope demonstrates the *order
    reduction* to ~5/2 caused by the non-smooth surface behaviour
    P ~ (R-r)^{5/2}.
    """
    e_cd = eos_mod.ConstantDensityEOS.from_cgs(1.0e15)
    p_c = 0.1 * e_cd.eps0
    ref_cd = solve_star(e_cd, p_c=p_c, method="DOP853", rtol=1e-13,
                        p_surface=p_c * 1e-12, tidal=False, inertia=False)
    e_poly = eos_mod.get_eos("poly-nr")
    rho_c = float(u.density_cgs_to_geom(5.0e15))
    ref_poly = solve_star(e_poly, rho_c=rho_c, method="DOP853", rtol=1e-12,
                          tidal=False, inertia=False)
    hs = np.array([0.4, 0.2, 0.1, 0.05, 0.025] if quick else
                  [0.4, 0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625])
    errors: dict = {}
    for method in ("euler", "heun", "rk4"):
        err = []
        for h in hs:
            s = solve_star(e_cd, p_c=p_c, method=method, h=float(h),
                           p_surface=p_c * 1e-12, tidal=False, inertia=False)
            err.append(abs(s.M - ref_cd.M) / ref_cd.M)
        errors[method] = np.array(err)
        order = np.polyfit(np.log(hs), np.log(np.maximum(err, 1e-16)), 1)[0]
        log.info("convergence %-6s: measured order = %.2f", method, order)
    err = []
    for h in hs:
        s = solve_star(e_poly, rho_c=rho_c, method="rk4", h=float(h),
                       tidal=False, inertia=False)
        err.append(abs(s.M - ref_poly.M) / ref_poly.M)
    errors["rk4 (polytrope)"] = np.array(err)
    order = np.polyfit(np.log(hs), np.log(np.maximum(err, 1e-16)), 1)[0]
    log.info("convergence rk4/polytrope: measured order = %.2f "
             "(mass is insensitive to the non-smooth surface since "
             "dm/dr -> 0 there)", order)
    plots.plot_convergence(hs, errors, FIGDIR)
    write_csv(RESDIR / "convergence_fixed_step.csv",
              ["h_km"] + list(errors.keys()),
              [hs] + [errors[m] for m in errors])

    # work-precision over adaptive methods (SLy 1.4-ish model)
    e_sly = eos_mod.get_eos("sly")
    eps_c = float(u.density_cgs_to_geom(1.0e15))
    ref = solve_star(e_sly, eps_c=eps_c, method="DOP853", rtol=1e-12,
                     tidal=False, inertia=False)
    tols = np.array([1e-4, 1e-6, 1e-8, 1e-10] if quick else
                    [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11])
    data: dict = {}
    for method in ("RK45", "DOP853", "Radau", "BDF", "LSODA"):
        nr, er = [], []
        for tol in tols:
            try:
                s = solve_star(e_sly, eps_c=eps_c, method=method,
                               rtol=float(tol), tidal=False, inertia=False)
                nr.append(s.n_rhs)
                er.append(abs(s.M - ref.M) / ref.M)
            except Exception as exc:
                log.warning("work-precision %s tol=%.0e failed: %s",
                            method, tol, exc)
        data[method] = {"nrhs": np.array(nr), "err": np.array(er)}
    plots.plot_work_precision(data, FIGDIR)
    rows = [{"method": m, "rtol": float(t), "nrhs": int(n), "err_M": float(e_)}
            for m, d in data.items()
            for t, n, e_ in zip(tols[: len(d["nrhs"])], d["nrhs"], d["err"])]
    write_kv_csv(RESDIR / "work_precision.csv", rows)


def sensitivity_study(quick: bool) -> None:
    """!Sensitivity of (M, R) to rtol, starting radius and surface cut."""
    e = eos_mod.get_eos("sly")
    eps_c = float(u.density_cgs_to_geom(1.0e15))
    ref = solve_star(e, eps_c=eps_c, rtol=1e-12, tidal=False, inertia=False)
    tols = np.geomspace(1e-4, 1e-11, 5 if quick else 8)
    dm, dr = [], []
    for tol in tols:
        s = solve_star(e, eps_c=eps_c, rtol=float(tol),
                       tidal=False, inertia=False)
        dm.append(abs(s.M - ref.M) / ref.M)
        dr.append(abs(s.R - ref.R) / ref.R)
    plots.plot_sensitivity({"rtol": tols, "dM": np.array(dm),
                            "dR": np.array(dr)}, FIGDIR)
    rows = [{"rtol": float(t), "dM_over_M": float(a), "dR_over_R": float(b)}
            for t, a, b in zip(tols, dm, dr)]

    # starting radius r0 and surface pressure cut
    for r0 in (1e-8, 1e-6, 1e-4, 1e-2):
        s = solve_star(e, eps_c=eps_c, r0=r0, rtol=1e-10,
                       tidal=False, inertia=False)
        rows.append({"rtol": f"r0={r0:g} km",
                     "dM_over_M": abs(s.M - ref.M) / ref.M,
                     "dR_over_R": abs(s.R - ref.R) / ref.R})
    for fac in (0.1, 1.0, 10.0, 100.0):
        s = solve_star(e, eps_c=eps_c, rtol=1e-10,
                       p_surface=e.p_surface * fac,
                       tidal=False, inertia=False)
        rows.append({"rtol": f"P_surf x {fac:g}",
                     "dM_over_M": abs(s.M - ref.M) / ref.M,
                     "dR_over_R": abs(s.R - ref.R) / ref.R})
    write_kv_csv(RESDIR / "sensitivity.csv", rows)


def main(argv: list | None = None) -> int:
    """!Entry point; returns a process exit code."""
    ap = argparse.ArgumentParser(description="TOV neutron-star pipeline")
    ap.add_argument("--quick", action="store_true",
                    help="coarse grids (smoke test)")
    args = ap.parse_args(argv)
    n_pts = 25 if args.quick else 60

    t0 = time.perf_counter()
    plots.setup_style()
    FIGDIR.mkdir(exist_ok=True)
    RESDIR.mkdir(exist_ok=True)

    log.info("=== 1/6 EOS figures ===")
    all_eos = [eos_mod.get_eos(n) for n in
               ("sly", "sly-pp", "apr", "fermi", "poly-nr")]
    plots.plot_eos_comparison(all_eos, FIGDIR)
    plots.plot_sound_speed(all_eos, FIGDIR)
    plots.plot_adiabatic_index(all_eos, FIGDIR)

    log.info("=== 2/6 sequences ===")
    seqs = build_sequences(n_pts)
    export_sequences(seqs)
    seq_list = [seqs[n][1] for n in ("sly", "apr", "fermi", "poly-nr")]
    plots.plot_mass_radius(seq_list, FIGDIR)
    plots.plot_sequences_vs_density(seq_list, FIGDIR)
    plots.plot_tidal([seqs["sly"][1], seqs["apr"][1]], FIGDIR)
    plots.plot_inertia_binding([seqs["sly"][1], seqs["apr"][1]], FIGDIR)

    log.info("=== 3/6 canonical star + Newtonian comparison ===")
    canonical_star(seqs)
    newton_comparison(max(20, n_pts // 2))

    log.info("=== 4/6 validation ===")
    rows = val.run_all_validations(seqs["sly"][1], seqs["apr"][1],
                                   seqs["sly"][0], seqs["apr"][0])
    for name in ("sly", "apr", "fermi"):
        e, s = seqs[name]
        b = val.buchdahl_check(s)
        mm = maximum_mass(e, s)
        c = val.causality_check(e, mm["eps_c_cgs"])
        rows.append({"benchmark": f"{e.name} Buchdahl C_max", "numeric":
                     b["C_max"], "reference": 4.0 / 9.0,
                     "rel_error": b["C_max"] / (4.0 / 9.0),
                     "source": "Buchdahl (1959)"})
        rows.append({"benchmark": f"{e.name} max c_s^2/c^2", "numeric":
                     c["cs2_max"], "reference": 1.0,
                     "rel_error": c["cs2_max"], "source": "causality"})
    write_kv_csv(RESDIR / "validation.csv", rows)

    log.info("=== 5/6 convergence + sensitivity ===")
    convergence_study(args.quick)
    sensitivity_study(args.quick)

    log.info("=== 6/6 summary table ===")
    summary = []
    for name in ("sly", "apr", "fermi", "poly-nr"):
        e, s = seqs[name]
        mm = maximum_mass(e, s)
        row = {"EOS": e.name, "M_max_Msun": round(mm["M_max"], 4),
               "R_at_Mmax_km": round(mm["R"], 3),
               "eps_c_Mmax_gcm3": f"{mm['eps_c_cgs']:.3e}",
               "C_at_Mmax": round(mm["compactness"], 4)}
        try:
            st = solve_at_mass(e, s, 1.4)
            row.update({"R_1.4_km": round(st.R, 3),
                        "Lambda_1.4": round(float(st.Lambda), 1),
                        "I_1.4_1e45gcm2": round(float(st.I_45), 3),
                        "z_surf_1.4": round(st.z_surf, 4),
                        "Eb_1.4_Msun": round(st.binding_energy, 4)})
        except ValueError:
            row.update({"R_1.4_km": "n/a", "Lambda_1.4": "n/a",
                        "I_1.4_1e45gcm2": "n/a", "z_surf_1.4": "n/a",
                        "Eb_1.4_Msun": "n/a"})
        summary.append(row)
    write_kv_csv(RESDIR / "summary.csv", summary)
    for row in summary:
        log.info("%s", row)
    log.info("pipeline finished in %.1f s", time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
