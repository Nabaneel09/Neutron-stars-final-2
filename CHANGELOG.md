# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project adheres to
semantic versioning.

## [1.0.0] — 2026-07-08

### Added
- Six equations of state behind one abstract interface: constant density,
  Γ = 5/3 neutron polytrope (K from first principles), relativistic neutron
  Fermi gas, SLy via the Haensel–Potekhin analytic fit, and SLy/AP4 piecewise
  polytropes (Read et al. 2009) matched to the SLy crust.
- TOV integrator with metric potential, baryon mass, tidal response y(r) and
  Hartle frame-dragging integrated in a single pass; Newtonian branch.
- Hand-written Euler / Heun / RK4 with order-preserving bisection event
  location; adaptive RK45 / DOP853 / Radau / BDF / LSODA via SciPy.
- Regularized u = m/r³ formulation curing the O(h²) order reduction at the
  coordinate singularity r = 0 (measured orders 0.99 / 2.01 / 3.90).
- Validation ladder: interior Schwarzschild (~1e-13), Lane–Emden (~1e-8),
  OV maximum mass (6e-5), SLy/AP4 literature comparison (0.1–1 %).
- Sensitivity studies (tolerance, starting radius, surface cut), convergence
  and work–precision figures, 34-test pytest suite.
- LaTeX report with the complete TOV derivation; Doxygen configuration;
  25-minute defense presentation; CI workflow; pre-commit tooling.

### Fixed (during development — kept for pedagogical value)
- O(h²) error floor from linear event interpolation → bisection refinement.
- Discontinuous RHS beyond the surface corrupting adaptive event location →
  smooth continuation (EOS lookups clamped, dynamical P kept).
- Catastrophic cancellation of the k₂(C, y) formula at C ≪ 1 → documented,
  tested via linear extrapolation in C.
- AP4 crust-core junction not found when using the full SLy fit as "crust" →
  last crust polytrope extended to the Γ₁ segment (Read et al. prescription).
