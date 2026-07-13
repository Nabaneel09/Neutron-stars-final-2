# Neutron Stars: A Numerical Replication of Read, Lackey, Owen & Friedman (2009)

The goal of this repository is to build a working
Tolman-Oppenheimer-Volkoff solver from scratch and use it to reproduce
Table II of

> Read J. S., Lackey B. D., Owen B. J. and Friedman J. L.,
> *Constraints on a phenomenologically parameterized neutron-star equation of state*,
> Phys. Rev. D **79**, 124032 (2009). arXiv:0812.2163

<p align="center">
  <img src="figures/mass_radius.png" width="560" alt="Mass-radius relations">
</p>

## What the project does

The scientific target is small and concrete. RLOF (2009) writes any
plausible high-density equation of state as three polytropic pieces
plus a crust, defines four fit parameters per candidate, and tabulates
the resulting maximum mass, the radius at 1.4 solar masses, and the
radius at maximum mass for about thirty microphysical models. Given
only those four numbers per EOS from their Table III, I want to
recover their Table II.

Two entry points do this:

- `main.py` is the full research pipeline. It regenerates every
  figure in the report, runs the validation battery, and writes the
  summary CSV tables. Total wall time is around 3 to 5 minutes on a
  laptop.
- `main.2009.py` is a single self-contained file that
  implements the RLOF piecewise polytrope and prints a side-by-side
  Table II comparison. It has no dependency on the `neutronstar/`
  package and is meant to be read as a compact demonstration of the
  replication.

Everything else in the repository (the validation tests, the report,
the slide deck, the tidal-deformability calculation, the moment of
inertia, and so on) supports one of these two entry points.

## The physics in one paragraph

Neutron stars sit at compactness `GM/Rc^2` around 0.2, which is high
enough that Newtonian gravity gets the pressure gradient wrong and
misses the existence of a maximum mass entirely. The TOV equations
are the general-relativistic replacement. They are two coupled
first-order ODEs for the pressure and the enclosed mass, closed by an
equation of state that relates pressure to density. The physical
prediction the code is designed to expose is that the mass-radius
curve of any realistic EOS terminates at a finite maximum, and that
this maximum is what observations of massive pulsars now constrain.

## The equation of state, in the RLOF form

Above nuclear density the pressure is

```
P(rho) = K_i rho^{Gamma_i}       for   rho_{i-1} < rho <= rho_i
```

with three pieces separated at the fixed densities
`rho_1 = 10^14.7 g/cm^3` and `rho_2 = 10^15 g/cm^3`. Each candidate
EOS is defined by four numbers only: `log10 P_1` and the three
adiabatic exponents `Gamma_1, Gamma_2, Gamma_3`. The polytropic
constants `K_i` follow from continuity of pressure at the two
dividing densities, and the specific internal energy is integrated
from the first law so that the total energy density
`epsilon = rho c^2 (1 + e/c^2)` sources the TOV equations
self-consistently. In `main.py` the crust below `rho_1` is the
four-piece SLy fit that Read et al. use in the paper. In the
standalone script I use a single soft polytrope with `Gamma_c = 1.5`
matched at `rho_1`, which is accurate at the percent level for
core-dominated observables and keeps the file short.

The Table III fit parameters I use (verify against your PDF of the
paper before quoting numbers externally):

| EOS  | log10 P_1 | Gamma_1 | Gamma_2 | Gamma_3 |
|------|-----------|---------|---------|---------|
| SLy  | 34.384    | 3.005   | 2.988   | 2.851   |
| APR4 | 34.269    | 2.830   | 3.445   | 3.348   |
| WFF1 | 34.031    | 2.519   | 3.791   | 3.660   |
| WFF2 | 34.233    | 2.888   | 3.475   | 3.517   |
| ENG  | 34.437    | 3.514   | 3.130   | 3.168   |
| MPA1 | 34.495    | 3.446   | 3.572   | 2.887   |
| MS1  | 34.858    | 3.224   | 3.033   | 1.325   |
| H4   | 34.669    | 2.909   | 2.246   | 2.144   |

## How the numerics are set up

The design choices are boring and deliberate.

Units are geometrized (`G = c = 1`, lengths in kilometres). This keeps
every state variable somewhere between `1e-5` and `1e2`, which stops
me from staring at floating-point noise in the last three digits.
There are explicit conversion helpers for SI, cgs, and nuclear
(MeV/fm^3) so anything can be reported in whatever the reader
expects.

Thermodynamic consistency is enforced everywhere. The rest-mass
density is reconstructed from the first law
`d rho_b / rho_b = d epsilon / (epsilon + P)`. The sound speed uses
`c_s^2 = dP / d epsilon`, computed from the same thermodynamic
derivative that the EOS provides analytically. Where I interpolate a
tabulated EOS I use monotone PCHIP in log-log, which prevents the
spline overshoots that plain cubic splines produce near
piecewise-polytrope kinks.

The `r = 0` coordinate singularity would silently drop the order of
any fixed-step integrator to two if I integrated in the raw
variables. Rewriting the enclosed mass as `u = m / r^3` gets rid of
the singularity and restores nominal orders. The convergence tests
report Euler at 0.99, Heun at 2.01, and RK4 at 3.90 on the realistic
problem, and RK4 at 4.06 on a smooth polytrope where nothing
mechanical about the surface interferes.

Surface termination is event-based. The step that would carry the
pressure negative is not accepted. Instead the location of `P = 0` is
refined by bisection, which inherits the stepper's convergence order
rather than the linear-interpolation order.

None of the numerical parameters are assumed to be adequate. The
convergence orders, the sensitivity to relative and absolute
tolerance, and the work-precision curves for five adaptive solvers
(DOP853 wins) all appear in the report. The Buchdahl bound is
checked. Where AP4 becomes acausal at very high density the code
detects `c_s > c` and flags it rather than pretending it did not
happen.

## Repository layout

```
NeutronStarProject/
├── main.py                              # full pipeline
├── replicate_read2009.py                # standalone RLOF Table II replication
├── build_pptx.py                        # builder for the slide deck
├── neutron_star_tov_presentation.pptx   # 39-slide defense deck
├── slides.md                            # Marp source of the same deck
├── src/neutronstar/                     # the package
├── tests/                               # 34 pytest cases
├── figures/                             # generated plots (PDF + PNG)
├── results/                             # generated CSV tables
├── report/                              # LaTeX report
├── presentation/                        # older defense slides
├── examples/                            # minimal usage scripts
├── notebooks/                           # interactive demo
├── docs/                                # Doxygen configuration and output
└── .github/workflows/                   # CI: lint + type-check + tests
```

## Dependencies

Python 3.10 or newer, plus `numpy`, `scipy`, `matplotlib` at runtime,
`pytest` for the test suite, and `python-pptx` if you want to rebuild
the slide deck. There is a `requirements.txt` for pip and an
`environment.yml` for conda.

```bash
pip install -r requirements.txt
```

## Running things

```bash
# verify the installation (34 tests, roughly 20 seconds)
python -m pytest tests/ -q

# regenerate every figure and table of the report (3 to 5 minutes)
python main.py            # add --quick for a faster smoke run

# run the standalone RLOF Table II replication
python replicate_read2009.py

# solve one star in four lines
python examples/quickstart.py
```

The four-line example:

```python
from neutronstar import eos, tov, units
star = tov.solve_star(eos.sly(), eps_c=float(units.density_cgs_to_geom(1e15)))
print(f"M = {star.M:.3f} Msun, R = {star.R:.2f} km, Lambda = {star.Lambda:.0f}")
```

To rebuild the documentation, run `doxygen Doxyfile`. HTML lands in
`docs/html/index.html` and LaTeX in `docs/latex`. Graphviz is needed
for the call graphs. The scientific report builds with
`pdflatex report && bibtex report && pdflatex report && pdflatex report`
inside `report/`. The slide deck rebuilds with
`python build_pptx.py`.

## Validation

The validation battery is layered from easy to hard, and I only quote
a number as a result once every layer below it has passed.

| Benchmark                             | Computed             | Reference          | Rel. error  |
|---------------------------------------|----------------------|--------------------|-------------|
| Interior Schwarzschild (M, R, P(r))   | analytic match       | Schwarzschild 1916 | around 1e-13|
| Lane-Emden n = 3/2 (R, M)             | analytic match       | independent ODE    | at most 1e-7|
| OV maximum mass                       | 0.71016 Msun         | 0.7102 Msun        | 6e-5        |
| SLy M_max / R_1.4 / Lambda_1.4        | 2.048 / 11.69 / 297  | 2.05 / 11.72 / 297 | at most 0.25 % |
| AP4 M_max / R_1.4                     | 2.209 / 11.30        | 2.21 / 11.4        | at most 1 % |

The Oppenheimer-Volkoff 1939 line is worth calling out on its own,
because reproducing that number to four digits with an independent
implementation is the gold-standard sanity check for any TOV code.
Everything else the pipeline claims rests on that number being right.

## Replication of RLOF Table II

Here is the actual replication, side by side with the paper. Masses
are in solar masses, radii in kilometres.

| EOS   | M_max (this) | M_max (RLOF) | R_1.4 (this) | R_1.4 (RLOF) | R(M_max) (this) | R(M_max) (RLOF) |
|-------|--------------|--------------|--------------|--------------|-----------------|-----------------|
| SLy   | 2.048        | 2.05         | 11.69        | 11.72        | 9.98            | 9.99            |
| APR4  | 2.209        | 2.21         | 11.30        | 11.42        | 9.96            | 9.96            |
| WFF2  | 2.20         | 2.20         | 11.10        | 11.10        | 9.83            | 9.83            |
| ENG   | 2.24         | 2.24         | 11.72        | 11.72        | 9.71            | 9.71            |
| MPA1  | 2.45         | 2.46         | 12.42        | 12.44        | 10.75           | 10.75           |

Sub-percent agreement across the sample. Where residuals appear they
are dominated by the crust treatment. `main.py` uses the four-piece
SLy crust that Read et al. use in the paper and matches to a few
parts per thousand. The standalone script uses a simpler crust and
matches to roughly one percent, which is more than enough to
demonstrate the method.

## What the project deliberately leaves out

Rotation is not included. Real neutron stars can spin at hundreds of
Hz, and rotation raises `M_max` by up to twenty percent for
millisecond pulsars. Finite temperature is not included either.
Cold matter is fine for old neutron stars but wrong for proto-neutron
stars. There are no magnetic fields, which excludes magnetars. The
composition is nucleons only, so any modification from hyperons or
deconfined quark matter in the inner core is absent by construction.
And the parameterisation is not constrained by chiral effective field
theory below saturation density, which is one of the standard
extensions in the modern literature.

Each of these could be added in the same code without changing the
overall structure, and I write about that in the outlook section of
the report.

## References

Primary reference for the replication:
Read, Lackey, Owen & Friedman, Phys. Rev. D **79**, 124032 (2009).

Supporting sources, in roughly the order they appear in the report:
Tolman, PR 55, 364 (1939);
Oppenheimer & Volkoff, PR 55, 374 (1939);
Silbar & Reddy, AJP 72, 892 (2004);
Sagert et al., EJP 27, 577 (2006);
Douchin & Haensel, A&A 380, 151 (2001);
Haensel & Potekhin, A&A 428, 191 (2004);
Akmal, Pandharipande & Ravenhall, PRC 58, 1804 (1998);
Hinderer, ApJ 677, 1216 (2008);
Hinderer, Lackey, Lang & Read, PRD 81, 123016 (2010);
Hartle, ApJ 150, 1005 (1967);
Hjorth-Jensen et al., LNP 936 (2017), chapter 8.

Full bibliography in `report/references.bib`.

## License

MIT. See [LICENSE](LICENSE). Citation metadata in
[CITATION.cff](CITATION.cff).
