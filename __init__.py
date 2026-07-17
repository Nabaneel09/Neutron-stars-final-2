"""! @mainpage Neutron Stars: Numerical Solution of the TOV Equations
@brief Python package solving the Tolman-Oppenheimer-Volkoff equations.

Computational Nuclear Physics project (University of Bologna, P. Finelli).

@section modules Modules
- constants.py  -- CODATA/IAU physical constants
- units.py      -- SI / cgs / geometrized / nuclear unit conversions
- eos.py        -- six equations of state
- solver.py     -- hand-written Euler / Heun / RK4 integrators
- tov.py        -- TOV integration, tidal Love number, moment of inertia
- validation.py -- analytic and literature benchmarks
- plotting.py   -- publication-quality figures

@section quick Quick start
@code{.py}
from neutronstar import eos, tov, units
star = tov.solve_star(eos.sly(), eps_c=units.density_cgs_to_geom(1.0e15))
print(star.M, star.R, star.Lambda)
@endcode
"""

from . import constants, eos, plotting, solver, tov, units, utils, validation

__version__ = "1.0.0"
__all__ = ["constants", "units", "eos", "solver", "tov", "validation",
           "plotting", "utils", "__version__"]
