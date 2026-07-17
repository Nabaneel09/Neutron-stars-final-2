"""! @file solver.py
@brief Explicit fixed-step ODE integrators written from first principles.

These are used for the *method-comparison and convergence study*
(Euler, improved Euler/Heun, classical RK4). Production runs use the
adaptive SciPy integrators (RK45, DOP853, Radau, BDF, LSODA) through
:func:`neutronstar.tov.solve_star`.

Each stepper advances y' = f(r, y) by one step h and has the standard
local truncation error:

| method  | local error | global error | f-evals/step |
|---------|-------------|--------------|--------------|
| euler   | O(h^2)      | O(h)         | 1            |
| heun    | O(h^3)      | O(h^2)       | 2            |
| rk4     | O(h^5)      | O(h^4)       | 4            |
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

RHS = Callable[[float, np.ndarray], np.ndarray]
StopFn = Callable[[float, np.ndarray], float]


def euler_step(f: RHS, r: float, y: np.ndarray, h: float) -> np.ndarray:
    """!Forward Euler step, y_{n+1} = y_n + h f(r_n, y_n).

    @param f  Right-hand side.
    @param r  Current abscissa.
    @param y  Current state.
    @param h  Step size.
    @return   State at r + h. Complexity: 1 RHS evaluation.
    """
    return y + h * f(r, y)


def heun_step(f: RHS, r: float, y: np.ndarray, h: float) -> np.ndarray:
    """!Improved Euler (Heun) step -- 2nd-order predictor-corrector.

    Predictor: y* = y + h f(r, y); corrector averages the two slopes.
    """
    k1 = f(r, y)
    k2 = f(r + h, y + h * k1)
    return y + 0.5 * h * (k1 + k2)


def rk4_step(f: RHS, r: float, y: np.ndarray, h: float) -> np.ndarray:
    """!Classical fourth-order Runge-Kutta step.

    Four stages sampling the slope at the interval ends and midpoint,
    combined with Simpson-like weights 1/6 (1, 2, 2, 1).
    """
    k1 = f(r, y)
    k2 = f(r + 0.5 * h, y + 0.5 * h * k1)
    k3 = f(r + 0.5 * h, y + 0.5 * h * k2)
    k4 = f(r + h, y + h * k3)
    return y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


STEPPERS = {"euler": euler_step, "heun": heun_step, "rk4": rk4_step}


def integrate_fixed(
    f: RHS,
    r0: float,
    y0: np.ndarray,
    h: float,
    stop: StopFn,
    method: str = "rk4",
    max_steps: int = 2_000_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """!Integrate y' = f(r, y) with a fixed step until `stop` changes sign.

    The stopping function plays the role of a SciPy event: integration
    terminates on the first step where `stop(r, y)` becomes <= 0. The
    final point is then located by *bisection on the step size*, taking
    trial steps of the same method from the last interior point, so the
    event location inherits the full order of the stepper (a linear
    interpolation here would impose an O(h^2) error floor that masks the
    O(h^4) convergence of RK4).

    @param f          Right-hand side.
    @param r0         Initial abscissa.
    @param y0         Initial state vector.
    @param h          Fixed step size (> 0).
    @param stop       Scalar function, positive inside the domain.
    @param method     One of "euler", "heun", "rk4".
    @param max_steps  Hard iteration cap.
    @return           (r, Y) arrays with the accepted steps; the last row
                      is the interpolated stopping point.
    @throws ValueError    for an unknown method or non-positive h.
    @throws RuntimeError  if the stop condition is not met in max_steps.

    Complexity: O(N_steps * cost(f)); memory O(N_steps * len(y)).
    """
    if h <= 0.0:
        raise ValueError("step size h must be positive")
    try:
        step = STEPPERS[method]
    except KeyError as exc:
        raise ValueError(f"unknown fixed-step method '{method}'") from exc

    rs = [r0]
    ys = [np.asarray(y0, dtype=float)]
    r, y = r0, np.asarray(y0, dtype=float)
    for _ in range(max_steps):
        y_new = step(f, r, y, h)
        r_new = r + h
        s_new = stop(r_new, y_new)
        if s_new <= 0.0:  # crossed the surface inside (r, r+h]
            lo, hi = 0.0, h  # bisect on the size of the final step
            y_end, r_end = y_new, r_new
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                y_try = step(f, r, y, mid)
                if stop(r + mid, y_try) <= 0.0:
                    hi, y_end, r_end = mid, y_try, r + mid
                else:
                    lo = mid
                if hi - lo < 1e-15 * h:
                    break
            rs.append(r_end)
            ys.append(y_end)
            return np.asarray(rs), np.asarray(ys)
        r, y = r_new, y_new
        rs.append(r)
        ys.append(y)
    raise RuntimeError(
        f"fixed-step integration did not reach the surface in {max_steps} steps"
    )
