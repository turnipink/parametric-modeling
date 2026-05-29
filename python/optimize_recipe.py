"""Recipe optimization on top of the fitted surrogate.

Goal: bring sheet resistance back into the yield window
      Rs in [RS_TARGET - RS_TOL, RS_TARGET + RS_TOL]
while obeying hard process constraints:
      Xj                            <= XJ_MAX           (shallow junction)
      D_eff(T,t,dose) * t           <= DT_BUDGET_MAX    (cm^2; dimensionally
                                                         correct thermal budget
                                                         that correctly ranks
                                                         spike vs. soak anneals)
      and all knobs inside their physical bounds.

The Dt-budget constraint uses the *physics-model's* D_eff function so it
travels with whichever surrogate is being optimized.  We minimize squared
distance to target Rs with SLSQP because the model is cheap to query
thousands of times.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import physics_model as pm
from config import (
    INPUT_BOUNDS, INPUT_NAMES,
    RS_TARGET, RS_TOL, XJ_MAX, DT_BUDGET_MAX,
)


def _to_df(x):
    return pd.DataFrame({n: [v] for n, v in zip(INPUT_NAMES, x)})


def optimize(predict_fn, phys_params, x0=None, n_restarts: int = 16, seed: int = 1):
    """predict_fn(df) -> dict with 'Rs' and 'Xj' arrays.

    phys_params is the fitted PhysicsParams instance used to evaluate
    the Dt-budget constraint (which depends on D_eff and is not a
    property of the NN-only surrogate).
    """
    bounds = [INPUT_BOUNDS[n] for n in INPUT_NAMES]
    rng = np.random.default_rng(seed)

    def objective(x):
        out = predict_fn(_to_df(x))
        return float((out["Rs"][0] - RS_TARGET) ** 2)

    def dt_budget(x):
        # x = [dose, T_anneal, t_anneal, pO2]
        return float(pm.D_eff_dt(phys_params, x[0], x[1], x[2]))

    cons = [
        {"type": "ineq",
         "fun":  lambda x: XJ_MAX - float(predict_fn(_to_df(x))["Xj"][0])},
        {"type": "ineq",
         "fun":  lambda x: DT_BUDGET_MAX - dt_budget(x)},
    ]

    best = None
    for k in range(n_restarts):
        if x0 is not None and k == 0:
            start = np.asarray(x0, dtype=float)
        else:
            start = np.array([rng.uniform(lo, hi) for (lo, hi) in bounds])
        try:
            res = minimize(
                objective, start, method="SLSQP",
                bounds=bounds, constraints=cons,
                options={"maxiter": 300, "ftol": 1e-9},
            )
        except Exception:
            continue
        if not res.success:
            continue
        if best is None or res.fun < best.fun:
            best = res

    if best is None:
        raise RuntimeError("Optimizer failed from every restart.")

    rec = dict(zip(INPUT_NAMES, [float(v) for v in best.x]))
    pred = predict_fn(_to_df(best.x))
    rec["Rs_pred"]   = float(pred["Rs"][0])
    rec["Xj_pred"]   = float(pred["Xj"][0])
    rec["Dt_budget"] = dt_budget(best.x)
    rec["in_spec"]   = bool(abs(pred["Rs"][0] - RS_TARGET) <= RS_TOL
                            and pred["Xj"][0] <= XJ_MAX
                            and rec["Dt_budget"] <= DT_BUDGET_MAX)
    return rec
