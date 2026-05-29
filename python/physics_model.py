"""Physics-shaped parametric surrogate.

We fit *coefficients of equations we already trust* rather than a
black-box regressor.  The form is dictated by domain knowledge:

    Xj      = 2 * sqrt( D0 * exp(-Ea_diff / kT) * t )           # Gaussian diffusion
    f_act   = 1 - exp( -( t / (tau0 * exp(Ea_act / kT)) )^beta )# stretched activation
    Na_bulk = dose * f_act / Xj                                 # active carriers
    mu      = mu0 / (1 + (Na_bulk / Nref)^gamma)                # mobility roll-off
    Rs      = 1 / (q * dose * f_act * mu) * (P / P0)^p_exp

Free parameters fit from sparse data:
    theta = [log10(D0), Ea_diff,
             log10(tau0), Ea_act, beta,
             log10(mu0), gamma,
             p_exp]

We fit log(Rs) and log(Xj) to stabilize residuals across decades.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json

import numpy as np
from scipy.optimize import least_squares

from config import KB_EV, Q_E

P_REF = 760.0  # Torr
N_REF = 1.0e19  # cm^-3, mobility roll-off knee


@dataclass
class PhysicsParams:
    log10_D0:    float
    Ea_diff:     float
    log10_tau0:  float
    Ea_act:      float
    beta:        float
    log10_mu0:   float
    gamma:       float
    p_exp:       float


def predict(params: PhysicsParams, dose, T, t, P):
    dose = np.asarray(dose, dtype=float)
    T    = np.asarray(T,    dtype=float)
    t    = np.asarray(t,    dtype=float)
    P    = np.asarray(P,    dtype=float)

    D0   = 10.0 ** params.log10_D0
    tau0 = 10.0 ** params.log10_tau0
    mu0  = 10.0 ** params.log10_mu0

    D_diff = D0 * np.exp(-params.Ea_diff / (KB_EV * T))
    Xj_cm  = 2.0 * np.sqrt(np.maximum(D_diff * t, 1.0e-30))
    Xj_nm  = Xj_cm * 1.0e7

    tau = tau0 * np.exp(params.Ea_act / (KB_EV * T))
    f_a = 1.0 - np.exp(-np.power(np.maximum(t / tau, 1.0e-30), params.beta))
    f_a = np.clip(f_a, 1.0e-6, 1.0)

    Na_bulk = dose * f_a / np.maximum(Xj_cm, 1.0e-7)
    mu = mu0 / (1.0 + (Na_bulk / N_REF) ** params.gamma)

    Rs = 1.0 / (Q_E * dose * f_a * mu) * (P / P_REF) ** params.p_exp
    return {"Rs": Rs, "Xj": Xj_nm, "f_act": f_a}


def fit(df_inputs, Rs_meas, Xj_meas):
    """Fit physics coefficients to sparse measurements.

    df_inputs: pandas DataFrame with columns dose, T_anneal, t_anneal, P_chamber.
    Returns (PhysicsParams, info_dict).
    """
    dose = df_inputs["dose"].to_numpy()
    T    = df_inputs["T_anneal"].to_numpy()
    t    = df_inputs["t_anneal"].to_numpy()
    P    = df_inputs["P_chamber"].to_numpy()

    log_Rs_meas = np.log(np.asarray(Rs_meas))
    log_Xj_meas = np.log(np.asarray(Xj_meas))

    # Physically plausible starting point — deliberately *not* the true values
    # so the fit has work to do.
    theta0 = np.array([
        np.log10(1.0),    # log10(D0)
        3.0,              # Ea_diff
        np.log10(1e-12),  # log10(tau0)
        2.2,              # Ea_act
        1.0,              # beta
        np.log10(300.0),  # log10(mu0)
        0.25,             # gamma
        0.0,              # p_exp
    ])
    lower = np.array([-3.0, 2.0, -16.0, 1.5, 0.3, np.log10(50.0),  0.05, -0.5])
    upper = np.array([ 2.0, 5.0,  -8.0, 4.0, 2.0, np.log10(2000.0), 0.6,  0.5])

    def residuals(theta):
        p = PhysicsParams(*theta)
        pred = predict(p, dose, T, t, P)
        r_Rs = np.log(pred["Rs"]) - log_Rs_meas
        r_Xj = np.log(pred["Xj"]) - log_Xj_meas
        # Rs is the yield KPI — weight it slightly higher.
        return np.concatenate([1.0 * r_Rs, 0.5 * r_Xj])

    res = least_squares(
        residuals, theta0, bounds=(lower, upper),
        method="trf", x_scale="jac", max_nfev=10000,
    )
    params = PhysicsParams(*res.x)
    info = {
        "success": bool(res.success),
        "cost":    float(res.cost),
        "n_iter":  int(res.nfev),
        "message": str(res.message),
    }
    return params, info


def to_json(params: PhysicsParams) -> str:
    return json.dumps(asdict(params), indent=2)
