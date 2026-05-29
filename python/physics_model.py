"""Physics-shaped parametric surrogate.

We fit *coefficients of equations we already trust* rather than a
black-box regressor.  Updated relative to the first cut after
independent technical review:

  * Junction depth uses the Gaussian profile crossing background
    doping N_bg, so Xj is dose-dependent the way a real junction is.
  * Transient enhanced diffusion (TED) is modeled explicitly via a
    {311}-defect dissolution time -- the equilibrium Arrhenius D
    alone underestimates Xj by 10x-100x at short dwell times.
  * Pressure knob is O2 *partial* pressure, not total chamber
    pressure (oxidation is what perturbs Rs at the surface).

    D_eq(T)            = D0 * exp(-Ea_diff / kT)
    tau_311(T)         = tau0_311 * exp(Ea_311 / kT)
    E_TED(T,t,dose)    = 1 + A_TED * (dose/dose0)^p_TED * exp(-t/tau_311)
    D_eff              = D_eq * E_TED
    Cs(T,t,dose)       = (dose * f_act) / sqrt(pi * D_eff * t)
    Xj                 = sqrt(4 * D_eff * t * ln(Cs / N_bg))
    f_act(T,t)         = 1 - exp(-(t / (tau0 e^{Ea_act/kT}))^beta)
    mu(N_a)            = mu0 / (1 + (N_a / N_ref)^gamma)
    Rs                 = 1/(q dose f_act mu) * (pO2 / pO2_ref)^p_exp

Free coefficients fit from sparse data:
    theta = [log D0, Ea_diff,
             log A_TED, p_TED, log tau0_311, Ea_311,
             log tau0,  Ea_act, beta,
             log mu0,   gamma,
             p_exp]

We fit log(Rs) and log(Xj) to stabilize residuals across decades.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json

import numpy as np
from scipy.optimize import least_squares

from config import KB_EV, Q_E, N_BG, PO2_REF

N_REF = 1.0e19   # cm^-3, mobility roll-off knee
DOSE0 = 1.0e15   # cm^-2, reference dose for TED scaling


@dataclass
class PhysicsParams:
    log10_D0:       float
    Ea_diff:        float
    log10_A_TED:    float
    p_TED:          float
    log10_tau0_311: float
    Ea_311:         float
    log10_tau0:     float
    Ea_act:         float
    beta:           float
    log10_mu0:      float
    gamma:          float
    p_exp:          float


def predict(params: PhysicsParams, dose, T, t, pO2):
    dose = np.asarray(dose, dtype=float)
    T    = np.asarray(T,    dtype=float)
    t    = np.asarray(t,    dtype=float)
    pO2  = np.asarray(pO2,  dtype=float)

    D0       = 10.0 ** params.log10_D0
    A_TED    = 10.0 ** params.log10_A_TED
    tau0_311 = 10.0 ** params.log10_tau0_311
    tau0     = 10.0 ** params.log10_tau0
    mu0      = 10.0 ** params.log10_mu0

    # Equilibrium + TED-enhanced diffusivity
    D_eq    = D0 * np.exp(-params.Ea_diff / (KB_EV * T))
    tau_311 = tau0_311 * np.exp(params.Ea_311 / (KB_EV * T))
    E_TED   = 1.0 + A_TED * (dose / DOSE0) ** params.p_TED \
                  * np.exp(-t / tau_311)
    D_eff   = D_eq * E_TED
    Dt      = np.maximum(D_eff * t, 1.0e-30)

    # Activation fraction (stretched exponential)
    tau = tau0 * np.exp(params.Ea_act / (KB_EV * T))
    f_a = 1.0 - np.exp(-np.power(np.maximum(t / tau, 1.0e-30), params.beta))
    f_a = np.clip(f_a, 1.0e-6, 1.0)

    # Dose-dependent Xj: where Gaussian tail crosses background
    Na_sheet = dose * f_a
    Cs       = Na_sheet / np.sqrt(np.pi * Dt)
    ratio    = np.maximum(Cs / N_BG, 1.0 + 1.0e-9)
    Xj_cm    = np.sqrt(4.0 * Dt * np.log(ratio))
    Xj_nm    = Xj_cm * 1.0e7

    # Mobility roll-off vs bulk active carriers
    Na_bulk = Na_sheet / np.maximum(Xj_cm, 1.0e-7)
    mu      = mu0 / (1.0 + (Na_bulk / N_REF) ** params.gamma)

    Rs = 1.0 / (Q_E * Na_sheet * mu) * (pO2 / PO2_REF) ** params.p_exp

    return {"Rs": Rs, "Xj": Xj_nm, "f_act": f_a, "D_eff": D_eff, "Dt": Dt}


def D_eff_dt(params: PhysicsParams, dose, T, t):
    """Thermal-budget proxy: D_eff(T,t,dose) * t  [cm^2]."""
    D0       = 10.0 ** params.log10_D0
    A_TED    = 10.0 ** params.log10_A_TED
    tau0_311 = 10.0 ** params.log10_tau0_311
    D_eq    = D0 * np.exp(-params.Ea_diff / (KB_EV * T))
    tau_311 = tau0_311 * np.exp(params.Ea_311 / (KB_EV * T))
    E_TED   = 1.0 + A_TED * (dose / DOSE0) ** params.p_TED \
                  * np.exp(-t / tau_311)
    return D_eq * E_TED * t


def fit(df_inputs, Rs_meas, Xj_meas):
    """Fit physics coefficients to sparse measurements.

    df_inputs: pandas DataFrame with columns dose, T_anneal, t_anneal, pO2.
    Returns (PhysicsParams, info_dict).
    """
    dose = df_inputs["dose"].to_numpy()
    T    = df_inputs["T_anneal"].to_numpy()
    t    = df_inputs["t_anneal"].to_numpy()
    pO2  = df_inputs["pO2"].to_numpy()

    log_Rs_meas = np.log(np.asarray(Rs_meas))
    log_Xj_meas = np.log(np.asarray(Xj_meas))

    # Physically plausible starting point -- deliberately *not* the true
    # values so the fit has work to do.
    theta0 = np.array([
        np.log10(1.0),    # log10(D0)
        3.0,              # Ea_diff
        np.log10(10.0),   # log10(A_TED)
        0.5,              # p_TED
        np.log10(1e-13),  # log10(tau0_311)
        3.4,              # Ea_311
        np.log10(1e-12),  # log10(tau0)
        2.2,              # Ea_act
        1.0,              # beta
        np.log10(300.0),  # log10(mu0)
        0.25,             # gamma
        0.0,              # p_exp
    ])
    lower = np.array([
        -3.0, 2.0,
        -1.0, 0.1, -16.0, 2.5,
        -16.0, 1.5, 0.3,
        np.log10(50.0),  0.05,
        -0.5,
    ])
    upper = np.array([
         2.0, 5.0,
         3.0, 1.5,  -8.0, 4.5,
         -8.0, 4.0, 2.0,
        np.log10(2000.0), 0.6,
         0.5,
    ])

    def residuals(theta):
        p = PhysicsParams(*theta)
        pred = predict(p, dose, T, t, pO2)
        r_Rs = np.log(pred["Rs"]) - log_Rs_meas
        r_Xj = np.log(pred["Xj"]) - log_Xj_meas
        # Rs is the yield KPI -- weight it slightly higher.
        return np.concatenate([1.0 * r_Rs, 0.5 * r_Xj])

    res = least_squares(
        residuals, theta0, bounds=(lower, upper),
        method="trf", x_scale="jac", max_nfev=20000,
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
