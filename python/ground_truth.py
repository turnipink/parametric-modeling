"""Stand-in for the *expensive* TCAD / CAE simulator.

In a real fab this would be a coupled process+device solver
(Sentaurus, Silvaco, COMSOL, an in-house defect-kinetics code, etc.)
costing minutes to hours per point.

Here we hard-code a physics-flavored ground truth so the rest of the
pipeline (LHS sampling, parametric fitting, optimization) has something
honest to chew on.  The truth model includes:

  * Equilibrium Arrhenius diffusion of boron in Si.
  * Transient enhanced diffusion (TED) from implant-damage-driven
    self-interstitial supersaturation; enhancement factor decays as
    the {311} extended defects dissolve (Frank-Turnbull / kick-out
    picture).  Without this term, Xj is systematically underestimated
    by 10x-100x at short dwell times in the 950-1100 C window.
  * Dose-dependent junction depth from the Gaussian profile crossing
    substrate background doping N_bg:
        Cs = (dose * f_act) / sqrt(pi * D_eff * t)
        Xj = sqrt(4 * D_eff * t * ln(Cs / N_bg))
  * Stretched-exponential dopant activation kinetics.
  * Caughey-Thomas mobility roll-off vs active carrier density.
  * Weak Rs perturbation from O2 *partial* pressure (oxidation pile-up
    at the surface).  Total chamber pressure is not used.

Small log-normal process noise is added so the surrogate has to
extract signal the way it would in a real run.
"""

from __future__ import annotations

import numpy as np

from config import KB_EV, Q_E, N_BG, PO2_REF

# "True" physical parameters the surrogate is NOT told about.
_TRUE = dict(
    # Equilibrium B-in-Si diffusion
    D0       = 0.76,     # cm^2/s   pre-exponential
    Ea_diff  = 3.46,     # eV       activation energy

    # Transient enhanced diffusion (TED)
    A_TED    = 35.0,     # peak enhancement factor scale at dose0
    p_TED    = 0.55,     # dose exponent (more damage -> more interstitials)
    dose0    = 1.0e15,   # cm^-2    reference dose for TED scaling
    tau0_311 = 4.0e-13,  # s        {311} dissolution Arrhenius prefactor
    Ea_311   = 3.6,      # eV       {311} dissolution activation energy

    # Activation kinetics
    tau0     = 2.0e-13,  # s        prefactor
    Ea_act   = 2.55,     # eV       activation energy
    beta     = 0.85,     # stretched-exponential exponent

    # Mobility roll-off
    mu0      = 470.0,    # cm^2/(V.s)  low-field hole mobility scale
    gamma    = 0.18,     # roll-off exponent

    # Surface chemistry: weak Rs sensitivity to O2 partial pressure
    p_exp    = 0.05,
)

# --- Systematic effects structurally absent from the surrogate's equations ---
# These represent real process physics a 12-parameter Arrhenius surrogate cannot
# represent, giving the hybrid NN genuine residual signal to learn.
#
# (1) Dose-dependent activation saturation: boron-interstitial cluster (BIC)
#     trapping reduces the electrically active fraction at high implant doses.
#     The surrogate's f_act(T,t) has no dose dependence -- this is
#     structurally unrepresentable by any combination of the 12 parameters.
_C_SAT     = 0.10     # ±10 % activation correction across the dose range
_LOG_D_CTR = 14.5     # log10 pivot ≈ 3e14 cm^-2 (centre of dose range)
_LOG_D_SCL = 0.6      # width in decades of the saturation roll-off
#
# (2) Nonlinear O2 partial-pressure effect: Langmuir-type adsorption gives a
#     quadratic correction in log(pO2) space that a pure power-law surrogate
#     cannot fit.
_C_QUAD    = 0.005    # ~+11 % at 0.01 Torr, ~+3 % at 10 Torr, 0 at 1 Torr


def _D_eq(T):
    """Equilibrium Arrhenius diffusivity (cm^2/s)."""
    return _TRUE["D0"] * np.exp(-_TRUE["Ea_diff"] / (KB_EV * T))


def _ted_enhancement(T, t, dose):
    """Time- and temperature-dependent TED enhancement factor.

    Enhancement decays as {311} interstitial clusters dissolve with
    time constant tau_311(T) = tau0_311 * exp(Ea_311 / kT).
    Returns a dimensionless multiplier >= 1.
    """
    tau_311 = _TRUE["tau0_311"] * np.exp(_TRUE["Ea_311"] / (KB_EV * T))
    decay = np.exp(-t / tau_311)
    return 1.0 + _TRUE["A_TED"] * (dose / _TRUE["dose0"]) ** _TRUE["p_TED"] * decay


def _D_eff(T, t, dose):
    return _D_eq(T) * _ted_enhancement(T, t, dose)


def _activation_fraction(T, t):
    tau = _TRUE["tau0"] * np.exp(_TRUE["Ea_act"] / (KB_EV * T))
    return 1.0 - np.exp(-((t / tau) ** _TRUE["beta"]))


def _mobility(Na):
    return _TRUE["mu0"] / (1.0 + (Na / 1.0e19) ** _TRUE["gamma"])


def simulate(dose, T_anneal, t_anneal, pO2, noise=0.02, rng=None):
    """Run the 'expensive' simulator at one or many points.

    All inputs may be scalars or 1-D arrays of equal length.
    Returns dict with arrays Rs [Ohm/sq] and Xj [nm].
    """
    dose     = np.atleast_1d(np.asarray(dose,     dtype=float))
    T_anneal = np.atleast_1d(np.asarray(T_anneal, dtype=float))
    t_anneal = np.atleast_1d(np.asarray(t_anneal, dtype=float))
    pO2      = np.atleast_1d(np.asarray(pO2,      dtype=float))

    f_act      = _activation_fraction(T_anneal, t_anneal)
    # BIC clustering: dose-dependent activation saturation
    # (surrogate's f_act has no dose dependence -- structurally unrepresentable)
    f_sat      = 1.0 - _C_SAT * np.tanh((np.log10(dose) - _LOG_D_CTR) / _LOG_D_SCL)
    Na_sheet   = dose * f_act * np.clip(f_sat, 0.5, 1.5)    # cm^-2
    D_eff_arr  = _D_eff(T_anneal, t_anneal, dose)             # cm^2/s
    Dt         = D_eff_arr * t_anneal                          # cm^2

    # Gaussian profile peak concentration (cm^-3)
    Cs = Na_sheet / np.sqrt(np.pi * np.maximum(Dt, 1.0e-30))
    # Junction depth where Gaussian tail crosses N_bg
    ratio = np.maximum(Cs / N_BG, 1.0 + 1.0e-9)               # ensure Xj > 0
    Xj_cm = np.sqrt(4.0 * Dt * np.log(ratio))
    Xj_nm = Xj_cm * 1.0e7

    # Active bulk carrier density for mobility roll-off
    Na_bulk = Na_sheet / np.maximum(Xj_cm, 1.0e-7)
    mu = _mobility(Na_bulk)

    Rs = 1.0 / (Q_E * Na_sheet * mu)
    # pO2 effect: power-law base + nonlinear Langmuir correction in log(pO2)
    # space.  The surrogate fits only the linear (power-law) term.
    log_pO2_norm = np.log(pO2 / PO2_REF)
    Rs = Rs * (pO2 / PO2_REF) ** _TRUE["p_exp"] \
             * (1.0 + _C_QUAD * log_pO2_norm ** 2)

    if noise > 0:
        rng = (rng if isinstance(rng, np.random.Generator)
               else np.random.default_rng(rng if isinstance(rng, (int, np.integer))
                                          else None))
        Rs    = Rs    * np.exp(rng.normal(0.0, noise,        size=Rs.shape))
        Xj_nm = Xj_nm * np.exp(rng.normal(0.0, noise * 0.5,  size=Xj_nm.shape))

    return {"Rs": Rs, "Xj": Xj_nm}


if __name__ == "__main__":
    out = simulate(2e15, 1323.0, 5.0, 1.0, noise=0.0)
    print(f"Rs={out['Rs'][0]:.1f} Ohm/sq   Xj={out['Xj'][0]:.2f} nm")
