"""Stand-in for the *expensive* TCAD / CAE simulator.

In a real fab this would be a coupled process+device solver
(Sentaurus, Silvaco, Synopsys TCAD, COMSOL, an in-house ab-initio
defect kinetics code, etc.) costing minutes to hours per point.

Here we hard-code a physics-flavored ground truth so the rest of the
pipeline (LHS sampling, parametric fitting, optimization) has something
honest to chew on.  Small process noise is added so the surrogate must
extract signal the way it would in a real run.
"""

from __future__ import annotations

import numpy as np

from config import KB_EV, Q_E

# "True" physical parameters the surrogate is NOT told about.
_TRUE = dict(
    D0       = 0.76,     # cm^2/s   pre-exponential for B diffusion in Si
    Ea_diff  = 3.46,     # eV       activation energy for diffusion
    tau0     = 2.0e-13,  # s        Arrhenius prefactor for activation kinetics
    Ea_act   = 2.55,     # eV       activation energy for dopant activation
    beta     = 0.85,     # stretched-exponential exponent
    mu0      = 470.0,    # cm^2/(V.s) low-field hole mobility scale
    gamma    = 0.18,     # mobility roll-off vs active carrier density
    p_ref    = 760.0,    # Torr     reference pressure
    p_exp    = 0.05,     # weak pressure dependence (ambient oxidation tail)
)


def _diffusivity(T):
    return _TRUE["D0"] * np.exp(-_TRUE["Ea_diff"] / (KB_EV * T))


def _activation_fraction(T, t):
    tau = _TRUE["tau0"] * np.exp(_TRUE["Ea_act"] / (KB_EV * T))
    return 1.0 - np.exp(-((t / tau) ** _TRUE["beta"]))


def _mobility(Na):
    # Caughey-Thomas-flavored simple roll-off
    return _TRUE["mu0"] / (1.0 + (Na / 1.0e19) ** _TRUE["gamma"])


def simulate(dose, T_anneal, t_anneal, P_chamber, noise=0.02, rng=None):
    """Run the 'expensive' simulator at one or many points.

    All inputs may be scalars or 1-D arrays of equal length.
    Returns dict with arrays Rs [Ohm/sq] and Xj [nm].
    """
    dose      = np.atleast_1d(np.asarray(dose, dtype=float))
    T_anneal  = np.atleast_1d(np.asarray(T_anneal, dtype=float))
    t_anneal  = np.atleast_1d(np.asarray(t_anneal, dtype=float))
    P_chamber = np.atleast_1d(np.asarray(P_chamber, dtype=float))

    D    = _diffusivity(T_anneal)                  # cm^2/s
    Xj_cm = 2.0 * np.sqrt(D * t_anneal)            # cm
    Xj_nm = Xj_cm * 1.0e7                          # nm

    f_act = _activation_fraction(T_anneal, t_anneal)
    Na_sheet = dose * f_act                        # cm^-2 (sheet)
    # convert to bulk-ish density for mobility model
    Na_bulk = Na_sheet / np.maximum(Xj_cm, 1.0e-7)
    mu = _mobility(Na_bulk)

    Rs = 1.0 / (Q_E * Na_sheet * mu)               # Ohm/sq
    # weak pressure correction: more oxidation -> small Rs uptick
    Rs = Rs * (P_chamber / _TRUE["p_ref"]) ** _TRUE["p_exp"]

    if noise > 0:
        rng = np.random.default_rng(rng if isinstance(rng, (int, np.integer)) else None) \
              if not isinstance(rng, np.random.Generator) else rng
        Rs   = Rs   * np.exp(rng.normal(0.0, noise, size=Rs.shape))
        Xj_nm = Xj_nm * np.exp(rng.normal(0.0, noise * 0.5, size=Xj_nm.shape))

    return {"Rs": Rs, "Xj": Xj_nm}


if __name__ == "__main__":
    out = simulate(2e15, 1323.0, 5.0, 760.0, noise=0.0)
    print(f"Rs={out['Rs'][0]:.1f} Ohm/sq   Xj={out['Xj'][0]:.2f} nm")
