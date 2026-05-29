"""Shared physical constants and input/output specifications.

Example process: Post-implant Rapid Thermal Anneal (RTA) for an
ultra-shallow boron junction. We model the link between recipe
knobs and two electrical KPIs.

Reviewer-driven updates vs. the first cut:
  * Pressure knob is now O2 *partial* pressure (oxidation is what
    drives the small surface Rs perturbation; total chamber pressure
    is not the physically meaningful variable -- 1 Torr N2 and
    2 Torr O2 have wildly different surface chemistries).
  * Junction depth uses the Gaussian profile crossing background
    doping, so Xj depends on dose (see physics_model.py).
  * Transient enhanced diffusion (TED) is modeled explicitly via
    a {311}-defect dissolution time -- equilibrium Arrhenius alone
    underestimates Xj at short dwell times by 10x-100x.
  * Thermal-budget constraint is now the Dt-integral (cm^2),
    which is dimensionally correct and has the right isocurves
    for spike vs. soak anneals.

    Inputs (recipe knobs)
    ---------------------
    dose      implant dose             [1e14 .. 5e15 cm^-2]
    T_anneal  RTA peak temperature     [1223 .. 1373 K]   (~950..1100 C)
    t_anneal  RTA dwell time           [0.5 .. 30 s]
    pO2       O2 partial pressure      [0.01 .. 10 Torr]  (balance N2)

    Outputs (KPIs)
    --------------
    Rs        sheet resistance         [Ohm/sq]
    Xj        junction depth           [nm]
"""

from __future__ import annotations

# Boltzmann constant in eV/K
KB_EV = 8.617333262e-5

# Elementary charge in Coulombs
Q_E = 1.602176634e-19

# Substrate background doping (n-type wafer for a p+ S/D extension)
N_BG = 1.0e15   # cm^-3

# Reference O2 partial pressure for the Rs perturbation term
PO2_REF = 1.0   # Torr

# Recipe-knob bounds used everywhere (sampling, fitting, optimization)
INPUT_BOUNDS = {
    "dose":     (1.0e14, 5.0e15),   # cm^-2
    "T_anneal": (1223.0, 1373.0),   # K
    "t_anneal": (0.5,    30.0),     # s
    "pO2":      (0.01,   10.0),     # Torr  (balance: inert N2)
}

INPUT_NAMES = list(INPUT_BOUNDS.keys())
OUTPUT_NAMES = ["Rs", "Xj"]

# Yield window the fab is trying to hit (post-anneal target for the
# shallow boron extension)
RS_TARGET   = 120.0   # Ohm/sq
RS_TOL      = 12.0    # +/- Ohm/sq
XJ_MAX      = 30.0    # nm  (shallow-junction ceiling)

# Dimensionally correct thermal-budget proxy: time-integrated effective
# diffusivity (cm^2).  For an isothermal RTA step this is just
# D_eff(T, t, dose) * t.  Replaces the old (and dimensionally nonsense)
# T_anneal * t_anneal form; this version has the right isocurves --
# a hot short spike correctly outranks a warm long soak.
DT_BUDGET_MAX = 1.5e-13   # cm^2
