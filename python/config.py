"""Shared physical constants and input/output specifications.

Example process: Post-implant Rapid Thermal Anneal (RTA) for an
ultra-shallow boron junction. We model the link between recipe
knobs and two electrical KPIs:

    Inputs (recipe knobs)
    ---------------------
    dose         implant dose                [1e14 .. 5e15 cm^-2]
    T_anneal     RTA peak temperature        [1223 .. 1373 K]   (~950..1100 C)
    t_anneal     RTA dwell time              [0.5 .. 30 s]
    P_chamber    chamber pressure            [1 .. 760 Torr]

    Outputs (KPIs)
    --------------
    Rs           sheet resistance            [Ohm/sq]
    Xj           junction depth              [nm]
"""

from __future__ import annotations

# Boltzmann constant in eV/K
KB_EV = 8.617333262e-5

# Elementary charge in Coulombs
Q_E = 1.602176634e-19

# Recipe-knob bounds used everywhere (sampling, fitting, optimization)
INPUT_BOUNDS = {
    "dose":      (1.0e14, 5.0e15),   # cm^-2
    "T_anneal":  (1223.0, 1373.0),   # K
    "t_anneal":  (0.5,    30.0),     # s
    "P_chamber": (1.0,    760.0),    # Torr
}

INPUT_NAMES = list(INPUT_BOUNDS.keys())
OUTPUT_NAMES = ["Rs", "Xj"]

# Yield window the fab is trying to hit
RS_TARGET   = 90.0    # Ohm/sq
RS_TOL      = 8.0     # +/- Ohm/sq
XJ_MAX      = 12.0    # nm (must stay shallow)
THERMAL_BUDGET_MAX = 18000.0  # T_anneal * t_anneal upper bound (process limit)
