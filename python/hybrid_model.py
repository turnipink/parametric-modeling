"""Hybrid surrogate: physics model + small NN that learns the residual.

This is the pattern most production virtual-metrology stacks land on:
the physics carries extrapolation and interpretability, the NN absorbs
the unmodeled second-order effects (tool drift, edge effects, chamber
seasoning, etc.) without being allowed to hallucinate the bulk trend.
"""

from __future__ import annotations

import numpy as np

import physics_model as pm
import nn_model as nm


class HybridModel:
    def __init__(self, phys_params: pm.PhysicsParams, residual_model):
        self.phys = phys_params
        self.residual = residual_model

    def predict(self, df_inputs):
        phys_pred = pm.predict(
            self.phys,
            df_inputs["dose"], df_inputs["T_anneal"],
            df_inputs["t_anneal"], df_inputs["P_chamber"],
        )
        delta = nm.predict(self.residual, df_inputs)
        # residual model learned log-residuals — multiplicative correction
        return {
            "Rs": phys_pred["Rs"] * delta["Rs"],
            "Xj": phys_pred["Xj"] * delta["Xj"],
        }


def fit(df_inputs, Rs_meas, Xj_meas):
    phys, info = pm.fit(df_inputs, Rs_meas, Xj_meas)
    phys_pred = pm.predict(
        phys,
        df_inputs["dose"], df_inputs["T_anneal"],
        df_inputs["t_anneal"], df_inputs["P_chamber"],
    )
    # Residual targets: measured / physics-predicted (multiplicative)
    r_Rs = np.asarray(Rs_meas) / phys_pred["Rs"]
    r_Xj = np.asarray(Xj_meas) / phys_pred["Xj"]
    residual_model = nm.fit(df_inputs, r_Rs, r_Xj, seed=1)
    return HybridModel(phys, residual_model), info
