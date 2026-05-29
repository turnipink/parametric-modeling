"""Black-box neural-net baseline (small MLP via scikit-learn).

Used to *compare* against the physics surrogate so we can have an
honest conversation about extrapolation and sample efficiency.
"""

from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def _features(df):
    # Use log of the multiplicative knobs — gives the NN a fair shot at small N.
    X = np.column_stack([
        np.log10(df["dose"].to_numpy()),
        df["T_anneal"].to_numpy(),
        np.log10(df["t_anneal"].to_numpy()),
        np.log10(df["pO2"].to_numpy()),
    ])
    return X


def fit(df_inputs, Rs_meas, Xj_meas, seed: int = 0):
    X = _features(df_inputs)
    y = np.column_stack([np.log(Rs_meas), np.log(Xj_meas)])
    model = Pipeline([
        ("scale", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(32, 32),
            activation="tanh",
            solver="lbfgs",         # small data -> lbfgs converges nicely
            max_iter=4000,
            random_state=seed,
            alpha=1.0e-3,
        )),
    ])
    model.fit(X, y)
    return model


def predict(model, df_inputs):
    X = _features(df_inputs)
    yhat = model.predict(X)
    return {"Rs": np.exp(yhat[:, 0]), "Xj": np.exp(yhat[:, 1])}
