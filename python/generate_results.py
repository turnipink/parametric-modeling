"""End-to-end pipeline runner.

  1. LHS-sample the recipe space and call the "expensive" simulator.
  2. Fit three surrogates: physics, neural-net, hybrid.
  3. Score them on a held-out random test set.
  4. Optimize the recipe with each surrogate against the yield target.
  5. Emit data/samples.csv, data/fit_params.json, and
     assets/data/results.json for the website to render.

Run from the repo root:
    /usr/bin/python3 python/generate_results.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from dataclasses import asdict

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from config import (
    INPUT_NAMES, INPUT_BOUNDS,
    RS_TARGET, RS_TOL, XJ_MAX, DT_BUDGET_MAX,
)
import sampling
import ground_truth as gt
import physics_model as pm
import nn_model as nm
import hybrid_model as hm
import optimize_recipe as opt


# ---------- helpers ----------------------------------------------------------

def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mape = float(np.mean(np.abs(err) / np.maximum(np.abs(y_true), 1e-12)) * 100.0)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"rmse": rmse, "mape_pct": mape, "r2": r2}


def random_grid(n, seed):
    rng = np.random.default_rng(seed)
    cols = {}
    for name in INPUT_NAMES:
        lo, hi = INPUT_BOUNDS[name]
        if name in ("dose", "t_anneal", "pO2"):
            cols[name] = np.exp(rng.uniform(np.log(lo), np.log(hi), size=n))
        else:
            cols[name] = rng.uniform(lo, hi, size=n)
    return pd.DataFrame(cols)


# ---------- main pipeline ----------------------------------------------------

def main():
    rng_seed = 42
    n_train = 40              # sparse — the whole point of the example
    n_test  = 400

    out_data_dir = ROOT / "data"
    out_web_dir  = ROOT / "assets" / "data"
    out_data_dir.mkdir(parents=True, exist_ok=True)
    out_web_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: LHS sampling + (mock) expensive simulator
    train_X = sampling.latin_hypercube(n_train, seed=rng_seed)
    sim     = gt.simulate(
        train_X["dose"], train_X["T_anneal"],
        train_X["t_anneal"], train_X["pO2"],
        noise=0.025, rng=rng_seed,
    )
    train = train_X.copy()
    train["Rs"] = sim["Rs"]
    train["Xj"] = sim["Xj"]
    train.to_csv(out_data_dir / "samples.csv", index=False)

    # ---- Step 2a: physics fit
    phys, phys_info = pm.fit(train_X, train["Rs"], train["Xj"])
    (out_data_dir / "fit_params.json").write_text(pm.to_json(phys))

    # ---- Step 2b: black-box NN
    nn = nm.fit(train_X, train["Rs"], train["Xj"], seed=0)

    # ---- Step 2c: hybrid
    hyb, _ = hm.fit(train_X, train["Rs"], train["Xj"])

    # ---- Step 3: hold-out scoring on a fresh random grid
    test_X = random_grid(n_test, seed=rng_seed + 1)
    test_truth = gt.simulate(
        test_X["dose"], test_X["T_anneal"],
        test_X["t_anneal"], test_X["pO2"],
        noise=0.0,
    )
    phys_pred = pm.predict(
        phys, test_X["dose"], test_X["T_anneal"],
        test_X["t_anneal"], test_X["pO2"],
    )
    nn_pred  = nm.predict(nn, test_X)
    hyb_pred = hyb.predict(test_X)

    scores = {
        "physics": {
            "Rs": metrics(test_truth["Rs"], phys_pred["Rs"]),
            "Xj": metrics(test_truth["Xj"], phys_pred["Xj"]),
        },
        "neural_net": {
            "Rs": metrics(test_truth["Rs"], nn_pred["Rs"]),
            "Xj": metrics(test_truth["Xj"], nn_pred["Xj"]),
        },
        "hybrid": {
            "Rs": metrics(test_truth["Rs"], hyb_pred["Rs"]),
            "Xj": metrics(test_truth["Xj"], hyb_pred["Xj"]),
        },
    }

    # ---- Extrapolation stress test:
    # push T 5% beyond the training upper bound; black-box NNs typically fall apart.
    extrap = test_X.copy()
    extrap["T_anneal"] = INPUT_BOUNDS["T_anneal"][1] * 1.05
    extrap_truth = gt.simulate(
        extrap["dose"], extrap["T_anneal"], extrap["t_anneal"], extrap["pO2"],
        noise=0.0,
    )
    extrap_scores = {
        "physics":    {"Rs": metrics(extrap_truth["Rs"],
                                     pm.predict(phys, extrap["dose"], extrap["T_anneal"],
                                                extrap["t_anneal"], extrap["pO2"])["Rs"])},
        "neural_net": {"Rs": metrics(extrap_truth["Rs"], nm.predict(nn, extrap)["Rs"])},
        "hybrid":     {"Rs": metrics(extrap_truth["Rs"], hyb.predict(extrap)["Rs"])},
    }

    # ---- Step 4: optimize a yield-recovery recipe with the hybrid model
    def predict_hyb(df):  return hyb.predict(df)
    def predict_phys(df):
        out = pm.predict(phys, df["dose"], df["T_anneal"], df["t_anneal"], df["pO2"])
        return {"Rs": out["Rs"], "Xj": out["Xj"]}

    recipe_hybrid = opt.optimize(predict_hyb,  phys, n_restarts=24, seed=11)
    recipe_phys   = opt.optimize(predict_phys, phys, n_restarts=24, seed=12)

    # Verify the recommended recipe against the (here-known) ground truth
    for rec in (recipe_hybrid, recipe_phys):
        truth = gt.simulate(rec["dose"], rec["T_anneal"], rec["t_anneal"],
                            rec["pO2"], noise=0.0)
        rec["Rs_truth"] = float(truth["Rs"][0])
        rec["Xj_truth"] = float(truth["Xj"][0])

    # ---- 1-D scan for the website plot: sweep T at the optimized dose/time/pO2
    T_grid = np.linspace(INPUT_BOUNDS["T_anneal"][0], INPUT_BOUNDS["T_anneal"][1], 60)
    base = recipe_hybrid
    scan_df = pd.DataFrame({
        "dose":     np.full_like(T_grid, base["dose"]),
        "T_anneal": T_grid,
        "t_anneal": np.full_like(T_grid, base["t_anneal"]),
        "pO2":      np.full_like(T_grid, base["pO2"]),
    })
    truth_scan = gt.simulate(scan_df["dose"], scan_df["T_anneal"],
                             scan_df["t_anneal"], scan_df["pO2"], noise=0.0)
    phys_scan = pm.predict(phys, scan_df["dose"], scan_df["T_anneal"],
                           scan_df["t_anneal"], scan_df["pO2"])
    nn_scan   = nm.predict(nn, scan_df)
    hyb_scan  = hyb.predict(scan_df)

    # ---- pack everything for the front-end
    results = {
        "process": {
            "name": "Post-implant Rapid Thermal Anneal (RTA), shallow boron junction",
            "inputs":  INPUT_NAMES,
            "bounds":  INPUT_BOUNDS,
            "outputs": ["Rs (Ohm/sq)", "Xj (nm)"],
            "spec": {
                "Rs_target":     RS_TARGET,
                "Rs_tol":        RS_TOL,
                "Xj_max":        XJ_MAX,
                "Dt_budget_max": DT_BUDGET_MAX,
            },
        },
        "training": {
            "n_samples": n_train,
            "samples":   train.to_dict(orient="list"),
        },
        "physics_params": asdict(phys),
        "fit_info":       phys_info,
        "scores":         scores,
        "extrap_scores":  extrap_scores,
        "holdout": {
            "truth_Rs":   test_truth["Rs"].tolist(),
            "physics_Rs": phys_pred["Rs"].tolist(),
            "nn_Rs":      nn_pred["Rs"].tolist(),
            "hybrid_Rs":  hyb_pred["Rs"].tolist(),
            "truth_Xj":   test_truth["Xj"].tolist(),
            "physics_Xj": phys_pred["Xj"].tolist(),
            "nn_Xj":      nn_pred["Xj"].tolist(),
            "hybrid_Xj":  hyb_pred["Xj"].tolist(),
        },
        "temperature_scan": {
            "T":         T_grid.tolist(),
            "truth_Rs":  truth_scan["Rs"].tolist(),
            "physics_Rs": phys_scan["Rs"].tolist(),
            "nn_Rs":     nn_scan["Rs"].tolist(),
            "hybrid_Rs": hyb_scan["Rs"].tolist(),
            "base_recipe": {k: base[k] for k in INPUT_NAMES},
        },
        "optimized_recipes": {
            "physics_only": recipe_phys,
            "hybrid":       recipe_hybrid,
        },
    }

    (out_web_dir / "results.json").write_text(json.dumps(results, indent=2))

    # ---- console summary
    def pp_score(tag, s):
        print(f"  {tag:<11s}  RMSE={s['rmse']:8.3f}   MAPE={s['mape_pct']:5.2f}%   R^2={s['r2']:.4f}")

    print("\n=== Hold-out scores on Rs (Ohm/sq) ===")
    for k in ("physics", "neural_net", "hybrid"):
        pp_score(k, scores[k]["Rs"])
    print("\n=== Hold-out scores on Xj (nm) ===")
    for k in ("physics", "neural_net", "hybrid"):
        pp_score(k, scores[k]["Xj"])
    print("\n=== Extrapolation (T pushed +5% beyond training) on Rs ===")
    for k in ("physics", "neural_net", "hybrid"):
        pp_score(k, extrap_scores[k]["Rs"])

    print("\n=== Optimized recipe (hybrid surrogate) ===")
    for k, v in recipe_hybrid.items():
        print(f"  {k:<10s} {v}")

    print(f"\nWrote: {out_data_dir/'samples.csv'}")
    print(f"Wrote: {out_data_dir/'fit_params.json'}")
    print(f"Wrote: {out_web_dir/'results.json'}")


if __name__ == "__main__":
    main()
