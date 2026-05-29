"""Latin Hypercube Sampling of the recipe space.

The point of LHS (vs. a full factorial or random) is to *buy information*:
every projection onto a single input axis is uniformly covered, so each
expensive simulation purchases maximum coverage of the marginal space.

We log-space the dose and time knobs because they vary over orders of
magnitude and the underlying physics is multiplicative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import qmc

from config import INPUT_BOUNDS, INPUT_NAMES

# Which knobs should be sampled in log-space.
_LOG_KNOBS = {"dose", "t_anneal", "P_chamber"}


def latin_hypercube(n_samples: int, seed: int = 7) -> pd.DataFrame:
    sampler = qmc.LatinHypercube(d=len(INPUT_NAMES), seed=seed)
    u = sampler.random(n=n_samples)  # uniform [0,1]^d
    cols = {}
    for j, name in enumerate(INPUT_NAMES):
        lo, hi = INPUT_BOUNDS[name]
        if name in _LOG_KNOBS:
            cols[name] = np.exp(np.log(lo) + u[:, j] * (np.log(hi) - np.log(lo)))
        else:
            cols[name] = lo + u[:, j] * (hi - lo)
    return pd.DataFrame(cols)


if __name__ == "__main__":
    df = latin_hypercube(40)
    print(df.describe())
