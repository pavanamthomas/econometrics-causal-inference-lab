"""Propensity scores, overlap, IPW, nearest-neighbour matching, and balance.

These procedures adjust for observed covariates. They do not address
selection on unobservables. Overlap (positivity) is part of the identifying
content; trimming is a sensitivity choice, not a free lunch.

Copyright 2026 Dr. Pavanam Thomas
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors


def propensity_scores(
    df: pd.DataFrame,
    treat: str = "d",
    covariates: list[str] | None = None,
) -> np.ndarray:
    """Logistic propensity scores fitted on the listed covariates."""
    covs = covariates or ["x"]
    x = df[covs].to_numpy(dtype=float)
    d = df[treat].to_numpy(dtype=int)
    clf = LogisticRegression(solver="lbfgs", max_iter=400)
    clf.fit(x, d)
    return clf.predict_proba(x)[:, 1]


def overlap_summary(ps: np.ndarray, treat: np.ndarray | pd.Series) -> dict[str, float]:
    """Support of fitted scores by treatment status."""
    d = np.asarray(treat).astype(int)
    p = np.asarray(ps, dtype=float)
    return {
        "min_treated": float(p[d == 1].min()) if np.any(d == 1) else float("nan"),
        "max_treated": float(p[d == 1].max()) if np.any(d == 1) else float("nan"),
        "min_control": float(p[d == 0].min()) if np.any(d == 0) else float("nan"),
        "max_control": float(p[d == 0].max()) if np.any(d == 0) else float("nan"),
        "share_ps_below_0_05": float(np.mean(p < 0.05)),
        "share_ps_above_0_95": float(np.mean(p > 0.95)),
    }


def clip_propensity(ps: np.ndarray, eps: float = 0.02) -> np.ndarray:
    """Clip scores away from 0/1. Documented trimming, not a hidden default."""
    return np.clip(np.asarray(ps, dtype=float), eps, 1.0 - eps)


def standardized_mean_difference(
    x: np.ndarray,
    treat: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """Standardized mean difference of a single covariate (optionally weighted)."""
    d = treat.astype(int)
    w = np.ones(len(d), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    xt = x[d == 1]
    xc = x[d == 0]
    wt = w[d == 1]
    wc = w[d == 0]
    mt = float(np.average(xt, weights=wt))
    mc = float(np.average(xc, weights=wc))
    vt = _weighted_var(xt, wt)
    vc = _weighted_var(xc, wc)
    denom = np.sqrt(0.5 * (vt + vc))
    if denom <= 1e-12:
        return 0.0
    return float((mt - mc) / denom)


def _weighted_var(x: np.ndarray, w: np.ndarray) -> float:
    m = float(np.average(x, weights=w))
    return float(np.average((x - m) ** 2, weights=w))


def ipw_att(
    df: pd.DataFrame,
    y: str = "y",
    treat: str = "d",
    ps: np.ndarray | None = None,
    covariates: list[str] | None = None,
    clip: float = 0.02,
) -> dict[str, float]:
    """IPW estimator of ATT: treated weight 1, control weight e/(1-e)."""
    d = df[treat].to_numpy(dtype=int)
    yv = df[y].to_numpy(dtype=float)
    scores = clip_propensity(ps if ps is not None else propensity_scores(df, treat, covariates), eps=clip)
    w = np.where(d == 1, 1.0, scores / (1.0 - scores))
    att = float(np.average(yv[d == 1]) - np.average(yv[d == 0], weights=w[d == 0]))
    return {"att": att, "n_treated": float(d.sum()), "n_control": float((d == 0).sum())}


def nearest_neighbor_att(
    df: pd.DataFrame,
    y: str = "y",
    treat: str = "d",
    ps: np.ndarray | None = None,
    covariates: list[str] | None = None,
) -> dict[str, float]:
    """1:1 nearest-neighbour matching on the propensity score, with replacement."""
    d = df[treat].to_numpy(dtype=int)
    yv = df[y].to_numpy(dtype=float)
    scores = ps if ps is not None else propensity_scores(df, treat, covariates)
    scores = np.asarray(scores, dtype=float).reshape(-1, 1)
    treated_ps = scores[d == 1]
    control_ps = scores[d == 0]
    control_y = yv[d == 0]
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn.fit(control_ps)
    _, idx = nn.kneighbors(treated_ps)
    matched = control_y[idx[:, 0]]
    att = float(yv[d == 1].mean() - matched.mean())
    return {"att": att, "n_treated": float(d.sum())}


def balance_table(
    df: pd.DataFrame,
    treat: str = "d",
    covariates: list[str] | None = None,
    weights: np.ndarray | None = None,
) -> pd.DataFrame:
    """SMD for each covariate, unweighted or with ATT-style IPW weights."""
    covs = covariates or ["x"]
    d = df[treat].to_numpy(dtype=int)
    rows = []
    for c in covs:
        x = df[c].to_numpy(dtype=float)
        rows.append(
            {
                "covariate": c,
                "smd": standardized_mean_difference(x, d, weights=weights),
                "weighted": weights is not None,
            }
        )
    return pd.DataFrame(rows)


def att_ipw_weights(df: pd.DataFrame, treat: str, ps: np.ndarray, clip: float = 0.02) -> np.ndarray:
    """ATT IPW weights used for balance checks after weighting."""
    d = df[treat].to_numpy(dtype=int)
    scores = clip_propensity(ps, eps=clip)
    return np.where(d == 1, 1.0, scores / (1.0 - scores))


SENSITIVITY_LIMITS = (
    "Matching and IPW adjust for the covariates that enter the propensity score. "
    "They are silent about unobserved confounders. Overlap failures cannot be "
    "repaired by a more flexible classifier. Trimming changes the target population. "
    "A balance table is a diagnostic, not a proof of unconfoundedness."
)
