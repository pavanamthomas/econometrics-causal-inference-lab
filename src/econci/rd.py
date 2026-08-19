"""Sharp regression discontinuity: local linear estimation and bandwidth checks.

Identification is at the cutoff. Local linear fits inside a bandwidth are the
object of interest. Global polynomials and mean comparisons far from the cutoff
extrapolate the running-variable relationship and should not be reported as
the RD parameter.

Copyright 2026 Dr. Pavanam Thomas
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def local_linear_rd(
    df: pd.DataFrame,
    y: str = "y",
    running: str = "running",
    cutoff: float = 0.0,
    bandwidth: float = 0.75,
    kernel: str = "triangular",
) -> dict[str, float]:
    """Local linear RD with a treated-side intercept shift and slope change.

    For |x - c| <= h:
        y = a + tau * 1{x>=c} + bl *(x-c) + br * 1{x>=c}*(x-c) + e
    Triangular kernel: w = 1 - |x-c|/h (zero outside the window). Uniform
    kernel: equal weights inside the window.
    """
    x = df[running].to_numpy(dtype=float) - cutoff
    d = (df[running].to_numpy(dtype=float) >= cutoff).astype(float)
    yv = df[y].to_numpy(dtype=float)
    dist = np.abs(x)
    inside = dist <= bandwidth
    if kernel == "triangular":
        w = np.where(inside, 1.0 - dist / bandwidth, 0.0)
    elif kernel == "uniform":
        w = inside.astype(float)
    else:
        raise ValueError("kernel must be 'triangular' or 'uniform'")
    if np.sum(w > 0) < 20:
        return {
            "tau": float("nan"),
            "se": float("nan"),
            "bandwidth": float(bandwidth),
            "n_used": float(np.sum(w > 0)),
            "kernel": kernel,
        }
    xmat = np.column_stack([np.ones_like(x), d, x, d * x])
    res = sm.WLS(yv, xmat, weights=w).fit()
    return {
        "tau": float(res.params[1]),
        "se": float(res.bse[1]),
        "bandwidth": float(bandwidth),
        "n_used": float(np.sum(w > 0)),
        "kernel": kernel,
        "pvalue": float(res.pvalues[1]),
    }


def bandwidth_sensitivity(
    df: pd.DataFrame,
    bandwidths: list[float],
    y: str = "y",
    running: str = "running",
    cutoff: float = 0.0,
    kernel: str = "triangular",
) -> pd.DataFrame:
    """Local linear tau-hat across bandwidths."""
    rows = [local_linear_rd(df, y=y, running=running, cutoff=cutoff, bandwidth=h, kernel=kernel) for h in bandwidths]
    return pd.DataFrame(rows)


def naive_far_difference(
    df: pd.DataFrame,
    y: str = "y",
    running: str = "running",
    cutoff: float = 0.0,
    margin: float = 1.0,
) -> dict[str, float]:
    """Mean difference using only observations at least `margin` from the cutoff.

    This is an extrapolation diagnostic, not an RD estimator. In the laboratory
    DGP it mixes the jump with the quadratic running-variable path.
    """
    left = df[df[running] <= cutoff - margin]
    right = df[df[running] >= cutoff + margin]
    if left.empty or right.empty:
        return {"far_diff": float("nan"), "n_left": float(len(left)), "n_right": float(len(right))}
    return {
        "far_diff": float(right[y].mean() - left[y].mean()),
        "n_left": float(len(left)),
        "n_right": float(len(right)),
        "margin": float(margin),
    }


def global_polynomial_jump(
    df: pd.DataFrame,
    y: str = "y",
    running: str = "running",
    cutoff: float = 0.0,
    degree: int = 3,
) -> dict[str, float]:
    """Global polynomial with a jump at the cutoff (still not local).

    Included to show that using the whole support is a different estimand from
    a local linear fit. It is not recommended as the primary RD report.
    """
    x = df[running].to_numpy(dtype=float) - cutoff
    d = (df[running].to_numpy(dtype=float) >= cutoff).astype(float)
    yv = df[y].to_numpy(dtype=float)
    cols = [np.ones_like(x), d]
    for p in range(1, degree + 1):
        cols.append(x ** p)
        cols.append(d * (x ** p))
    xmat = np.column_stack(cols)
    res = sm.OLS(yv, xmat).fit()
    return {"tau_global": float(res.params[1]), "se": float(res.bse[1]), "degree": float(degree)}


def rd_bin_means(
    df: pd.DataFrame,
    y: str = "y",
    running: str = "running",
    cutoff: float = 0.0,
    n_bins: int = 24,
) -> pd.DataFrame:
    """Binned means of the outcome on each side of the cutoff (graphical RD)."""
    work = df.copy()
    work["_side"] = np.where(work[running] >= cutoff, "right", "left")
    pieces = []
    for side, sub in work.groupby("_side"):
        if sub.empty:
            continue
        cats, edges = pd.cut(sub[running], bins=n_bins // 2, retbins=True)
        grouped = sub.groupby(cats, observed=False)[y].agg(["mean", "count"])
        grouped = grouped.reset_index()
        grouped["mid"] = grouped.iloc[:, 0].apply(lambda iv: iv.mid if hasattr(iv, "mid") else np.nan)
        grouped["side"] = side
        pieces.append(grouped[["mid", "mean", "count", "side"]])
    if not pieces:
        return pd.DataFrame(columns=["mid", "mean", "count", "side"])
    return pd.concat(pieces, ignore_index=True)
