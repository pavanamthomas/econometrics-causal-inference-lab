"""Instrumental variables: first stage, 2SLS, and invalid/weak designs.

Relevance (first-stage F) is necessary for precision and for the usual
normal approximation. It does not establish exclusion, instrument
exogeneity, or monotonicity. Those are identifying assumptions, not test
statistics.

Under a binary instrument and binary treatment, 2SLS estimates a LATE for
compliers when exclusion, independence, and monotonicity hold. The linear
homogeneous DGP used in this laboratory makes 2SLS consistent for the
common slope; that is a teaching simplification, not a claim that LATE and
ATE coincide in applications.

Copyright 2026 Dr. Pavanam Thomas
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def first_stage(
    df: pd.DataFrame,
    endog: str = "x",
    instrument: str = "z",
    exog: list[str] | None = None,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """OLS first stage of the endogenous regressor on the instrument and exog."""
    cols = [instrument, *(exog or [])]
    x = sm.add_constant(df[cols], has_constant="add")
    return sm.OLS(df[endog].to_numpy(dtype=float), x).fit()


def first_stage_f(results: sm.regression.linear_model.RegressionResultsWrapper, instrument: str = "z") -> dict[str, float]:
    """Partial F on the excluded instrument in a just-identified first stage."""
    hyp = f"{instrument} = 0"
    w = results.f_test(hyp)
    f_stat = float(np.asarray(w.fvalue).squeeze())
    pvalue = float(np.asarray(w.pvalue).squeeze())
    return {"f_stat": f_stat, "pvalue": pvalue, "t_stat": float(results.tvalues[instrument])}


def two_sls(
    df: pd.DataFrame,
    y: str = "y",
    endog: str = "x",
    instrument: str = "z",
    exog: list[str] | None = None,
) -> dict[str, float | pd.DataFrame]:
    """Just-identified 2SLS with homoskedastic textbook standard errors.

    Residuals are formed from the original endogenous regressor, not from the
    fitted first-stage values. First-stage F is returned alongside the second
    stage so that relevance can be inspected separately from the point estimate.
    """
    extra = list(exog or [])
    z_cols = extra + [instrument]
    x_cols = extra + [endog]
    yv = df[y].to_numpy(dtype=float)
    x = sm.add_constant(df[x_cols], has_constant="add").to_numpy(dtype=float)
    z = sm.add_constant(df[z_cols], has_constant="add").to_numpy(dtype=float)
    names = ["const", *x_cols]

    p_z = z @ np.linalg.inv(z.T @ z) @ z.T
    xtpx = x.T @ p_z @ x
    beta = np.linalg.solve(xtpx, x.T @ p_z @ yv)
    resid = yv - x @ beta
    n, k = x.shape
    sigma2 = float(resid @ resid / (n - k))
    vcov = sigma2 * np.linalg.inv(xtpx)
    se = np.sqrt(np.diag(vcov))
    tstats = beta / se
    pvalues = 2.0 * stats.t.sf(np.abs(tstats), df=n - k)

    fs = first_stage(df, endog=endog, instrument=instrument, exog=extra)
    fs_f = first_stage_f(fs, instrument=instrument)

    table = pd.DataFrame(
        {
            "variable": names,
            "coef": beta,
            "se": se,
            "t": tstats,
            "pvalue": pvalues,
        }
    )
    endog_row = table.loc[table["variable"] == endog].iloc[0]
    ols = sm.OLS(yv, sm.add_constant(df[x_cols], has_constant="add")).fit()
    return {
        "table": table,
        "beta_endog": float(endog_row["coef"]),
        "se_endog": float(endog_row["se"]),
        "ols_endog": float(ols.params.iloc[-1] if hasattr(ols.params, "iloc") else np.asarray(ols.params)[-1]),
        "first_stage_f": float(fs_f["f_stat"]),
        "first_stage_f_pvalue": float(fs_f["pvalue"]),
        "n": float(n),
    }


def ols_endog_slope(df: pd.DataFrame, y: str = "y", endog: str = "x") -> float:
    """OLS slope on the endogenous regressor (inconsistent under endogeneity)."""
    x = sm.add_constant(df[[endog]], has_constant="add")
    res = sm.OLS(df[y].to_numpy(dtype=float), x).fit()
    return float(res.params.iloc[-1] if hasattr(res.params, "iloc") else np.asarray(res.params)[-1])


def assumptions_text() -> str:
    """Identifying language for IV, kept next to the estimator on purpose."""
    return (
        "Relevance: the instrument must shift the endogenous regressor "
        "(first-stage F is a diagnostic for this, with the usual weak-IV caveats). "
        "Exogeneity and exclusion: the instrument must be independent of the "
        "structural error and must affect the outcome only through the endogenous "
        "regressor. First-stage strength does not establish either condition. "
        "Monotonicity: in a binary-instrument / binary-treatment design, no defiers. "
        "LATE: under those conditions 2SLS estimates the average effect for compliers, "
        "which need not equal the ATE. The linear homogeneous DGP in this lab is a "
        "simplification in which that distinction collapses."
    )
