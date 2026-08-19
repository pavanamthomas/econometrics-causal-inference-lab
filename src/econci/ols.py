"""Ordinary least squares, residual diagnostics, and influence measures.

Estimation uses statsmodels. Robust covariance choices (HC1, HC3) and
heteroskedasticity tests are available because conventional (nonrobust)
standard errors are not automatic.

A correctly specified linear DGP is provided for recovery checks. A
deliberately misspecified specification (linear fit when the mean is
quadratic) is used so that diagnostics can fail in a known way.

Copyright 2026 Dr. Pavanam Thomas
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor


def fit_ols(
    df: pd.DataFrame,
    formula: str,
    cov_type: str = "nonrobust",
) -> RegressionResultsWrapper:
    """Fit OLS by formula. cov_type may be nonrobust, HC1, or HC3."""
    model = smf.ols(formula, data=df)
    if cov_type == "nonrobust":
        return model.fit()
    return model.fit(cov_type=cov_type)


def residual_diagnostics(results: RegressionResultsWrapper) -> pd.DataFrame:
    """Residuals, fitted values, and basic residual moments."""
    resid = np.asarray(results.resid)
    fitted = np.asarray(results.fittedvalues)
    out = pd.DataFrame({"fitted": fitted, "residual": resid})
    out.attrs["mean_residual"] = float(np.mean(resid))
    out.attrs["sd_residual"] = float(np.std(resid, ddof=1))
    out.attrs["corr_fitted_residual"] = float(np.corrcoef(fitted, resid)[0, 1])
    return out


def breusch_pagan_test(results: RegressionResultsWrapper) -> dict[str, float]:
    """Breusch-Pagan heteroskedasticity test on the fitted OLS model."""
    lm, lm_pval, fval, f_pval = het_breuschpagan(results.resid, results.model.exog)
    return {
        "lm": float(lm),
        "lm_pvalue": float(lm_pval),
        "f": float(fval),
        "f_pvalue": float(f_pval),
    }


def white_test(results: RegressionResultsWrapper) -> dict[str, float]:
    """White heteroskedasticity test (no cross terms beyond statsmodels default)."""
    lm, lm_pval, fval, f_pval = het_white(results.resid, results.model.exog)
    return {
        "lm": float(lm),
        "lm_pvalue": float(lm_pval),
        "f": float(fval),
        "f_pvalue": float(f_pval),
    }


def vif_table(df: pd.DataFrame, exog_cols: list[str]) -> pd.DataFrame:
    """Variance inflation factors for the listed exogenous columns (plus constant)."""
    x = sm.add_constant(df[exog_cols], has_constant="add")
    rows = []
    for i, name in enumerate(x.columns):
        if name == "const":
            continue
        rows.append({"variable": name, "vif": float(variance_inflation_factor(x.values, i))})
    return pd.DataFrame(rows)


def influence_table(results: RegressionResultsWrapper) -> pd.DataFrame:
    """Leverage, Cook's distance, and studentized residuals."""
    infl = OLSInfluence(results)
    cooks = np.asarray(infl.cooks_distance[0])
    leverage = np.asarray(infl.hat_matrix_diag)
    student = np.asarray(infl.resid_studentized_internal)
    n = len(cooks)
    return pd.DataFrame(
        {
            "leverage": leverage,
            "cooks_distance": cooks,
            "studentized_residual": student,
            "flag_cooks_4n": cooks > (4.0 / n),
        }
    )


def ramsey_reset(results: RegressionResultsWrapper, power: int = 3) -> dict[str, float]:
    """Ramsey RESET: add powers 2..power of fitted values and test them jointly.

    This is a specification diagnostic, not a substitute for an identifying
    argument. Rejection indicates that the conditional mean is poorly captured
    by the current linear index, not that a particular omitted variable is
    the cause.
    """
    if power < 2:
        raise ValueError("power must be at least 2")
    y = np.asarray(results.model.endog, dtype=float)
    x = np.asarray(results.model.exog, dtype=float)
    yhat = np.asarray(results.fittedvalues, dtype=float)
    extras = np.column_stack([yhat ** p for p in range(2, power + 1)])
    x_aug = np.column_stack([x, extras])
    unrestricted = sm.OLS(y, x_aug).fit()
    ssr_r = float(results.ssr)
    ssr_u = float(unrestricted.ssr)
    q = extras.shape[1]
    n, k_u = x_aug.shape
    f_stat = ((ssr_r - ssr_u) / q) / (ssr_u / (n - k_u))
    pvalue = float(stats.f.sf(f_stat, q, n - k_u))
    return {"f_stat": float(f_stat), "pvalue": pvalue, "df_num": float(q), "df_den": float(n - k_u)}


def added_quadratic_test(df: pd.DataFrame, y: str, x: str) -> dict[str, float]:
    """Nested F-test of x^2 in y ~ x + x^2 versus y ~ x.

    Used for the omitted-quadratic laboratory design, where the extra term is
    the object known to be missing from the misspecified linear fit.
    """
    restricted = smf.ols(f"{y} ~ {x}", data=df).fit()
    work = df.copy()
    work["_x2"] = work[x] ** 2
    unrestricted = smf.ols(f"{y} ~ {x} + _x2", data=work).fit()
    f_stat, pvalue, df_diff = unrestricted.compare_f_test(restricted)[:3]
    return {
        "f_stat": float(f_stat),
        "pvalue": float(pvalue),
        "df_diff": float(df_diff),
        "quadratic_coef": float(unrestricted.params["_x2"]),
    }


def coefficient_table(results: RegressionResultsWrapper) -> pd.DataFrame:
    """Point estimates, standard errors, t-statistics, and p-values."""
    ci = results.conf_int()
    return pd.DataFrame(
        {
            "variable": results.params.index,
            "coef": results.params.values,
            "se": results.bse.values,
            "t": results.tvalues.values,
            "pvalue": results.pvalues.values,
            "ci_low": ci.iloc[:, 0].values,
            "ci_high": ci.iloc[:, 1].values,
        }
    )
