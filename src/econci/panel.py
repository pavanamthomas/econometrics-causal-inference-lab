"""Panel estimators: pooled OLS, entity fixed effects, and random effects.

A panel follows the same entities over time. A repeated cross-section draws
new units each period and does not support within-entity demeaning.

Entity fixed effects use only within-unit variation. Random effects is
feasible GLS (Swamy-Arora-style variance components) and is inconsistent for
the slope when entity effects are correlated with the regressor. Clustered
standard errors treat the entity as the sampling cluster.

Copyright 2026 Dr. Pavanam Thomas
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def _cluster_fit(y: pd.Series, x: pd.DataFrame, groups: pd.Series):
    return sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": groups})


def fit_pooled_ols(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    entity_col: str,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Pooled OLS with cluster-robust standard errors at the entity level."""
    y = df[y_col].astype(float)
    x = sm.add_constant(df[x_cols].astype(float), has_constant="add")
    return _cluster_fit(y, x, df[entity_col])


def fit_entity_fe(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    entity_col: str,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Within (entity demeaning) estimator with clustered standard errors.

    Time-invariant covariates are absorbed and cannot be identified.
    Interpretation is within-unit: how y moves with x after removing each
    entity's mean.
    """
    means = df.groupby(entity_col)[[y_col, *x_cols]].transform("mean")
    y = (df[y_col] - means[y_col]).astype(float)
    x = (df[x_cols] - means[x_cols]).astype(float)
    return _cluster_fit(y, x, df[entity_col])


def fit_random_effects(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    entity_col: str,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Feasible GLS random-effects estimator for a balanced panel.

    Residual variance is taken from the within regression. The between
    variance is taken from entity means. theta = 1 - sigma_e / sqrt(T sigma_a^2
    + sigma_e^2). Quasi-demeaned OLS is then clustered by entity.
    """
    n_entities = df[entity_col].nunique()
    t_bar = df.groupby(entity_col).size().mean()
    fe = fit_entity_fe(df, y_col, x_cols, entity_col)
    n = len(df)
    k = len(x_cols)
    sigma_e2 = float(np.sum(fe.resid ** 2) / (n - n_entities - k))
    entity_means = df.groupby(entity_col)[[y_col, *x_cols]].mean()
    yb = entity_means[y_col].to_numpy(dtype=float)
    xb = sm.add_constant(entity_means[x_cols], has_constant="add").to_numpy(dtype=float)
    between = sm.OLS(yb, xb).fit()
    sigma_between2 = float(np.sum(between.resid ** 2) / (n_entities - k - 1))
    sigma_a2 = max(sigma_between2 - sigma_e2 / t_bar, 0.0)
    theta = 1.0 - np.sqrt(sigma_e2) / np.sqrt(t_bar * sigma_a2 + sigma_e2)

    means = df.groupby(entity_col)[[y_col, *x_cols]].transform("mean")
    y = (df[y_col] - theta * means[y_col]).astype(float)
    x_d = (df[x_cols] - theta * means[x_cols]).astype(float)
    x = sm.add_constant(x_d, has_constant="add")
    results = _cluster_fit(y, x, df[entity_col])
    results.theta = float(theta)
    results.sigma_e2 = sigma_e2
    results.sigma_a2 = sigma_a2
    return results


def hausman_fe_re(
    fe_results: sm.regression.linear_model.RegressionResultsWrapper,
    re_results: sm.regression.linear_model.RegressionResultsWrapper,
    slope_names: list[str],
) -> dict[str, float]:
    """Hausman-style contrast of overlapping slope coefficients.

    H = (b_fe - b_re)' (V_fe - V_re)^{-1} (b_fe - b_re). If V_fe - V_re is
    not positive definite, the statistic is reported as NaN and the point
    estimates should be compared directly.
    """
    b_fe = fe_results.params
    b_re = re_results.params
    fe_vec = b_fe.reindex(slope_names).to_numpy(dtype=float)
    re_vec = b_re.reindex(slope_names).to_numpy(dtype=float)
    v_fe = fe_results.cov_params().reindex(index=slope_names, columns=slope_names).to_numpy(dtype=float)
    v_re = re_results.cov_params().reindex(index=slope_names, columns=slope_names).to_numpy(dtype=float)

    diff = fe_vec - re_vec
    vdiff = v_fe - v_re
    eigs = np.linalg.eigvalsh(vdiff)
    k = len(slope_names)
    if np.min(eigs) <= 1e-12:
        return {
            "hausman": float("nan"),
            "pvalue": float("nan"),
            "df": float(k),
            "min_eig_vdiff": float(np.min(eigs)),
            "fe_minus_re": float(diff[0]) if k == 1 else float("nan"),
        }
    inv = np.linalg.inv(vdiff)
    h = float(diff.T @ inv @ diff)
    pvalue = float(stats.chi2.sf(h, k))
    return {
        "hausman": h,
        "pvalue": pvalue,
        "df": float(k),
        "min_eig_vdiff": float(np.min(eigs)),
        "fe_minus_re": float(diff[0]) if k == 1 else float("nan"),
    }


def coef_row(label: str, results: sm.regression.linear_model.RegressionResultsWrapper, name: str) -> dict[str, float | str]:
    """Extract one named coefficient for comparison tables."""
    return {
        "estimator": label,
        "variable": name,
        "coef": float(results.params[name]),
        "se": float(results.bse[name]),
    }
