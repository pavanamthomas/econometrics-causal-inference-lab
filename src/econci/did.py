"""Difference-in-differences, event studies, and staggered-treatment illustrations.

Two-by-two DiD with a single treatment date is the textbook case. Event-study
leads are pre-treatment coefficients; they are diagnostics for parallel trends,
not a proof of it.

Conventional two-way fixed effects (TWFE) with a static treated dummy is not
automatically valid under staggered adoption and heterogeneous treatment
effects. The group-time ATT routine below is an educational aggregation of
simple 2x2 contrasts against the never-treated group. It is not a production
Callaway-Sant'Anna estimator: no not-yet-treated comparison, no doubly robust
score, and no CS inference.

Copyright 2026 Dr. Pavanam Thomas
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import RegressionResultsWrapper


def fit_did_2x2(
    df: pd.DataFrame,
    y: str = "y",
    treated_group: str = "treated_group",
    post: str = "post",
    cluster: str | None = "unit",
) -> RegressionResultsWrapper:
    """Regression representation of 2x2 DiD: y ~ G + Post + G:Post.

    The interaction is the DiD estimand. Clustered standard errors use `cluster`
    if provided.
    """
    work = df.copy()
    work["_did"] = work[treated_group] * work[post]
    formula = f"{y} ~ {treated_group} + {post} + _did"
    model = smf.ols(formula, data=work)
    if cluster is None:
        return model.fit()
    return model.fit(cov_type="cluster", cov_kwds={"groups": work[cluster]})


def did_att_from_means(df: pd.DataFrame) -> float:
    """Algebraic 2x2 DiD of cell means (equals the interaction in balanced 2x2)."""
    g1 = df["treated_group"] == 1
    g0 = df["treated_group"] == 0
    post = df["post"] == 1
    pre = df["post"] == 0
    dt = df.loc[g1 & post, "y"].mean() - df.loc[g1 & pre, "y"].mean()
    dc = df.loc[g0 & post, "y"].mean() - df.loc[g0 & pre, "y"].mean()
    return float(dt - dc)


def build_event_study_design(
    df: pd.DataFrame,
    k_min: int = -4,
    k_max: int = 3,
    omit: int = -1,
    unit: str = "unit",
    period: str = "period",
    g: str = "g",
    ever: str = "ever_treated",
) -> pd.DataFrame:
    """Construct event-time dummies and a static treatment indicator.

    Static `treated` equals 1 only when the unit is ever treated and
    calendar time is weakly after adoption (period >= g). Pre-adoption
    observations of eventually treated units are coded 0. That coding is
    the no-future-treatment-leakage property tested in this laboratory.

    Event-time dummies 1{period - g = k} are defined only for ever-treated
    units. Never-treated units have all event dummies equal to 0. The omitted
    category is `omit` (default -1). Endpoints are binned: k <= k_min and
    k >= k_max.
    """
    out = df.copy()
    g_num = pd.to_numeric(out[g], errors="coerce")
    rel = out[period].astype(float) - g_num
    ever_mask = out[ever].astype(int) == 1
    adopted = ever_mask & g_num.notna() & (out[period].astype(float) >= g_num)
    out["treated"] = adopted.astype(int)
    out["rel"] = rel

    ks = list(range(k_min, k_max + 1))
    if omit in ks:
        ks.remove(omit)
    dummy_names: list[str] = []
    for k in ks:
        name = _rel_name(k)
        dummy_names.append(name)
        if k == k_min:
            flag = ever_mask & (rel <= k)
        elif k == k_max:
            flag = ever_mask & (rel >= k)
        else:
            flag = ever_mask & (rel == k)
        out[name] = flag.fillna(False).astype(int)
    out.attrs["event_dummy_names"] = dummy_names
    out.attrs["omit"] = omit
    return out


def _rel_name(k: int) -> str:
    if k < 0:
        return f"rel_m{abs(k)}"
    return f"rel_p{k}"


def fit_event_study(
    df: pd.DataFrame,
    dummy_names: list[str] | None = None,
    y: str = "y",
    unit: str = "unit",
    period: str = "period",
) -> tuple[RegressionResultsWrapper, pd.DataFrame]:
    """Two-way FE event-study regression on previously built dummies.

    Returns the fitted model and a coefficient table for the relative-time
    dummies (pre-treatment leads and post-treatment lags).
    """
    work = df.copy()
    names = dummy_names if dummy_names is not None else list(work.attrs.get("event_dummy_names", []))
    if not names:
        names = [c for c in work.columns if c.startswith("rel_m") or c.startswith("rel_p")]
    y_star, x_star = two_way_demean(work, y, names, unit, period)
    results = sm.OLS(y_star, x_star).fit(
        cov_type="cluster",
        cov_kwds={"groups": work[unit]},
    )
    rows = []
    for name in names:
        k = _name_to_k(name)
        rows.append(
            {
                "term": name,
                "k": k,
                "coef": float(results.params[name]),
                "se": float(results.bse[name]),
                "pvalue": float(results.pvalues[name]),
            }
        )
    coefs = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    return results, coefs


def _name_to_k(name: str) -> int:
    if name.startswith("rel_m"):
        return -int(name.replace("rel_m", ""))
    if name.startswith("rel_p"):
        return int(name.replace("rel_p", ""))
    raise ValueError(f"unrecognized event dummy name: {name}")


def two_way_demean(
    df: pd.DataFrame,
    y: str,
    x_cols: list[str],
    unit: str,
    period: str,
) -> tuple[pd.Series, pd.DataFrame]:
    """Two-way within transformation: z_it - z_i. - z_.t + z_.."""
    cols = [y, *x_cols]
    unit_mean = df.groupby(unit)[cols].transform("mean")
    time_mean = df.groupby(period)[cols].transform("mean")
    grand = df[cols].mean()
    demeaned = df[cols] - unit_mean - time_mean + grand
    return demeaned[y], demeaned[x_cols]


def joint_pretrend_test(
    results: RegressionResultsWrapper,
    lead_names: list[str],
) -> dict[str, float]:
    """Wald/F test that all listed pre-treatment event-time coefficients are zero."""
    if not lead_names:
        return {"f_stat": float("nan"), "pvalue": float("nan"), "df_num": 0.0}
    hyp = " = ".join(lead_names) + " = 0"
    w = results.f_test(hyp)
    return {
        "f_stat": float(np.asarray(w.fvalue).squeeze()),
        "pvalue": float(np.asarray(w.pvalue).squeeze()),
        "df_num": float(len(lead_names)),
    }


def lead_names_from_dummies(dummy_names: list[str], omit: int = -1) -> list[str]:
    """Event-time dummy names with k < 0 (and k != omit, which is already absent)."""
    leads = []
    for name in dummy_names:
        k = _name_to_k(name)
        if k < 0:
            leads.append(name)
    return leads


def placebo_timing_2x2(
    df: pd.DataFrame,
    fake_post_from: int,
    period: str = "period",
    y: str = "y",
    treated_group: str = "ever_treated",
    cluster: str = "unit",
) -> RegressionResultsWrapper:
    """Placebo 2x2 using only periods strictly before true adoption.

    `fake_post_from` is the calendar period at which the placebo post dummy
    switches on. Rows with period >= min(g) among treated units are dropped
    so that no actual post-treatment outcome enters the placebo regression.
    """
    g_min = pd.to_numeric(df["g"], errors="coerce").min()
    pre = df[df[period] < g_min].copy()
    if pre.empty:
        raise ValueError("no strictly pre-adoption periods available for placebo timing")
    pre["treated_group"] = pre[treated_group].astype(int)
    pre["post"] = (pre[period] >= fake_post_from).astype(int)
    if pre["post"].nunique() < 2:
        raise ValueError("placebo split does not create both pre and post cells")
    return fit_did_2x2(pre, y=y, treated_group="treated_group", post="post", cluster=cluster)


def fit_twfe_static(
    df: pd.DataFrame,
    y: str = "y",
    treat: str = "treated",
    unit: str = "unit",
    period: str = "period",
) -> RegressionResultsWrapper:
    """Conventional TWFE: two-way demeaned OLS of y on the static treated dummy."""
    y_star, x_star = two_way_demean(df, y, [treat], unit, period)
    return sm.OLS(y_star, x_star).fit(
        cov_type="cluster",
        cov_kwds={"groups": df[unit]},
    )


def group_time_att_educational(
    df: pd.DataFrame,
    y: str = "y",
    unit: str = "unit",
    period: str = "period",
    g: str = "g",
) -> pd.DataFrame:
    """Educational group-time ATT using never-treated units as the comparison.

    For each treated cohort g and each t >= g:
        ATT(g,t) = (Ybar_{g,t} - Ybar_{g, g-1}) - (Ybar_{never,t} - Ybar_{never, g-1})

    This is a transparent simulation device. It is not the Callaway and
    Sant'Anna (2021) estimator: it does not use not-yet-treated units, does
    not estimate a doubly robust score, and does not provide CS standard errors.
    """
    work = df.copy()
    g_num = pd.to_numeric(work[g], errors="coerce")
    never = work[g_num.isna()]
    if never.empty:
        raise ValueError("educational group-time ATT requires a never-treated group")
    never_means = never.groupby(period)[y].mean()
    cohorts = sorted(g_num.dropna().unique().tolist())
    rows: list[dict[str, float]] = []
    for cohort in cohorts:
        g_int = int(cohort)
        base = g_int - 1
        if base not in never_means.index:
            continue
        treated_c = work[g_num == cohort]
        n_units = treated_c[unit].nunique()
        c_means = treated_c.groupby(period)[y].mean()
        if base not in c_means.index:
            continue
        post_times = [int(t) for t in sorted(c_means.index) if int(t) >= g_int]
        for t in post_times:
            if t not in never_means.index:
                continue
            att_gt = (c_means[t] - c_means[base]) - (never_means[t] - never_means[base])
            rows.append(
                {
                    "g": float(g_int),
                    "t": float(t),
                    "att_gt": float(att_gt),
                    "n_units": float(n_units),
                }
            )
    return pd.DataFrame(rows)


def aggregate_group_time_att(att_gt: pd.DataFrame) -> dict[str, float]:
    """Simple cohort-size weighted mean of educational ATT(g,t) cells."""
    if att_gt.empty:
        return {"att_simple": float("nan"), "n_cells": 0.0}
    w = att_gt["n_units"].to_numpy(dtype=float)
    a = att_gt["att_gt"].to_numpy(dtype=float)
    return {"att_simple": float(np.sum(w * a) / np.sum(w)), "n_cells": float(len(att_gt))}


def event_study_coef_table_with_omitted(
    coefs: pd.DataFrame,
    omit: int = -1,
) -> pd.DataFrame:
    """Insert the omitted event time as a zero for plotting."""
    row = pd.DataFrame([{"term": _rel_name(omit), "k": omit, "coef": 0.0, "se": 0.0, "pvalue": np.nan}])
    out = pd.concat([coefs, row], ignore_index=True).sort_values("k").reset_index(drop=True)
    return out
