"""Simulated data-generating processes for the econometrics lab.

Every function returns a pandas DataFrame whose columns are documented in the
docstring. Parameters that define the causal or structural object of interest
(slopes, ATT, LATE-relevant first stage, RD jump) are arguments, so tests can
compare estimators to known truth.

Samples are simulated. They are not observational microdata.

Copyright 2026 Dr. Pavanam Thomas
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_SEED = 42


def make_rng(seed: int = DEFAULT_SEED) -> np.random.Generator:
    """Return a NumPy Generator. Callers should pass an explicit seed."""
    return np.random.default_rng(seed)


def simulate_ols_linear(
    n: int = 800,
    beta0: float = 1.0,
    beta1: float = 2.0,
    beta2: float = -0.5,
    sigma: float = 1.0,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Correctly specified linear DGP: y = beta0 + beta1 x1 + beta2 x2 + e."""
    rng = make_rng(seed)
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    e = rng.normal(0.0, sigma, n)
    y = beta0 + beta1 * x1 + beta2 * x2 + e
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def simulate_ols_omitted_quadratic(
    n: int = 800,
    beta0: float = 1.0,
    beta_x: float = 0.4,
    beta_x2: float = 1.4,
    sigma: float = 0.7,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Nonlinear DGP: y = beta0 + beta_x x + beta_x2 x^2 + e.

    A linear projection of y on x is misspecified. Residual diagnostics and an
    added-quadratic test are expected to detect the omitted curvature.
    """
    rng = make_rng(seed)
    x = rng.uniform(-2.0, 2.0, n)
    e = rng.normal(0.0, sigma, n)
    y = beta0 + beta_x * x + beta_x2 * (x ** 2) + e
    return pd.DataFrame({"y": y, "x": x, "x2": x ** 2})


def simulate_ols_heteroskedastic(
    n: int = 800,
    beta0: float = 1.0,
    beta1: float = 1.5,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Linear mean with variance increasing in |x| (for HC and BP illustrations)."""
    rng = make_rng(seed)
    x = rng.normal(0.0, 1.0, n)
    e = rng.normal(0.0, 0.4 + 1.2 * np.abs(x), n)
    y = beta0 + beta1 * x + e
    return pd.DataFrame({"y": y, "x": x})


def simulate_binary_choice(
    n: int = 900,
    beta0: float = -0.4,
    beta1: float = 1.1,
    beta2: float = -0.7,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Latent-index binary choice with logistic errors (logit-aligned DGP).

    y = 1{beta0 + beta1 x1 + beta2 x2 + nu > 0}, nu ~ logistic.
    x1 and x2 are independent of nu. Average marginal effects on P(y=1) are
    well-defined ceteris paribus effects in this DGP; classification metrics
    (AUC, confusion matrix) are not identifying arguments.
    """
    rng = make_rng(seed)
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    nu = rng.logistic(0.0, 1.0, n)
    latent = beta0 + beta1 * x1 + beta2 * x2 + nu
    y = (latent > 0.0).astype(int)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "latent": latent})


def simulate_panel(
    n_entities: int = 60,
    n_periods: int = 8,
    beta: float = 1.5,
    fe_x_loading: float = 0.7,
    sigma_a: float = 1.0,
    sigma_e: float = 1.0,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Balanced panel with entity effects correlated with x.

    y_it = a_i + beta x_it + e_it,  x_it = fe_x_loading * a_i + u_it.

    Entity fixed effects recover beta from within-unit variation. Pooled OLS
    is inconsistent for beta when fe_x_loading != 0. This is a panel (same
    entities over time), not a repeated cross-section.
    """
    rng = make_rng(seed)
    entities = np.arange(n_entities)
    periods = np.arange(n_periods)
    a = rng.normal(0.0, sigma_a, n_entities)
    rows: list[dict[str, float | int]] = []
    for i, a_i in zip(entities, a):
        for t in periods:
            u = rng.normal(0.0, 1.0)
            x = fe_x_loading * a_i + u
            e = rng.normal(0.0, sigma_e)
            y = a_i + beta * x + e
            rows.append({"entity": int(i), "period": int(t), "y": y, "x": x, "a": float(a_i)})
    return pd.DataFrame(rows)


def simulate_few_treated_clusters(
    n_clusters: int = 8,
    n_per_cluster: int = 16,
    n_treated_clusters: int = 2,
    beta: float = 1.0,
    sigma_a: float = 2.5,
    sigma_e: float = 1.0,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Cluster-level treatment with a small number of treated clusters.

    y_ig = a_g + beta * d_g + e_ig. Treatment is constant inside a cluster.
    Conventional cluster-robust Wald intervals for beta can undercover when
    few clusters are treated. The laboratory uses this DGP to compare that
    Wald interval with a wild cluster bootstrap, not as a real programme.
    """
    if not 1 <= n_treated_clusters < n_clusters:
        raise ValueError("need at least one treated and one untreated cluster")
    rng = make_rng(seed)
    rows: list[dict[str, float | int]] = []
    for g in range(n_clusters):
        treated = int(g < n_treated_clusters)
        a_g = float(rng.normal(0.0, sigma_a))
        for _ in range(n_per_cluster):
            e = float(rng.normal(0.0, sigma_e))
            y = a_g + beta * treated + e
            rows.append(
                {
                    "entity": int(g),
                    "treated": treated,
                    "y": y,
                    "n_treated_clusters": int(n_treated_clusters),
                    "true_beta": float(beta),
                }
            )
    return pd.DataFrame(rows)


def simulate_did_2x2(
    n_treat: int = 120,
    n_control: int = 120,
    att: float = 2.0,
    parallel_trends: bool = True,
    trend_gap: float = 1.2,
    sigma: float = 1.0,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Two groups, two periods. Treatment occurs for the treated group in period 1.

    Untreated potential outcomes:
        Y(0)_it = a_i + b_t + 1{treated} * extra_trend * period + e_it
    extra_trend is 0 if parallel_trends is True, otherwise trend_gap.

    Observed outcome adds att for treated units in the post period.
    Long format with columns: unit, period, treated_group, post, y.
    """
    rng = make_rng(seed)
    extra = 0.0 if parallel_trends else trend_gap
    n = n_treat + n_control
    treated_group = np.array([1] * n_treat + [0] * n_control)
    a = rng.normal(0.0, 1.0, n)
    b = {0: 0.0, 1: 0.8}
    rows: list[dict[str, float | int]] = []
    for i in range(n):
        for period in (0, 1):
            e = rng.normal(0.0, sigma)
            y0 = a[i] + b[period] + extra * treated_group[i] * period + e
            treated_now = int(treated_group[i] == 1 and period == 1)
            y = y0 + att * treated_now
            rows.append(
                {
                    "unit": i,
                    "period": period,
                    "treated_group": int(treated_group[i]),
                    "post": period,
                    "y": y,
                    "true_att": att,
                }
            )
    return pd.DataFrame(rows)


def simulate_did_two_controls(
    n_treat: int = 80,
    n_control_a: int = 80,
    n_control_b: int = 80,
    att: float = 2.0,
    sigma: float = 1.0,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """2x2 DiD with two never-treated comparison groups (A and B).

    Parallel trends hold for both comparison groups by construction.
    Column `control_pool` is 'A', 'B', or 'treated'.
    """
    rng = make_rng(seed)
    labels = (
        ["treated"] * n_treat
        + ["A"] * n_control_a
        + ["B"] * n_control_b
    )
    n = len(labels)
    a = rng.normal(0.0, 1.0, n)
    b = {0: 0.0, 1: 0.6}
    rows: list[dict[str, float | int | str]] = []
    for i, lab in enumerate(labels):
        treated_group = int(lab == "treated")
        for period in (0, 1):
            e = rng.normal(0.0, sigma)
            y0 = a[i] + b[period] + e
            y = y0 + att * treated_group * period
            rows.append(
                {
                    "unit": i,
                    "period": period,
                    "treated_group": treated_group,
                    "post": period,
                    "control_pool": lab,
                    "y": y,
                }
            )
    return pd.DataFrame(rows)


def simulate_event_study(
    n_treat: int = 70,
    n_control: int = 70,
    n_periods: int = 10,
    treat_period: int = 6,
    att: float = 2.0,
    parallel_trends: bool = True,
    pretrend_slope: float = 0.35,
    sigma: float = 1.0,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Common-timing event-study DGP with a never-treated group.

    Periods are 0 .. n_periods-1. Treated units adopt at `treat_period`.
    If parallel_trends is True, untreated paths differ only by unit and time
    effects. If False, treated units have an extra linear drift of
    pretrend_slope per period in Y(0).

    Static treatment dummy is 1 only for t >= G among ever-treated units.
    """
    rng = make_rng(seed)
    n = n_treat + n_control
    ever = np.array([1] * n_treat + [0] * n_control)
    g = np.where(ever == 1, treat_period, np.nan)
    a = rng.normal(0.0, 1.0, n)
    time_effects = rng.normal(0.0, 0.15, n_periods)
    rows: list[dict[str, float | int]] = []
    for i in range(n):
        for t in range(n_periods):
            extra = 0.0
            if not parallel_trends and ever[i] == 1:
                extra = pretrend_slope * t
            e = rng.normal(0.0, sigma)
            treated = int(ever[i] == 1 and t >= treat_period)
            y = a[i] + time_effects[t] + extra + att * treated + e
            gi = treat_period if ever[i] == 1 else np.nan
            rows.append(
                {
                    "unit": i,
                    "period": t,
                    "ever_treated": int(ever[i]),
                    "g": gi,
                    "treated": treated,
                    "y": y,
                }
            )
    return pd.DataFrame(rows)


def simulate_staggered_heterogeneous(
    n_early: int = 50,
    n_late: int = 50,
    n_never: int = 50,
    n_periods: int = 12,
    g_early: int = 4,
    g_late: int = 8,
    att_early: float = 4.0,
    att_late: float = -1.5,
    sigma: float = 0.8,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Staggered adoption with cohort-specific constant treatment effects.

    Conventional TWFE (y on unit FE, time FE, and a static treated dummy) need
    not recover a convex combination of the cohort ATTs. This DGP is intended
    to make that distinction visible. It is not a claim about a real policy.
    """
    rng = make_rng(seed)
    cohorts = (
        [(g_early, att_early)] * n_early
        + [(g_late, att_late)] * n_late
        + [(np.nan, 0.0)] * n_never
    )
    n = len(cohorts)
    a = rng.normal(0.0, 1.0, n)
    time_effects = np.linspace(0.0, 0.5, n_periods)
    rows: list[dict[str, float | int]] = []
    for i, (g, tau) in enumerate(cohorts):
        ever = int(not np.isnan(g))
        for t in range(n_periods):
            treated = int(ever == 1 and t >= g)
            e = rng.normal(0.0, sigma)
            y = a[i] + time_effects[t] + tau * treated + e
            rows.append(
                {
                    "unit": i,
                    "period": t,
                    "g": g if ever else np.nan,
                    "ever_treated": ever,
                    "treated": treated,
                    "true_cohort_att": tau if ever else np.nan,
                    "y": y,
                }
            )
    return pd.DataFrame(rows)


def simulate_iv(
    n: int = 600,
    beta: float = 1.0,
    pi: float = 1.0,
    endog_corr: float = 0.7,
    exclusion_violation: float = 0.0,
    sigma_z: float = 1.0,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Linear IV DGP with a single endogenous regressor and a single instrument.

    z ~ N(0, sigma_z^2), independent of structural shocks unless used elsewhere.
    (u, e) jointly normal with Corr(u, e) = endog_corr.
    x = pi z + u
    y = beta x + exclusion_violation * z + e

    Designs:
      valid:   pi reasonably large, exclusion_violation = 0
      weak:    pi near 0, exclusion_violation = 0
      invalid: pi large, exclusion_violation != 0

    First-stage strength does not establish exclusion or exogeneity.
    Under validity, 2SLS is consistent for beta (here the ATE, because the
    design is linear and homogeneous). LATE language is reserved for binary
    instrument / binary treatment designs; see iv.assumptions_text().
    """
    rng = make_rng(seed)
    z = rng.normal(0.0, sigma_z, n)
    cov = np.array([[1.0, endog_corr], [endog_corr, 1.0]])
    shocks = rng.multivariate_normal(np.zeros(2), cov, n)
    u = shocks[:, 0]
    e = shocks[:, 1]
    x = pi * z + u
    y = beta * x + exclusion_violation * z + e
    return pd.DataFrame({"y": y, "x": x, "z": z})


def simulate_sharp_rd(
    n: int = 1200,
    cutoff: float = 0.0,
    tau: float = 2.0,
    x_low: float = -2.0,
    x_high: float = 2.0,
    sigma: float = 0.7,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Sharp RD: treatment is 1{running >= cutoff}.

    y = 1 + 0.9 running + 0.45 running^2 + tau * treated + e

    The jump at the cutoff is tau. Comparisons far from the cutoff mix the
    jump with the running-variable relationship and should not be read as
    the RD parameter.
    """
    rng = make_rng(seed)
    running = rng.uniform(x_low, x_high, n)
    treated = (running >= cutoff).astype(int)
    e = rng.normal(0.0, sigma, n)
    y = 1.0 + 0.9 * running + 0.45 * (running ** 2) + tau * treated + e
    return pd.DataFrame(
        {
            "y": y,
            "running": running,
            "treated": treated,
            "cutoff": cutoff,
            "true_tau": tau,
        }
    )


def simulate_selection_observables(
    n: int = 800,
    att: float = 1.5,
    beta_x: float = 1.2,
    selection: float = 1.4,
    sigma: float = 0.9,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Treatment selected on an observed covariate; unconfoundedness holds.

    x ~ N(0, 1)
    P(d=1 | x) = logistic(selection * x)
    y = att * d + beta_x * x + e

    Naive treated-control mean differences confound att with E[x|d=1]-E[x|d=0].
    IPW / matching on x is consistent for att in this DGP (overlap permitting).
    """
    rng = make_rng(seed)
    x = rng.normal(0.0, 1.0, n)
    z = rng.normal(0.0, 1.0, n)
    ps_true = 1.0 / (1.0 + np.exp(-selection * x))
    d = (rng.uniform(0.0, 1.0, n) < ps_true).astype(int)
    e = rng.normal(0.0, sigma, n)
    y = att * d + beta_x * x + e
    return pd.DataFrame(
        {
            "y": y,
            "d": d,
            "x": x,
            "z": z,
            "ps_true": ps_true,
            "true_att": att,
        }
    )
