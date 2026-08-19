"""Instrumental variables: valid, weak, and invalid designs.

Copyright 2026 Dr. Pavanam Thomas
"""

from econci import dgp, iv

import numpy as np


def test_valid_iv_closer_to_truth_than_ols_under_endogeneity() -> None:
    df = dgp.simulate_iv(n=900, beta=1.0, pi=1.0, endog_corr=0.75, exclusion_violation=0.0, seed=42)
    est = iv.two_sls(df)
    ols_slope = iv.ols_endog_slope(df)
    assert abs(float(est["beta_endog"]) - 1.0) < abs(ols_slope - 1.0)
    assert abs(float(est["beta_endog"]) - 1.0) < 0.15
    assert abs(ols_slope - 1.0) > 0.2
    assert float(est["first_stage_f"]) > 10.0


def test_weak_iv_has_low_first_stage_f() -> None:
    df = dgp.simulate_iv(n=500, beta=1.0, pi=0.06, endog_corr=0.7, exclusion_violation=0.0, seed=42)
    est = iv.two_sls(df)
    assert float(est["first_stage_f"]) < 10.0


def test_invalid_iv_is_biased() -> None:
    df = dgp.simulate_iv(n=900, beta=1.0, pi=1.0, endog_corr=0.2, exclusion_violation=1.2, seed=42)
    est = iv.two_sls(df)
    assert abs(float(est["beta_endog"]) - 1.0) > 0.4


def test_first_stage_strength_is_not_validity() -> None:
    text = iv.assumptions_text().lower()
    assert "does not establish" in text
    invalid = dgp.simulate_iv(n=700, pi=1.2, exclusion_violation=1.0, endog_corr=0.1, seed=42)
    est = iv.two_sls(invalid)
    assert float(est["first_stage_f"]) > 10.0
    assert abs(float(est["beta_endog"]) - 1.0) > 0.3


def test_anderson_rubin_covers_truth_when_iv_is_valid() -> None:
    df = dgp.simulate_iv(n=900, beta=1.0, pi=1.0, endog_corr=0.75, exclusion_violation=0.0, seed=42)
    p_at_truth = iv.anderson_rubin_pvalue(df, 1.0)
    assert p_at_truth > 0.05
    grid = np.linspace(-0.5, 2.5, 61)
    interval = iv.anderson_rubin_interval(df, grid, alpha=0.05)
    assert interval["n_accepted"] > 0
    assert interval["low"] <= 1.0 <= interval["high"]


def test_anderson_rubin_interval_is_wider_than_wald_when_iv_is_weak() -> None:
    df = dgp.simulate_iv(n=500, beta=1.0, pi=0.06, endog_corr=0.7, exclusion_violation=0.0, seed=42)
    est = iv.two_sls(df)
    assert float(est["first_stage_f"]) < 10.0
    wald_lo, wald_hi = iv.wald_interval_2sls(est)
    grid = np.linspace(-8.0, 10.0, 181)
    ar = iv.anderson_rubin_interval(df, grid, alpha=0.05)
    ar_width = ar["high"] - ar["low"]
    wald_width = wald_hi - wald_lo
    assert np.isfinite(ar_width)
    assert ar_width > wald_width
