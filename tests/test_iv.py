"""Instrumental variables: valid, weak, and invalid designs.

Copyright 2026 Dr. Pavanam Thomas
"""

from econci import dgp, iv


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
