"""OLS recovery and misspecification diagnostics.

Copyright 2026 Dr. Pavanam Thomas
"""

from econci import dgp, ols


def test_ols_recovers_known_slope() -> None:
    df = dgp.simulate_ols_linear(n=1600, beta1=2.0, beta2=-0.5, sigma=1.0, seed=42)
    fit = ols.fit_ols(df, "y ~ x1 + x2", cov_type="HC1")
    assert abs(float(fit.params["x1"]) - 2.0) < 0.12
    assert abs(float(fit.params["x2"]) - (-0.5)) < 0.12


def test_misspecification_diagnostic_fires_on_omitted_quadratic() -> None:
    df = dgp.simulate_ols_omitted_quadratic(n=900, beta_x2=1.4, seed=42)
    diag = ols.added_quadratic_test(df, y="y", x="x")
    assert diag["pvalue"] < 1e-6
    linear = ols.fit_ols(df, "y ~ x")
    reset = ols.ramsey_reset(linear, power=3)
    assert reset["pvalue"] < 1e-4


def test_quadratic_specification_recovers_known_curvature() -> None:
    """Correction of the omitted-quadratic design: fit the term that was missing."""
    df = dgp.simulate_ols_omitted_quadratic(n=1200, beta_x=0.4, beta_x2=1.4, seed=42)
    corrected = ols.fit_ols(df, "y ~ x + x2")
    assert abs(float(corrected.params["x2"]) - 1.4) < 0.12
    assert abs(float(corrected.params["x"]) - 0.4) < 0.12
    linear = ols.fit_ols(df, "y ~ x")
    reset = ols.ramsey_reset(linear, power=3)
    assert reset["pvalue"] < 1e-4


def test_robust_se_and_influence_run() -> None:
    df = dgp.simulate_ols_heteroskedastic(n=500, seed=42)
    hc3 = ols.fit_ols(df, "y ~ x", cov_type="HC3")
    classical = ols.fit_ols(df, "y ~ x", cov_type="nonrobust")
    assert float(hc3.bse["x"]) != float(classical.bse["x"])
    bp = ols.breusch_pagan_test(classical)
    assert bp["lm_pvalue"] < 0.05
    infl = ols.influence_table(classical)
    assert "cooks_distance" in infl.columns
    vif = ols.vif_table(dgp.simulate_ols_linear(n=200, seed=42), ["x1", "x2"])
    assert (vif["vif"] > 0).all()
