"""Matching and IPW: covariate balance under simulated selection on observables.

Copyright 2026 Dr. Pavanam Thomas
"""

from econci import dgp, matching


def test_ipw_reduces_covariate_imbalance() -> None:
    df = dgp.simulate_selection_observables(n=900, att=1.5, selection=1.5, seed=42)
    ps = matching.propensity_scores(df, treat="d", covariates=["x"])
    smd_before = matching.standardized_mean_difference(df["x"].to_numpy(), df["d"].to_numpy())
    weights = matching.att_ipw_weights(df, "d", ps)
    smd_after = matching.standardized_mean_difference(df["x"].to_numpy(), df["d"].to_numpy(), weights=weights)
    assert abs(smd_after) < abs(smd_before)
    assert abs(smd_before) > 0.25


def test_ipw_att_near_truth_under_unconfoundedness() -> None:
    df = dgp.simulate_selection_observables(n=1200, att=1.5, beta_x=1.2, selection=1.3, seed=42)
    ps = matching.propensity_scores(df, treat="d", covariates=["x"])
    est = matching.ipw_att(df, ps=ps)
    naive = df.loc[df["d"] == 1, "y"].mean() - df.loc[df["d"] == 0, "y"].mean()
    assert abs(est["att"] - 1.5) < abs(naive - 1.5)
    nn = matching.nearest_neighbor_att(df, ps=ps)
    assert abs(nn["att"] - 1.5) < abs(naive - 1.5)


def test_overlap_summary_and_limits_text() -> None:
    df = dgp.simulate_selection_observables(n=400, seed=42)
    ps = matching.propensity_scores(df, covariates=["x"])
    summary = matching.overlap_summary(ps, df["d"])
    assert "min_treated" in summary
    assert "unobserved" in matching.SENSITIVITY_LIMITS.lower()
