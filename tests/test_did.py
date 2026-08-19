"""Difference-in-differences: 2x2 recovery, parallel-trend violations, event-study coding.

Copyright 2026 Dr. Pavanam Thomas
"""

import numpy as np

from econci import dgp, did


def test_2x2_recovers_known_att_under_parallel_trends() -> None:
    df = dgp.simulate_did_2x2(n_treat=180, n_control=180, att=2.0, parallel_trends=True, sigma=1.0, seed=42)
    fit = did.fit_did_2x2(df)
    means = did.did_att_from_means(df)
    assert abs(float(fit.params["_did"]) - 2.0) < 0.25
    assert abs(means - 2.0) < 0.25
    assert abs(float(fit.params["_did"]) - means) < 1e-8


def test_parallel_trends_violation_biases_did() -> None:
    df = dgp.simulate_did_2x2(
        n_treat=180,
        n_control=180,
        att=2.0,
        parallel_trends=False,
        trend_gap=1.3,
        sigma=0.7,
        seed=42,
    )
    fit = did.fit_did_2x2(df)
    assert abs(float(fit.params["_did"]) - 2.0) > 0.6


def test_event_study_no_future_treatment_leakage() -> None:
    df = dgp.simulate_event_study(n_treat=40, n_control=40, n_periods=10, treat_period=6, att=2.0, seed=42)
    design = did.build_event_study_design(df, k_min=-4, k_max=3, omit=-1)
    pre = design[(design["ever_treated"] == 1) & (design["period"] < design["g"])]
    assert len(pre) > 0
    assert (pre["treated"] == 0).all()
    post = design[(design["ever_treated"] == 1) & (design["period"] >= design["g"])]
    assert (post["treated"] == 1).all()
    never = design[design["ever_treated"] == 0]
    assert (never["treated"] == 0).all()
    dummy_cols = [c for c in design.columns if c.startswith("rel_")]
    assert (never[dummy_cols].to_numpy() == 0).all()


def test_event_study_pretrends_near_zero_when_parallel() -> None:
    df = dgp.simulate_event_study(
        n_treat=90,
        n_control=90,
        n_periods=10,
        treat_period=6,
        att=2.0,
        parallel_trends=True,
        sigma=0.8,
        seed=42,
    )
    design = did.build_event_study_design(df)
    names = list(design.attrs["event_dummy_names"])
    _, coefs = did.fit_event_study(design, dummy_names=names)
    leads = coefs[coefs["k"] < 0]
    assert leads["coef"].abs().mean() < 0.35


def test_educational_group_time_not_cs_label() -> None:
    assert "not" in did.group_time_att_educational.__doc__.lower()
    assert "callaway" in did.group_time_att_educational.__doc__.lower()


def test_staggered_twfe_can_miss_cohort_atts() -> None:
    df = dgp.simulate_staggered_heterogeneous(
        n_early=60,
        n_late=60,
        n_never=60,
        att_early=4.0,
        att_late=-1.5,
        seed=42,
    )
    twfe = did.fit_twfe_static(df)
    att_gt = did.group_time_att_educational(df)
    agg = did.aggregate_group_time_att(att_gt)
    twfe_est = float(twfe.params["treated"])
    # Known cohort ATTs have opposite signs; TWFE need not equal either.
    assert abs(twfe_est - 4.0) > 0.8
    assert abs(twfe_est - (-1.5)) > 0.8
    assert np.isfinite(agg["att_simple"])
