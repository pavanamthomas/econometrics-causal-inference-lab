"""Panel: within-unit recovery versus pooled bias under correlated entity effects.

Copyright 2026 Dr. Pavanam Thomas
"""

from econci import dgp, panel


def test_entity_fe_recovers_within_slope() -> None:
    df = dgp.simulate_panel(n_entities=160, n_periods=10, beta=1.5, fe_x_loading=0.7, seed=42)
    fe = panel.fit_entity_fe(df, "y", ["x"], "entity")
    assert abs(float(fe.params["x"]) - 1.5) < 0.12


def test_pooled_biased_when_entity_effects_correlate_with_x() -> None:
    df = dgp.simulate_panel(n_entities=80, n_periods=8, beta=1.5, fe_x_loading=0.7, seed=42)
    pooled = panel.fit_pooled_ols(df, "y", ["x"], "entity")
    fe = panel.fit_entity_fe(df, "y", ["x"], "entity")
    assert abs(float(pooled.params["x"]) - 1.5) > abs(float(fe.params["x"]) - 1.5)
    assert abs(float(pooled.params["x"]) - 1.5) > 0.15


def test_random_effects_and_clustered_se() -> None:
    df = dgp.simulate_panel(n_entities=50, n_periods=6, beta=1.5, seed=42)
    re = panel.fit_random_effects(df, "y", ["x"], "entity")
    fe = panel.fit_entity_fe(df, "y", ["x"], "entity")
    assert float(re.params["x"]) != 0.0
    assert float(fe.bse["x"]) > 0.0
    haus = panel.hausman_fe_re(fe, re, ["x"])
    assert "hausman" in haus
