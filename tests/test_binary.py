"""Binary models: prediction is not identification.

Copyright 2026 Dr. Pavanam Thomas
"""

from econci import binary, dgp


def test_logit_and_probit_fit_and_ame() -> None:
    df = dgp.simulate_binary_choice(n=900, seed=42)
    logit = binary.fit_logit(df, "y ~ x1 + x2")
    probit = binary.fit_probit(df, "y ~ x1 + x2")
    assert float(logit.params["x1"]) > 0
    assert float(probit.params["x1"]) > 0
    ame = binary.average_marginal_effects(logit)
    assert not ame.empty


def test_classification_metrics_are_not_vacuous() -> None:
    df = dgp.simulate_binary_choice(n=900, seed=42)
    logit = binary.fit_logit(df, "y ~ x1 + x2")
    p = binary.predicted_probabilities(logit)
    cm = binary.confusion_at_threshold(df["y"], p, threshold=0.5)
    roc = binary.roc_auc(df["y"], p)
    cal = binary.calibration_table(df["y"], p, n_bins=8)
    assert cm["accuracy"] > 0.6
    assert roc["auc"] > 0.7
    assert cal["n"].sum() == len(df)
    assert "causal" in binary.CLASSIFICATION_VERSUS_CAUSAL.lower() or "treatment" in binary.CLASSIFICATION_VERSUS_CAUSAL.lower()
