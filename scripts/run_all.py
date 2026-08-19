"""Run the simulated econometrics laboratory and write tables and figures.

Deterministic seeds (default 42). Matplotlib uses the Agg backend so the
script can run without a display.

Copyright 2026 Dr. Pavanam Thomas
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from econci import binary, dgp, did, iv, matching, ols, panel, plots, rd  # noqa: E402

FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
SEED = 42

METHODS_NOTE = """
METHODS NOTE
Problem -> formalization -> assumptions -> computation/estimation -> validation -> interpretation -> limitations

This run used documented simulated data-generating processes (seed=42 unless a
routine states otherwise). Tables and figures in outputs/ are finite-sample
computational artifacts. They are not estimates from observational
administrative data and should not be cited as empirical findings.

Classification metrics (ROC/AUC, confusion matrices) are not identifying
arguments. Conventional two-way fixed effects is not automatically valid
under staggered adoption with heterogeneous treatment effects. First-stage
strength does not establish instrument validity. Sharp RD identifies a jump
at the cutoff, not an effect far from it.
""".strip()


def _write(df: pd.DataFrame, name: str) -> Path:
    TAB.mkdir(parents=True, exist_ok=True)
    path = TAB / name
    df.to_csv(path, index=False)
    return path


def run_ols() -> None:
    linear = dgp.simulate_ols_linear(n=800, seed=SEED)
    fit = ols.fit_ols(linear, "y ~ x1 + x2", cov_type="HC1")
    _write(ols.coefficient_table(fit), "ols_coefficients.csv")
    _write(ols.vif_table(linear, ["x1", "x2"]), "ols_vif.csv")
    _write(ols.influence_table(fit), "ols_influence.csv")
    resid = ols.residual_diagnostics(fit)
    plots.plot_residuals_vs_fitted(
        resid["fitted"].to_numpy(),
        resid["residual"].to_numpy(),
        FIG / "ols_residuals_correct.png",
        title="Residuals versus fitted (correctly specified linear DGP)",
    )

    miss = dgp.simulate_ols_omitted_quadratic(n=800, seed=SEED)
    bad = ols.fit_ols(miss, "y ~ x")
    resid_bad = ols.residual_diagnostics(bad)
    plots.plot_residuals_vs_fitted(
        resid_bad["fitted"].to_numpy(),
        resid_bad["residual"].to_numpy(),
        FIG / "ols_residuals_misspecified.png",
        title="Residuals versus fitted (linear fit, quadratic DGP)",
    )
    quad = ols.added_quadratic_test(miss, "y", "x")
    reset = ols.ramsey_reset(bad, power=3)
    bp = ols.breusch_pagan_test(bad)
    _write(
        pd.DataFrame(
            [
                {"diagnostic": "added_quadratic_pvalue", "value": quad["pvalue"]},
                {"diagnostic": "reset_pvalue", "value": reset["pvalue"]},
                {"diagnostic": "breusch_pagan_pvalue", "value": bp["lm_pvalue"]},
            ]
        ),
        "ols_misspecification_diagnostics.csv",
    )

    hetero = dgp.simulate_ols_heteroskedastic(n=800, seed=SEED)
    classical = ols.fit_ols(hetero, "y ~ x", cov_type="nonrobust")
    hc3 = ols.fit_ols(hetero, "y ~ x", cov_type="HC3")
    _write(
        pd.DataFrame(
            [
                {"cov_type": "nonrobust", "se_x": float(classical.bse["x"])},
                {"cov_type": "HC3", "se_x": float(hc3.bse["x"])},
            ]
        ),
        "ols_robust_se.csv",
    )


def run_binary() -> None:
    df = dgp.simulate_binary_choice(n=900, seed=SEED)
    logit = binary.fit_logit(df, "y ~ x1 + x2")
    probit = binary.fit_probit(df, "y ~ x1 + x2")
    p = binary.predicted_probabilities(logit)
    ame = binary.average_marginal_effects(logit)
    _write(ame, "binary_ame_logit.csv")
    cm = binary.confusion_at_threshold(df["y"], p)
    _write(pd.DataFrame([cm]), "binary_confusion.csv")
    roc = binary.roc_auc(df["y"], p)
    cal = binary.calibration_table(df["y"], p)
    _write(cal, "binary_calibration.csv")
    _write(
        pd.DataFrame(
            [
                {"model": "logit", "x1": float(logit.params["x1"]), "x2": float(logit.params["x2"])},
                {"model": "probit", "x1": float(probit.params["x1"]), "x2": float(probit.params["x2"])},
                {"model": "logit_auc_classification_only", "x1": float(roc["auc"]), "x2": float("nan")},
            ]
        ),
        "binary_coefficients.csv",
    )
    plots.plot_roc(roc["fpr"], roc["tpr"], float(roc["auc"]), FIG / "binary_roc.png")
    plots.plot_calibration(cal, FIG / "binary_calibration.png")


def run_panel() -> None:
    df = dgp.simulate_panel(n_entities=60, n_periods=8, beta=1.5, seed=SEED)
    pooled = panel.fit_pooled_ols(df, "y", ["x"], "entity")
    fe = panel.fit_entity_fe(df, "y", ["x"], "entity")
    re = panel.fit_random_effects(df, "y", ["x"], "entity")
    rows = [
        panel.coef_row("pooled_ols", pooled, "x"),
        panel.coef_row("entity_fe", fe, "x"),
        panel.coef_row("random_effects", re, "x"),
    ]
    _write(pd.DataFrame(rows), "panel_comparison.csv")
    haus = panel.hausman_fe_re(fe, re, ["x"])
    _write(pd.DataFrame([haus]), "panel_hausman.csv")


def run_did() -> None:
    two = dgp.simulate_did_2x2(n_treat=150, n_control=150, att=2.0, parallel_trends=True, seed=SEED)
    fit = did.fit_did_2x2(two)
    violated = dgp.simulate_did_2x2(
        n_treat=150, n_control=150, att=2.0, parallel_trends=False, trend_gap=1.2, seed=SEED
    )
    fit_bad = did.fit_did_2x2(violated)
    _write(
        pd.DataFrame(
            [
                {
                    "design": "parallel_trends",
                    "did_interaction": float(fit.params["_did"]),
                    "se": float(fit.bse["_did"]),
                    "true_att": 2.0,
                },
                {
                    "design": "parallel_trends_violated",
                    "did_interaction": float(fit_bad.params["_did"]),
                    "se": float(fit_bad.bse["_did"]),
                    "true_att": 2.0,
                },
            ]
        ),
        "did_2x2.csv",
    )

    ev = dgp.simulate_event_study(
        n_treat=80, n_control=80, n_periods=10, treat_period=6, att=2.0, parallel_trends=True, seed=SEED
    )
    design = did.build_event_study_design(ev)
    names = list(design.attrs["event_dummy_names"])
    results, coefs = did.fit_event_study(design, dummy_names=names)
    leads = did.lead_names_from_dummies(names)
    pretrend = did.joint_pretrend_test(results, leads)
    plot_df = did.event_study_coef_table_with_omitted(coefs, omit=-1)
    _write(plot_df, "event_study_coefs.csv")
    _write(pd.DataFrame([pretrend]), "event_study_pretrend_test.csv")
    plots.plot_event_study(plot_df, FIG / "event_study.png")

    placebo = did.placebo_timing_2x2(ev, fake_post_from=3)
    _write(
        pd.DataFrame(
            [{"placebo_did": float(placebo.params["_did"]), "se": float(placebo.bse["_did"])}]
        ),
        "did_placebo_timing.csv",
    )

    two_c = dgp.simulate_did_two_controls(seed=SEED)
    primary = did.fit_did_2x2(two_c[two_c["control_pool"].isin(["treated", "A"])])
    alt = did.fit_did_2x2(two_c[two_c["control_pool"].isin(["treated", "B"])])
    _write(
        pd.DataFrame(
            [
                {"comparison": "control_A", "did": float(primary.params["_did"])},
                {"comparison": "control_B", "did": float(alt.params["_did"])},
            ]
        ),
        "did_alternative_comparison.csv",
    )

    sens_rows = []
    for att in (0.5, 1.0, 2.0):
        for sigma, sseed in ((0.7, SEED), (1.4, SEED + 1)):
            sim = dgp.simulate_did_2x2(n_treat=120, n_control=120, att=att, sigma=sigma, seed=sseed)
            est = did.fit_did_2x2(sim)
            sens_rows.append(
                {
                    "true_att": att,
                    "sigma": sigma,
                    "estimate": float(est.params["_did"]),
                    "se": float(est.bse["_did"]),
                }
            )
    _write(pd.DataFrame(sens_rows), "did_sensitivity.csv")

    stag = dgp.simulate_staggered_heterogeneous(seed=SEED)
    twfe = did.fit_twfe_static(stag)
    att_gt = did.group_time_att_educational(stag)
    agg = did.aggregate_group_time_att(att_gt)
    _write(att_gt, "did_group_time_att_educational.csv")
    early = stag.loc[stag["g"] == 4, "true_cohort_att"].iloc[0]
    late = stag.loc[stag["g"] == 8, "true_cohort_att"].iloc[0]
    _write(
        pd.DataFrame(
            [
                {"object": "twfe_static_treated", "value": float(twfe.params["treated"])},
                {"object": "educational_group_time_att_simple", "value": agg["att_simple"]},
                {"object": "known_att_early_cohort", "value": float(early)},
                {"object": "known_att_late_cohort", "value": float(late)},
            ]
        ),
        "did_staggered_twfe_vs_group_time.csv",
    )
    plots.plot_staggered_comparison(
        ["TWFE", "Educ. group-time", "Early ATT", "Late ATT"],
        [float(twfe.params["treated"]), float(agg["att_simple"]), float(early), float(late)],
        FIG / "staggered_twfe_comparison.png",
    )


def run_iv() -> None:
    rows = []
    designs = {
        "valid": dgp.simulate_iv(n=700, beta=1.0, pi=1.0, endog_corr=0.7, exclusion_violation=0.0, seed=SEED),
        "weak": dgp.simulate_iv(n=700, beta=1.0, pi=0.06, endog_corr=0.7, exclusion_violation=0.0, seed=SEED),
        "invalid": dgp.simulate_iv(n=700, beta=1.0, pi=1.0, endog_corr=0.2, exclusion_violation=1.2, seed=SEED),
    }
    for name, df in designs.items():
        est = iv.two_sls(df)
        rows.append(
            {
                "design": name,
                "true_beta": 1.0,
                "ols": est["ols_endog"],
                "two_sls": est["beta_endog"],
                "two_sls_se": est["se_endog"],
                "first_stage_f": est["first_stage_f"],
            }
        )
    _write(pd.DataFrame(rows), "iv_comparison.csv")


def run_rd() -> None:
    df = dgp.simulate_sharp_rd(n=1200, cutoff=0.0, tau=2.0, seed=SEED)
    local = rd.local_linear_rd(df, cutoff=0.0, bandwidth=0.75)
    far = rd.naive_far_difference(df, cutoff=0.0, margin=1.1)
    glob = rd.global_polynomial_jump(df, cutoff=0.0, degree=3)
    _write(
        pd.DataFrame(
            [
                {"estimator": "local_linear_h075", "estimate": local["tau"], "se": local["se"]},
                {"estimator": "naive_far_difference", "estimate": far["far_diff"], "se": float("nan")},
                {"estimator": "global_cubic_jump", "estimate": glob["tau_global"], "se": glob["se"]},
            ]
        ),
        "rd_estimates.csv",
    )
    _write(rd.bandwidth_sensitivity(df, bandwidths=[0.4, 0.6, 0.8, 1.0, 1.4]), "rd_bandwidth.csv")
    bins = rd.rd_bin_means(df, cutoff=0.0)
    plots.plot_rd(df, bins, cutoff=0.0, path=FIG / "rd_binscatter.png")


def run_matching() -> None:
    df = dgp.simulate_selection_observables(n=900, att=1.5, seed=SEED)
    ps = matching.propensity_scores(df, treat="d", covariates=["x"])
    overlap = matching.overlap_summary(ps, df["d"])
    _write(pd.DataFrame([overlap]), "matching_overlap.csv")
    w = matching.att_ipw_weights(df, "d", ps)
    before = matching.balance_table(df, treat="d", covariates=["x"])
    after = matching.balance_table(df, treat="d", covariates=["x"], weights=w)
    before["stage"] = "unweighted"
    after["stage"] = "ipw_att"
    _write(pd.concat([before, after], ignore_index=True), "matching_balance.csv")
    ipw = matching.ipw_att(df, ps=ps)
    nn = matching.nearest_neighbor_att(df, ps=ps)
    naive = df.loc[df["d"] == 1, "y"].mean() - df.loc[df["d"] == 0, "y"].mean()
    _write(
        pd.DataFrame(
            [
                {"estimator": "naive_mean_difference", "att": float(naive)},
                {"estimator": "ipw_att", "att": ipw["att"]},
                {"estimator": "nn_ps_att", "att": nn["att"]},
                {"estimator": "true_att", "att": 1.5},
            ]
        ),
        "matching_att.csv",
    )
    plots.plot_propensity_overlap(ps, df["d"].to_numpy(), FIG / "propensity_overlap.png")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    run_ols()
    run_binary()
    run_panel()
    run_did()
    run_iv()
    run_rd()
    run_matching()
    print(METHODS_NOTE)
    print(f"Wrote figures to {FIG}")
    print(f"Wrote tables to {TAB}")


if __name__ == "__main__":
    main()
