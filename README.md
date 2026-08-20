# Econometrics and Causal Inference Lab

[![CI](https://github.com/pavanamthomas/econometrics-causal-inference-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/pavanamthomas/econometrics-causal-inference-lab/actions)

Reproducible Python studies in econometrics, causal inference, diagnostics, robustness, and research-design validation.

Dr. Pavanam Thomas · [pavanamthomas](https://github.com/pavanamthomas) · thomaspavanam@gmail.com  
Copyright 2026 · MIT License

Applied work often reports a regression coefficient as if it were an identified treatment effect. This laboratory treats identification as a research-design claim: a question, an estimand, an assignment process, and explicit assumptions, then estimation, diagnostics, robustness, and a limited interpretation.

The package covers OLS with residual and influence diagnostics; logit and probit with average marginal effects kept separate from classification metrics; entity fixed effects; 2x2 DiD, event study, placebo timing, and a staggered TWFE caution; 2SLS including weak and invalid instruments, with Anderson–Rubin inversion when the first stage is weak; sharp local linear RD; IPW and nearest-neighbour matching; and a wild-cluster Rademacher percentile interval on a few-treated-cluster DGP. Statsmodels is used for likelihood and least squares; 2SLS and two-way demeaning are written out.

All samples are simulated from documented DGPs in `src/econci/dgp.py` (`numpy.random.Generator`, default seed 42). They are not observational microdata. Finite-sample tables and figures are computational artifacts. Each design states the condition that maps identifying variation to an estimand (parallel trends, exclusion, continuity at a cutoff, unconfoundedness and overlap). Conventional TWFE is not automatically valid under heterogeneous staggered treatment. Residual plots and balance tables are descriptive; ROC/AUC is a classification summary. Causal language is used only when the estimand and identifying assumptions are stated and the estimator matches that estimand.

Start with `CASE_STUDY.md`, `docs/failures_and_corrections.md`, `docs/causal_inference_checklist.md`, `ROADMAP.md`, `src/econci/did.py`, and `tests/`. Remaining bounds are in `ROADMAP.md` (issues #1–#5 are closed). Failures the laboratory is designed to exhibit stay in `docs/failures_and_corrections.md`. A numerical change needs a test that would have failed before the change. CI on `main` runs pytest, `python scripts/run_all.py`, and an optional base-R 2x2 DiD job; that is not evidence about an application. See `docs/lab_process.md`.

```bash
pip install -e ".[dev]"
pytest -q
python scripts/run_all.py
```

## Summary

The laboratory is a source-layout Python package (`econci`) plus a case study, a design checklist, and a deterministic reproduction script. It is written as research code: type hints, docstrings, and tests that compare estimators to known truth on simulated designs.

A 2x2 DiD recovers a known ATT when parallel trends hold by construction and is biased when they do not. An event-study construction codes the static treatment dummy as 0 in pre-adoption periods (no future-treatment leakage). A staggered-adoption simulation with opposite-signed cohort ATTs shows that conventional TWFE need not recover those cohort effects. An educational group-time ATT aggregation is labelled as such; it is not a production Callaway–Sant'Anna estimator.

## Designs in the package

- Specification, robust standard errors (HC1/HC3), heteroskedasticity tests, VIF, Cook's distance and leverage, Ramsey RESET and an added-quadratic test on a deliberately misspecified model
- Logit and probit, average marginal effects, predicted probabilities, calibration; explicit separation from causal claims
- Pooled OLS, entity fixed effects, feasible GLS random effects, clustered standard errors, wild-cluster Rademacher percentile intervals on a few-treated-cluster DGP; panel versus repeated cross-section
- DiD regression representation, event-study leads, joint pre-trend test, placebo timing, alternative comparison group, sensitivity to effect size and noise
- First-stage F, 2SLS, weak-IV and exclusion-violation designs; LATE language kept distinct from a linear homogeneous teaching DGP
- Sharp RD local linear estimation, bandwidth sensitivity, graphical bins, no-extrapolation warning
- Propensity overlap, IPW ATT, nearest-neighbour matching, covariate balance, stated limits under unobserved confounding

## Methodology

The protocol is written out in `docs/causal_inference_checklist.md`. Estimation without an estimand and identifying assumptions is treated as incomplete.

## Structure

```
README.md
LICENSE
CITATION.cff
pyproject.toml
ROADMAP.md
CONTRIBUTING.md
CASE_STUDY.md
docs/causal_inference_checklist.md
docs/failures_and_corrections.md
docs/lab_process.md
docs/data_policy.md
src/econci/          # package: dgp, ols, binary, panel, did, iv, rd, matching, plots
scripts/run_all.py   # writes outputs/figures and outputs/tables
tests/               # design-based recovery and diagnostic checks
notebooks/           # short illustrations that import econci
r/optional_twfe_base_r.R   # optional; R is not required
data/README.md       # no stored microdata
```

## Reproducibility

Python 3.11 or newer.

```
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q
python scripts/run_all.py
```

CI (`.github/workflows/ci.yml`): `ubuntu-latest`, Python 3.11, `pip install -e ".[dev]"`, `pytest -q`, `python scripts/run_all.py`.

Optional R script: `r/optional_twfe_base_r.R` illustrates the same 2x2 regression in base R. The `optional-r` GitHub Actions job runs `Rscript` on that file. Python tests remain the required local path; R is not needed to run `pytest`.

## Example outputs

Do not quote numbers from memory. After `python scripts/run_all.py`, inspect:

- `outputs/tables/ols_coefficients.csv`, `ols_misspecification_diagnostics.csv`
- `outputs/tables/did_2x2.csv`, `event_study_coefs.csv`, `did_staggered_twfe_vs_group_time.csv`
- `outputs/tables/iv_comparison.csv`, `rd_bandwidth.csv`, `matching_balance.csv`
- `outputs/figures/ols_residuals_misspecified.png`, `event_study.png`, `rd_binscatter.png`, `propensity_overlap.png`

Those files are simulation artifacts. They change only if the DGP, seed, or estimator changes.

## Validation

Tests compare estimators to known DGP parameters and check that diagnostics fail in the designs where they should fail (omitted quadratic; parallel-trend violation; weak and invalid instruments; far-from-cutoff RD contrasts; event-study dummy coding). Passing tests validate the laboratory, not an empirical application.

## Assumptions

Assumptions are design-specific and are stated next to the corresponding module. Across the lab:

- Simulated assignment is known; that knowledge is not available in observational work.
- Clustering is at the entity/unit level in panel and DiD routines.
- IPW clipping at a documented epsilon is a trimming choice that changes the target population.
- Educational group-time ATT uses never-treated units only and does not implement Callaway–Sant'Anna inference.

## Limitations

- External validity is not identified. Every sample is simulated.
- Parallel-trend tests, first-stage F, overlap plots, and balance tables can fail to reject a false assumption or can reject a true one in finite samples.
- Staggered TWFE is shown to be misleading in one heterogeneous-effects DGP; that is a counterexample, not a complete taxonomy of negative weights.
- The educational group-time aggregation is a teaching device, not a substitute for a production group-time estimator.
- Matching and IPW do not address selection on unobservables.
- Sharp RD does not identify effects away from the cutoff.
- Valid 2SLS in the linear homogeneous DGP is not a license to treat 2SLS as ATE in an application where LATE is the relevant parameter.

## Case study and reproduction

1. Read `CASE_STUDY.md` (one complete DiD argument, including what cannot be concluded).
2. Skim `docs/failures_and_corrections.md` and `docs/causal_inference_checklist.md`.
3. Open `src/econci/did.py` and `tests/test_did.py`.
4. Run `pip install -e ".[dev]" && python -m pytest -q && python scripts/run_all.py`.
5. If useful, `notebooks/01_ols_diagnostics.ipynb` and `notebooks/02_did_event_study.ipynb`.

Related work (sparse): statistical protocol in [statistical-reasoning-validation](https://github.com/pavanamthomas/statistical-reasoning-validation); evaluation design in [ai-response-evaluation-benchmarks](https://github.com/pavanamthomas/ai-response-evaluation-benchmarks). Formal optimization arguments in Lean are a separate line of work ([lean4-optimization-economics](https://github.com/pavanamthomas/lean4-optimization-economics)) and are not modified here.
