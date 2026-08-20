# Roadmap

Current as of August 2026. This is a bound on the laboratory, not a product backlog.

## In scope now

- Simulated DGPs with documented assignment (OLS, binary, panel, 2x2 DiD, event study, staggered TWFE caution, IV, sharp RD, IPW/matching).
- Tests that recover known parameters when the design is correctly specified, and that fail in the intended direction when it is not.
- Wild-cluster Rademacher percentile intervals compared with cluster-robust Wald on a few-treated-cluster DGP.
- Case study and design checklist: `CASE_STUDY.md`, `docs/causal_inference_checklist.md`.
- CI: install, `python -m pytest`, `python scripts/run_all.py`.

## Failures that are part of the design

These are not tickets to “fix the estimator so it always looks good”:

- Linear OLS on an omitted quadratic: RESET and the added-quadratic test reject.
- 2x2 DiD under a parallel-trend violation: the ATT is biased.
- Conventional TWFE under staggered adoption with opposite-signed cohort ATTs: TWFE matches neither cohort.
- Weak IV: first-stage F is below the conventional threshold used in the test.
- Invalid IV with a strong first stage: 2SLS remains biased.

Details: `docs/failures_and_corrections.md`.

## Remaining bounds

GitHub issues #1–#5 were closed when the corresponding tests were added
(educational group-time cells, Anderson–Rubin inversion, quadratic
specification recovery, optional-R CI job, wild-cluster Rademacher interval).
The following are still bounds, not silent TODOs in the estimator code:

1. Educational group-time ATT still uses never-treated units only. It is not Callaway–Sant’Anna inference.
2. Stock–Yogo tabulations are not implemented; weak-IV correction is the Anderson–Rubin inversion.
3. The wild cluster interval uses Rademacher percentile coefficients, not Webb weights and not a restricted bootstrap-t.

## Explicitly not in scope

- Observational microdata or a claim about a real programme.
- Treating TWFE as a group-time ATT under heterogeneous staggered adoption.
- SQL warehouses. No panel store is required for these DGPs.
- Auto-generated co-author metadata or editor-tool files.

When an open item is closed, the closing comment should name the test that now locks the behaviour, or state that the item remains a documented limitation.
