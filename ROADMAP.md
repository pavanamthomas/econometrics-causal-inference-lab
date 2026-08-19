# Roadmap

Current as of August 2026. This is a bound on the laboratory, not a product backlog.

## In scope now

- Simulated DGPs with documented assignment (OLS, binary, panel, 2x2 DiD, event study, staggered TWFE caution, IV, sharp RD, IPW/matching).
- Tests that recover known parameters when the design is correctly specified, and that fail in the intended direction when it is not.
- Recruiter artifacts: `CASE_STUDY.md`, `docs/causal_inference_checklist.md`.
- CI: install, `python -m pytest`, `python scripts/run_all.py`.

## Failures that are part of the design

These are not tickets to “fix the estimator so it always looks good”:

- Linear OLS on an omitted quadratic: RESET and the added-quadratic test reject.
- 2x2 DiD under a parallel-trend violation: the ATT is biased.
- Conventional TWFE under staggered adoption with opposite-signed cohort ATTs: TWFE matches neither cohort.
- Weak IV: first-stage F is below the conventional threshold used in the test.
- Invalid IV with a strong first stage: 2SLS remains biased.

Details: `docs/failures_and_corrections.md`.

## Open (issues)

Tracked as GitHub issues, not as silent placeholders in code:

1. Educational group-time ATT versus a production staggered estimator (Callaway–Sant’Anna or an equivalent). Out of the current dependency budget; the code must keep the “not CS” label until that changes.
2. Weak-IV behaviour is flagged by first-stage F. A Stock–Yogo tabulation or Anderson–Rubin interval is not implemented.
3. Optional `r/optional_twfe_base_r.R` is not in CI. R is not a required runtime.
4. Clustered event-study inference with few treated clusters: the laboratory uses conventional clustering; wild cluster bootstrap is not implemented.

## Explicitly not in scope

- Observational microdata or a claim about a real programme.
- Treating TWFE as a group-time ATT under heterogeneous staggered adoption.
- SQL warehouses. No panel store is required for these DGPs.
- Auto-generated co-author metadata or editor-tool files.

When an open item is closed, the closing comment should name the test that now locks the behaviour, or state that the item remains a documented limitation.
