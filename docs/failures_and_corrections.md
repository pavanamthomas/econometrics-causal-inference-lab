# Failures and corrections

This note records specification failures that the laboratory is designed to exhibit. Each row is a **correction of interpretation or specification**, not a software defect to hide.

Passing tests mean the failure still occurs under the stated DGP. If a future change makes the failure disappear, the corresponding test should fail until the documentation is updated.

| What was tried | How it failed | Diagnostic | Correction | Locked by | What remains unknown |
| --- | --- | --- | --- | --- | --- |
| OLS of `y` on `x` when the DGP is quadratic in `x` | Linear fit is the wrong conditional mean | Added-quadratic test; Ramsey RESET | Include the nonlinear term, or treat the linear coefficient as a projection, not a structural slope | `tests/test_ols.py::test_misspecification_diagnostic_fires_on_omitted_quadratic` | Which nonlinear form is right in an application; RESET is not a specification search |
| 2x2 DiD when untreated and treated trends diverge by construction | ATT target 2.0 is not recovered | Parallel-trend violation in the DGP; event-study leads when that design is used | Do not report the coefficient as ATT; the identifying assumption failed | `tests/test_did.py::test_parallel_trends_violation_biases_did` | A pre-trend test can fail to reject a false assumption in short panels |
| Static TWFE on staggered adoption with ATT_early = 4 and ATT_late = −1.5 | TWFE matches neither cohort ATT | Comparison to educational group-time aggregation | Do not read TWFE as a group-time ATT under heterogeneous staggered adoption | `tests/test_did.py::test_staggered_twfe_can_miss_cohort_atts` | Negative-weight taxonomy for other timing designs; this is one counterexample |
| 2SLS with a weak instrument | First-stage F below 10 in the test DGP | First-stage F | Do not treat the 2SLS point as precise; weakness is not cured by a large reduced-form | `tests/test_iv.py::test_weak_iv_has_low_first_stage_f` | Finite-sample IV distributions; Stock–Yogo critical values are not tabulated here |
| 2SLS with exclusion violated and a strong first stage | Point estimate stays far from 1.0 | First-stage F can still be large | Relevance is not validity | `tests/test_iv.py::test_first_stage_strength_is_not_validity` | Exclusion is not testable from the first stage |
| Global polynomial contrast far from an RD cutoff | Extrapolation error relative to local linear at the cutoff | Bandwidth table; binscatter | Restrict the estimand to a neighbourhood of the cutoff | `tests/test_rd.py` | Optimal bandwidth in observational RD; here the cutoff and DGP are known |
| Event-study coded with future treatment in pre-adoption periods | That coding leaks post-adoption status into the pre window | Pre-period `treated == 0` for ever-treated units | Build the static dummy from calendar time versus adoption date only | `tests/test_did.py::test_event_study_no_future_treatment_leakage` | Calendar-time confounding in real panels |

Process: `docs/lab_process.md`. Open extensions: `ROADMAP.md`.
