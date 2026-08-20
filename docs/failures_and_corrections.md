# Failures and corrections

This note records specification failures that the laboratory is designed to exhibit. Each row is a **correction of interpretation or specification**, not a software defect to hide.

Passing tests mean the failure still occurs under the stated DGP. If a future change makes the failure disappear, the corresponding test should fail until the documentation is updated.

| What was tried | How it failed | Diagnostic | Correction | Locked by | What remains unknown |
| --- | --- | --- | --- | --- | --- |
| OLS of `y` on `x` when the DGP is quadratic in `x` | Linear fit is the wrong conditional mean | Added-quadratic test; Ramsey RESET | Fit `y ~ x + x2`; the quadratic coefficient recovers the DGP | `tests/test_ols.py::test_quadratic_specification_recovers_known_curvature` | Which nonlinear form is right in an application |
| 2x2 DiD when untreated and treated trends diverge by construction | ATT target 2.0 is not recovered | Parallel-trend violation in the DGP | Do not report the coefficient as ATT | `tests/test_did.py::test_parallel_trends_violation_biases_did` | A pre-trend test can fail to reject a false assumption |
| Static TWFE on staggered adoption with ATT_early = 4 and ATT_late = -1.5 | TWFE matches neither cohort ATT | Comparison to educational group-time aggregation | Read never-treated group-time cells, not TWFE, for cohort signs | `tests/test_did.py::test_educational_group_time_recovers_cohort_signs` | This is not Callaway-Sant'Anna inference |
| 2SLS with a weak instrument | First-stage F below 10; Wald interval is too tight | First-stage F | Invert Anderson-Rubin over a grid | `tests/test_iv.py::test_anderson_rubin_interval_is_wider_than_wald_when_iv_is_weak` | Stock-Yogo critical values are not tabulated |
| 2SLS with exclusion violated and a strong first stage | Point estimate stays far from 1.0 | First-stage F can still be large | Relevance is not validity | `tests/test_iv.py::test_first_stage_strength_is_not_validity` | Exclusion is not testable from the first stage |
| Global polynomial contrast far from an RD cutoff | Extrapolation error relative to local linear at the cutoff | Bandwidth table; binscatter | Restrict the estimand to a neighbourhood of the cutoff | `tests/test_rd.py` | Optimal bandwidth in observational RD |
| Event-study coded with future treatment in pre-adoption periods | That coding leaks post-adoption status into the pre window | Pre-period `treated == 0` for ever-treated units | Build the static dummy from calendar time versus adoption date only | `tests/test_did.py::test_event_study_no_future_treatment_leakage` | Calendar-time confounding in real panels |
| Cluster-robust Wald with two treated clusters | Coverage of the Wald interval for the cluster-level treatment coefficient falls below 0.85 on the designed DGP | Few-treated-cluster DGP | Wild-cluster Rademacher percentile interval | `tests/test_wild_cluster.py::test_cluster_wald_undercovers_with_few_treated_clusters` | Webb weights; restricted wild bootstrap-t |

Process: `docs/lab_process.md`. Open extensions: `ROADMAP.md`.
