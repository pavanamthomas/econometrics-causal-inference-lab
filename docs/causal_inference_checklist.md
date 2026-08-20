# Causal inference checklist

Copyright 2026 Dr. Pavanam Thomas

This checklist is the laboratory's working protocol. It is a sequence of
research decisions, not a menu of regression commands. Each item asks for an
object that can be written down before software is run.

## 1. Research question

State the question in terms of a well-defined intervention, population, and
outcome. Separate a descriptive comparison from a causal query. If the
question is predictive (who is classified as y=1?), stop using causal
language.

## 2. Estimand

Name the target: ATE, ATT, LATE, RD jump at a cutoff, a group-time ATT, or a
descriptive contrast. Write it as a functional of potential outcomes or of a
conditional distribution. Do not let software defaults choose the estimand.

## 3. Treatment assignment

How is treatment assigned in the design you actually have? Group-level
policy timing, a cutoff rule, an instrument, or selection on observables are
different assignment processes. In this repository, assignment is simulated
and documented in `src/econci/dgp.py`.

## 4. Identifying variation

State the comparison that is supposed to stand in for the missing
counterfactual: untreated group over time, observations just below a cutoff,
compliers shifted by an instrument, or matched units with similar propensity
scores. If you cannot point to the variation, you do not have a design.

## 5. Assumptions

Write the assumptions that map the identifying variation to the estimand.
Examples: parallel trends in untreated potential outcomes; exclusion,
exogeneity, relevance, and (where applicable) monotonicity; continuity of
conditional mean potential outcomes at the cutoff; unconfoundedness and
overlap. Conventional two-way fixed effects with a static treated dummy is
not automatically valid under staggered adoption with heterogeneous
treatment effects.

## 6. Estimator

Choose an estimator that matches the estimand, not an estimator that is
convenient. Report the regression representation where it exists (2x2 DiD as
an interaction; 2SLS; local linear RD). Record clustering, kernels,
bandwidths, and weights.

## 7. Diagnostics

Inspection is part of estimation. Residual plots, RESET or added-variable
tests, VIF, influence, first-stage F, event-study pre-treatment
coefficients, placebo timing, overlap histograms, and covariate balance
tables are diagnostics. A diagnostic that "passes" does not prove the
identifying assumption.

## 8. Robustness

Change one design choice at a time: bandwidth, comparison group, effect size
and noise in a sensitivity grid, HC1 versus HC3, IPW versus nearest
neighbour, TWFE versus an explicit group-time aggregation. Report when
conclusions move.

## 9. Interpretation

State what was estimated, for whom, and under which assumption. Keep
classification performance (AUC, confusion matrix) separate from causal
claims. Do not read a coefficient as an ATT because the command is called
`did` or `iv`.

## 10. Limitations

List what the design cannot answer: effects far from an RD cutoff; ATE when
only LATE is identified; staggered TWFE under heterogeneous adoption;
selection on unobservables after matching on x; real-world external
validity when the sample is simulated. This laboratory never claims
empirical evidence about an actual market or program.

Related protocol discussion, in a statistical rather than design register:
https://github.com/pavanamthomas/statistical-reasoning-validation
