# Case study: a simulated event-study difference-in-differences

Copyright 2026 Dr. Pavanam Thomas

This note walks one design from question to a limited conclusion. The sample
is simulated. It is not an evaluation of an actual statute, plant opening, or
transfer program. Finite-sample numbers belong in `outputs/tables` after
`python scripts/run_all.py`; they are not restated here as if they were
empirical findings.

The walk-through below is one identification argument: research question,
estimand, assignment, assumptions, estimator, checks, and a limited
conclusion.

## 1. Research question

In a two-group panel with a single adoption date, what is the average effect
of the discrete policy shift on the outcome of adopting units after
adoption?

The question is causal. A descriptive analogue (did mean outcomes rise more
in the adopting group?) is well-defined without potential outcomes, but it
is not the same object.

## 2. Formalization (estimand)

Units $i$ are observed in calendar periods $t = 0,\ldots,T-1$. A subset is
ever treated and adopts at a common date $G$. The remainder is never
treated. Potential outcomes $Y_{it}(0)$ and $Y_{it}(1)$ denote the outcome
without and with the policy in force.

The target is the average treatment effect on the treated in post-adoption
periods,

\[
\mathrm{ATT} = \mathbb{E}[Y_{it}(1) - Y_{it}(0) \mid G_i = G,\, t \ge G].
\]

In the laboratory DGP the effect is constant after adoption, so collapsing
the pre period and the post period recovers the same ATT as a 2x2 design.
The event-study representation keeps calendar time in order to inspect
pre-adoption coefficients.

## 3. Treatment assignment and identifying variation

Assignment is at the group-time level: all treated units switch at $G$;
controls never switch. There is no individual-level randomization.

The comparison that stands in for $Y_{it}(0)$ among adopters after $G$ is
the time path of the never-treated group, after removing unit and time
effects. That is the identifying variation.

The DGP is `econci.dgp.simulate_event_study`. When `parallel_trends=True`,
untreated potential outcomes differ only by unit and time effects. When
`parallel_trends=False`, adopters have an extra linear drift in $Y(0)$.
Those two designs exist so that the same estimator can be shown to recover
ATT in one case and to fail in the other.

## 4. Assumptions

Parallel trends in untreated potential outcomes:

\[
\mathbb{E}[Y_{it}(0) - Y_{is}(0) \mid \text{ever treated}]
=
\mathbb{E}[Y_{it}(0) - Y_{is}(0) \mid \text{never treated}]
\]

for the times $s,t$ used in the contrast. No anticipation is also required
if pre-adoption event-time coefficients are to have mean zero under the
null. SUTVA is built into the simulation (no interference, a single
treatment version).

These assumptions are true by construction in the baseline DGP. That is a
validation device. It is not evidence that parallel trends holds in an
observational setting with the same regression command.

## 5. Estimation

Two representations are estimated, both clustered at the unit level.

**2x2 collapse.** Define $\mathrm{Post}_t = 1\{t \ge G\}$ and
$\mathrm{Group}_i = 1\{i \text{ ever treated}\}$. The regression

\[
Y_{it} = \alpha + \gamma \mathrm{Group}_i + \lambda \mathrm{Post}_t
+ \delta (\mathrm{Group}_i \times \mathrm{Post}_t) + \varepsilon_{it}
\]

has $\delta$ equal to the 2x2 DiD of cell means in a balanced design
(`econci.did.fit_did_2x2`, `econci.did.did_att_from_means`).

**Event study.** Relative time $k = t - G$ for adopters. Dummies
$D_{it}^k = 1\{t-G = k\}$ are defined only for ever-treated units;
never-treated units have all such dummies equal to 0. The omitted category
is $k = -1$. Endpoints are binned. The two-way fixed-effects regression on
those dummies is `econci.did.fit_event_study`.

The static treatment dummy used in diagnostics equals 1 only when
$t \ge G$ for ever-treated units. Eventually treated units are coded 0
before adoption. That is the no-future-treatment-leakage property checked
in `tests/test_did.py`.

## 6. Diagnostics and robustness

- Pre-treatment event-study coefficients and a joint Wald/F test that all
  leads are zero (`joint_pretrend_test`). Failure to reject is consistent
  with parallel trends in this sample; it does not prove the assumption.
- Placebo timing: the same 2x2 regression on strictly pre-adoption periods
  with a fake post date (`placebo_timing_2x2`). Under the baseline DGP the
  placebo interaction should be near zero.
- Alternative comparison group: a separate DGP with two never-treated pools
  (`simulate_did_two_controls`). The DiD is repeated against pool A and
  against pool B.
- Sensitivity: 2x2 DiD is re-estimated on DGPs that vary the known ATT and
  the noise scale (`outputs/tables/did_sensitivity.csv` after reproduction).
- Negative control for the assumption: the same 2x2 estimator on
  `parallel_trends=False`. The interaction is then a mix of ATT and the
  extra treated drift; tests require that this object not recover the known
  ATT.

A staggered-adoption companion design (`simulate_staggered_heterogeneous`)
is not this case study's estimand. It is reported in the same reproduction
script to record a separate point: conventional TWFE with a static treated
dummy need not recover cohort-specific ATTs when effects are heterogeneous.
The educational group-time table in that script is a transparent
never-treated contrast. It is not Callaway and Sant'Anna (2021).

## 7. Interpretation

Under the baseline simulated design, the 2x2 interaction is an estimator of
the ATT defined above. Event-study post-period coefficients estimate how
that effect is distributed across time since adoption in a DGP where the
effect is in fact constant. Pre-period coefficients are not treatment
effects; they are specification diagnostics.

Classification language is out of place here. There is no ROC curve in this
case study because the object is not a predicted label.

## 8. What the case study distinguishes

The following five objects must not be collapsed into a single sentence of
the form "we find a significant effect."

**Estimated effect.** The 2x2 interaction (and the post-adoption event-time
coefficients) computed from the simulated sample. Point estimates and
standard errors are in `outputs/tables/did_2x2.csv` and
`outputs/tables/event_study_coefs.csv` after running the reproduction
script. They are finite-sample draws, not a policy parameter from the
world.

**Identifying assumption.** Parallel trends in untreated potential
outcomes, plus no anticipation for the event-study leads. The estimator
does not create that assumption.

**Diagnostic evidence.** Pre-trend coefficients, the joint lead test, the
placebo interaction, and the alternative-control comparison. These can
fail to detect a violation. In the companion DGP where parallel trends is
false by construction, the 2x2 estimator is biased for ATT; that is the
relevant warning, not a successful pre-trend test in the baseline draw.

**Uncertainty.** Cluster-robust standard errors at the unit level, and the
sensitivity table over ATT and noise. Uncertainty statements do not cover
failure of parallel trends, interference, or a different adoption process.

**What cannot be concluded.**

- Nothing about a real labor market, tax, or transfer program.
- Nothing about units that are never in the simulated population.
- Nothing about dynamic treatment effects in a DGP that, in the baseline,
  imposes a constant ATT.
- Nothing about staggered adoption from this common-timing design. The
  staggered TWFE illustration is a different DGP; even there, TWFE is not
  a group-time ATT.
- A small placebo estimate does not prove that parallel trends would hold
  after adoption.
- Statistical significance of $\delta$ is not identification.

The disciplined conclusion is therefore conditional: *if* untreated paths
would have moved in parallel, *then* the 2x2 functional equals ATT in this
design, and the laboratory DGP is one environment where that premise is
true by construction. Moving the same regression to observational data
requires a separate argument that the premise is plausible there. That
argument is not supplied by `fit_did_2x2`.
