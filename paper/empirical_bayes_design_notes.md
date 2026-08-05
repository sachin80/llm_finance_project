# Empirical Bayes Design Notes

## Document status

> **Historical design rationale; partially superseded.**
>
> For current notation, estimator hierarchy, zero-floor treatment, overlap
> framework, implementation order, and the primary Jansen reference edition,
> follow `AUTHORITATIVE_PROJECT_DECISIONS_2026-08-04.md`.
>
> Where this document uses tilde notation for historical shrinkage, introduces
> an additional current-period forecast coefficient, treats the zero floor as
> primary, or conflicts with the accepted phased implementation order, those
> passages are historical and are not authoritative.

## Document role

This document records the methodological rationale, statistical assumptions,
validation requirements, and intended interpretation of the empirical-Bayes
calibration layer for the SEC filing signal project.

It is **not** the authoritative source for function signatures, output filenames,
or test interfaces. The implementation and tests define the executable contract.

The factor-selection audit is complete. The frozen primary factor is
`sec_similarity_x_exposure`, supplied to the estimator through the existing
signal panel and caller-provided `signal_name`.

The initial implementation is limited to:

1. Shrunk univariate regression
2. Q5-premium empirical Bayes

It stops at saved point-in-time estimated-return panels and does not integrate
the optimizer or backtest.

---

## Methodological references

Primary empirical-Bayes reference:

- Efron, Bradley, and Trevor Hastie. *Computer Age Statistical Inference:
  Algorithms, Evidence, and Data Science*. Cambridge University Press, 2016.
  - Chapter 6: Empirical Bayes
  - Chapter 7: James–Stein Estimation and Ridge Regression
  - Chapter 21: Empirical Bayes Estimation Strategies

Supporting project references:

- Stefan Jansen, *Machine Learning for Algorithmic Trading*: main trading-ML and
  implementation reference.
- Marcos López de Prado, *Advances in Financial Machine Learning*: leakage,
  overlapping labels, sample dependence, purging/embargo logic, and backtest
  overfitting.
- Boyd-related convex optimization material: optimizer and risk-modeling
  motivation.

These references support different layers of the project and should not be
treated as interchangeable.

---

## Current project status

The factor-selection audit is complete.

Primary selected factor:

```text
sec_similarity_x_exposure
```

Economic definition:

> Point-in-time filing-summary theme similarity multiplied by LLM-assessed
> exposure materiality.

Existing saved panels may use names such as:

```text
similarity_x_exposure_score_alpha
sec_similarity_x_exposure_alpha
```

The empirical-Bayes module should not contain project-specific alias-resolution
logic. The caller supplies:

```text
signal_panel
signal_name
```

and the supplied signal name is retained in metadata.

Do not rerun:

- SEC retrieval
- OpenAI calls
- filing parsing
- signal construction
- existing diagnostics

The initial empirical-Bayes implementation should use saved project outputs only.

---

## Factor-selection evidence

The factor-selection audit supports `sec_similarity_x_exposure` as the primary
empirical-Bayes factor because it provides the strongest overall combination of:

- positive point-in-time rank information;
- strong 63-day upper-quintile separation;
- improvement over static TF-IDF and embedding baselines;
- persistence after sector neutralization;
- broad coverage across the S&P 500 universe.

The factor should be interpreted primarily as an **upper-tail selector**, not as
evidence of a perfectly monotonic or perfectly linear signal-return relationship.

The audit sample contains a limited number of evaluation dates, and the 63-day
outcomes overlap through time. Therefore, the audit supports factor selection,
but it does not eliminate:

- selection bias;
- cross-sectional dependence;
- temporal dependence;
- the need for untouched walk-forward evaluation.

Empirical Bayes calibrates the selected factor. It does not validate the earlier
factor-selection process.

---

## Confirmed estimator hierarchy

Use the following ordering:

1. **Primary:** shrunk univariate regression for 21-day and 63-day
   estimated-return panels.
2. **Second:** Q5-premium empirical Bayes for the upper-tail interpretation.
3. **Third:** full five-quantile empirical Bayes only after the first two
   estimators are validated.
4. **Sensitivity only:** raw standardized signal into the optimizer.

No full five-quantile placeholder or unused abstraction is needed in the first
implementation.

Regression is primary because it produces a continuous stock-level
expected-return vector. Q5-premium empirical Bayes is required because the
strongest diagnostic evidence is concentrated in the upper signal quintile.

---

## Why empirical Bayes is the appropriate first estimator

The immediate problem is primarily a **noisy calibration problem**, not a
fully identified dynamic-state estimation problem.

At each completed rebalance date, the project obtains a noisy estimate of either:

- the cross-sectional signal-return slope; or
- the Q5-versus-non-Q5 active-return premium.

The number of stocks in each cross-section may be large, but the number of
effectively independent historical rebalance periods is limited. Returns are
noisy, firms share common shocks, firms repeat across dates, and forward-return
windows can overlap.

Empirical Bayes directly models:

\[
\text{observed monthly estimate}
=
\text{underlying monthly effect}
+
\text{sampling uncertainty}.
\]

It then estimates the population distribution of the completed historical
effects and uses that distribution to stabilize calibration.

This is a narrower and more defensible first assumption than asserting that the
coefficient follows a particular latent transition law through time.

Recursive least squares would introduce an exponential forgetting rule. A
Kalman filter would introduce a state-transition equation and process-noise
parameter. These may later be useful as dynamic benchmarks, but the current
evidence establishes noisy historical calibration and upper-tail behavior, not a
well-identified dynamic law.

---

## Common point-in-time data contract

Let:

\[
X_{i,t}
\]

denote the cross-sectionally standardized SEC similarity-times-exposure signal
for stock \(i\) at rebalance date \(d_t\).

For horizon \(H\in\{21,63\}\), define:

\[
R_{i,t}^{(H)}
=
\frac{P_{i,e_{t,H}}}{P_{i,d_t}}-1,
\]

where \(e_{t,H}\) is exactly \(H\) trading days after \(d_t\).

Use the cross-sectionally demeaned return as the primary calibration target:

\[
Y_{i,t}^{(H)}
=
R_{i,t}^{(H)}
-
\overline R_t^{(H)}.
\]

This is an expected active-return target. It is appropriate for a fully
invested cross-sectional optimizer because each forecast vector is centered
relative to the current equal-weight universe.

Estimate 21-day and 63-day models separately.

Do not pool across:

- horizons;
- different return targets;
- raw and sector-neutral signal variants;
- materially different signal transformations.

---

## Strict completed-outcome rule

At forecast date \(d_T\), historical rebalance date \(d_t\) is eligible only
when:

\[
e_{t,H}<d_T.
\]

The strict inequality is intentional.

A target whose end date equals the current forecast date is excluded.

For each horizon:

\[
\mathcal W_{T,H}
=
\{t:e_{t,H}<d_T\}.
\]

Use an expanding training window containing all eligible historical periods.

Default minimum:

\[
|\mathcal W_{T,H}|\geq 12.
\]

Before the threshold is reached, the forecast row remains `NaN`.

Do not backfill warm-up periods using full-sample or future estimates.

The minimum of 12 periods is an implementation threshold, not a claim that 12
periods provide strong statistical identification.

---

## Shared normal-normal empirical-Bayes model

Each completed historical period produces:

\[
\widehat\theta_t
\]

with estimated sampling variance:

\[
s_t^2.
\]

For regression:

\[
\widehat\theta_t=\widehat\beta_{t,H}.
\]

For Q5 premium:

\[
\widehat\theta_t=\widehat\delta_{t,H}.
\]

Use:

\[
\widehat\theta_t\mid\theta_t
\sim
\mathcal N(\theta_t,s_t^2),
\]

and:

\[
\theta_t
\sim
\mathcal N(\mu,\tau^2).
\]

For the first implementation, estimate:

\[
\widehat\tau^2
=
\max\left[
0,\,
\operatorname{Var}(\widehat\theta_t)
-
\operatorname{mean}(s_t^2)
\right].
\]

Define random-effects precision weights:

\[
a_t
=
\frac{1}{\widehat\tau^2+s_t^2}.
\]

Estimate the prior mean as:

\[
\widehat\mu
=
\frac{\sum_t a_t\widehat\theta_t}
{\sum_t a_t}.
\]

For a completed historical period, its posterior mean is:

\[
\widetilde\theta_t
=
B_t\widehat\theta_t
+
(1-B_t)\widehat\mu,
\]

where:

\[
B_t
=
\frac{\widehat\tau^2}
{\widehat\tau^2+s_t^2}.
\]

A noisier completed estimate is pulled more strongly toward the estimated prior
mean.

This is empirical Bayes because \(\mu\) and \(\tau^2\) are estimated from the
observed collection of completed historical estimates.

---

## Critical forecasting distinction

At a new rebalance date \(d_T\), there is no contemporaneous:

\[
\widehat\theta_T,
\]

because the corresponding future return has not occurred.

Therefore, the point-in-time forecast is **not** a posterior update for period
\(T\).

The valid forecast is:

\[
\widehat\theta_{T\mid T^-}
=
\widehat\mu_{T,H},
\]

where the prior mean is estimated only from completed historical periods in
\(\mathcal W_{T,H}\).

Historical posterior estimates are diagnostic outputs. They must not be applied
backward to generate an apparently point-in-time forecast for the same period
whose future return was used to calculate the direct estimate.

In particular, the forecast must not use:

\[
\widetilde\beta_T
\]

or:

\[
\widetilde\delta_T,
\]

because neither exists before the current period's future return is observed.

This is the key distinction between:

- shrinking completed historical measurements; and
- forecasting a new, unobserved period.

---

## Primary estimator: shrunk univariate regression

For each completed rebalance date, estimate:

\[
Y_{i,t}^{(H)}
=
\beta_{t,H}X_{i,t}
+
\epsilon_{i,t,H}.
\]

Because both \(X\) and \(Y\) are cross-sectionally centered, the primary
specification has no intercept.

The monthly coefficient is:

\[
\widehat\beta_{t,H}
=
\frac{
\sum_i X_{i,t}Y_{i,t}^{(H)}
}{
\sum_i X_{i,t}^2
}.
\]

Estimate an HC1 heteroskedasticity-robust standard error:

\[
s_{\beta,t,H}.
\]

The monthly-estimate table should retain:

- rebalance date;
- target-end date;
- horizon;
- coefficient;
- robust standard error;
- stock count;
- residual variance;
- \(R^2\).

At forecast date \(d_T\), estimate the empirical-Bayes prior using only completed
historical coefficients.

The raw forecast coefficient is:

\[
\widehat\beta_{T,H}^{\mathrm{raw}}
=
\widehat\mu_{\beta,T,H}.
\]

The project calibrates an existing positive-ranking signal rather than allowing
the calibration layer to reverse its meaning.

Therefore:

\[
\widehat\beta_{T,H}^{\mathrm{applied}}
=
\max
\left(
0,\widehat\beta_{T,H}^{\mathrm{raw}}
\right).
\]

If the completed historical evidence is nonpositive, the signal is turned off
rather than inverted.

The stock-level expected active return is:

\[
\widehat Y_{i,T}^{(H,\mathrm{reg})}
=
\widehat\beta_{T,H}^{\mathrm{applied}}
X_{i,T}.
\]

Because \(X_{i,T}\) is centered, this vector has cross-sectional mean zero.

Its units are estimated 21-day or 63-day active returns, not arbitrary signal
scores.

---

## Interpretation of the positive-scale floor

The floor at zero is a project design rule, not a general empirical-Bayes
theorem.

It encodes the prior economic decision that:

- larger SEC similarity-times-exposure scores should not be deliberately
  inverted;
- insufficient or negative historical calibration should deactivate the
  factor;
- a contrarian interpretation would require a separate model-selection
  decision.

Retain both:

```text
prior_mean_raw
applied_scale
```

A negative raw estimate must not be discarded or overwritten.

Later analysis should report how often the floor is active. Frequent activation
would indicate instability in the selected factor or a violation of the
expanding-window comparability assumption.

---

## Second estimator: Q5-premium empirical Bayes

Use the same ranking and `qcut` convention as the existing diagnostics.

Let:

\[
I_{i,t}
=
\mathbf 1\{i\in Q5_t\}.
\]

For each completed period, estimate:

\[
\widehat\delta_{t,H}
=
\overline Y_{Q5,t}^{(H)}
-
\overline Y_{\neg Q5,t}^{(H)}.
\]

Use the Welch sampling variance:

\[
s_{\delta,t,H}^2
=
\frac{s_{Q5,t}^2}{n_{Q5,t}}
+
\frac{s_{\neg Q5,t}^2}{n_{\neg Q5,t}}.
\]

Fit the same normal-normal empirical-Bayes model over completed historical
premiums.

At forecast date \(d_T\):

\[
\widehat\delta_{T,H}^{\mathrm{raw}}
=
\widehat\mu_{\delta,T,H},
\]

and:

\[
\widehat\delta_{T,H}^{\mathrm{applied}}
=
\max
\left(
0,\widehat\delta_{T,H}^{\mathrm{raw}}
\right).
\]

Construct:

\[
u_{i,T}
=
\widehat\delta_{T,H}^{\mathrm{applied}}I_{i,T}.
\]

Benchmark-center the vector:

\[
\widehat Y_{i,T}^{(H,Q5)}
=
u_{i,T}
-
\frac{1}{N_T}\sum_j u_{j,T}.
\]

Equivalently, if \(p_T\) is the fraction assigned to Q5:

\[
\widehat Y_{i,T}^{(H,Q5)}
=
\widehat\delta_{T,H}^{\mathrm{applied}}
\left(
I_{i,T}-p_T
\right).
\]

This produces:

- positive estimated active returns for Q5;
- small negative active-return estimates for non-Q5;
- exactly zero cross-sectional mean.

The Q5 estimator directly tests the diagnostic finding that the strongest
predictive effect is concentrated in the upper signal quintile.

---

## What HC1 does and does not solve

HC1 addresses cross-sectional heteroskedasticity within each completed monthly
regression.

It does not fully resolve:

- common shocks across stocks;
- repeated observations of the same firms;
- overlapping 21-day or 63-day outcomes;
- serial dependence among monthly estimates;
- factor-selection bias.

HC1 is nevertheless a transparent first implementation because the
empirical-Bayes layer requires a relative measure of each monthly estimate's
sampling uncertainty.

Before strong paper-level inferential claims are made, later robustness work
should consider a time-block procedure applied to the sequence of monthly
estimates, such as:

- contiguous-date bootstrap;
- horizon-aware block bootstrap;
- another documented dependence-robust sensitivity analysis.

This later robustness work should not silently replace the baseline estimator.

---

## Prior-variance boundary behavior

The method-of-moments estimate may produce:

\[
\operatorname{Var}(\widehat\theta_t)
\leq
\operatorname{mean}(s_t^2),
\]

which gives:

\[
\widehat\tau^2=0.
\]

This is not necessarily a numerical error. It means that the observed dispersion
of completed monthly estimates is no greater than the dispersion attributed to
estimated sampling uncertainty.

When this happens:

- historical effects fully pool toward the prior mean;
- historical shrinkage weights become zero;
- the boundary condition must be recorded;
- the implementation must not create a negative variance.

Prior-collapse periods must remain auditable in calibration history or metadata.

---

## Expanding-window assumption

The baseline expanding window assumes that all completed historical periods
remain relevant to the current forecast.

This is an exchangeability or comparability assumption. It is weaker than a
fully specified state-transition model, but it is still an assumption.

Potential violations include:

- structural changes in the AI investment cycle;
- changes in disclosure practices;
- changes in market volatility;
- changes in universe composition;
- changes in the economic meaning of the theme.

Do not replace the expanding-window baseline before it is implemented and
validated.

Afterward, rolling-window or time-decayed variants may be introduced as
predeclared robustness comparisons. They should not be repeatedly tuned using
final portfolio performance.

---

## Factor-selection bias

The factor-selection audit explains why `sec_similarity_x_exposure` was selected
among the candidate factors.

It does not create untouched out-of-sample evidence.

Empirical Bayes reduces calibration noise, but it does not correct for selecting
the factor after comparing multiple diagnostics and candidate factors.

The following should remain frozen for the primary analysis:

- primary signal definition;
- 21-day and 63-day horizons;
- active-return target;
- expanding-window baseline;
- minimum completed-period rule;
- regression EB as the primary estimator;
- Q5-premium EB as the required alternative;
- method-of-moments shrinkage;
- positive-scale floor.

Later changes made after inspecting final results must be labeled exploratory or
sensitivity analysis.

---

## Sector-neutral robustness

The primary target is the cross-sectionally demeaned active return.

This is not the same as a sector-neutral target.

The explicitly sector-neutral signal or target should remain a separate
robustness specification. Do not combine raw and sector-neutral monthly estimates
in one empirical prior.

Each materially different signal or target definition represents a different
shrinkage population and should be calibrated separately.

---

## Required audit trail

The result bundle should preserve:

```text
estimated_returns
monthly_estimates
calibration_history
metadata
```

The direct completed-period measurements, historical posterior diagnostics, and
point-in-time forecasts must remain distinct.

At minimum, the audit trail should make it possible to reconstruct:

- the supplied signal panel and name;
- the horizon;
- the target definition;
- the forecast date;
- the eligible completed periods;
- the last completed target-end date;
- the number of training periods;
- the monthly direct estimates;
- their standard errors;
- the estimated prior mean;
- the estimated prior variance;
- the raw forecast scale;
- the applied nonnegative scale;
- the resulting stock-level forecast vector.

`calibration_history` is the main point-in-time audit trail.

---

## Why RLS and Kalman filtering remain later benchmarks

Recursive least squares primarily answers:

> How should older observations lose influence through time?

A forgetting-factor RLS model imposes exponential decay:

\[
w_j=\lambda^j.
\]

A Kalman filter primarily answers:

> How does an unobserved coefficient evolve from one date to the next?

A local-level model would impose:

\[
\beta_t=\beta_{t-1}+w_t.
\]

Both require additional temporal assumptions and tuning or estimation.

The current empirical-Bayes model answers the narrower question:

> Given completed, noisy historical effect estimates, what population-level
> effect is supported after accounting for their estimated precision?

That question matches the current evidence and provides a transparent baseline.

RLS or a local-level Kalman filter may be implemented later only after:

1. direct monthly estimates have been inspected;
2. the empirical-Bayes panels pass all point-in-time tests;
3. there is evidence of temporal dynamics after accounting for sampling
   uncertainty;
4. dynamic benchmarks can be evaluated without changing the primary SEC signal.

A nonlinear Kalman filter should not be introduced unless a specific nonlinear
state or measurement equation is independently motivated.

---

## Interpretation for the paper

The selected factor may be described as:

> A point-in-time interaction between filing-summary theme similarity and
> LLM-assessed exposure materiality, selected because it exhibited the strongest
> overall combination of cross-sectional rank information, upper-quintile return
> separation, incremental value over static baselines, and persistence after
> sector controls.

The empirical-Bayes layer may be described as:

> A point-in-time calibration procedure that estimates the historical
> population mean and dispersion of noisy monthly signal-return relationships
> and converts the selected SEC filing signal into 21-day and 63-day expected
> active-return panels.

The Q5-premium estimator may be described as:

> A complementary empirical-Bayes calibration designed to test whether
> predictive separation is concentrated in the upper signal quintile rather
> than following a uniformly linear cross-sectional relationship.

Do not claim that empirical Bayes:

- creates the filing signal;
- proves the selected factor is valid;
- eliminates factor-selection bias;
- establishes stationarity;
- corrects all dependence from overlapping outcomes;
- supplies a posterior estimate for an unobserved current-period coefficient;
- replaces untouched walk-forward validation;
- establishes that the signal-return relationship is fully linear.

The main research contribution remains the point-in-time SEC filing signal and
its predictive evidence. Empirical Bayes is a supporting calibration and
noise-reduction layer.

---

## Immediate implementation boundary

Implement only:

```text
src/empirical_bayes.py
tests/test_empirical_bayes.py
```

The module should:

- build forward active-return targets;
- enforce exact trading-day horizons;
- enforce the strict completed-outcome rule;
- estimate completed monthly regression slopes;
- estimate completed monthly Q5 premiums;
- estimate the shared normal-normal empirical-Bayes prior;
- produce point-in-time regression and Q5 estimated-return panels;
- preserve the full audit trail;
- save the agreed result bundle.

Do not yet:

- integrate the optimizer;
- run the backtest;
- add full five-quantile empirical Bayes;
- add RLS;
- add a Kalman filter;
- alter the SEC pipeline;
- rerun the LLM extraction process.

The next objective is to implement the agreed baseline correctly and verify that
every forecast can be reproduced using only information available at its
forecast date.
