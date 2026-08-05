
# AUTHORITATIVE PROJECT DECISIONS

Date: 2026-08-04

## Purpose

This document resolves conflicts among older project chats, July design notes, August methodology documents, and the latest accepted revision package.

It does not replace the executable repository or saved empirical outputs. It identifies which methodological and notational decisions are currently authoritative.

## Source hierarchy

When project materials conflict, use the following order:

1. This authoritative-decisions document.
2. `notes_080226_revision_inline_latex_fixed.txt`.
3. Executable repository state and saved result files for claims about what has actually been implemented or produced.
4. `notes_080226.docx`.
5. `empirical_bayes_design_notes.md`.
6. `referee_review.md`, which is advisory rather than automatically binding.
7. July design notes and older project chats.
8. Reference books and papers.

Code and saved outputs determine what currently exists.

The latest accepted methodology documents determine what should be implemented next.

Older notation and rejected methodological branches must not be revived when they conflict with this document.

## Paper framing

The paper is primarily a methodology-and-measurement paper.

Its principal contribution is a point-in-time and auditable process that converts SEC-filing thematic measurements into expected active returns and constrained portfolio weights.

The AI-infrastructure application is an empirical demonstration over a short 2023–2026 sample.

The paper does not claim to establish a permanent new alpha premium.

## Frozen signal decision

The selected factor is:

`sec_similarity_x_exposure`

The stored alpha-panel key is:

`similarity_x_exposure_score_alpha`

The documented candidate set contains nine raw factors.

The factor search is frozen.

Do not add new SEC factors, new horizons, opaque composite scores, LightGBM models, neural models, or alternative thematic taxonomies before completing the current paper.

## Return horizons

Estimate the 21-trading-day and 63-trading-day horizons separately.

Do not pool the horizons.

## Canonical notation

$$i$$ is the stock index.

$$k$$ is the filing index.

$$s$$ is a completed historical rebalance date.

$$t$$ is the current forecast date.

$$h$$ is the fixed forward-return horizon.

$$e_h(s)$$ is the trading date on which the forward return beginning at $$s$$ is complete.

The completed historical estimation set is:

$$W_h(t)=\left\{s:e_h(s)<t\right\}.$$

$$z_{i,t}$$ is the standardized SEC signal for stock $$i$$ at forecast date $$t$$.

$$y_{i,s}^{(h)}$$ is the realized forward active return for stock $$i$$ beginning at completed historical date $$s$$.

The direct historical cross-sectional slope is:

$$\widehat{\beta}_s^{(h)}.$$

Its estimated sampling variance is:

$$
v_s^{(h)}.
$$

The estimated between-date variance is:

$$
\widehat{\tau}_{\beta,t}^{2(h)}.
$$

The estimated population mean slope is:

$$
\widehat{\mu}_{\beta,t}^{(h)}.
$$

The empirical-Bayes estimate of the latent historical slope belonging to completed date $$s$$, using information through date $$t$$, is:


$$
\widehat{\beta}_{s\mid t}^{EB,(h)}=B_{s,t}^{(h)}\widehat{\beta}_s^{(h)}+\left(1-B_{s,t}^{(h)}\right)\widehat{\mu}_{\beta,t}^{(h)}.
$$

Do not replace this with a tilde notation.

Do not introduce $$\widehat b_t$$ or another alias for $$\widehat{\mu}_{\beta,t}^{(h)}$$.

The unrestricted stock-level expected active return is:

$$
\widehat{\mu}_{i,t}^{(h)}=\widehat{\mu}_{\beta,t}^{(h)}z_{i,t}.
$$

## Current-period forecast interpretation

At date $$t$$, the direct slope $$\widehat{\beta}_t^{(h)}$$ is unavailable because the forward return beginning at $$t$$ has not yet been observed.

Under the exchangeable random-effects model, the predicted slope for the new period is:

$$
\widehat{\mu}_{\beta,t}^{(h)}.
$$

The retrospectively shrunk historical slopes

$$
\widehat{\beta}_{s\mid t}^{EB,(h)}
$$

 are substantive empirical-Bayes estimates of completed historical effects. They are not forecasts that were available at historical date $$s$$.

## Primary empirical-Bayes estimator

Ordinary inverse-total-variance random-effects empirical Bayes is the primary estimator.

The baseline weight is:

$$
a_{s,t}^{(h)}=\frac{1}{\widehat{\tau}_{\beta,t}^{2(h)}+v_s^{(h)}}.
$$

The estimated population mean is:

$$
\widehat{\mu}_{\beta,t}^{(h)}=\frac{\sum_{s\in W_h(t)}a_{s,t}^{(h)}\widehat{\beta}_s^{(h)}}{\sum_{s\in W_h(t)}a_{s,t}^{(h)}}.
$$

Do not replace the between-date sample variance with a Newey-West long-run variance.

## Zero-floor decision

The unrestricted empirical-Bayes forecast is the primary statistical estimate.

The optional nonnegative slope is:

$$
\widehat{\mu}_{\beta,t}^{+(h)}=\max\left\{0,\widehat{\mu}_{\beta,t}^{(h)}\right\}.
$$

The corresponding stock-level portfolio sensitivity is:

$$
\widehat{\mu}_{i,t}^{+(h)}=\widehat{\mu}_{\beta,t}^{+(h)}z_{i,t}.
$$

The zero floor is not part of empirical Bayes.

Save and report both the unrestricted and zero-floor versions.

## Overlap framework

The completed-outcome rule prevents unfinished future labels from entering estimation.

It does not make completed historical outcomes independent.

Concurrency and average uniqueness quantify redundancy among overlapping completed labels.

Ordinary empirical Bayes remains the primary estimator.

Uniqueness-weighted empirical Bayes is a sensitivity analysis inspired by López de Prado. It is not a standard empirical-Bayes result.

Use moving-block or stationary-block bootstrap methods for dependence-aware uncertainty.

Use verified non-overlapping 63-day sequences as an additional robustness analysis.

Do not use sequential bootstrap as the primary confidence-interval procedure.

## Accepted implementation order

Phase 1: Focused signal contract and measurement audits.

Phase 2: Dependence-aware primary signal inference and search control.

Phase 3: Core empirical-Bayes implementation and tests.

Phase 4: Concurrency, uniqueness, block-bootstrap, and non-overlapping-label robustness.

Phase 5: Point-in-time calibration validation.

Phase 6: Optimizer integration and final paper exhibits.

## Deferred work

Do not implement before the current paper deadline:

1. New SEC signal discovery.
2. Additional return horizons.
3. LightGBM or neural return models.
4. Kalman, RLS, AR, MA, regime-switching, or state-space slope models.
5. Full five-quantile empirical Bayes as the primary estimator.
6. A complete anonymized OpenAI rerun.
7. Open-weight LLM replication.
8. Ontology or knowledge-graph expansion.
9. Large optimizer redesign before calibration is complete.
