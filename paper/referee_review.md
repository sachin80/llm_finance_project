# Referee-Style Review: LLM-Derived SEC Thematic Exposure → Calibrated Expected Returns → MVO

**Reviewed:** methodology document dated 07/25/26
**Target:** SSRN posting, end of August 2026
**Reviewer stance:** first-principles, adversarial-but-constructive. I am reading this the way a discussant at a quant conference or a referee at *JFDS / Journal of Investment Management / Financial Analysts Journal* would read it.

---

## 0. Bottom line up front

**The pipeline is well-engineered and the point-in-time discipline is better than most work in this space. The empirical claim it currently makes is not supported by the sample you have.**

Three sentences that summarize everything below:

1. **Your headline results are almost certainly statistically insignificant** once you account for 63-day overlapping returns, 36 rebalance dates, and selection over nine candidate factors. My estimate of your true |t| on the 63D IC is **1.2–2.4**, against a bar of 2.8–3.3.
2. **You have not addressed LLM look-ahead bias**, which is the single most likely reason a referee rejects an LLM-in-finance paper in 2026. Asking a 2025/2026-vintage model to score "AI infrastructure exposure" from a 2023 10-K means the model already knows who won.
3. **Your framing is wrong for what you actually have.** You are writing a factor-discovery paper. You have the sample size for a *methodology and measurement* paper. Change the frame and the paper becomes publishable; keep the frame and it gets shot down on power alone.

There is a real paper here. It needs a different set of claims, roughly six specific robustness exhibits, and one genuinely novel theoretical observation that you are currently sitting on without noticing (see §4.3).

---

## 1. What is genuinely good — do not lose these in the rewrite

These are your defensible assets. Lead with them.

**1.1 Point-in-time construction with an explicit availability lag.** `d_f + Δ ≤ t` with forward-fill is correct and many published papers get this wrong. Say so explicitly, and report the *distribution of filing age at rebalance* (see §5.3).

**1.2 The feature was engineered before returns were observed.** This matters and you should make more of it. But be precise: the *feature* was pre-specified; the *choice among nine* was not. Those are different, and conflating them will get flagged. Write one paragraph that draws the line honestly.

**1.3 The economic logic of the interaction is sound.** `similarity × materiality` is the right shape. Generic AI hand-waving in a 10-K should not score unless the filing also concedes material economic exposure. This is a real idea, not a data-mined one.

**1.4 Freezing the factor.** Rare, credible, and worth a dedicated subsection ("Pre-specification and factor freeze"). This is the closest thing finance has to pre-registration. Report the freeze date.

**1.5 You report the sector-neutral number even though it is much worse.** 6.49% → 2.84% is a painful disclosure and you made it anyway. Keep doing that. It buys enormous credibility.

**1.6 The estimation layer is a real gap in the applied literature.** Most factor papers stop at the ranking and hand-wave the step to `μ`. You are formalizing it. That is the actual contribution — see §7.

---

## 2. Threat #1 (existential): LLM look-ahead bias

**This must be the first robustness section in the paper. Not an appendix.**

Your pipeline sends 10-K/10-Q text to an OpenAI model and asks it to (a) summarize and (b) assign an exposure materiality score. If that model's pretraining corpus extends past your sample dates — and for a 2023–2026 sample it almost certainly does — then the model is not reasoning from the filing. It is partly recalling that NVIDIA, Vertiv, Broadcom, and Super Micro subsequently became AI-infrastructure winners.

This is a well-established and actively-studied failure mode. <cite index="9-1">Glasserman and Lin distinguish two forms: look-ahead bias, in which the model has specific knowledge of the returns that followed an event, and a "distraction effect," in which the model's general knowledge of the named company interferes with its reading of the text</cite>. <cite index="2-1">Leakage of post-event information from the pretraining corpus inflates estimated forecast performance and leads to disappointing out-of-sample results</cite>.

Note the distraction effect is *also* a problem for you, and it cuts differently: <cite index="9-1">in-sample, Glasserman and Lin find anonymized headlines actually outperform, indicating distraction mattered more than look-ahead, and the effect was strongest for large companies about which the model has more general knowledge</cite>. Your universe is ~499 large caps. This applies directly.

### What you must do

**2.1 Anonymization ablation (mandatory).** Re-run the summarization and exposure-scoring stages with company identifiers stripped from the filing text — name, ticker, subsidiary names, executive names, headquarters city, and any distinctive product names. <cite index="4-1">The reasoning is that if a prompt contains no identifying information, the model's generation cannot correlate with future events tied to that identifier</cite>. Report IC and Q5-Q1 for anonymized vs. non-anonymized. **If the signal is materially weaker anonymized, you have found leakage and must report it.** If it survives, you have a much stronger paper than you have now.

Practical note: full anonymization of a 10-K is hard (a filing describing "our GPU data center interconnect business" is self-identifying). Do your best, document the procedure, and treat it as a lower bound on leakage.

**2.2 Model-level recall diagnostic.** There is a portable test built on <cite index="5-1">a date-only recall query containing the firm's name, ticker, and target date and nothing else</cite> — you ask the model directly for the outcome with no filing context and measure how well its answers predict realized returns. That gives you a memorization proxy to subtract or at least report alongside. A related approach <cite index="7-1">prompts LLMs to retrieve historical stock returns without context to construct a proxy for memorization-driven predictability, and finds look-ahead bias is more severe at lower frequencies and for larger models</cite>. 63-day horizons + large models + mega-cap universe is exactly the worst corner of that grid for you.

**2.3 Post-cutoff subsample.** State the knowledge cutoff of the specific model version you used. Split your 36 dates into pre-cutoff and post-cutoff. Report both. This subsample will be tiny and underpowered — say so — but not reporting it is worse.

**2.4 Consider an open-weight replication.** Run the whole thing with an open model whose cutoff you can pin (and which you can cite by exact checkpoint). This doubles as a **reproducibility** fix: an OpenAI model version gets deprecated and your paper becomes literally unreproducible. Pin `model=`, `temperature=`, `seed=`, prompt text verbatim in an appendix, and archive the raw JSON outputs.

**2.5 Where did the theme text come from?** If you wrote the AI-infrastructure theme description in 2025/2026 — mentioning HBM, liquid cooling, power density, networking — you embedded hindsight into the *target vector itself*, independent of the LLM. State when the theme text was written, by whom, and report sensitivity to at least three alternative wordings (a naive one, yours, and one written from a 2022 vantage point).

---

## 3. Threat #2: statistical power

![Figure 1](fig1_power.png)

### 3.1 The overlap problem is not a footnote

You sample signals monthly and measure 63-trading-day forward returns. Consecutive observations share ~2/3 and ~1/3 of their return windows. This induces a mechanical MA(2) structure with ρ₁ = 2/3, ρ₂ = 1/3, giving a variance inflation factor of **2.93** and an **SE inflation of 1.71×**.

Your effective independent sample is not 36. It is roughly 12 — and after the 12-period EB warm-up and the incomplete-outcome tail, your *live out-of-sample* forecast record is about **7 independent 63-day blocks.**

![Figure 2](fig2_sample.png)

### 3.2 What your t-stat probably is

You report a mean rank IC of 0.0703 but **never report its dispersion**, which means the number as printed is uninterpretable. Working backwards:

| assumed monthly IC vol | naive t | overlap-adjusted t |
|---|---|---|
| 0.05 | 8.44 | 4.93 |
| 0.08 | 5.27 | 3.08 |
| 0.10 | 4.22 | 2.47 |
| **0.12** | **3.52** | **2.05** |
| **0.15** | **2.81** | **1.64** |
| 0.20 | 2.11 | 1.23 |

For a single thematic factor over 2023–2026 with N≈499, monthly IC volatility of 0.10–0.16 is the realistic range. **Your adjusted t is most likely between 1.2 and 2.5.**

### 3.3 What to report instead

Replace the bare "0.07030" (drop the spurious precision — four decimals on an IC estimated from 12 independent observations is not credible) with:

- mean IC, **IC standard deviation**, IC information ratio
- **Newey–West t-statistic**, lag = 3 for 63D (or Hansen–Hodrick / Hodrick 1992 standard errors, which are better-behaved for overlapping returns)
- % of months with positive IC, and a sign test
- a **block bootstrap** confidence interval with block length ≥ 3 months
- the same set for the Q5-Q1 spread and the sector-neutral spread

Also fix the annualization. 2.84% per 63D naively annualizes to 11.4%/yr, but you should never annualize a 12-observation mean without also showing the confidence band, which will span zero.

---

## 4. Threat #3: multiple testing and the meaning of "frozen"

![Figure 3](fig3_multiple_testing.png)

You selected `sec_similarity_x_exposure` as the winner from seven SEC-derived factors and two baselines. That is nine explicit trials — Bonferroni threshold **|t| = 2.77**.

But the honest trial count is much larger. Every one of these was a choice made with some visibility into results:

- embedding model
- summarization prompt design and output schema
- exposure materiality scale and its numeric encoding
- interaction functional form (product vs. sum vs. conditional gate)
- availability lag Δ
- **horizon** — you write that 63D is primary "because the current diagnostics are materially stronger at that horizon." That is horizon selection on the outcome. Count it.

Realistically 25–100 trials → required |t| of **3.0–3.5**, which coincides with the Harvey–Liu–Zhu bar for a newly proposed factor.

**Freezing after the selection does not undo the selection.** It prevents *future* mining, which is valuable and worth stating, but a referee will make exactly this point and you should make it first.

### What to do

- Publish the **full grid of all nine factors** with IC, t-stat, and spread. Transparency here is far more persuasive than a clean single result.
- Apply an explicit haircut (Harvey–Liu "Lucky Factors" bootstrap, or a deflated-Sharpe-style adjustment).
- Add a **candidate-count disclosure paragraph**: "we estimate the effective number of specifications searched at approximately N; under a Bonferroni adjustment this implies a required |t| of X."

---

## 5. Threat #4: is this just a levered semiconductor bet in the AI boom?

### 5.1 The sector decomposition already tells you the answer is partly yes

![Figure 5](fig5_spread.png)

**56% of your headline spread is a sector tilt.** That single number is the most informative statistic in the document, and it needs to be in the abstract.

### 5.2 Regime confounding

Your 36 dates span roughly mid-2023 to mid-2026 — that is, the entire sample is the AI capex boom. You are testing an AI-infrastructure exposure factor exclusively during the period in which AI-infrastructure exposure paid. There is no regime in which it did not pay, so you cannot say anything about whether the factor works or whether you have measured a single realized theme return.

This is worth engaging with directly, because there is a specific literature saying thematic exposure is *negatively* predictive on average: <cite index="16-1">Ben-David et al. find specialized ETFs tend to underperform over their first five years, roughly a 30% risk-adjusted loss, linked to overvaluation of underlying stocks at launch, with providers catering to extrapolative beliefs by issuing ETFs focused on attention-grabbing themes</cite>. Your paper is, structurally, an argument that a thematic tilt earns positive alpha. You need to say why yours differs — the honest answer is probably "our sample is the boom" and you should say that.

### 5.3 The baselines you compare against are too weak

Two static Yahoo-description baselines is not a serious control. A referee will ask for these four, and the first two are the ones that can kill the paper:

1. **Keyword count.** Frequency of "artificial intelligence" / "machine learning" / "GPU" / "accelerated computing" in the 10-K, standardized. **If your LLM pipeline does not clearly beat grep, the entire apparatus is unjustified.** This is the ablation that determines whether the paper is interesting.
2. **GICS sub-industry dummy.** Long semiconductors + semiconductor equipment, short everything else, same weighting scheme. If your signal does not beat this, you have an expensive industry classifier.
3. **Factor-model alpha.** Regress the Q5-Q1 series on FF5 + MOM, and separately on FF5 + MOM + an AI-theme proxy (an AI thematic ETF or an equal-weight basket of the obvious names). Report alpha and t-stat. Do this for both the raw and sector-neutral legs.
4. **Standard alternatives.** Momentum, analyst revisions, and a simple size/value control, to show the signal is not a repackaging.

There is also directly-adjacent prior work you must cite and differentiate from — for example, work <cite index="16-1">constructing objective, data-driven AI stock indices from 10-K filings, motivated by the observation that AI thematic index selection criteria are often opaque and rely on vague language about companies that "could benefit" from AI adoption</cite>. Your differentiator is the *materiality gate* and the *point-in-time forward-fill*; make that explicit rather than letting a referee discover the overlap.

---

## 6. Methodological issues in the estimation layer

This is the part you asked about most directly, so I'm going deeper here.

### 6.1 The empirical Bayes model assumes independence across dates. Overlapping returns violate it.

Your variance decomposition is
$$\operatorname{Var}(\hat\beta_s) = \tau^2 + \mathbb{E}[s_s^2]$$
and you estimate $\hat\tau^2 = \max(0,\ \widehat{\operatorname{Var}}(\hat\beta) - \overline{s^2})$. This is the DerSimonian–Laird estimator, and it requires the $\hat\beta_s$ to be independent draws.

With 63-day horizons sampled monthly, $\operatorname{Cov}(\hat\beta_s, \hat\beta_{s+1}) \neq 0$ — consecutive slopes are estimated on overlapping return windows and share realized shocks. Two consequences:

- **The sample variance of $\hat\beta$ is contaminated by the overlap covariance**, so $\hat\tau^2$ is biased. Direction depends on the sign of the induced autocovariance; in a trending theme it will typically be *inflated*, which means **you are shrinking too little**.
- $\overline{s^2}$ is a within-date HC1 quantity that does not capture the cross-date dependence at all, so the two terms in your decomposition are not on the same footing.

**Fixes, in order of effort:**
- *Cheapest:* estimate the EB layer on **non-overlapping 63D dates only** (every third month). You lose 2/3 of the observations but the model is then correctly specified. Report both.
- *Better:* replace $\widehat{\operatorname{Var}}(\hat\beta)$ with a Newey–West / HAC long-run variance of the $\hat\beta_s$ series, then subtract $\overline{s^2}$.
- *Best:* a small hierarchical model with an AR or MA(2) structure on $\beta_s$. Given 36 observations this may be over-ambitious, but even stating it as the correct specification and explaining why you approximate is good practice.

Also: with $T_{\text{eff}} \approx 12$, $\hat\tau^2$ is extremely imprecise and the $\max(0,\cdot)$ floor will bind often. Report how frequently it binds, and add a robustness check with a Paule–Mandel or REML estimator, which handle the boundary better.

### 6.2 Shrinkage mechanics — build this figure, it's a good one

![Figure 4](fig4_shrinkage.png)

The document explains shrinkage well in prose but has no visual. Panel A (weight vs. noise ratio) and Panel B (the shrinkage "spaghetti" plot) are standard, immediately legible, and will make the EB section land. Build the real version from your data.

### 6.3 **The observation you are sitting on: the EB layer is a leverage knob under unconstrained MVO**

This is the most important technical point in the review, and it is also your best opportunity.

Your linear estimator produces $\hat\mu_{i,t} = \bar b_t \cdot z_{i,t}$. Under unconstrained mean-variance with a volatility target:

$$w \propto \Sigma^{-1}\hat\mu = \bar b_t\,\Sigma^{-1} z \quad\Longrightarrow\quad \frac{w}{\sigma_{\text{target}}}\ \text{is independent of } \bar b_t$$

**The scalar cancels exactly.** Every piece of empirical Bayes machinery in Part 2 — the slopes, the shrinkage, the precision weighting, the zero floor — produces a portfolio *identical* to the one you would get by feeding raw $z$ into the optimizer.

![Figure 7](fig7_optimizer.png)

Panel A shows three wildly different $\bar b$ values producing one indistinguishable weight curve. Panel B adds a 2% position box and a gross-exposure cap and the curves separate.

**Do not hide this. Make it a result.** The correct framing:

> "We show that under unconstrained mean-variance optimization with a volatility target, the calibration scalar cancels and the empirical Bayes layer is observationally equivalent to feeding the standardized signal directly. Calibration becomes economically material only when (i) position or gross-leverage constraints bind, (ii) a turnover penalty is present, (iii) risk aversion is fixed rather than vol-targeted, or (iv) the signal is blended with other alpha sources, where relative units determine relative weight."

Point (iv) is your strongest justification for the whole exercise and it is currently absent from the document. **Cardinal calibration exists so you can combine signals.** Say that.

Then demonstrate it: run the optimizer under both a vol target and a realistic constrained specification, and show the weights and realized performance diverge in the second case. That is a genuine, novel, publishable finding, and it converts your weakest section into your most interesting one.

### 6.4 The linear EB forecast cannot improve ranking. Be precise about this.

![Figure 6](fig6_estimators.png)

The document says the estimation layer will "preserve the ranking information while reducing noise." For the linear model, ranking preservation is **trivially true by construction** (monotone transform under $\bar b > 0$), and there is **no cross-sectional noise reduction at all** — the shrinkage operates on the time series of the coefficient, not on the cross-section of stocks. Claiming otherwise will read as confusion.

What is true, and what you should write:
- The linear EB layer changes **units and scale only**.
- Noise reduction happens in the **time series of $\bar b_t$** — it stabilizes the magnitude of your bet across periods.
- The **Q5-premium model is a genuinely different object**: it is a step function, it does change the effective ranking (collapsing within-group ordering), and it encodes a non-linear view. Contrasting the two is a legitimate empirical question.

### 6.5 Smaller estimation-layer points

- **No-intercept regression is fine.** Both $y$ and $z$ are centered by construction, so with-intercept and without-intercept OLS coincide. But say why in one line rather than asserting it.
- **$\hat\beta_s$ is essentially the IC in return units.** Since $z$ has unit cross-sectional variance, $\hat\beta_s \approx \frac{1}{N}\sum_i y_{i}z_{i}$ — the covariance of active return with the standardized signal. This is a nice interpretive bridge between Stage 2 and Stage 3, and it means your IC diagnostics and your slope estimates are two views of the same object. Say so.
- **Outliers.** OLS slopes on a 2023–2026 cross-section containing NVIDIA will be dominated by a handful of names. HC1 fixes the standard error, not the point estimate. Add winsorization at 1%/99% on both $z$ and $y$, and report a Huber-robust or rank-regression version as a robustness row.
- **The zero floor $\bar b_t = \max(0, \hat\mu_\beta)$.** Defensible, but it is an ex-post-motivated sign restriction. Report how often it binds (I suspect never, in which case it is cosmetic — say that) and show results without it.
- **The 12-period warm-up.** State plainly that this leaves ~24 live forecast dates and ~7 independent blocks. The document already flags this with a "(?)" — resolve it in the text rather than leaving the reader to wonder.

---

## 7. Concrete pipeline issues that may be affecting your numbers right now

These are specific, checkable, and two of them could materially change your results.

### 7.1 **10-K vs 10-Q heterogeneity — check this today**

A 10-K contains Item 1 (Business) and full Item 1A (Risk Factors). A 10-Q contains neither in full form. Your summaries — and therefore your cosine similarities — will differ **systematically by form type**. Since you forward-fill "the latest filing," your signal for a given stock **mechanically jumps between a 10-K level and a 10-Q level** four times a year.

That is a source of pure measurement noise, potentially of turnover, and possibly of spurious cross-sectional variation (a firm whose latest filing is a 10-K vs. one whose latest is a 10-Q are not comparable).

**Check:** mean and SD of similarity by form type. If they differ, standardize *within form type* before the cross-sectional z-score, or include a form-type fixed effect.

### 7.2 **Is the interaction really just the exposure score?**

Cosine similarities from modern embedding models are notoriously compressed — typically in a narrow band like [0.72, 0.88]. Your exposure materiality score is ordinal with a much wider relative range. In the product $\text{sim} \times \text{exposure}$, the **exposure term will dominate the variance**, and your "interaction" may be numerically indistinguishable from exposure alone.

**Check:** correlation between `sec_similarity_x_exposure` and `exposure` alone. If ρ > 0.95, your central engineered feature is a relabeled LLM materiality score and the paper's framing must change. You presumably have exposure-alone among your seven factors — report that comparison explicitly.

**Fix if needed:** rank-transform or z-score the similarity within date before multiplying, so both components contribute comparable variance. Report the cross-sectional distribution of raw similarity in an appendix figure.

### 7.3 Filing staleness

With forward-fill, a 10-K-only filer's signal can be ~12 months old. Report the **distribution of filing age at rebalance**, and a robustness cut excluding filings older than 6 months. This also lets you make a *positive* argument you currently miss: because the signal only updates on new filings, **turnover should be very low**, which is a genuine advantage against transaction costs. You measured turnover in Stage 2 but never report the number — do.

### 7.4 Survivorship in the 499

"499 stocks" reads like current S&P 500 constituents minus one. If you took today's index membership and backfilled 36 months, you have severe survivorship *and* look-ahead bias — the index committee added the AI winners *because* they won.

**You need point-in-time index membership**, delisted names included, and an explicit statement of universe construction: source, membership dates, delisting return handling, price adjustment source, and any liquidity/price filters.

### 7.5 Costs and capacity

Figure 5B shows the sector-neutral leg's survival under cost assumptions. You need to fill in the real numbers: annual two-way turnover, cost assumption in bps with justification, borrow cost on the short leg, and a **net** spread. A gross 11.4%/yr sector-neutral number with unreported costs is not an economic claim.

---

## 8. The calibration validation you said you're running — what to actually report

![Figure 8](fig8_calibration.png)

You wrote that you are "testing it to confirm that it produces calibrated expected returns." Here is the specific exhibit set:

**8.1 Mincer–Zarnowitz regression.** Pool all out-of-sample forecasts and estimate
$$y_{i,t+h} = a + c\,\hat\mu_{i,t} + \varepsilon$$
Test $a = 0$ and $c = 1$ jointly, with standard errors clustered by date (essential — cross-sectional residuals are highly correlated within a date). $c < 1$ means over-confident forecasts and you should shrink harder; $c > 1$ means you are leaving magnitude on the table. Panel A above shows the three shapes.

**8.2 Decile calibration plot.** Bin by predicted $\hat\mu$, plot decile mean predicted vs. decile mean realized, with error bars. Overlay the 45° line. This is the single most persuasive calibration exhibit and it takes ten minutes to produce.

**8.3 Out-of-sample $R^2$.** Campbell–Thompson style, against a zero-forecast benchmark. **Expect something like 0.1–0.5%.** That is normal for cross-sectional return forecasting and you should say so preemptively, because an unqualified "$R^2 = 0.003$" invites a dismissive reading.

**8.4 Realized-to-predicted spread ratio.** Compare the predicted Q5-Q1 spread to the realized one. If realized/predicted is systematically below 1, your shrinkage is insufficient.

**8.5 IC term structure.** You only report 21D and 63D. Panel B shows the shape a referee expects. Produce it for {5, 10, 21, 42, 63, 126, 252}. If the IC peaks exactly at your chosen horizon and decays sharply on both sides, that is evidence of horizon selection; if it is a smooth hump, that is evidence of a real slow-moving signal. **This figure is diagnostic of your own selection problem** — which is exactly why you should run it.

**8.6 Rolling $\bar b_t$ plot.** Time series of $\bar b_t$ with a confidence band, plus the count of completed periods on a second axis. Shows the reader the estimator warming up and stabilizing.

---

## 9. Reframing: the change that makes this publishable

You currently write as though the deliverable is "we found a factor." With ~7 independent 63-day observations, that claim cannot be supported and defending it will consume the entire referee exchange.

**Reframe as a measurement-and-methodology paper.** Proposed title direction:

> *From Filing Text to Portfolio Weights: A Leakage-Audited Pipeline for LLM-Derived Thematic Exposure*

Proposed contribution list:

1. A **point-in-time, materiality-gated** construction of thematic exposure from SEC filings, with the availability lag and forward-fill made explicit.
2. A **leakage audit** — anonymization ablation and memorization diagnostics — that we argue should be standard for any LLM-derived signal. (This alone is a contribution; almost nobody does it properly.)
3. A **formalization of the calibration step** from cross-sectional score to expected active return via empirical Bayes, including the result that under unconstrained vol-targeted MVO the calibration scalar cancels, so the layer matters only under binding constraints or multi-signal blending.
4. An **honest, underpowered empirical demonstration** on 2023–2026 US large caps, with full disclosure of sample limitations, sector decomposition, and multiple-testing haircuts.

Every one of those survives a small sample. The current framing does not.

Add a **Limitations** section that says, in your own voice, before a referee says it: 36 dates, ~7 independent blocks, single AI-boom regime, 56% sector tilt, selection over N specifications, LLM pretraining overlap. Papers that pre-empt their own critique get read charitably. Papers that hide it get rejected.

---

## 10. Positioning against the literature

You currently cite Jansen's ML-for-trading workflow, which is a textbook, not a research reference. You need a real related-work section. At minimum, engage with:

**LLM look-ahead bias:** Glasserman & Lin (2023/24) on anonymization; Sarkar & Vafa (2024) on lookahead in pretrained models; the more recent date-only recall diagnostics. **This is where your novelty claim is strongest — very few applied factor papers do this audit.**

**LLMs and the cross-section:** Chen, Kelly & Xiu on expected returns and LLMs; Kim, Muhn & Nikolaev on financial statement analysis with LLMs; Lopez-Lira & Tang on ChatGPT and return prediction.

**Text-based firm characterization:** Hoberg & Phillips text-based industry classification (the intellectual ancestor of similarity-based exposure); Bybee, Kelly et al. on topic-model-derived state variables.

**AI exposure measurement specifically:** Eisfeldt, Schubert & Zhang on generative AI and firm values; Babina, Fedyk, He & Hodson on AI adoption; and the 10-K-based AI index construction work noted in §5.3, which is your closest direct predecessor.

**Multiple testing:** Harvey, Liu & Zhu (2016); Harvey & Liu, "Lucky Factors"; Novy-Marx on backtesting with multiple signals.

**Overlapping returns:** Hodrick (1992); Hansen & Hodrick; Newey & West (1987).

**Optimizer fragility:** Michaud (1989) on the Markowitz optimization enigma; Best & Grauer on input sensitivity; Britten-Jones on weight standard errors. These directly support your §6.3 argument that calibrated inputs matter only in specific settings.

**Thematic investing skepticism:** Ben-David et al. on specialized ETF underperformance; Blitz on thematic investing. Engaging with the bear case makes you look like a researcher rather than a promoter.

*(Verify each citation before use — I'm giving you author-topic pointers, not verified bibliographic entries.)*

---

## 11. Presentation issues

- **The LaTeX is entirely broken in the document I received.** Every symbol is missing — `Let  index the stocks available at rebalance date `. Whatever export path you used stripped the math. Fix before anyone else reads it.
- **Two unresolved `(?)` markers** in the text (the 12-period rule, and the "not a regression from one monthly return series" note). Resolve them.
- **Spurious precision:** 0.07030, 6.49%, 2.84%. Report as 0.070, 6.5%, 2.8%, always with an interval.
- **Missing definitions:** the exposure materiality scale (0–3? 1–5? what do the levels mean, verbatim from the prompt?); the embedding function; handling of stocks with no filing in the window; tie-breaking in quintile assignment.
- **The example numbers are inconsistent.** The opening narrative uses a NVIDIA score of 1.7, the Part 1 worked example derives 2.0. Pick one.
- **Structure.** The document is currently a lab notebook — "Current Status," "Next Steps," "Stage 3." Restructure into paper form: Introduction, Data, Signal Construction, Leakage Audit, Signal Diagnostics, Calibration, Portfolio Construction, Robustness, Limitations, Conclusion.

---

## 12. Priority checklist for the end-of-August deadline

Given ~5 weeks, here is what I would actually do, in order. Items 1–5 are non-negotiable.

| # | Task | Effort | Why |
|---|---|---|---|
| 1 | Anonymization ablation for leakage | 3–5 days | Existential. Referees will ask first. |
| 2 | Report IC dispersion, NW t-stats, block-bootstrap CIs everywhere | 1–2 days | Your headline numbers are currently uninterpretable. |
| 3 | Keyword-count and GICS-dummy baselines | 2 days | Determines whether the LLM apparatus is justified at all. |
| 4 | Confirm point-in-time universe; document survivorship handling | 1–2 days | Silent killer. |
| 5 | Reframe as methodology paper + write Limitations section | 2 days | Changes what you have to defend. |
| 6 | Check 10-K/10-Q similarity heterogeneity; check sim×exposure vs exposure correlation | half day | Cheap; may change your central feature. |
| 7 | FF5+MOM+AI-proxy alpha on the Q5-Q1 series | 1 day | Standard; its absence is glaring. |
| 8 | Calibration exhibits (§8.1–8.4) | 2 days | Directly what you're already working on. |
| 9 | Non-overlapping EB robustness | 1 day | Fixes the independence violation. |
| 10 | Constrained-vs-unconstrained optimizer experiment (§6.3) | 2–3 days | Turns your weakest section into a real contribution. |
| 11 | Multiple-testing haircut + full nine-factor grid | 1 day | Pre-empts the obvious attack. |
| 12 | Turnover, costs, net spread | 1 day | Economic significance. |
| 13 | Fix LaTeX, restructure, related-work section | 3–4 days | Presentation. |
| 14 | IC term structure figure | half day | Expected exhibit. |

If time runs short, **cut items 10 and 14 before you cut 1–5.**

---

## 13. Final assessment

**Is this research-worthy?** Yes — as a methodology and measurement paper with an honest, underpowered empirical illustration. The pipeline discipline is real, the materiality-gating idea is economically sensible, the factor freeze is genuinely good practice, and the leakage audit (once you do it) would be a contribution in its own right because so few applied papers bother.

**Is it research-worthy as currently framed?** No. As a factor-discovery paper it has ~7 independent observations, a single-regime sample, a 56% sector tilt, an unaddressed LLM leakage channel, and a selection-over-nine problem. Any competent referee finds all five.

The gap between those two answers is almost entirely a *framing and disclosure* problem, not a *research quality* problem. That is a good position to be in five weeks out.

One last thing. The instinct behind the honest reporting you've already done — showing the 2.84% next to the 6.49% — is the right instinct, and it is rarer than you'd think. Lean into it harder. The most credible version of this paper is the one that says clearly: *here is a carefully-built pipeline, here is everything we did to try to break it, here is what survived, and here is why 36 months cannot settle the question.* That paper gets cited. The one that claims a new alpha from 36 months does not.
