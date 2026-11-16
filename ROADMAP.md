# EECE 490 research roadmap

**Status** — Research roadmap.

## What the proposal commits to

The source is [the official proposal](eece490_proposal.pdf), pp. 1–4. Section and figure references below refer to that document. The milestone order, candidate equations beyond the cited trend-filtering method, safeguards and completion criteria are roadmap recommendations.

| Proposal evidence | Implication for the plan |
| --- | --- |
| §§1–2 and Fig. 1 | Use information available through time $t$ to forecast future trend segments and associated uncertainty. The conceptual system maps multiple markets to latent regimes and depicts forecasts at $t+k_1,t+k_2,t+k_3$, with asymmetric, regime-dependent envelopes and conditional distributions that may be skewed, heavy-tailed or multimodal. Equities, gold and FX appear in the figure. |
| §4, Figs. 2–3 and Table 2 | The preliminary dataset comprises 501 daily OHLCV rows each for SPY and QQQ, Oct. 21, 2024–Oct. 20, 2026, from Yahoo Finance via yfinance. The proposal reports zero missing OHLCV entries. The closing-price plots show an overall rise and a sharp April 2025 decline and recovery. The return histograms are centered near zero with large moves, and reported daily standard deviations are 1.0416% for SPY and 1.3755% for QQQ. These are descriptive observations about the preliminary sample. |
| §4 and §5 | Construct direction, slope and duration of the next trend segment using ℓ₁ trend filtering [12]. Compare linear and multinomial logistic regression, LSTM, reconstructive AE/U-Net and JEPA. §5 calls this a five-tier hierarchy and mentions contrastive embeddings, while the numbered list specifies four families with linear and logistic grouped together. The exact status of a separate contrastive benchmark needs resolution. |
| §3 and Table 1 | Stock-JEPA/TS-JEPA motivate latent prediction, TNC motivates a contrastive regime comparison, conditioned flows/MANF motivate flexible densities, ARTEMIS motivates continuous-time dynamics, and Market-1T/LaT-PFN motivate cross-asset and multi-horizon representations. The table contrasts these along representation, distribution, horizon and volatility capability. References [9] and [10] add market-state JEPA and neural Lévy SDE context. Treat the table's limitations and hybrid design as proposal claims to check during literature review. |
| §6 and Table 3 | Use chronological 75/15/10 train/validation/test, training-only normalization and feature selection, boundary controls for labels and horizons, and an untouched test. Report directional accuracy, net PnL with costs and slippage, and maximum drawdown. Compare JEPA with polynomial regression and a raw-OHLCV-head ablation, and disaggregate calm/high-volatility performance. The proposal expects consistent out-of-sample gains, with direction and drawdown considered alongside PnL. |
| §§7–8 and Table 4 | Start on a laptop with Python and maintain code, setup and data-loading instructions in the project repository. The original schedule proceeds through data/EDA, regressions, further models, progress report, robustness, poster, presentation and final report. The completion criteria below guide progression through the project. |

## Execution and review protocol

- Work on one milestone at a time. At each milestone, review its artifacts, test results, figures, limitations and decision-log changes. If evidence fails a completion criterion, revise within that milestone or agree on a documented scope change.
- Version raw-data manifests, label and split definitions, features, configs, seeds, package versions and run outputs. Keep origin-level predictions with asset, target ID, timestamps and model/run ID so comparisons can be reproduced.
- Use chronological development folds drawn from the permitted development data, with an explicit role for the fixed validation period. Keep the final test out of tuning, calibration, model choice and interim reporting. Any authorized protocol amendment must be logged before test access.
- Proposed layout once implementation begins: `configs/`, `src/`, `tests/`, `data/` for snapshots/manifests subject to source terms, and `reports/` for review packets. The folders and experiments are future deliverables.

## 1. Specify the forecast and the labels

**Objective.** Define the immediate fitted movement from the after-close forecast origin $t$ to the first admissible future fitted knot $b>t$.

**Work and decisions.** Write a formal contract for the information set ℱₜ, the price transform, forecast issuance and earliest trade time, anchor, horizon and label maturity. Select the first knot $t<b\le U=t+H-g$. Define remaining duration $d_t=b-t$ and slope $s_t=(\widehat x_b^{(t)}-\widehat x_t^{(t)})/d_t$. Specify the flat band, right-censoring and maturity $m(t)=t+H$. Keep fitted labels separate from causal features and executable returns. Document the secondary fixed-horizon return benchmark and distinguish it from later multi-horizon trend forecasts. Review the proposal's cited trend filtering and boundary methods [12, 15]. Use the candidate formulation and open questions in D01–D04.

**Deliverables.** Label/evaluation design note, hand-worked examples and timeline diagrams for up, down, flat, unchanged, boundary-crossing and censored cases. Record expected label and maturity time for each example. This milestone is specification work.

**Depends on.** Project roadmap.

**Acceptance.** The same example yields one reproducible interpretation from its written rules. Every target has units, an origin, a last required bar and an eligibility rule. The protocol explains how overlapping labels, retrospective knots and right censoring are handled. Define the initial task and the mapping to the later multi-horizon task.

## 2. Collect, audit and freeze the data and split protocol

**Objective.** Establish reproducible inputs and determine how much usable evidence the project has.

**Work and decisions.** Build a versioned ingest and snapshot pipeline, the specified immediate-movement labeler and chronological 75/15/10 split manifests. Record source, dates, adjustment flags, calendar, timezone, hashes and package version. Reproduce proposal Figs. 2–3 and Table 2 where the preliminary snapshot can be recovered. Fit preprocessing and select label settings on permissible training data.

**Dataset audit.** Investigate daily SPY and QQQ histories from January 1, 2010 through the latest completed trading session available in 2026 at the snapshot date. Treat these dates as a candidate range. Verify provider availability, session coverage, raw and adjusted price conventions, corporate actions, missing or duplicate bars, OHLC consistency and adjustment-vintage limits. Compare the expanded candidate with the proposal's 501 observations per ETF under the same label protocol. Record coverage and counts on permissible development data and use only anticipated support for the locked test.

**Label-stability diagnostics.** On training/design data, predeclare sweeps of the smoothing penalty $\lambda$, historical support $B$, future support $H$ and knot tolerance $\tau$. Compare configurations on common eligible origins and report each configuration's own coverage. Separately compare neighboring origins, such as 100 on bars 81–130 and 101 on bars 82–131. Use a common comparison interval strictly after both origins and no later than either endpoint cutoff. Report pairs separately when a boundary has already passed the later origin or lies outside the common interval. For the remaining pairs, report exact boundary disagreement among pairs with two comparable boundaries, absolute shifts in sessions, and changes between completion and censoring with counts and denominators. Summarize slope, remaining duration and direction changes under the sensitivity sweeps.

**Sample accounting.** Tabulate eligible, completed, censored and purged origins and Up/Down/Flat counts by ETF and development fold. Report distinct fitted episodes alongside origin counts using a documented grouping rule, with exact terminal dates as an auditable proxy and a sensitivity check for nearby boundaries. A fitted episode can yield several origins with different durations. Overlapping windows and shared ETF shocks limit the number of independent examples.

**Deliverables.** Data and label code, deterministic and boundary fixtures, snapshot and split manifests, a causal availability ledger, and a dataset and label-stability report. Include overlaid adjacent-window fits with knots, a sensitivity grid or heatmap, distributions of boundary shifts and label attributes, and a sample sufficiency table for both histories.

**Depends on.** Defined label contract in M1.

**Acceptance.** A rerun reproduces manifests and labels. Fixtures cover stable and shifted adjacent knots, exact eligibility, right-censoring, split purges and invariance to observations outside declared support. The report contains the parameter sweeps, boundary disagreement and shift diagnostics with explicit denominators, completion and censoring rates, distinct episode and origin counts, and directional-class balance. Document the dataset and label settings chosen from this evidence, or narrow the experiment if support is insufficient (D02–D06). Keep the test locked.

## 3. Establish reproducible classical baselines

**Objective.** Measure how much predictability is available from simple, auditable models.

**Work and decisions.** Implement linear regression for fitted slope and remaining duration and multinomial logistic regression for up/down/flat. Score fixed-horizon returns as a separate secondary benchmark. Include train-only constant/majority and a causal persistence reference where well-defined. Build the polynomial comparator required by §6, with degree and regularization chosen on development data. Specify feature history, duration positivity, class handling and the trading mapping. Forecast with the same origins and covariate availability that later models will use. Start with slope/duration MAE in native units, directional accuracy, confusion matrix and class-balanced diagnostics. Report development net PnL and drawdown from executable fills under a common timing, cost and slippage policy.

**Deliverables.** Training/evaluation scripts, configs, origin-level predictions and deterministic smoke tests. Produce fold scorecards, residual and duration-error plots, confusion matrices and net equity/drawdown curves. Document naïve reference construction and model selection.

**Depends on.** M2 data and split freeze.

**Acceptance.** One documented run reproduces the scorecard to declared tolerances. Each task has a naïve reference and the appropriate regression baseline. Polynomial and later comparisons share eligible target IDs, source inputs and scoring rules. All reported results are development results, with valid denominators, execution timing and cost assumptions.

## 4. Test supervised sequence learning with an LSTM

**Objective.** Measure the incremental value of ordered history and nonlinear temporal dynamics.

**Work and decisions.** Train a compact LSTM on causal windows for the same immediate-movement outputs. Predefine lookback candidates, loss weights or separate heads, early stopping and a finite laptop-sized tuning budget. Give the linear/polynomial comparison access to the same source history via a documented representation. Compare against the classical baselines and, if feasible, a parameter-conscious nonrecurrent or order ablation using development data.

**Deliverables.** Window loader, LSTM, configs and causal alignment tests. Report learning curves, multi-seed paired fold comparisons, error slices and compute cost.

**Depends on.** M3 scoring and data contract.

**Acceptance.** Window and target alignment tests pass, all matched comparisons use the same forecast IDs, and variance across folds and seeds is visible. Record a conclusion about the value of recurrence and whether model capacity or sample support limits it. Improvement is an empirical outcome.

## 5. Test reconstructive representations with an autoencoder

**Objective.** Determine whether reconstructing historical sequences yields transferable trend features before testing predictive embeddings.

**Work and decisions.** Start with a compact sequence AE trained to reconstruct past input windows on training data. Define reconstruction scaling, bottleneck dimension and the downstream training protocol. Use a frozen-encoder probe as the primary representation comparison, with any fine-tuning reported separately. Consider U-Net if warranted by the study design, with checks that skip paths do not bypass the representation being tested. Compare downstream immediate-movement forecasts against a raw-window or raw-OHLCV head and the LSTM with matched forecast IDs and a documented head/input design.

**Deliverables.** AE training and probe code, reconstruction and downstream learning curves, bottleneck comparisons, residual examples and run-cost notes.

**Depends on.** M4 completion and the M3–M4 benchmark record.

**Acceptance.** Train-only reconstruction and downstream split isolation are verified. Reconstruction quality and forecasting utility are scored separately. Document whether reconstruction carries useful trend signal and freeze the AE comparator before JEPA work.

## 6. Test point-forecast JEPA and isolate representation gains

**Objective.** Test whether predicting future embeddings improves immediate-movement forecasts.

**Work and decisions.** Define context and future-target windows, encoders, latent predictor, latent loss and collapse diagnostics. A candidate is a context encoder with a predictor and an EMA target encoder with stopped target gradients. Future windows are training targets with explicit maturity timestamps. At inference, forecast from the context through $t$. Use matched downstream heads where feasible and run the proposal's raw OHLCV ablation and polynomial comparison. Decide whether a TNC-style comparator is warranted after reviewing the cited work and checking support and compute. Compare the smallest single-asset and SPY/QQQ context settings as separately labeled experiments if feasible.

**Deliverables.** JEPA pretraining and head code, window/maturity tests, latent variance and collapse diagnostics, ablation tables, seed-level development results and resource notes.

**Depends on.** Completed M3–M5 baselines and D07–D09.

**Acceptance.** A test demonstrates that future target windows are unavailable to inference and stay inside permitted training intervals during pretraining. Compare the full model with the required polynomial and raw OHLCV controls on the same task, plus LSTM and AE references. Document evidence for or against incremental value before adding distribution heads.

## 7. Forecast distributions, multiple horizons and volatility envelopes

**Objective.** Extend point forecasts into verifiable conditional uncertainty forecasts.

**Work and decisions.** Define the distribution of immediate-movement attributes, then specify later multi-horizon targets and what the envelope covers. Later experiments may include complete subsequent segments. Start with empirical or simple conditional distribution references and quantile or parametric heads. Consider a conditioned flow or neural SDE only after reviewing the literature and demonstrating enough data and compute to estimate added value. Handle discrete duration, flat segments and coherence between direction and slope. Use proper scores matched to outputs, such as log loss/Brier for direction, pinball loss for quantiles and CRPS for continuous marginals with a forecast CDF. For claimed joint forecasts, add a joint scoring rule and check dependence. Evaluate coverage and width together by horizon. Any calibration uses permitted development data with separate fitting and scoring roles.

**Deliverables.** Distribution heads, calibration/evaluation code, fan charts, reliability and coverage plots, interval-width summaries, proper-score comparisons, and illustrative calm/high-volatility cases.

**Depends on.** M6, supported sample size and D10–D11.

**Acceptance.** Forecasts have valid support, ordered quantiles, clear interval semantics and usable coverage denominators. Score each horizon against a simple probabilistic reference on matched forecast IDs. Separate marginal intervals from simultaneous path coverage. Document which Figure 1 capabilities the tested design supports and which require further data or research.

## 8. Run the locked evaluation and deliver the research record

**Objective.** Assess generalization and interpret forecasting and trading results jointly.

**Work and decisions.** Freeze model selection, metrics, comparison margins, eligible IDs, costs and slippage, execution/position policy, uncertainty method and regime definitions before opening the test. Run the prespecified evaluation for classical baselines, LSTM, AE, JEPA, the raw-feature ablation and specified distribution forecasts. Pair comparisons by timestamp/forecast ID and account for dependent observations and common SPY/QQQ exposure. Report any preplanned cost or regime sensitivity with support counts.

**Deliverables.** Locked-run manifest, per-origin forecasts and trades, aggregate and regime tables, uncertainty intervals where estimable, equity/drawdown and calibration figures, final report, poster, presentation and reproduction instructions. Include a progress-report snapshot during development at a milestone review.

**Depends on.** M1–M7 completion and a defined evaluation plan (D12–D13).

**Acceptance.** The frozen protocol reproduces the final tables and figures. Report immediate fitted-slope and remaining-duration errors, directional accuracy, net PnL from executable fills, max drawdown, probabilistic scores and coverage for applicable models. Report fixed-horizon return benchmark scores separately. Interpret JEPA's comparison with polynomial regression and the ablations under prespecified criteria. Report negative or inconclusive results and limits on generalization when warranted. Any later exploratory analysis is labeled separately.
