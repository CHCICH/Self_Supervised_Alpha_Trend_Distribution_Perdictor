# EECE 490 research decision log

The Milestone 1 target and timing rules are specified in [the formulation](MILESTONE_1_FORMULATION.md). Numerical settings and later experiment choices remain provisional where indicated. The [proposal](eece490_proposal.pdf) supplies the cited requirements and motivating design. For each resolution, record the selected rule, rationale, alternatives, evidence, approving reviewer, date and affected artifact/config versions. Validate the provisional D01–D04 settings during the data audit and D05–D08 before baseline training. Resolve later decisions before their respective experiments.

## D01 — Forecast origin and estimand · M1

- **Proposal.** §1 conditions forecasts on history through $t$. §4 specifies slope, duration and direction of a future trend segment. Figure 1 shows multiple future horizons.
- **Specified.** Issue after the completed daily bar is available for each ETF. The movement starts at $t$ and ends at the first admissible fitted knot $b>t$. A knot at $t$ marks the outgoing movement, and earlier knots do not shift the start. Use provisional $L=20$ causal feature rows.
- **Secondary benchmark.** Use close-to-close log returns over provisional $h=5,10,20$ sessions with maturity $t+h$ and horizon-specific purging. Report return forecasts separately from primary label scores.

## D02 — Trend-filtering operator and label units · M1

- **Specified.** Fit the proposal's $\ell_1$ trend filter independently by ETF on log adjusted closes over $[t-B+1,t+H]$. Detect centered second differences above $\tau$, excluding fit endpoints. Start without knot merging. For a completed case, $d_t=b-t$ is remaining duration in trading-session intervals and $s_t=(\widehat x_b^{(t)}-\widehat x_t^{(t)})/d_t$ is the average fitted log slope per session. Label Up for $s_t>\varepsilon$, Down for $s_t<-\varepsilon$, and Flat otherwise.
- **Provisional.** $B=20$, $\tau=10^{-6}$, $\varepsilon=0.0005$. Use a training-only design prefix to scale and select $\lambda_j$ for each ETF from a common multiplier grid. Require the Milestone 2 stability report before freezing the rule and comparing forecasts.
- **Required stability checks.** On training/design data, predeclare sweeps over $\lambda,B,H,\tau$. Include sensitivity figures and quantify changes in boundaries, completion, slope, remaining duration and direction on matched eligible origins, alongside configuration-specific coverage. Adjacent origins 100 and 101 use bars 81–130 and 82–131 respectively. Different fits can move the boundary of the same apparent episode.
- **Required adjacent-window diagnostics.** Compare endpoints over the shared eligible interval. Report pairs with an elapsed endpoint or an endpoint outside the shared interval separately. Report the exact disagreement rate among pairs with two comparable endpoints, shifts in sessions, and completed/censored transitions with denominators. Deliver paired-fit overlays, shift distributions and fixtures for stable and shifted knots. Record the chosen settings and the supporting evidence.

## D03 — Label maturity, future support and censoring · M1

- **Specified.** Set $T=t+H$, $U=T-g$ and accept the first valid knot in $(t,U]$. If the complete fit has no eligible knot, record right-censoring and follow-up $U-t$ with no exact target tuple. Record split purges and insufficient support separately.
- **Maturity.** Freeze the outcome when the entire fit is available at $m(t)=T$, even if the selected endpoint is earlier. Store the snapshot, parameters, support and availability time. Revisions use a separate version.
- **Provisional.** $H=30$, $g=2$. Audit completion and split eligibility on development data. Score exact attributes conditionally on completion, report completion rates and usable-label counts, and consider a censor-aware duration analysis separately.

## D04 — Information set, preprocessing and leakage controls · M1–M2

- **Open.** Set issue and execution timestamps, same-session OHLCV availability, rolling feature warm-up rules, adjustment-vintage policy and cross-market joins. Define which previously observed bars may warm up later windows. The retrospective fitted value $\widehat x_t^{(t)}$ belongs to label construction and is excluded from causal features. Address leakage from centered indicators, revised data, scaling, feature selection, regime cutoffs, pretraining target windows and calibration.
- **Candidate.** Use point-in-time availability timestamps, training-only fit operations, forward chronology, trailing windows and explicit context/outcome intervals. Permit historically available warm-up bars with frozen transforms. If historical vendor vintages are unavailable, document the limitation and audit corporate-action sensitivity.
- **Resolve with.** An availability ledger and temporal invariance tests that perturb rows after the forecast origin or the relevant fit boundary.

## D05 — Data identity and sample sufficiency · M2

- **Proposal.** §4 and Table 2 report 501 rows per ETF and zero missing OHLCV. Figure 1 depicts additional markets.
- **Candidate histories.** Compare the proposal's 501 observations per ETF with candidate daily SPY and QQQ coverage from January 1, 2010 through the latest completed trading session available in 2026 at the snapshot date. Verify availability, actual covered dates and sessions, provider adjustment conventions, corporate actions, missing or duplicate bars, quality and adjustment-vintage limits.
- **Required sample accounting.** On development data, report candidate, eligible, completed, censored and purged counts, rates and class balance by ETF and fold. Report distinct episodes alongside origin counts. Document the grouping rule and sensitivity to nearby fitted boundaries. Overlapping origins and correlated ETFs limit independence even with a longer history.
- **Resolve with.** A verified snapshot manifest and pilot-versus-expanded comparison under the same label protocol. Set minimum usable support before model comparisons. Justify the dataset scope using development diagnostics and anticipated test support, with the test held out.

## D06 — Chronological split and model-selection protocol · M2

- **Open.** Choose dates and rounding for 75/15/10, split anchors by session, define expanding folds and how the fixed validation set is used, and decide any extra gap beyond exact label-support purging. Decide whether there is a preapproved final train-plus-validation refit and how calibration remains isolated.
- **Candidate.** Keep both assets on common date cutoffs. Use inner chronological folds in the training period where feasible, reserve validation for model selection or calibration under a documented allocation, and hold out the final test. Split controls apply to all training targets, including JEPA pretraining. Choose one common forecast-ID set for primary comparisons and disclose any auxiliary coverage.
- **Resolve with.** Versioned split definitions and a fold table showing context allowances, outcome cutoffs and excluded origins.

## D07 — Comparator fairness and resource budget · M3–M6

- **Open.** Define OHLCV/return features, static summaries versus flattened causal windows, polynomial degree/regularization, lookbacks, head capacity, tuning budgets, seeds and runtime limits. Determine how to evaluate slope-derived direction versus a dedicated classifier.
- **Candidate.** Lock shared asset universe, raw information, timestamps, label IDs and scoring. Document model-specific representations. Use prespecified laptop-sized budgets and report compute. Compare frozen embeddings and raw inputs with the same downstream head family where feasible, disclose capacity differences and keep fine-tuning separate.
- **Resolve with.** A model-by-input and tuning matrix before training, plus matched baseline and ablation manifests.

## D08 — Primary point metrics and failure handling · M3

- **Proposal.** §6 specifies directional accuracy, net PnL and maximum drawdown and calls for lower forecast errors in the JEPA comparison. It does not define slope or duration error.
- **Candidate.** Use slope and duration MAE in native units, optionally RMSE as a tail-sensitive diagnostic, and directional accuracy plus class support and a confusion matrix. Predefine how duration predictions satisfy positivity, how missing forecasts are counted, how censored labels are scored, and whether aggregation weights assets, origins or unique segments.
- **Resolve with.** A scoring contract with test cases and declared denominators. Lock the primary scoring and model-selection metrics before the first baseline run.

## D09 — Architecture scope and literature verification · M5–M6

- **Open.** Reconcile the four numbered model families, §5's five-tier language and contrastive comparison, and Table 1's hybrid JEPA, flow and neural SDE description. Decide if regime discovery is a measured output or an interpretation of embeddings. Specify frozen versus end-to-end JEPA evaluation.
- **Candidate.** Complete linear/logistic, LSTM and AE before JEPA. Make AE versus JEPA and raw-OHLCV versus JEPA explicit ablations. Verify Table 1's claims against the cited sources, recording task, dataset, horizon and metrics for each. Add TNC-style contrastive learning only if its role and budget are approved. Consider U-Net, flows and SDEs as separate experiments.
- **Resolve with.** A literature/comparator matrix and a JEPA specification covering future support, latent loss, target-encoder update, collapse tests, probe protocol and resource limits.

## D10 — Probabilistic target, horizon and coherence · M7

- **Open.** Predict separate marginals or a joint distribution over slope, direction and remaining duration of the immediate movement. Decide whether horizons index segments, session offsets or both. Decide how to model censored/discrete durations and how direction relates to slope.
- **Candidate.** Start with distribution references and simple outputs for supported horizons. Define coherent direction probabilities through slope bands or a conditional factorization if modeling jointly. Report baseline-relative proper scores and calibration. Escalate to flows/SDEs only with a supported modeling and evaluation question.
- **Resolve with.** A mathematical output schema with support constraints, calibration split and scoring plan, including a decision about joint dependence.

## D11 — Volatility envelope semantics · M7

- **Proposal.** Figure 1 depicts asymmetric, potentially regime-dependent envelopes. It leaves the covered variable and probability semantics open.
- **Open.** Distinguish predictive bands for future price/return, intervals around a latent trend, and forecasts of realized volatility. Decide whether coverage is pointwise by horizon or simultaneous over a path, and choose nominal levels and units.
- **Candidate.** Start with conditional quantile bands for an explicitly named cumulative log-return outcome at fixed offsets. If realized volatility is also a target, specify its forward window and estimator separately. Evaluate coverage and width, and use a path-wise score only for a stated simultaneous band.
- **Resolve with.** A worked fan chart with a formula for the covered outcome and an exact empirical coverage calculation.

## D12 — Trading simulation and regimes · M3 design, M8 freeze

- **Open.** Define signal-to-position mapping, flat handling, long/short constraints, next eligible fill, costs, slippage, holding/exit logic, overlapping positions, exposure and PnL aggregation. Define calm/high-volatility cutoff, measurement horizon and minimum regime support.
- **Candidate.** Use a common position policy with executable observed prices, realistic timing, transaction costs and slippage. Exit decisions use information available at execution. Retrospective fitted levels and boundaries belong to label construction. Document turnover and cost units. Compute drawdown from the net equity path as the maximum fractional decline from a running peak. Define regime at issuance from trailing volatility with a training-derived cutoff. If realized future regimes are analyzed, report them separately as retrospective slices.
- **Resolve with.** A hand-calculated trading ledger and regime assignment examples before development backtests, then a locked simulation config before test.

## D13 — Evidence standard and final test · M8

- **Proposal.** §6 requires consistent out-of-sample gains and considers direction and drawdown alongside net PnL.
- **Open.** Define primary comparisons and improvement margins, what consistency means across horizons/assets/folds, acceptable drawdown, multiplicity treatment and what evidence is insufficient. Decide how to quantify uncertainty with overlapping windows and synchronized asset observations.
- **Candidate.** Prespecify paired differences and effect sizes, with dependence-aware uncertainty such as date blocks resampled jointly across assets if support permits. Freeze block strategy and score hierarchy before test access. Report low-power results as inconclusive where appropriate and preserve a clear distinction between preplanned and exploratory analyses.
- **Resolve with.** An approved evaluation manifest stating when the hypothesis will be supported, unsupported or inconclusive, and who authorizes opening the test.
