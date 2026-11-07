
# Forecasting immediate market movement

EECE 490 · Milestone 1 mathematical formulation

## 1. Forecast origin and available information

The research proposal asks for the slope, duration and direction of a future market trend using information available through the forecast origin. It proposes daily OHLCV for SPY and QQQ, $`\ell_1`$ trend filtering for labels, and chronological evaluation. This revision defines the primary target as the fitted movement from the close of session $`t`$ to the first valid fitted boundary strictly after $`t`$. A forecast can describe the remaining movement of a trend already underway. Later multi-horizon and probabilistic experiments remain distinct from this initial point-forecast task.

Use an exchange-session index $`t`$ and issue one forecast per eligible ETF after its completed daily bar is available. Let $`\mathcal F_t`$ contain the observations and fitted parameters available at that issuance time. For an initial own-ETF specification, construct a trailing sequence of $`L`$ causal feature rows $`z_i`$ from OHLCV and returns:

```math
X_t=(z_{t-L+1},\ldots,z_t),\qquad L=20\ \text{provisionally}.
```

All raw observations used by $`X_t`$, including feature warm-up, must be available at issuance. A same-date observation from a second market can enter a later cross-asset comparison only if its actual availability precedes issuance. Forecast labels use closing levels. Any later trading simulation must separately define execution prices and timing, with the next eligible open as a candidate fill.

### Worked example · Building the input at session 100

The following numbers and all worked examples in this note are illustrative. Issue the forecast once bar 100 is complete and available. The sequence contains rows 81 through 100, a total of $`100-81+1=20`$ rows. If $`z_{81}`$ includes a one-session log return, it also needs the close at 80. With illustrative closes 99 and 100 at sessions 80 and 81, that feature is $`\log(100/99)\approx0.01005`$. Session 101's open and all later realized observations are outside $`\mathcal F_{100}`$.

The label fit will eventually produce $`\widehat x^{(100)}_{100}`$. Its time index is 100, but its value depends on future data and stays in the label pipeline. The model sees the observed history through 100.

## 2. Finite-support trend filtering

For each ETF and each origin $`t`$, use a versioned series of positive adjusted closing prices $`P_i`$. Write $`p_i=\log P_i`$ to distinguish the log-price input to the filter from the supervised target $`y_t`$. For provisional $`B=20`$, $`H=30`$ and $`g=2`$, define

```math
T=t+H,\qquad U=T-g,\qquad I_t=\{t-B+1,\ldots,T\}.
```

Here $`B`$ counts historical sessions including $`t`$. $`H`$ counts future observations used by the fit. $`U`$ is the last admissible boundary, leaving $`g`$ terminal observations after it. A window at $`t=100`$ uses sessions 81 through 130 and admits future boundaries at 101 through 128. The guard keeps eligible knots away from the fit edge. Its adequacy is an empirical question.

Fit a separate trend $`\widehat x^{(t)}`$ on $`I_t`$ using the squared-error and absolute second-difference objective in reference [2], cited as reference [12] in the proposal:

```math
\widehat x^{(t)}
=\mathop{\mathrm{arg\,min}}\limits_{x\in\mathbb R^{B+H}}
\left\{\frac12\sum_{i\in I_t}(p_i-x_i)^2
+\lambda_j\!\!\sum_{i:\ i-1,i,i+1\in I_t}
  |x_{i+1}-2x_i+x_{i-1}|\right\}.
```

The squared-error term makes the minimizer unique. The penalty favors equal increments over stretches of the fitted series. At interior session $`i`$, a knot is recorded when

```math
\left|\widehat x^{(t)}_{i+1}-2\widehat x^{(t)}_i+\widehat x^{(t)}_{i-1}\right|>\tau,
\qquad \tau=10^{-6}\ \text{provisionally}.
```

Use centered knot indices, exclude the two fit endpoints, and start without knot merging or a minimum segment length. Verify numerical error in second differences is small relative to $`\tau`$, and audit near-threshold or closely spaced knots. The log-price scale makes a given slope comparable across different price levels.

### Worked example · Locating a fitted knot

Fitted log levels $`0.000,0.002,0.004,0.003`$ have successive increments $`0.002,0.002,-0.001`$. The centered differences are $`0`$ at the second session and $`-0.003`$ at the third. Only the third session is a knot at the provisional tolerance.

For asset $`j`$, a candidate penalty scale is $`\lambda_j=\kappa\,\mathrm{median}|\Delta p_{j,i}|`$, using a positive scale estimated in a training-only design prefix. Prespecify a grid for $`\kappa`$, examine completion, durations, class support and sensitivity, and freeze the chosen rule before comparing forecasts. A larger penalty can remove knots and reduce the number of usable labels.

## 3. Primary target and boundary cases

Let $`K_t`$ be the interior knots from the frozen fit for origin $`t`$. With full observation support, the completed movement ends at

```math
b=\min\{i\in K_t:t\lt i\le U\}.
```

If the set is empty, record the outcome as right-censored at $`U`$. The target starts at $`t`$, and endpoint eligibility depends on the first admissible future knot. A knot at $`t`$ is the starting boundary for the outgoing fitted movement. A knot before $`t`$ provides retrospective context. The first admissible knot strictly after $`t`$ determines the end.

For a completed case, define the principal target $`y_t=(s_t,d_t,c_t)`$ by

```math
\begin{aligned}
d_t &= b-t,\\
s_t &= \frac{\widehat x^{(t)}_b-\widehat x^{(t)}_t}{b-t},\\
c_t &=
\begin{cases}
\text{Up},   & s_t>\varepsilon,\\
\text{Down}, & s_t\lt-\varepsilon,\\
\text{Flat}, & |s_t|\le\varepsilon,
\end{cases}
\qquad \varepsilon=0.0005\ \text{provisionally}.
\end{aligned}
```

Duration counts remaining trading-session intervals. The endpoint levels are both used to measure the average log-price slope. Approximately $`10{,}000s_t`$ gives basis points per session for small slopes. Direction is a deterministic class label derived from the fitted slope. The endpoint $`b`$, both fitted levels and the direction become available as supervised outcomes after the label window matures.

<!-- figure: forecast -->
    Session        81                100     104                 128  130
    Past input     |------------------|
    Target                            |=======|
    Future label data                 |-------------------------------|
    Terminal guard                                               |----|

**Figure 1.** Session 100 anchors the movement. Sessions 101–130 contribute only to the retrospective label, including the guard after 128. The label matures when bar 130 is available. Spacing is schematic.

### Worked examples · Three locations of the origin

| Case | Boundary rule | Result |
| --- | --- | --- |
| Origin 100, future knots 104 and 112 | First knot after 100 is 104 | Movement 100–104, duration 4 |
| Trend began at 96, origin 102, next knot 104 | Use 102 as the starting level | Movement 102–104, duration 2. The full fitted segment spans 8 sessions. |
| Knot at 100, next knot at 104 | Ignore 100 as an endpoint | Outgoing movement 100–104, duration 4 |

For the first row, fitted levels 4.6000 and 4.6080 give $`(4.6080-4.6000)/4=0.0020`$ log units per session, or about 20 basis points per session, so the class is Up. A change of $`-0.0080`$ over four sessions gives $`-0.0020`$, or Down. A change of $`0.0016`$ gives $`0.0004`$, or Flat. At either exact threshold $`\pm0.0005`$, the class is Flat.

### Observed prices, fitted labels and trading returns

The observed close $`C_t`$ is a market price recorded for session $`t`$. The versioned adjusted close $`P_t`$ used for labeling may differ because of corporate-action adjustments. The fitted value $`\widehat x_t^{(t)}`$ is a smoothed estimate in log-price units obtained jointly from the whole label window, including observations after $`t`$. It belongs exclusively to the retrospective label pipeline and must be excluded from the causal feature set at issuance.

The slope $`s_t`$ measures the fitted log-price change per session from $`t`$ to $`b`$. A strategy return depends on its entry and exit fills, the position and execution costs. A trading simulation uses actual executable prices, realistic timing, transaction costs and slippage. The exit rule must be executable without knowing the retrospective boundary.

**Illustration at origin 100.** Suppose the observed close at 100 is 100 USD, the next open is 102 USD and the close at 104 is 101 USD. The fitted log values at 100 and 104 are 4.6000 and 4.6080, so the fitted slope is $`0.0020`$ per session and the fitted cumulative change is $`\exp(0.008)-1\approx+0.80\%`$. A hypothetical long strategy enters at the open of 101 and exits at the close of 104 under a four-session holding rule fixed at issuance. Its exit coincides with the fitted knot by construction of this example. Its gross simple return is $`101/102-1\approx-0.98\%`$. With a combined illustrative cost and slippage of 10 basis points per side applied to each fill, its net simple return is

```math
\frac{101(1-0.001)}{102(1+0.001)}-1\approx-1.18\%.
```

## 4. Right-censoring and label maturity

For an origin with full permitted labeling support, let $`S_t=K_t\cap(t,U]`$. Record the event flag $`\delta_t`$ and follow-up $`v_t`$ as

```math
(\delta_t,v_t)=
\begin{cases}
(1,\min S_t-t), & S_t\neq\varnothing,\\
(0,U-t), & S_t=\varnothing.
\end{cases}
```

If $`\delta_t=1`$, then $`b=\min S_t`$ and $`d_t=v_t`$. If $`\delta_t=0`$, the frozen fit has no admissible knot through $`U`$. The exact duration, slope and direction labels are unavailable.

| Status | Illustrative case at $`t=100`$ | Recorded outcome |
| --- | --- | --- |
| Completed | First valid knot at 104 | $`\delta_t=1,\ v_t=4`$, with $`y_t`$ |
| Right-censored | No valid knot through 128. A knot at 129 also falls outside the eligible range. | $`\delta_t=0,\ v_t=28`$, with no exact target tuple |
| Insufficient or disallowed support | Fit requires bar 130 and the partition ends at 125 | Record the exclusion separately from censoring |

Issue forecasts by the eligibility rules available at the origin. Match models on the same completed cases when scoring exact labels and explicitly label the scores as conditional on completion. Report candidate origins, missing support, split purges, censored counts, completed counts and usable labels. Compute completion rate as completed divided by completed plus censored, with counts by asset, partition and, where supported, origin-time regime. Completed cases can overrepresent short movements. The primary scorecard cannot establish performance on all origins. A future censor-aware duration analysis can use the observed follow-up times. An executable trading rule uses information available at issuance and during execution.

### Worked example · An early endpoint with a later maturity

The maturity session is $`m(t)=T=t+H`$ after the full window is available under a frozen rule. For origin 100, an endpoint at 104 still matures at bar 130 because the optimization jointly estimates the trend over sessions 81–130. The guard bars 129 and 130 can affect $`\widehat x^{(100)}_{100}`$, $`\widehat x^{(100)}_{104}`$ and the knot test. Record the actual availability timestamp of the required data alongside $`m(t)`$.

## 5. Why retrospective knots can move

The trend filter estimates its fitted levels jointly. Extending its support can change both the fitted values and the knot locations inside the original window. The following exact example uses arbitrary illustrative log levels and $`\lambda=0.1`$.

| Observed log levels | Fitted levels | Centered differences |
| --- | --- | --- |
| $`0,0,0.7`$ | $`-0.10,0.20,0.60`$ | $`0.10`$ at session 2 |
| $`0,0,0.7,0`$ | $`-0.05,0.20,0.45,0.10`$ | $`0`$ at session 2 and $`-0.60`$ at session 3 |

For the first row, the second difference at session 2 is $`0.60-2(0.20)-0.10=0.10`$. After appending one observation it becomes $`0.45-2(0.20)-0.05=0`$. Session 2 loses its knot and session 3 gains one. If the illustrative origin is session 1, the first retrospective knot shifts from 2 to 3 across the two fits.

To verify the two solutions, write $`D`$ for the second-difference operator and $`p`$ for the observed vector. The convex optimality condition is $`x=p-D^\top q`$, where $`q_i=\lambda\mathrm{sign}((Dx)_i)`$ at nonzero differences and $`|q_i|\le\lambda`$ at zeros. The first solution uses $`q=(0.10)`$. The second uses $`q=(0.05,-0.10)`$. Substitution gives the listed fitted levels in both cases.

For example, origins 100 and 101 use windows 81–130 and 82–131. Even when both origins appear to lie in the same movement, their independently fitted boundaries can differ. The Milestone 2 stability report will quantify adjacent-boundary disagreements and shifts over shared eligible follow-up, distinguish naturally elapsed boundaries, and audit sensitivity to the penalty, both window lengths and knot tolerance. It will report completion, censoring, episode counts and class balance alongside eligible origin counts.

### What fixing the window changes

Once the parameters, data snapshot and support for an origin are fixed, its label is frozen. At $`t=100`$, adding session 131 to the archive does not change the label based on sessions 81–130. Expanding the fit or adopting a revised vendor snapshot defines a new label version and should be recorded separately. A retrospectively located knot is an outcome of this procedure. It supplies neither a live trigger nor a confirmed boundary at the time it occurred.

## 6. Chronological splits and reproducibility

Use the proposal's chronological 75/15/10 train, validation and test design as the starting split policy, with aligned session cutoffs across ETFs. Use training data only for feature scaling, selection and label design. Choose label settings in a training prefix available before the evaluated origins and freeze them across comparisons. Use chronological development folds for model selection under a declared validation policy. Keep the final test out of design and tuning.

Assign a row by its origin and require all future label support through $`m(t)`$ to lie in its assigned partition. Purge rows whose labeling window crosses the cutoff. Preserve the original support for all retained labels. Earlier bars can supply warm-up and retrospective left context for a later partition when those bars were available by issuance. They cannot introduce later observations into training or fit new preprocessing parameters.

### Worked example · Purging near a split boundary

If training ends at 125, origin 95 has maturity 125 and passes the right-boundary check. Origin 96 needs bar 126 and is purged even if its fitted endpoint would be 104. A validation forecast at 126 can use earlier bars in its past-only input, and its label needs its own full future support through 156.

<!-- figure: split -->
    Origin     Required maturity     Training cutoff     Decision
    95         125                   125                 Pass
    96         126                   125                 Purge

**Figure 2.** Check maturity against the training cutoff. Origin 95 passes the split boundary check.

Adjacent origins share bars and can refer to the same fitted episode. Origin-specific fits can also disagree about the shared boundary. Track exact support intervals and repeated or nearby boundaries. Compare models on paired origins and retain ETF alignment. Use dependence-aware uncertainty, such as jointly resampling contiguous date blocks across ETFs if enough blocks are available. Prespecify any additional embargo based on the fold design or remaining dependence. An embargo does not substitute for exact purging.

Record data provenance, trading calendar and timezone, price adjustments and vintage limits, raw input and label support, issue and maturity times, split cutoffs, versions of features and labels, solver settings and later experiment configurations. Require invariance checks for inputs after issuance, labels after maturity, and training artifacts under perturbations to validation or test observations. The 501-row pilot would give roughly 50 test sessions per ETF and at most about 20 eligible test origins with $`H=30`$, before censoring. Milestone 2 will compare the 501-row pilot with candidate daily SPY and QQQ histories from January 1, 2010 through the latest completed trading session available in 2026 at the snapshot date. The data audit will verify provider availability, adjustments, quality and coverage before choosing the dataset.

## 7. Secondary fixed-horizon benchmark

For the same forecast origin and versioned positive adjusted closing series, define close-to-close log returns at provisional horizons $`h=5,10,20`$:

```math
r_{t,h}=\log\!\left(\frac{P_{t+h}}{P_t}\right),\qquad m_h(t)=t+h.
```

The two closes determine the label directly once both are available in the specified data version. Every horizon has its own maturity and split cutoff. Fixed-horizon returns aggregate price movement over a known length, possibly across several fitted boundaries. The primary target instead describes the fitted slope and remaining duration through an unknown boundary.

### Worked example · Same origin, different outcomes

At $`t=100`$, suppose $`P_{100}=100`$ and closes at 105, 110 and 120 are 102, 99 and 104. The three log returns are about $`0.01980`$, $`-0.01005`$ and $`0.03922`$, maturing at sessions 105, 110 and 120. They are defined whether or not the primary target is completed. If the fitted boundary is at 104, the primary slope uses fitted levels at 100 and 104 and the remaining duration is 4.

Use the same causal input budget and chronological protocol. Compare each target with its own simple reference, such as zero return for $`r_{t,h}`$, and report benchmark results on all eligible origins and on matched origins when contrasting tasks. Different outcomes have different units and difficulty, so raw errors across tasks cannot isolate the benefit of segmentation. To assess incremental value later, compare a return-only forecast with one augmented by model-predicted trend attributes for the same fixed-return outcome, using matched origins and past-only or out-of-fold predictions as appropriate. Realized fitted attributes would leak.

## 8. Four baseline approaches

The planned sequence uses the same revised primary target and origin information. Slope is continuous. Remaining duration is observed as a positive integer on completed cases, although a regression prediction can be real-valued. Direction is a categorical Up, Down or Flat label.

| Approach | Role in the immediate-movement experiment |
| --- | --- |
| Linear regression | Fit separate numerical predictions for $`s_t`$ and $`d_t`$ from a fixed summary or flattened history derived from $`X_t`$. Specify how any duration prediction is constrained to valid support. |
| Multinomial logistic regression | Estimate class probabilities for $`c_t`$ from the same causal feature budget. Select the predicted class under a declared rule. |
| LSTM | Preserve the order of the 20 feature rows. Use regression and classification heads, or separate models, to predict the same labels. Specify loss weights if training jointly. |
| Autoencoder | Learn a compressed representation by reconstructing historical inputs on training windows. Attach downstream predictors for the same labels and compare a frozen encoder with raw-feature baselines. |

For example, a slope prediction of $`+0.0010`$ implies Up under the band, while an independent classifier might give Down probability 0.55 and select Down. Separate objectives can give different decisions even though the observed class is a function of the observed slope. Report both the direct classifier and the class induced by thresholding the slope forecast, with agreement on matched origins.

## 9. End-to-end example and settings to validate

1. **Issue.** After bar 100 is available, assemble feature rows 81–100. Generate a forecast from those rows and the frozen model parameters.
2. **Wait for maturity.** After bar 130 is available, read the versioned adjusted log prices for sessions 81–130. Fit the trend on those 50 observations and search only sessions 101–128 for the first valid endpoint.
3. **Construct the label.** The fitted knots after issuance are 104 and 112. Select 104. With fitted values 4.6000 and 4.6080 at 100 and 104, store $`y_{100}=(0.0020,4,\text{Up})`$, the completed flag and maturity 130.
4. **Assign and compare.** If origin 100 belongs to training and training ends at 140, the example passes the right-boundary check. With a training cutoff at 125, purge it. For the retained case, regressions use the numerical labels, the classifier uses Up, and later sequence and representation models use the same origin and outcomes.

Compare numerical errors in native units and report directional accuracy, class support and a confusion matrix. Keep the proposal's polynomial comparator and subsequent JEPA comparisons in their scheduled milestones. Their primary labels must follow this same definition.

| Provisional setting | What to examine in Milestone 2 |
| --- | --- |
| $`L=20`$, $`B=20`$ | Feature warm-up, historical context and sensitivity of fitted knots to left support |
| $`H=30`$, $`g=2`$ | Completion, guard sensitivity and the trade-off between censoring and split purges |
| $`\varepsilon=0.0005`$, $`\tau=10^{-6}`$ | Class counts, numerical tolerance and boundary stability |
| $`\lambda_j`$ and $`h=5,10,20`$ | Training-only penalty grid, segment support and usable rows per benchmark horizon |

Open choices include data length and adjustment vintage, penalty calibration, minimum sample and class support, duration prediction constraints, the treatment of censoring beyond conditional scoring, and dependence-aware uncertainty. Resolve data-dependent settings on permitted development history before comparing models.

### References

[1] EECE 490 project proposal, *Probabilistic Forecasting of Future Market Trends and Volatility Envelopes*, §§1, 4–6, Figure 1 and Table 3.

[2] S.-J. Kim, K. Koh, S. Boyd and D. Gorinevsky, “ℓ₁ Trend Filtering,” *SIAM Review* 51(2), 339–360 (2009). DOI 10.1137/070690274.
