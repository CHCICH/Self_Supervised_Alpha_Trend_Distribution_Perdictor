# Six-model stock forecasting comparison

All 20 stocks are modeled independently. The comparison covers opening
price, closing price, and daily high minus low in USD per share. The
existing shared Yahoo Finance history supplies the observations.

## Time-only forecasts

The full comparison issues forecasts after the December 31, 2025
close. The time figures show January–April 2026, and the loss table
scores all 219 sessions through November 13. The
[one-month gallery](../1month_figures_and_visualizations/index.html)
issues the time-only forecasts after October 12, 2026 and scores the
24 sessions from October 13 through November 13. The lagged models
in both galleries retain their original block issue dates.

Each time-only curve uses `h`, the number of trading sessions since
issuance, as its only forecasting input. Parameters are estimated
using observations available when the forecast is issued.

| Family | Inputs and parameters |
| --- | --- |
| Flat baseline | Last observed open or close for the corresponding price target, and the mean of the last 21 ranges for high–low. |
| Damped trend | `a + w × b × τ × (1 − exp(−h/τ))`. The open and close both use a 126-session fitted slope `b` with weight 0.5. Open uses the last five opens as its mean anchor `a` and τ = 63. Close uses the last close and τ = 21. For range, use its last 21 sessions as the mean anchor with weight 0.25. The one-month range slope uses 21 sessions and τ = 5, and the full-year range slope uses 126 sessions and τ = 126. |
| Shifted power curve | `A × (1 + 3h/248)^B`, fitted on log levels from the last 63 observed sessions. This was the leading long-horizon candidate for the price targets in the exploratory sweep. The range version is included as a comparison. |

The one-month gallery separately shows a natural cubic spline with
13 knots fitted to actual values within the displayed month. This
retrospective fit is excluded from forecast scores.

## Lagged models and block forecasts

The **linear regressive model**, **RNN**, and **LSTM** each get a
60-session window containing open, close, and high–low range.
The linear model is ordinary least squares on the flattened window.
The RNN and LSTM each use one recurrent layer with 24 hidden units
and a linear output head. All three fit direct predictions of changes
relative to the last input session, then add back those observed levels.
The lagged linear regression reports raw affine predictions. The RNN
and LSTM retain their zero floor.

Each lagged method is fitted once per stock to produce a 30-session
forecast vector. Use its first (h) predictions to evaluate block
size (h). This keeps the model fixed when comparing how performance
changes as forecasts are refreshed less often. For the RNN and LSTM,
select the number of epochs from 4, 8, 12, and 16 using the second half
of 2025 as validation. Then refit with training data from September 2020
through December 2025. Every target in every training window precedes
the corresponding validation or test cutoff. Neural training uses
standardized inputs and changes, Adam at 0.003, and batches of up to
128 windows.

For each test block size, start at the first 2026 session. The model
sees the previous 60 actual sessions and issues the whole block at once.
Once all (h) sessions have completed, move forward (h) sessions and
construct the next 60-session window from revealed observations.
The weights stay frozen. For example, with (h=5), the second forecast
is issued only after the fifth actual session of the first block.
The last block is truncated to the remaining observations in the dataset.

## Student’s t closing-price forecast

The closing-price panel uses a Student’s t increment model in the slot
previously occupied by the linear regressive model. The opening-price
and range panels still use the saved linear regressive forecasts.

Fit a separate Student’s t distribution by maximum likelihood for each
stock on 1,319 close changes. The underlying closes run from September
30, 2020 through December 31, 2025, inclusive. [Fitted parameters](student_t_parameters.csv)
include degrees of freedom, location, and scale. The mean and variance
columns are blank where the fitted distribution lacks a finite moment.

For every test block, start from the last actual close before the block.
Add one independent draw from that stock’s fitted distribution per
future session. After the block, anchor the next forecast to the new
actual close. At lead `j` the forecast median is the anchor plus `j`
times the fitted location. This is the line scored by MSE. The shaded
pointwise 80% interval is analytic for lead 1 and uses 50,000 seeded
simulated sums for longer leads. The fit and simulation use no 2026
price changes.

The [40 figures](index.html) include five horizon rows for the lagged
comparison for each stock (1, 5, 10, 20, and 30 sessions). Each row has
separate axes for the three targets. The [horizon explorer](all_horizons.html)
plots all tested sizes, including 2–4 and 6–9.

## Scores and predictions

- [Test MSE](test_mse.csv) gives each stock, target, and model. For each
  lagged block size it reports pooled MSE and MSE by lead within the block.
- [Summary MSE](summary_mse.csv) averages stock-level MSE and reports the
  median as well. Squared dollar errors give larger-priced stocks more
  influence on the mean.
- [Average MSE by block size](avg_mse_by_horizon.png) and its [chart values](mse_by_horizon.csv)
  report total squared error divided by the number of predicted values
  across all 20 stocks, separately for each target. The top row uses
  every prediction in each test block, including the final partial
  block. The bottom row uses the final prediction from each complete
  (K)-session block. Time-only predictions have the same pooled MSE
  for every (K). The endpoint score for those models uses the
  same dates as the lagged-model endpoint scores.
- [Block predictions](block_predictions.csv) records the session after
  whose close the block was issued, the target date, and the lead.
  Closing rows include the Student’s t median and pointwise 10th and
  90th percentiles. The linear column applies to opening price and range.
- [Time predictions](time_predictions.csv) contains the full scored 2026 period.
- [Model selection](model_selection.csv) records selected RNN/LSTM epoch counts.

Run `python -B stock_forecast_comparison.py` from the project directory
to reproduce the figures and scores.
To update just the closing-price comparator from the saved forecasts,
run `python -B -c "from stock_forecast_comparison import replace_linear_close; replace_linear_close()"`.
