# Self-Supervised Alpha Trend Distribution Predictor

EECE 490 research project on forecasting immediate market trends and their uncertainty. **Milestone 1 is complete. Milestone 2 has not started.** The mathematical specification is finalized. The data pipeline, model training, and experimental results are planned work.

## Forecasting task

After session $t$ closes, use information available at issuance to forecast the movement from $t$ to the first eligible future trend boundary $b>t$. The target is the fitted log-price slope from $t$ to $b$, the remaining duration $d_t=b-t$, and the direction (Up, Down, or Flat).

A fixed-window $\ell_1$ trend filter defines the labels retrospectively. If the fit has no eligible boundary by the cutoff, record a censored outcome and leave the exact target values unavailable. Fixed-horizon log returns at 5, 10, and 20 sessions are a separate benchmark.

## Milestones

| # | Milestone | Status |
| --- | --- | --- |
| 1 | Forecast and label specification | **Completed** |
| 2 | Data collection, quality and label-stability audits, chronological splits | **Not started** |
| 3 | Linear regression for slope and duration, multinomial logistic regression for direction, and a polynomial comparator | Planned |
| 4 | LSTM sequence baseline | Planned |
| 5 | Autoencoder and downstream prediction baseline | Planned |
| 6 | Joint-Embedding Predictive Architecture (JEPA) and representation comparisons | Planned |
| 7 | Probabilistic forecasts, multiple horizons, and volatility envelopes | Planned |
| 8 | Locked out-of-sample evaluation and research report | Planned |

## Planned data and evaluation

Use daily SPY and QQQ OHLCV from Yahoo Finance via yfinance, as proposed. Compare the 501-session pilot per ETF with a candidate history from January 2010 through the latest completed session available in 2026 at snapshot time. Verify coverage and adjustment conventions before selecting the dataset.

Start with chronological 75/15/10 training, validation, and test splits. Fit preprocessing on training data, purge examples whose label windows cross a split boundary, and reserve the test split for final evaluation. Compare models on matched origins, report label completion and censoring, and measure slope and duration error, directional accuracy, net PnL, and maximum drawdown. Trading simulations use executable prices with transaction costs and slippage.

Later work will estimate conditional distributions and volatility envelopes. Evaluation will check calibration and report interval coverage together with interval width.

## Research documents

| File | Purpose |
| --- | --- |
| [Research proposal](eece490_proposal.pdf) | Original project scope and methods |
| [Roadmap](ROADMAP.md) · [Milestones PDF](EECE490_Project_Milestones.pdf) · [LaTeX](EECE490_Project_Milestones.tex) | Eight milestones, deliverables, and completion criteria |
| [Milestone 1 formulation](MILESTONE_1_FORMULATION.md) · [PDF](MILESTONE_1_FORMULATION.pdf) · [LaTeX](MILESTONE_1_FORMULATION.tex) | Forecast origin, target definitions, censoring, and leakage controls |
| [Research decisions](RESEARCH_DECISIONS.md) | Methodological choices and settings to validate |

As development progresses, source code will go in **src/**, tests in **tests/**, experiment settings in **configs/**, dataset manifests and permitted snapshots in **data/**, and analysis outputs in **reports/**.

