# EECE 490 stock dataset

The baseline covers **20 companies and 1,539 shared U.S. trading sessions**
from 2020-09-30 through 2026-11-13, inclusive. The first date is Palantir's
listing date and the earliest shared session. The end date is the latest
completed session when this snapshot was built. The cohort is fixed.

Ticker order in both files:
AAPL, NVDA, MSFT, GOOGL, TSM, AVGO, META, MU, AMD, ASML, PLTR, ORCL, INTC,
AMAT, SAP, LRCX, TXN, PANW, IBM, CRWD.

The underlying data come from [Yahoo Finance daily historical prices](https://finance.yahoo.com/quote/AAPL/history/).
Dates are U.S. exchange session dates. Prices, ranges, and changes are in
USD per share. Volume is in shares. The files preserve Yahoo's OHLC quote fields and use its
split-adjusted Close field for changes.

| Field | Definition |
| --- | --- |
| `open`, `high`, `low`, `close` | Yahoo Finance daily quote fields |
| `high_low_range` | Session high minus session low |
| `close_change` | Session close minus the previous trading session's close |
| `volume` | Yahoo Finance daily trading volume in shares |

`close_change` is blank in the CSV and `NaN` in the tensor on the first
date for each ticker. Later changes use the preceding row for the same
ticker, including across weekends and holidays.

## CSV

`yahoo_finance_daily.csv` has 30,780 rows, with one row per date and ticker.
Dates are ISO 8601 (`YYYY-MM-DD`). Rows are in ascending date order, with
the fixed ticker order inside each date.

```python
import pandas as pd

data = pd.read_csv("data/yahoo_finance_daily.csv", parse_dates=["date"])
aapl = data.loc[data["ticker"].eq("AAPL")].set_index("date")
```

## PyTorch

`yahoo_finance_daily.pt` contains the same data as a CPU float64 tensor
of shape `[1539, 20, 7]` in date × ticker × feature order. The boolean
`observed` tensor is true for every date and ticker in the shared calendar.
The dictionary also includes axis labels, dates, tickers, company names,
feature names, feature units, currency, and source.

```python
import torch

data = torch.load("data/yahoo_finance_daily.pt", map_location="cpu", weights_only=True)
values = data["values"]
aapl = values[:, data["tickers"].index("AAPL"), :]
close = aapl[:, data["features"].index("close")]
volume = aapl[:, data["features"].index("volume")]
```

## Plots

- [Baseline gallery](../figure_and_visulization/index.html) uses this snapshot.
  Close changes use a maximum-likelihood normal fit with mean μ and
  standard deviation σ, where the fitted variance divides by n. Intraday
  ranges use a maximum-likelihood gamma fit with location fixed at zero,
  shape α, and rate β. The gamma scale is 1 / β.
- [Daily spotlight](../daily_spotlight/index.html) ranks the same 20 tickers
  using their latest completed session and shows two plots each for five stocks.
- [Forecast comparison](../forecast_comparison/index.html) uses this aligned
  snapshot for the four-model benchmark.
- [2026 recursive forecast](../forecast_2026/index.html) uses the aligned
  snapshot to evaluate forecasts generated from the December 2025 cutoff.
- Run `python stock_distributions.py baseline` from the project folder to
  redraw the baseline from the CSV. Run `python stock_distributions.py daily`
  to fetch Yahoo Finance data and refresh the spotlight.
