"""Fit six single-stock models and compare fixed-window 2026 forecasts."""

import csv
from collections import defaultdict
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t
import torch
from torch import nn

from stock_distributions import DATA, ROOT, STOCKS, read_snapshot


OUTPUT = ROOT / "forecast_comparison"
TARGETS = ("open", "close", "high_low_range")
LABELS = ("Opening price", "Closing price", "High–low range")
TIME_MODELS = ("Flat baseline", "Damped trend", "Shifted power curve")
LAG_MODELS = ("Linear regressive model", "RNN", "LSTM")
STUDENT_CLOSE = "Student's t close"
T_BOUNDS = ("student_t_p10", "student_t_p90")
STUDENT_COLOR = "#711e9d"
STUDENT_BAND = "#ee98c7"
STYLES = (("#b73d15", "-"), ("#2357b7", "--"), ("#08764c", "-."))
HORIZONS = (*range(1, 11), 20, 30)
DISPLAY_HORIZONS = (1, 5, 10, 20, 30)
LOOKBACK = 60
FORECAST = max(HORIZONS)
CHECKPOINTS = (4, 8, 12, 16)
HIDDEN = 24


def linear_fit(inputs, targets):
    """Fit affine least squares on standardized features."""
    center = inputs.mean(axis=0)
    scale = inputs.std(axis=0)
    scale[scale == 0] = 1
    x = (inputs - center) / scale
    intercept = targets.mean(axis=0)
    y = targets - intercept
    weights = np.linalg.lstsq(x, y, rcond=None)[0]

    def predict(values):
        return ((values - center) / scale) @ weights + intercept

    return predict


def fit_time_models(history, count, *, short_window=False):
    """Fit alternate time curves using only data known at the forecast origin."""
    if len(history) < 126 or np.any(history[-63:] <= 0):
        raise ValueError("Time forecasts require 126 sessions and positive recent values")
    h = np.arange(1, count + 1)
    flat = np.array([history[-1, 0], history[-1, 1], history[-21:, 2].mean()])
    damped = np.empty((count, len(TARGETS)))
    range_settings = (21, 5, .25, 21) if short_window else (126, 126, .25, 21)
    settings = ((126, 63, .5, 5), (126, 21, .5, 1), range_settings)
    for j, (window, tau, weight, anchor_window) in enumerate(settings):
        x = np.arange(window, dtype=float)
        x -= x.mean()
        slope = (x @ history[-window:, j]) / (x @ x)
        anchor = history[-anchor_window:, j].mean()
        damped[:, j] = anchor - weight * slope * tau * np.expm1(-h / tau)

    # A + B log(1 + 3z) in log levels, with z = 1 at the issue date.
    sessions = np.arange(-62, 1)
    terms = np.log1p(3 * (1 + sessions / 62))
    design = np.column_stack((np.ones(len(sessions)), terms))
    intercept, exponent = np.linalg.lstsq(design, np.log(history[-63:]), rcond=None)[0]
    power = np.exp(intercept + np.log1p(3 * (1 + h / 62))[:, None] * exponent)
    return dict(zip(TIME_MODELS, (np.tile(flat, (count, 1)), damped, power)))


def inputs_at(values, origins):
    return values[origins[:, None] - np.arange(LOOKBACK, 0, -1)]


def training_windows(values):
    origins = np.arange(LOOKBACK, len(values) - FORECAST + 1)
    inputs = inputs_at(values, origins)
    targets = values[origins[:, None] + np.arange(FORECAST)]
    return inputs, targets - inputs[:, -1:, :]


def linear_regressive(train):
    inputs, deltas = training_windows(train)
    predict_delta = linear_fit(inputs.reshape(len(inputs), -1),
                               deltas.reshape(len(inputs), -1))

    def predict(history):
        changes = predict_delta(history.reshape(len(history), -1))
        return history[:, -1:, :] + changes.reshape(-1, FORECAST, len(TARGETS))

    return predict


class Recurrent(nn.Module):
    def __init__(self, cell):
        super().__init__()
        self.cell = cell(len(TARGETS), HIDDEN, batch_first=True)
        self.output = nn.Linear(HIDDEN, FORECAST * len(TARGETS))

    def forward(self, history):
        hidden, _ = self.cell(history)
        return self.output(hidden[:, -1]).reshape(-1, FORECAST, len(TARGETS))


def as_tensor(values):
    return torch.tensor(values, dtype=torch.float32)


def neural_fit(train, cell, seed, epochs, validation_values=None):
    inputs, delta = training_windows(train)
    center = train.mean(axis=0)
    scale = train.std(axis=0)
    delta_scale = delta.std(axis=0)
    x, y = as_tensor((inputs - center) / scale), as_tensor(delta / delta_scale)
    torch.manual_seed(seed)
    order = torch.Generator().manual_seed(seed)
    model = Recurrent(cell)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    if validation_values is not None:
        origins = np.arange(len(train), len(validation_values), FORECAST)
        histories = inputs_at(validation_values, origins)
        indices = origins[:, None] + np.arange(FORECAST)
        available = indices < len(validation_values)
        true = validation_values[indices[available]]
        checkpoint_losses = []

    def predict(history):
        model.eval()
        with torch.inference_mode():
            changes = model(as_tensor((history - center) / scale)).numpy() * delta_scale
        return np.maximum(history[:, -1:, :] + changes, 0)

    for epoch in range(1, epochs + 1):
        model.train()
        for indices in torch.randperm(len(x), generator=order).split(128):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(x[indices]), y[indices])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if validation_values is not None and epoch in CHECKPOINTS:
            predicted = predict(histories)[available]
            validation_loss = np.mean(((predicted - true) / scale) ** 2)
            checkpoint_losses.append((float(validation_loss), epoch))
    if validation_values is not None:
        return min(checkpoint_losses)[1]
    return predict


def rolling_blocks(values, split, predict):
    """Each direct forecast sees the 60 actual sessions preceding its origin."""
    origins = np.arange(split, len(values))
    # Model weights stay fixed. Cache one direct block per possible origin.
    direct = predict(inputs_at(values, origins))
    if not np.isfinite(direct).all():
        raise ValueError("Nonfinite lagged-model output")
    return {width: assemble_blocks(direct, width) for width in HORIZONS}


def assemble_blocks(direct, width):
    offset = np.arange(len(direct))
    lead = offset % width
    return direct[offset - lead, lead]


def student_close_fit(training_close, seed):
    changes = np.diff(training_close)
    fits = (student_t.fit(changes, start) for start in (2, 5, 20))
    df, location, scale = max(fits, key=lambda params: student_t.logpdf(changes, *params).sum())
    rng = np.random.default_rng(seed)
    innovations = np.cumsum(rng.standard_t(df, (50_000, FORECAST)), axis=1) * scale
    radius = np.quantile(np.abs(innovations), 0.80, axis=0)
    radius[0] = student_t.ppf(0.90, df) * scale
    return (df, location, scale), radius


def student_close_blocks(closes, split, parameters, radius):
    offset = np.arange(len(closes) - split)
    location = parameters[1]
    blocks, intervals = {}, {}
    for width in HORIZONS:
        lead = offset % width
        previous_close = closes[split + offset - lead - 1]
        median = previous_close + (lead + 1) * location
        blocks[width] = median
        intervals[width] = np.column_stack((median - radius[lead], median + radius[lead]))
    return blocks, intervals


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.20)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)


def draw_lines(ax, days, values, predictions, target):
    ax.plot(days, values[:, target], color="#18283d", linewidth=1.35,
            alpha=.86, label="Actual")
    for (name, data), (color, line) in zip(predictions.items(), STYLES):
        if name == STUDENT_CLOSE:
            color = STUDENT_COLOR
        ax.plot(days, data[:, target], linewidth=2.0 if name == STUDENT_CLOSE else 1.5,
                color=color, linestyle=line, label=name)
    style_axis(ax)


def display_dates(ax, monthly=False, short_window=False):
    if short_window:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        return
    locator = mdates.MonthLocator(interval=1 if monthly else 3)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))


def time_plot(ticker, days, actual, forecasts, *, period=None, issued="December 31, 2025"):
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, layout="constrained")
    fig.suptitle(f"{ticker} / {STOCKS[ticker]}  ·  Time functions",
                 fontsize=21, fontweight="bold", color="#18283d")
    for i, ax in enumerate(axes):
        draw_lines(ax, days, actual, forecasts, i)
        ax.set_title(LABELS[i], loc="left", fontsize=12, fontweight="bold")
        ax.set_ylabel("USD per share")
    axes[0].legend(loc="best", ncols=2, fontsize=9)
    display_dates(axes[-1], monthly=True, short_window=period is not None)
    fig.supxlabel(f"{period or 'January–April 2026'}  ·  Forecast issued after the {issued} close")
    fig.savefig(OUTPUT / f"{ticker}_time.png", dpi=134, facecolor="white")
    plt.close(fig)


def lag_plot(ticker, days, actual, forecasts, *, period=None, block_starts=None, close_intervals=None):
    fig, axes = plt.subplots(5, 3, figsize=(17, 15), sharex=True)
    fig.subplots_adjust(top=.91, bottom=.067, left=.065, right=.982, hspace=.16, wspace=.18)
    fig.suptitle(f"{ticker} / {STOCKS[ticker]}  ·  Lagged model comparison",
                 y=.984, fontsize=21, fontweight="bold", color="#18283d")
    for row, width in enumerate(DISPLAY_HORIZONS):
        predictions = {model: blocks[width] for model, blocks in forecasts.items()}
        for target, ax in enumerate(axes[row]):
            if target == 1 and close_intervals is not None:
                lower, upper = close_intervals[width].T
                ax.fill_between(days, lower, upper, color=STUDENT_BAND, alpha=.53,
                                label="Student’s t 80% interval")
                comparison = {STUDENT_CLOSE if method == LAG_MODELS[0] else method: result
                              for method, result in predictions.items()}
            else:
                comparison = predictions
            draw_lines(ax, days, actual, comparison, target)
            if width >= 10 or (block_starts is not None and width > 1):
                boundaries = days[width::width] if block_starts is None else block_starts[width]
                for boundary in boundaries:
                    ax.axvline(boundary, color="#b9c3ce", linewidth=.55, alpha=.50)
            if row == 0:
                ax.set_title(LABELS[target], fontsize=12, fontweight="bold")
            if target == 0:
                ax.set_ylabel(f"{width} session{'s' if width != 1 else ''}\nUSD per share",
                              fontsize=10, fontweight="bold")
    display_dates(axes[-1, 0], short_window=period is not None)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if close_intervals is not None:
        extra_handles, extra_labels = axes[0, 1].get_legend_handles_labels()
        for handle, label in zip(extra_handles, extra_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .959),
               ncols=len(labels), fontsize=9, frameon=False)
    fig.supxlabel(f"{period or 'Target date in 2026'}  ·  Each forecast block uses the preceding 60 actual sessions",
                  y=.023)
    fig.savefig(OUTPUT / f"{ticker}_lagged.png", dpi=133, facecolor="white")
    plt.close(fig)


def append_scores(scores, ticker, method, width, actual, forecast, *, student_close=False):
    square = (actual - forecast) ** 2
    names = [STUDENT_CLOSE if student_close and key == "close" else method for key in TARGETS]
    scores.extend((ticker, name, key, width, "all", len(actual), float(square[:, i].mean()))
                  for i, (key, name) in enumerate(zip(TARGETS, names)))
    if width is not None:
        for lead in range(width):
            errors = square[lead::width]
            scores.extend((ticker, name, key, width, lead + 1, len(errors), float(errors[:, i].mean()))
                          for i, (key, name) in enumerate(zip(TARGETS, names)))


def write_csv(name, columns, rows):
    with (OUTPUT / name).open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(rows)


def write_mse_chart(scores, *, time_file=None, start=None, end=None, title=None, offset=0):
    grouped = defaultdict(list)
    for _, model, target, width, lead, count, mse in scores:
        grouped[model, target, width, lead].append((count, mse))
    time_errors = defaultdict(list)
    with (time_file or OUTPUT / "time_predictions.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            for model in TIME_MODELS:
                error = (float(row["actual"]) - float(row[model])) ** 2
                time_errors[row["ticker"], model, row["target"]].append((row["target_date"], error))

    def in_window(day):
        return (start is None or day >= start) and (end is None or day <= end)

    def pooled(groups):
        observations = sum(count for count, _ in groups)
        mse = sum(count * value for count, value in groups) / observations if observations else np.nan
        return observations, mse

    rows = []
    methods = (*TIME_MODELS, *LAG_MODELS)
    colours = ("#7362a0", "#975381", "#5d646e", "#b73d15", "#2357b7", "#08764c")
    lines = ("--", "-.", ":", "-", "-", "-")
    markers = ("v", "x", "+", "o", "s", "D")
    fig, axes = plt.subplots(2, 3, figsize=(14.6, 8.5), sharex=True)
    fig.subplots_adjust(left=.077, right=.98, top=.78, bottom=.118, hspace=.42, wspace=.30)
    fig.suptitle(title or "2026 forecast error by block size", y=.97, fontsize=23, fontweight="bold",
                 color="#18283d")
    fig.text(.5, .909, "Upper row · all predictions in each block     Lower row · only the K-th prediction",
             ha="center", fontsize=11, color="#415469")
    for column, (target, label) in enumerate(zip(TARGETS, LABELS)):
        for model, color, style, marker in zip(methods, colours, lines, markers):
            block_values, endpoint_values = [], []
            if target == "close" and model == LAG_MODELS[0] and (STUDENT_CLOSE, target, 1, "all") in grouped:
                model = STUDENT_CLOSE
                color = STUDENT_COLOR
            for k in HORIZONS:
                if model in TIME_MODELS:
                    whole = grouped[model, target, None, "all"]
                    start_index = (k - 1 - offset) % k
                    endpoints = [[error for day, error in errors[start_index::k] if in_window(day)]
                                 for (ticker, name, field), errors in time_errors.items()
                                 if name == model and field == target]
                    exact = [(len(errors), float(np.mean(errors))) for errors in endpoints if errors]
                else:
                    whole = grouped[model, target, k, "all"]
                    exact = grouped[model, target, k, k]
                n, block_mse = pooled(whole)
                n_exact, end_mse = pooled(exact)
                rows.append((target, model, k, n, block_mse, n_exact, end_mse))
                block_values.append(block_mse)
                endpoint_values.append(end_mse)
            for ax, y in zip(axes[:, column], (block_values, endpoint_values)):
                ax.plot(HORIZONS, y, color=color, linestyle=style, marker=marker,
                        markersize=4, linewidth=1.8, label=model)
        axes[0, column].set_title(label, fontsize=12, fontweight="bold")
        axes[1, column].set_xlabel("Block size K (trading sessions)")
        for ax in axes[:, column]:
            ax.set_yscale("log")
            ax.set_xticks((1, 5, 10, 20, 30))
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(alpha=.23)
    for ax in axes[:, 0]:
        ax.set_ylabel("MSE (USD² per share²)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    extra_handles, extra_labels = axes[0, 1].get_legend_handles_labels()
    for handle, label in zip(extra_handles, extra_labels):
        if label not in labels:
            handles.append(handle)
            labels.append(label)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .875),
               ncols=4, fontsize=9.2, frameon=False)
    fig.text(.5, .041, "Pooled across 20 stocks. Logarithmic MSE axis. The K-th error uses completed blocks only.",
             ha="center", fontsize=10, color="#415469")
    fig.savefig(OUTPUT / "avg_mse_by_horizon.png", dpi=145, facecolor="white")
    plt.close(fig)
    write_csv("mse_by_horizon.csv", ("target", "model", "k_sessions", "all_observations",
                                    "block_mse", "kth_observations", "kth_mse"), rows)


def write_results(scores, dates, split):
    summary = []
    student_close = any(row[1] == STUDENT_CLOSE for row in scores)
    comparisons = [(name, (None,)) for name in TIME_MODELS] + [(name, HORIZONS) for name in LAG_MODELS]
    for method, widths in comparisons:
        for width in widths:
            for target in TARGETS:
                shown = STUDENT_CLOSE if student_close and method == LAG_MODELS[0] and target == "close" else method
                group = [row[-1] for row in scores if row[1:5] == (shown, target, width, "all")]
                summary.append((shown, target, width, float(np.mean(group)), float(np.median(group))))
    student_link = (' · <a href="student_t_parameters.csv">Student’s t parameters</a>' if student_close else "")
    write_csv("summary_mse.csv", ("model", "target", "block_sessions", "mean_stock_mse",
                                 "median_stock_mse"), summary)
    rows = []
    for method, target, width, mean, median in summary:
        if width is None or width in DISPLAY_HORIZONS:
            label = "Time only" if width is None else str(width)
            rows.append(f"<tr><td>{method}</td><td>{target}</td><td>{label}</td>"
                        f"<td>{mean:,.2f}</td><td>{median:,.2f}</td></tr>")
    cards = "".join(
        f'<section><h2>{ticker} · {name}</h2><div>'
        f'<a href="{ticker}_time.png"><img loading="lazy" src="{ticker}_time.png" alt="Time functions"></a>'
        f'<a href="{ticker}_lagged.png"><img loading="lazy" src="{ticker}_lagged.png" alt="Lagged models"></a>'
        '</div></section>' for ticker, name in STOCKS.items()
    )
    (OUTPUT / "index.html").write_text(f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EECE 490 · Six-model comparison</title>
<style>
body{{margin:0;background:#edf2f7;color:#172740;font:16px/1.5 "Segoe UI",sans-serif}}
main{{max-width:1430px;margin:auto;padding:24px}}section{{background:white;border:1px solid #d5e0ea;
border-radius:10px;margin:20px 0;padding:20px}}h2{{margin-top:0}}
section div{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(470px,100%),1fr));gap:12px}}
img{{width:100%;display:block;border:1px solid #d5e0ea;box-sizing:border-box}}
table{{border-collapse:collapse;background:white}}td,th{{padding:6px 10px;border-bottom:1px solid #dce5ec;text-align:left}}
</style><main><h1>Six-model stock forecasting comparison</h1>
<p>20 stocks · 40 figures · opening price, closing price, and high–low range</p>
<p>Time forecasts use recent history through {dates[split-1]}. Lagged models fit on {dates[0]}–{dates[split-1]}.
All scores use {dates[split]}–{dates[-1]}.</p>
<p><a href="README.md">Methods and split</a> · <a href="test_mse.csv">Per-stock MSE</a> ·
<a href="summary_mse.csv">All-horizon summary</a> · <a href="block_predictions.csv">Dated block predictions</a> ·
<a href="all_horizons.html">Plot any horizon</a>{student_link}</p>
<p><a href="avg_mse_by_horizon.png">Average MSE by K</a> · <a href="mse_by_horizon.csv">Chart values</a></p>
<details><summary>MSE by method, target, and displayed block size</summary><table>
<thead><tr><th>Model</th><th>Target</th><th>Block size</th><th>Mean</th><th>Median</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></details>{cards}</main></html>""", encoding="utf-8")


def write_horizon_view(*, block_file=None, start=None, end=None):
    data = {ticker: {str(horizon): {target: [] for target in TARGETS} for horizon in HORIZONS}
            for ticker in STOCKS}
    dates = []
    with (block_file or OUTPUT / "block_predictions.csv").open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        revised = STUDENT_CLOSE in reader.fieldnames
        for row in reader:
            day = row["target_date"]
            if (start is not None and day < start) or (end is not None and day > end):
                continue
            ticker, horizon, target = (row[key] for key in ("ticker", "block_sessions", "target"))
            first = STUDENT_CLOSE if revised and target == "close" else LAG_MODELS[0]
            data[ticker][horizon][target].append(
                [round(float(row[key]), 4) for key in ("actual", first, *LAG_MODELS[1:])])
            if ticker == next(iter(STOCKS)) and horizon == "1" and target == TARGETS[0]:
                dates.append(day)

    payload = json.dumps({"data": data, "dates": dates, "targets": TARGETS, "labels": LABELS,
                          "series": ("Actual", LAG_MODELS[0], *LAG_MODELS[1:]), "horizons": HORIZONS,
                          "colors": ("#18283d", *(color for color, _ in STYLES)),
                          "studentColor": STUDENT_COLOR if revised else None},
                         separators=(",", ":"))
    template = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Block horizon explorer</title>
<style>
body{margin:0;background:#edf2f7;color:#172740;font:16px/1.5 "Segoe UI",sans-serif}
main{max-width:1180px;margin:auto;padding:24px}label{margin-right:16px}
select{font:inherit;margin-left:7px;padding:3px}canvas{display:block;width:100%;height:640px;background:white;
border:1px solid #d5e0ea;border-radius:9px;margin-top:16px}
.legend{display:flex;gap:16px;flex-wrap:wrap;margin-top:14px}.legend span{font-weight:600}
</style><main><p><a href="index.html">Back to the 40 figures</a></p><h1>Compare any forecast horizon</h1>
<label>Stock<select id="ticker"></select></label>
<label>Sessions per forecast block<select id="horizon"></select></label>
<div class="legend" id="legend"></div>
<canvas id="chart" role="img" aria-label="Actual and forecast prices and intraday ranges"></canvas>
</main><script>
const payload = __PAYLOAD__;
const ticker = document.querySelector("#ticker");
const horizon = document.querySelector("#horizon");
const canvas = document.querySelector("#chart");
const legend = document.querySelector("#legend");
for (const symbol of Object.keys(payload.data)) ticker.add(new Option(symbol, symbol));
for (const value of payload.horizons) horizon.add(new Option(value, value));
for (const [i, name] of payload.series.entries()) {
  const item = document.createElement("span");
  item.textContent = i === 1 && payload.studentColor ? name + " (open and range)" : name;
  item.style.color = payload.colors[i];
  legend.append(item);
}
if (payload.studentColor) {
  const item = document.createElement("span");
  item.textContent = "Student’s t close";
  item.style.color = payload.studentColor;
  legend.append(item);
}
function draw() {
  const width = canvas.clientWidth, height = canvas.clientHeight;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.ceil(width * ratio); canvas.height = Math.ceil(height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  const left = 76, right = width - 24, section = 198;
  const values = payload.data[ticker.value][horizon.value];
  canvas.setAttribute("aria-label", ticker.value + " results for " + horizon.value + "-session blocks");
  for (const [j, target] of payload.targets.entries()) {
    const rows = values[target];
    let low = Infinity, high = -Infinity;
    for (const row of rows) for (const value of row) {
      low = Math.min(low, value); high = Math.max(high, value);
    }
    const pad = (high - low) * 0.08 || 1;
    low -= pad; high += pad;
    const top = j * section + 42, bottom = top + 145;
    const x = (i) => left + i * (right - left) / (rows.length - 1);
    const y = (v) => bottom - (v - low) * (bottom - top) / (high - low);
    ctx.fillStyle = "#172740"; ctx.font = "bold 14px Segoe UI";
    ctx.fillText(payload.labels[j] + " (USD per share)", left, top - 15);
    for (let tick = 0; tick <= 4; tick++) {
      const value = low + (high - low) * tick / 4;
      const yy = y(value);
      ctx.beginPath(); ctx.moveTo(left, yy); ctx.lineTo(right, yy);
      ctx.strokeStyle = "#dce6ef"; ctx.lineWidth = 1; ctx.stroke();
      ctx.fillStyle = "#415469"; ctx.font = "11px Segoe UI";
      ctx.fillText(value.toLocaleString(undefined, {maximumFractionDigits: 2}), 12, yy + 4);
    }
    for (const [i, color] of payload.colors.entries()) {
      ctx.beginPath(); ctx.moveTo(x(0), y(rows[0][i]));
      for (let k = 1; k < rows.length; k++) ctx.lineTo(x(k), y(rows[k][i]));
      const student = target === "close" && i === 1 && payload.studentColor;
      ctx.strokeStyle = student || color; ctx.lineWidth = student ? 2.2 : (i === 0 ? 1.45 : 1.8);
      ctx.setLineDash([[], [], [5,3], [7,3,2,3]][i]); ctx.stroke(); ctx.setLineDash([]);
    }
    if (j === 2) {
      ctx.fillStyle = "#415469"; ctx.font = "11px Segoe UI"; ctx.textAlign = "center";
      for (let label = 0; label <= 4; label++) {
        const i = Math.round((rows.length - 1) * label / 4);
        ctx.fillText(payload.dates[i].slice(5), x(i), bottom + 19);
      }
      ctx.textAlign = "left";
    }
  }
}
ticker.addEventListener("change", draw); horizon.addEventListener("change", draw);
new ResizeObserver(draw).observe(canvas);
draw();
</script></html>"""
    (OUTPUT / "all_horizons.html").write_text(template.replace("__PAYLOAD__", payload), encoding="utf-8")


def replace_linear_close():
    """Update the closing-price comparator from saved forecasts and the historical closes."""
    panel = read_snapshot()
    date_keys = [row[0].isoformat() for row in panel["AAPL"]]
    indices = {day: index for index, day in enumerate(date_keys)}
    dates = np.asarray(date_keys)
    split, stop = [int(np.searchsorted(dates, boundary))
                   for boundary in ("2026-01-01", "2027-01-01")]
    dates = dates[:stop]
    test = dates[split:].astype("datetime64[D]")
    count = len(test)
    blocks, bounds, parameters = {}, {}, []
    for index, (ticker, rows) in enumerate(panel.items()):
        closes = np.array([row[4] for row in rows[:stop]])
        fit, radius = student_close_fit(closes[:split], 2025 + index)
        blocks[ticker], bounds[ticker] = student_close_blocks(closes, split, fit, radius)
        df, location, scale = fit
        parameters.append((ticker, dates[0], dates[split-1], split-1, df, location, scale,
                           location if df > 1 else "", scale**2 * df / (df-2) if df > 2 else ""))

    actuals = {ticker: np.full((count, 3), np.nan) for ticker in STOCKS}
    forecasts = {ticker: {method: {h: np.full((count, 3), np.nan) for h in DISPLAY_HORIZONS}
                          for method in LAG_MODELS} for ticker in STOCKS}
    rows = []
    target_columns = {target: i for i, target in enumerate(TARGETS)}
    with (OUTPUT / "block_predictions.csv").open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        headers = tuple(field for field in reader.fieldnames if field not in (STUDENT_CLOSE, *T_BOUNDS))
        for row in reader:
            ticker, target = row["ticker"], row["target"]
            width, lead = int(row["block_sessions"]), int(row["lead_sessions"])
            index = indices[row["target_date"]] - split
            if lead != index % width + 1 or row["issued_after_close"] != dates[split+index-lead]:
                raise ValueError(f"Unexpected forecast origin for {ticker} on {row['target_date']}")
            t_fields = ("", "", "")
            if target == "close":
                t_fields = (blocks[ticker][width][index], *bounds[ticker][width][index])
                row[LAG_MODELS[0]] = ""
            rows.append((*[row[key] for key in headers], *t_fields))
            if width in DISPLAY_HORIZONS:
                j = target_columns[target]
                actuals[ticker][index, j] = float(row["actual"])
                for method in LAG_MODELS:
                    value = t_fields[0] if target == "close" and method == LAG_MODELS[0] else row[method]
                    forecasts[ticker][method][width][index, j] = float(value)

    scores = []
    with (OUTPUT / "test_mse.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["target"] == "close" and row["model"] in (LAG_MODELS[0], STUDENT_CLOSE):
                continue
            width, lead = row["block_sessions"], row["lead_sessions"]
            scores.append((row["ticker"], row["model"], row["target"], int(width) if width else None,
                           lead if lead == "all" else int(lead), int(row["observations"]),
                           float(row["mse_usd_per_share_squared"])))
    for ticker, actual in actuals.items():
        if not np.isfinite(actual).all():
            raise ValueError(f"Missing actual values for {ticker}")
        for width, median in blocks[ticker].items():
            errors = (actual[:, 1] - median) ** 2
            scores.append((ticker, STUDENT_CLOSE, "close", width, "all", count, float(errors.mean())))
            for lead in range(width):
                selected = errors[lead::width]
                scores.append((ticker, STUDENT_CLOSE, "close", width, lead+1,
                               len(selected), float(selected.mean())))

    write_csv("student_t_parameters.csv", ("ticker", "first_price_date", "last_price_date",
                                           "close_changes", "df", "location_usd", "scale_usd",
                                           "mean_usd_if_finite", "variance_usd_squared_if_finite"),
              parameters)
    write_csv("block_predictions.csv", (*headers, STUDENT_CLOSE, *T_BOUNDS), rows)
    write_csv("test_mse.csv", ("ticker", "model", "target", "block_sessions", "lead_sessions",
                              "observations", "mse_usd_per_share_squared"), scores)
    for ticker in STOCKS:
        lag_plot(ticker, test, actuals[ticker], forecasts[ticker], close_intervals=bounds[ticker])
    write_results(scores, dates, split)
    write_horizon_view()
    write_mse_chart(scores)


def main():
    torch.set_num_threads(1)
    dataset = torch.load(DATA.with_suffix(".pt"), map_location="cpu", weights_only=True)
    dates = np.asarray(dataset["dates"])
    dates_dt = dates.astype("datetime64[D]")
    split = int(np.searchsorted(dates, "2026-01-01"))
    year_end = np.searchsorted(dates, "2027-01-01")
    dates, dates_dt = dates[:year_end], dates_dt[:year_end]
    assert tuple(dataset["tickers"]) == tuple(STOCKS) and 0 < split < len(dates)
    display = dates[split:] < "2026-05-01"
    fields = [dataset["features"].index(target) for target in TARGETS]
    panel = dataset["values"][:year_end, :, fields].numpy()
    if not dataset["observed"][:year_end].all() or not np.isfinite(panel).all():
        raise ValueError("A complete aligned dataset is required")
    OUTPUT.mkdir(exist_ok=True)
    scores, selections, block_rows, time_rows = [], [], [], []
    for index, ticker in enumerate(dataset["tickers"]):
        values = panel[:, index]
        train, actual = values[:split], values[split:]
        seed = 490 + index
        time = fit_time_models(train, len(actual))
        lagged = {LAG_MODELS[0]: rolling_blocks(values, split, linear_regressive(train))}
        for method, cell in zip(LAG_MODELS[1:], (nn.RNN, nn.LSTM)):
            val_start = int(np.searchsorted(dates, "2025-07-01"))
            epochs = neural_fit(values[:val_start], cell, seed, max(CHECKPOINTS),
                                validation_values=train)
            neural = neural_fit(train, cell, seed, epochs)
            lagged[method] = rolling_blocks(values, split, neural)
            selections.append((ticker, method, "all", "", "", "", epochs))
        for method, forecast in time.items():
            append_scores(scores, ticker, method, None, actual, forecast)
        for method, blocks in lagged.items():
            for width, forecast in blocks.items():
                append_scores(scores, ticker, method, width, actual, forecast)
        for i, day in enumerate(dates[split:]):
            for j, target in enumerate(TARGETS):
                time_rows.append((ticker, day, target, actual[i, j],
                                  *(time[method][i, j] for method in TIME_MODELS)))
                for width in HORIZONS:
                    lead = i % width + 1
                    block_rows.append((ticker, dates[split+i-lead], day, width, lead, target, actual[i, j],
                                       *(lagged[method][width][i, j] for method in LAG_MODELS)))
        time_plot(ticker, dates_dt[split:][display], actual[display],
                  {name: pred[display] for name, pred in time.items()})
        print(f"{ticker}: six methods, {len(HORIZONS)} block sizes", flush=True)
    write_csv("test_mse.csv", ("ticker", "model", "target", "block_sessions", "lead_sessions",
                              "observations", "mse_usd_per_share_squared"), scores)
    write_csv("model_selection.csv", ("ticker", "model", "target", "degree", "ridge_penalty",
                                     "validation_mse", "epochs"),
              selections)
    write_csv("time_predictions.csv", ("ticker", "target_date", "target", "actual", *TIME_MODELS), time_rows)
    write_csv("block_predictions.csv", ("ticker", "issued_after_close", "target_date", "block_sessions",
                                       "lead_sessions", "target", "actual", *LAG_MODELS), block_rows)
    replace_linear_close()


if __name__ == "__main__":
    main()
