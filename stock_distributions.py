"""Build the EECE 490 Yahoo Finance distribution plots and daily spotlight."""

import argparse
import csv
import html
import json
import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time, timedelta
from pathlib import Path
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import gamma, norm


ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "figure_and_visulization"
SPOTLIGHT = ROOT / "daily_spotlight"
DATA = ROOT / "data" / "yahoo_finance_daily.csv"
START = date(2020, 1, 1)
LISTINGS = {"PLTR": date(2020, 9, 30)}
NEW_YORK = ZoneInfo("America/New_York")
PRICES = ("open", "high", "low", "close")
FEATURES = (*PRICES, "high_low_range", "close_change")
FIELDS = (*FEATURES, "volume")
STOCKS = {
    "AAPL": "Apple", "NVDA": "NVIDIA", "MSFT": "Microsoft", "GOOGL": "Alphabet",
    "TSM": "Taiwan Semiconductor", "AVGO": "Broadcom", "META": "Meta Platforms",
    "MU": "Micron Technology", "AMD": "Advanced Micro Devices", "ASML": "ASML",
    "PLTR": "Palantir Technologies", "ORCL": "Oracle", "INTC": "Intel",
    "AMAT": "Applied Materials", "SAP": "SAP", "LRCX": "Lam Research",
    "TXN": "Texas Instruments", "PANW": "Palo Alto Networks", "IBM": "IBM",
    "CRWD": "CrowdStrike",
}
METRICS = ("close_change", "high_low_range")
INK, MUTED, GRID = "#172740", "#4c6075", "#e1e8ee"


def fetch_stock(ticker, start, end, *, require_open_in_range=True):
    def timestamp(day):
        return int(datetime.combine(day, time(), NEW_YORK).timestamp())

    params = urlencode({
        "period1": timestamp(start), "period2": timestamp(end + timedelta(days=1)),
        "interval": "1d", "includePrePost": "false",
    })
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?{params}"
    response = subprocess.run(
        ["curl.exe", "--fail", "--silent", "--show-error", "--retry", "3",
         "--max-time", "25", "--user-agent", "Mozilla/5.0", url],
        capture_output=True, text=True, check=True,
    )
    result = json.loads(response.stdout)["chart"]["result"][0]
    meta = result["meta"]
    if meta["symbol"] != ticker or meta["currency"] != "USD":
        raise ValueError(f"{ticker}: unexpected symbol or currency")
    quote = result["indicators"]["quote"][0]
    rows = []
    for index, unix_time in enumerate(result["timestamp"]):
        day = datetime.fromtimestamp(unix_time, NEW_YORK).date()
        if not start <= day <= end:
            continue
        values = [quote[column][index] for column in PRICES]
        if any(value is None or not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError(f"{ticker} {day}: invalid or missing OHLC")
        opening, high, low, close = values
        if not low <= close <= high or (require_open_in_range and not low <= opening <= high):
            raise ValueError(f"{ticker} {day}: inconsistent OHLC")
        volume = quote["volume"][index]
        if not isinstance(volume, int) or volume < 0:
            raise ValueError(f"{ticker} {day}: invalid or missing volume")
        rows.append((day, *values, volume))
    return rows


def fetch_panel(start, end, *, require_open_in_range=True):
    panel = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {
            ticker: pool.submit(fetch_stock, ticker, start, end,
                                require_open_in_range=require_open_in_range)
            for ticker in STOCKS
        }
        for ticker, job in jobs.items():
            try:
                panel[ticker] = job.result()
            except Exception as error:
                raise RuntimeError(f"Yahoo Finance download failed for {ticker}: {error}") from error
    validate_dates(panel)
    return panel


def validate_dates(panel):
    canonical = [row[0] for row in panel["AAPL"]]
    if len(canonical) < 100 or canonical != sorted(set(canonical)):
        raise ValueError("Missing, duplicate or unordered trading dates")
    for ticker, rows in panel.items():
        first = max(canonical[0], LISTINGS.get(ticker, START))
        if [row[0] for row in rows] != [day for day in canonical if day >= first]:
            raise ValueError(f"{ticker}: missing or unexpected trading dates")


def save_snapshot(panel):
    import torch

    tickers = list(panel)
    dates = [row[0].isoformat() for row in panel[tickers[0]]]
    values = np.full((len(dates), len(tickers), len(FIELDS)), np.nan)
    observed = np.zeros(values.shape[:2], dtype=bool)
    date_index = {day: index for index, day in enumerate(dates)}
    for ticker_index, ticker in enumerate(tickers):
        rows = panel[ticker]
        first = date_index[rows[0][0].isoformat()]
        prices = np.asarray([row[1:5] for row in rows])
        changes, ranges = measurements(rows)
        values[first:, ticker_index, :4] = prices
        values[first:, ticker_index, 4] = ranges
        values[first + 1:, ticker_index, 5] = changes
        values[first:, ticker_index, 6] = [row[5] for row in rows]
        observed[first:, ticker_index] = True

    DATA.parent.mkdir(exist_ok=True)
    csv_temp = DATA.with_suffix(".csv.tmp")
    pt_path = DATA.with_suffix(".pt")
    pt_temp = DATA.with_suffix(".pt.tmp")
    with csv_temp.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("date", "ticker", *FIELDS))
        for day_index, day in enumerate(dates):
            for ticker_index, ticker in enumerate(tickers):
                if observed[day_index, ticker_index]:
                    row = values[day_index, ticker_index].tolist()
                    row[-1] = int(row[-1])
                    writer.writerow((day, ticker, *(None if math.isnan(v) else v for v in row)))
    torch.save({
        "values": torch.from_numpy(values),
        "observed": torch.from_numpy(observed),
        "dates": dates,
        "tickers": tickers,
        "company_names": [STOCKS[ticker] for ticker in tickers],
        "features": list(FIELDS),
        "feature_units": ["USD/share"] * len(FEATURES) + ["shares"],
        "axes": ["date", "ticker", "feature"],
        "currency": "USD",
        "source": "Yahoo Finance daily OHLC",
    }, pt_temp)
    csv_temp.replace(DATA)
    pt_temp.replace(pt_path)


def read_snapshot():
    panel = {ticker: [] for ticker in STOCKS}
    with DATA.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            panel[row["ticker"]].append((
                date.fromisoformat(row["date"]), *(float(row[key]) for key in PRICES),
                int(row["volume"]),
            ))
    validate_dates(panel)
    return panel


def measurements(rows):
    _, high, low, close = np.asarray([row[1:5] for row in rows]).T
    return np.diff(close), high - low


def font(size, bold=False):
    folder = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    return ImageFont.truetype(str(folder / ("segoeuib.ttf" if bold else "segoeui.ttf")), size)


def draw_plot(path, ticker, metric, values, first_date, last_date):
    is_change = metric == "close_change"
    title = "Daily close-to-close change" if is_change else "Daily high–low range"
    formula = "Close today − previous close" if is_change else "Session high − session low"
    color, fill = ("#126975", "#a9d9db") if is_change else ("#72599c", "#dacfec")
    if is_change:
        mean, sd = norm.fit(values)
        fitted = norm(loc=mean, scale=sd)
        fit_name = "Normal MLE"
        parameters = [("Normal mean (μ)", f"${mean:,.3f}"),
                      ("Normal SD (σ)", f"${sd:,.3f}"),
                      ("Normal variance (σ²)", f"{sd ** 2:,.3f} USD²")]
    else:
        if np.any(values <= 0):
            raise ValueError(f"{ticker}: a zero-location gamma fit requires strictly positive ranges")
        alpha, _, scale = gamma.fit(values, floc=0)
        beta = 1 / scale
        fitted = gamma(alpha, scale=1 / beta)
        fit_name = "Gamma MLE"
        parameters = [("Gamma shape (α)", f"{alpha:,.3f}"),
                      ("Gamma rate (β)", f"{beta:,.3f} / USD"),
                      ("Fitted mean (α / β)", f"${alpha / beta:,.3f}")]
    bins = int(np.clip(len(np.histogram_bin_edges(values, bins="fd")) - 1, 24, 60))
    density, edges = np.histogram(values, bins=bins, density=True)
    low = min(float(edges[0]), float(fitted.ppf(0.0001)), 0)
    high = max(float(edges[-1]), float(fitted.ppf(0.9999)))
    padding = 0.035 * (high - low)
    low, high = (low - padding if is_change else 0), high + padding
    # A gamma with alpha < 1 has infinite density at zero. Draw from
    # half a screen pixel to the right so the plotted curve stays finite.
    curve_start = low + (high - low) / 2476 if not is_change and alpha < 1 else low
    x_values = np.linspace(curve_start, high, 900)
    curve = fitted.pdf(x_values)
    maximum = 1.12 * max(float(max(density)), float(max(curve)))

    im = Image.new("RGB", (1500, 930), "#f8fafc")
    draw = ImageDraw.Draw(im)

    def label(x, y, value, size, paint=INK, bold=False, anchor="la"):
        draw.text((x, y), value, font=font(size, bold), fill=paint, anchor=anchor)

    label(84, 45, f"{ticker}  /  {STOCKS[ticker]}", 29, color, True)
    label(84, 90, title, 52, bold=True)
    label(84, 159, f"{formula}  •  Data: {first_date:%d %b %Y} to {last_date:%d %b %Y}", 25, MUTED)

    labels = [("Observations", f"{len(values):,}"), *parameters]
    for index, (heading, value) in enumerate(labels):
        x = 84 + 336 * index
        draw.rounded_rectangle((x, 216, x + 312, 305), radius=10, fill="#edf2f7")
        label(x + 16, 226, heading, 20, MUTED)
        label(x + 16, 251, value.replace("$-", "−$"), 29, bold=True)

    left, right, top, bottom = 165, 1403, 401, 729

    def x_coord(x):
        return left + (x - low) * (right - left) / (high - low)

    def y_coord(y):
        return bottom - y * (bottom - top) / maximum

    draw.rectangle((left, top, right, bottom), fill="white")
    label(left, 350, "Density (1 / USD)", 21, MUTED)
    draw.rounded_rectangle((850, 354, 876, 373), radius=3, fill=fill, outline=color)
    label(886, 349, "Observed density", 21, MUTED)
    draw.line((1111, 364, 1150, 364), fill="#bb501f", width=4)
    label(1161, 349, fit_name, 21, MUTED)
    for y in np.linspace(0, maximum / 1.12, 5):
        ypos = y_coord(y)
        draw.line((left, ypos, right, ypos), fill=GRID, width=2)
        label(left - 14, ypos, f"{y:.3g}", 19, MUTED, anchor="rm")
    for index, height in enumerate(density):
        if height > 0:
            draw.rectangle(
                (x_coord(edges[index]), y_coord(height), x_coord(edges[index + 1]), bottom),
                fill=fill, outline=color, width=1,
            )
    if is_change:
        zero = x_coord(0)
        for y in range(top, bottom, 12):
            draw.line((zero, y, zero, min(y + 6, bottom)), fill="#71839a", width=2)
    draw.line([(x_coord(x), y_coord(y)) for x, y in zip(x_values, curve)],
              fill="#bb501f", width=5, joint="curve")
    draw.line((left, bottom, right, bottom), fill=MUTED, width=2)
    base = 10 ** math.floor(math.log10((high - low) / 8))
    step = next(base * unit for unit in (1, 2, 2.5, 5, 10) if base * unit >= (high - low) / 8)
    for tick in range(math.ceil(low / step), math.floor(high / step) + 1):
        value = tick * step
        x = x_coord(value)
        draw.line((x, bottom, x, bottom + 7), fill=MUTED, width=2)
        label(x, bottom + 13, f"{value:g}".replace("-", "−"), 20, MUTED, anchor="ma")
    label((left + right) / 2, 784, f"{'Price change' if is_change else 'Intraday range'} (USD per share)",
          23, INK, anchor="ma")

    if is_change:
        foot = (f"Up {np.mean(values > 0):.1%}   ·   Down {np.mean(values < 0):.1%}   "
                f"·   Flat {np.mean(values == 0):.1%} of observed changes")
        outside = np.mean(np.abs(values - mean) > 2 * sd)
        detail = f"Beyond 2 fitted SD from mean: {outside:.1%} observed vs 4.6% normal"
    else:
        foot = (f"Median range: ${np.median(values):,.2f} observed vs ${fitted.median():,.2f} gamma")
        detail = ("Gamma density rises without bound at zero when α < 1" if alpha < 1 else
                  "Gamma starts at zero and uses shape α and rate β (scale = 1/β)")
    label(84, 843, foot, 21, INK, True)
    label(84, 875, f"{detail}   ·   Source: Yahoo Finance daily OHLC", 20, MUTED)
    im.save(path, dpi=(150, 150))


def stock_plots(folder, ticker, rows, prefix):
    paths = []
    for metric, values in zip(METRICS, measurements(rows)):
        path = folder / f"{prefix}_{metric}.png"
        draw_plot(path, ticker, metric, values, rows[0][0], rows[-1][0])
        paths.append(path.name)
    return paths


def write_gallery(folder, page_title, description, entries):
    cards = []
    for ticker, note, files in entries:
        charts = "".join(f'<a href="{name}"><img src="{name}" alt="{metric}" loading="lazy"></a>'
                         for name, metric in zip(files, ("Close-to-close change", "High–low range")))
        cards.append(f"<section><h2>{ticker} · {STOCKS[ticker]}</h2><p>{html.escape(note)}</p>"
                     f'<div class="charts">{charts}</div></section>')
    (folder / "index.html").write_text(f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{page_title}</title>
<style>
body{{margin:0;background:#edf2f7;color:#172740;font:17px/1.5 "Segoe UI",sans-serif}}
main{{max-width:1450px;margin:auto;padding:24px}} h1{{line-height:1.15}}
section{{background:#fff;border:1px solid #d5e0ea;border-radius:12px;margin:20px 0;padding:20px}}
h2{{margin:0;font-size:23px}} p{{margin:6px 0 14px;color:#4c6075}}
.charts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(440px,100%),1fr));gap:14px}}
img{{width:100%;display:block;border:1px solid #d5e0ea;border-radius:5px;box-sizing:border-box}}
</style><main><h1>{page_title}</h1>{description}{''.join(cards)}</main></html>
""", encoding="utf-8")


def build_baseline(panel):
    BASELINE.mkdir(exist_ok=True)
    entries = [(ticker, "", stock_plots(BASELINE, ticker, rows, ticker))
               for ticker, rows in panel.items()]
    reference = panel["AAPL"]
    write_gallery(
        BASELINE, "EECE 490 · Daily stock distributions",
        f"<p>{reference[0][0]:%d %b %Y} to {reference[-1][0]:%d %b %Y} · 20 stocks · 40 plots</p>"
        "<p>Close changes use a maximum-likelihood normal fit. Intraday ranges use a maximum-likelihood "
        "gamma fit with location fixed at zero, shape α, and rate β. The first session supplies the "
        "starting close, so there is one fewer close change than high–low range per stock.</p>"
        '<p><a href="../daily_spotlight/index.html">Open the latest daily spotlight</a> · '
        '<a href="https://finance.yahoo.com/quote/AAPL/history/">Yahoo Finance historical prices</a></p>',
        entries,
    )
    print(f"Baseline: {len(panel)} stocks, {len(panel['AAPL'])} sessions, 40 plots")


def spotlight_score(rows):
    changes, ranges = measurements(rows)
    previous_close = np.asarray([row[4] for row in rows[:-1]])
    moves = 100 * changes / previous_close
    relative_ranges = 100 * ranges[1:] / previous_close
    percentiles = []
    for values in (np.abs(moves), relative_ranges):
        reference = values[-61:-1]
        percentile = np.mean(reference < values[-1]) + 0.5 * np.mean(reference == values[-1])
        percentiles.append(float(percentile))
    return max(percentiles), min(percentiles), moves[-1], relative_ranges[-1]


def build_spotlight(panel):
    day = panel["AAPL"][-1][0]
    ranking = sorted(
        [(ticker, *spotlight_score(rows)) for ticker, rows in panel.items()],
        key=lambda item: (-item[1], -item[2], item[0]),
    )
    SPOTLIGHT.mkdir(exist_ok=True)
    entries = []
    for rank, (ticker, score, _, move, relative_range) in enumerate(ranking[:5], start=1):
        note = (f"#{rank} · Close {move:+.2f}% · High–low {relative_range:.2f}% of previous close · "
                f"Unusualness score {score:.0%}")
        entries.append((ticker, note, stock_plots(SPOTLIGHT, ticker, panel[ticker], f"{rank:02d}")))
    write_gallery(
        SPOTLIGHT, f"Daily stock spotlight · {day:%d %b %Y}",
        "<p>Five stocks from the fixed cohort, with two distribution plots each using data from 2020 "
        "or the stock's listing date, whichever comes later.</p>"
        "<p>Ranked by the higher percentile of the latest absolute close change and high–low range "
        "against the previous 60 sessions for each stock. Both measures are relative to the previous "
        "close for ranking. The plots use dollars per share.</p>"
        '<p><a href="../figure_and_visulization/index.html">Open all 40 baseline plots</a></p>',
        entries,
    )
    print(f"Spotlight: {day}, {', '.join(item[0] for item in ranking[:5])}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("initial", "baseline", "daily", "export"))
    args = parser.parse_args()
    if args.mode in ("initial", "daily"):
        now = datetime.now(NEW_YORK)
        latest = now.date() - timedelta(days=int(now.time() < time(16, 30)))
        # The spotlight uses only the close and the reported high–low range.
        panel = fetch_panel(START, latest, require_open_in_range=args.mode == "initial")
        if panel["AAPL"][0][0] != date(2020, 1, 2):
            raise ValueError("Yahoo Finance did not return the first session of 2020")
        if args.mode == "initial":
            first_common = max(rows[0][0] for rows in panel.values())
            snapshot = {ticker: [row for row in rows if row[0] >= first_common]
                        for ticker, rows in panel.items()}
            validate_dates(snapshot)
            save_snapshot(snapshot)
        build_spotlight(panel)
    if args.mode in ("initial", "baseline"):
        build_baseline(read_snapshot())
    if args.mode == "export":
        save_snapshot(read_snapshot())
        print(f"Exported {DATA.name} and {DATA.with_suffix('.pt').name}")


if __name__ == "__main__":
    main()
