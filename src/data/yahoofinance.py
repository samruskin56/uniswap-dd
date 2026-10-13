import json
from datetime import datetime
import os
import sys
import bisect
import math
import yfinance as yf
import pandas as pd

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.helper_functions import ensure_directory, pull_progress, report_pull, report_errors


# Pull mcap to annualized metric multiple
@report_errors()
def pull_mcap_to_annualized_metric_multiple(tickers, metrics, start_date, end_date, yahoofinance_output_dir):
    failures = []

    for ticker in pull_progress(tickers, "Yahoo Finance multiples"):
        # One Ticker per symbol: yfinance caches quarterly_financials and info on
        # the object, so those downloads happen once rather than once per metric.
        tkr = yf.Ticker(ticker)
        for metric_name, metric_slug in metrics.items():
            try:
                # -----------------------------------
                # 1) Fetch the quarterly financials
                # -----------------------------------
                df = tkr.quarterly_financials

                # pick only the row labels that actually exist
                desired = ["Total Revenue", "Operating Revenue", "EBITDA"]
                available = df.index.intersection(desired)
                if available.empty:
                    raise KeyError(f"none of {desired} found in quarterly_financials")

                # pivot so dates → rows
                subset = df.loc[available].T

                # 2) Normalize index, fill NaNs
                subset.index = subset.index.strftime("%Y-%m-%d")
                subset = subset.where(pd.notnull(subset), None)
                quarterly_metrics = subset.to_dict(orient="index")

                # save quarterly metrics JSON
                qm_path = f"{yahoofinance_output_dir}/quarterly-metrics/{ticker}.json"
                ensure_directory(os.path.dirname(qm_path))
                with open(qm_path, "w") as f:
                    json.dump(quarterly_metrics, f, indent=2)

                # -----------------------------------
                # 3) Pull sharesOutstanding & history → market caps
                # -----------------------------------
                shares_out = tkr.info.get("sharesOutstanding")
                if shares_out is None:
                    raise ValueError("could not fetch sharesOutstanding")

                hist = tkr.history(start=start_date, end=end_date)
                hist["MarketCap"] = hist["Close"] * shares_out

                mc = {
                    dt.strftime("%Y-%m-%d"): cap
                    for dt, cap in zip(hist.index, hist["MarketCap"])
                }
                mc_path = f"{yahoofinance_output_dir}/{metric_slug}/{ticker}.json"
                ensure_directory(os.path.dirname(mc_path))
                with open(mc_path, "w") as f:
                    json.dump(mc, f, indent=2)

                # -----------------------------------
                # 4) Build daily mcap/metric multiple
                # -----------------------------------
                quarter_strs = sorted(quarterly_metrics.keys())
                quarter_dates = [datetime.strptime(q, "%Y-%m-%d") for q in quarter_strs]

                # re-download full history for daily multiples
                hist = tkr.history(start=quarter_strs[0],
                                   end=datetime.today().strftime("%Y-%m-%d"))
                if hasattr(hist.index, "tz"):
                    hist.index = hist.index.tz_localize(None)
                hist["MarketCap"] = hist["Close"] * shares_out

                daily_data = {}
                for dt, row in hist.iterrows():
                    date_str = dt.strftime("%Y-%m-%d")
                    idx = bisect.bisect_right(quarter_dates, dt) - 1
                    if idx < 0:
                        continue
                    q = quarter_strs[idx]
                    total_metric = quarterly_metrics[q].get(metric_name)
                    if not total_metric:
                        continue

                    mcap = row["MarketCap"]
                    ann_metric = total_metric * 4
                    multiple = mcap / ann_metric
                    daily_data[date_str] = {
                        "market_cap": mcap,
                        f"{metric_name}": total_metric,
                        f"annualized {metric_name}": ann_metric,
                        f"mcap-to-annualized-{metric_slug}": multiple
                    }

                # 5) 7-day trailing average
                dates = sorted(daily_data)
                vals = [daily_data[d][f"mcap-to-annualized-{metric_slug}"] for d in dates]
                trailing = {}
                for i, d in enumerate(dates):
                    if i < 6:
                        trailing[d] = None
                    else:
                        trailing[d] = sum(vals[i-6:i+1]) / 7

                # 6) merge & sanitize
                results = {}
                for d in dates:
                    results[d] = {
                        f"{ticker}-market-cap": daily_data[d]["market_cap"],
                        f"{ticker}-{metric_slug}": daily_data[d][f"{metric_name}"],
                        f"{ticker}-annualized-{metric_slug}": daily_data[d][f"annualized {metric_name}"],
                        f"{ticker}-mcap-to-annualized-{metric_slug}": daily_data[d][f"mcap-to-annualized-{metric_slug}"],
                        f"{ticker}-7d-avg": trailing[d]
                    }

                for metrics_data in results.values():
                    for k, v in metrics_data.items():
                        if isinstance(v, float) and math.isnan(v):
                            metrics_data[k] = None

                # 7) save final JSON — use metric_slug for path
                out_path = f"{yahoofinance_output_dir}/mcap-to-annualized-{metric_slug}/{ticker}.json"
                ensure_directory(os.path.dirname(out_path))
                with open(out_path, "w") as f:
                    json.dump(results, f, indent=2)

            except Exception as e:
                failures.append((metric_name, f"{ticker}: {e}"))

    metric_dirs = [f"mcap-to-annualized-{metric_slug}" for metric_slug in metrics.values()]
    report_pull(yahoofinance_output_dir, metric_dirs, failures)
