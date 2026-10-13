
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from matplotlib.dates import MonthLocator, YearLocator, MO
from datetime import datetime
import sys
import os
import numpy as np
from IPython.display import display
import re

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.helper_functions import space_x_axis_dates
from utils.helper_functions import format_large_numbers
from utils.helper_functions import format_multiples
from utils.helper_functions import format_multiple_value
from utils.helper_functions import format_percentages
from utils.helper_functions import format_date_with_suffix
from utils.helper_functions import format_logo_and_note
from utils.helper_functions import format_list_with_oxford_comma
from utils.helper_functions import get_last_data_date
from utils.helper_functions import load_metric_data
from utils.helper_functions import ensure_directory
from data.data_processing import weighted_average_multiple
from utils.helper_functions import report_errors, error_code

# SIMPLE DATA VISUALIZATIONS

# Function for plotting trailing average of a given metric
@report_errors()
def single_asset_daily_metric(metrics, assets, start_date, trailing_window, plots_filepath, date, annualize=False, index_from_first_date=False, combine_metrics=False):
    
    # If combine_metrics is True, create a combined chart with lines (fees, revenue, holders-revenue) and TVL area
    if combine_metrics:
        for asset_slug, asset_name in assets.items():
            try:
                # Define the metrics and their display names
                line_metrics = {
                    'fees': 'Fees',
                    'revenue': 'Revenue',
                    'holders-revenue': 'Holders Revenue',
                    'earnings': 'Earnings'          # revenue - incentives; plotted only where it differs from revenue
                }
                
                # Load and process each metric
                metric_data = {}
                for metric_id, metric_name in line_metrics.items():
                    try:
                        with open(f'data/raw/defillama/{metric_id}/{asset_slug}.json', "r") as f:
                            raw_data = json.load(f)
                        
                        df = pd.DataFrame(raw_data["totalDataChart"], columns=["timestamp", metric_id])
                        df["date"] = pd.to_datetime(df["timestamp"], unit="s")
                        df = df[["date", metric_id]]
                        
                        # Annualize if relevant
                        if annualize:
                            df[metric_id] = df[metric_id] * 365
                        
                        # Calculate rolling average
                        df[f'{metric_id}-{trailing_window}d'] = df[metric_id].rolling(window=trailing_window, min_periods=1).mean()
                        
                        metric_data[metric_id] = df[['date', f'{metric_id}-{trailing_window}d']].set_index('date')
                    except Exception as e:
                        print(f"Could not load {metric_id} for {asset_slug}: {e}")
                        continue
                
                # Load TVL data for the area
                tvl_data = None
                tvl_excluded = False
                try:
                    with open(f'data/raw/defillama/tvl/{asset_slug}.json', "r") as f:
                        tvl_raw = json.load(f)
                    
                    tvl_df = pd.DataFrame(tvl_raw["tvl"])
                    tvl_df["date"] = pd.to_datetime(tvl_df["date"], unit="s")
                    tvl_df = tvl_df.rename(columns={"totalLiquidityUSD": "tvl"})
                    tvl_df = tvl_df[["date", "tvl"]]
                    tvl_df[f'tvl-{trailing_window}d'] = tvl_df["tvl"].rolling(window=trailing_window, min_periods=1).mean()
                    
                    # Check if TVL has any non-zero values
                    if tvl_df[f'tvl-{trailing_window}d'].abs().max() > 1:
                        tvl_data = tvl_df[['date', f'tvl-{trailing_window}d']].set_index('date')
                    else:
                        print(f"Skipping TVL for {asset_slug}: all values are zero")
                        tvl_excluded = True
                except Exception as e:
                    print(f"Could not load TVL for {asset_slug}: {e}")
                    tvl_excluded = True
                
                # Combine all metrics into one dataframe
                if not metric_data:
                    print(f"No metric data available for {asset_slug}")
                    continue
                
                combined_df = pd.concat(metric_data.values(), axis=1)
                combined_df = combined_df.reset_index()
                combined_df = combined_df.fillna(0)
                
                # Merge TVL data if available
                if tvl_data is not None:
                    tvl_data = tvl_data.reset_index()
                    combined_df = pd.merge(combined_df, tvl_data, on="date", how="inner")
                
                # Filter by start_date
                combined_df = combined_df[combined_df['date'] >= start_date]
                
                if combined_df.empty:
                    print(f"No data after {start_date} for {asset_name}")
                    continue
                
                # Plot
                fig, ax1 = plt.subplots(figsize=(12, 6))
                
                # Check if we have valid TVL data to plot
                has_tvl = tvl_data is not None and f'tvl-{trailing_window}d' in combined_df.columns
                
                if has_tvl:
                    # DUAL AXIS MODE: TVL on left, metrics on right
                    # Plot TVL as area chart on LEFT axis (ax1)
                    ax1.fill_between(
                        combined_df["date"], 
                        combined_df[f"tvl-{trailing_window}d"],
                        color="#a4a2a8",
                        label="TVL"
                    )
                    ax1.set_ylabel(f"TVL")
                    tvl_max = combined_df[f"tvl-{trailing_window}d"].max()
                    ax1.yaxis.set_major_formatter(mtick.FuncFormatter(format_large_numbers(tvl_max)))
                    ax1.set_ylim(0)
                    
                    # Create secondary axis for line metrics on RIGHT
                    ax2 = ax1.twinx()
                    plot_ax = ax2  # Use ax2 for plotting metrics
                else:
                    # SINGLE AXIS MODE: No TVL, plot metrics on left axis only
                    plot_ax = ax1  # Use ax1 for plotting metrics
                    ax1.grid(axis="y", linestyle="--", alpha=0.5)     
                
                # Define line styles and colors for the three metrics
                line_styles = [
                    ('fees', 'Fees', '#1f1f60', 2, '-'),
                    ('revenue', 'Revenue', '#313EA5', 2, '-'),
                    ('holders-revenue', 'Holders Revenue', '#637BBF', 2, '-'),
                    ('earnings', 'Earnings', '#313EA5', 1.5, '--')
                ]
                
                # Track which metrics are actually plotted and which are excluded
                plotted_metrics = []
                excluded_metrics = []
                
                # Add TVL to excluded metrics if it was excluded
                if tvl_excluded:
                    excluded_metrics.append('TVL')
                
                # Plot lines for fees, revenue, and holders revenue
                # Only plot if the metric exists in the dataframe AND has non-zero values
                for metric_id, label, color, linewidth, linestyle in line_styles:
                    col_name = f'{metric_id}-{trailing_window}d'
                    if metric_id == 'earnings' and col_name in combined_df.columns:
                        # Earnings equals revenue when DefiLlama tracks no incentives; the line
                        # would sit exactly on top of revenue, so only plot it where it differs.
                        rev_col = f'revenue-{trailing_window}d'
                        if rev_col in combined_df.columns and (combined_df[col_name] - combined_df[rev_col]).abs().max() <= 1:
                            print(f"Skipping Earnings for {asset_slug}: equals revenue (no incentives tracked)")
                            continue
                    if col_name in combined_df.columns:
                        # Check if the column has any non-zero values (using a small threshold for floating point)
                        if combined_df[col_name].abs().max() > 1:
                            plot_ax.plot(
                                combined_df['date'],
                                combined_df[col_name],
                                color=color,
                                linewidth=linewidth,
                                linestyle=linestyle,
                                label=label
                            )
                            plotted_metrics.append(label)
                        else:
                            # Only track revenue and holders revenue as excluded (not fees)
                            if metric_id in ['revenue', 'holders-revenue']:
                                excluded_metrics.append(label)
                            print(f"Skipping {label} for {asset_slug}: all values are zero")
                    else:
                        # Only track revenue and holders revenue as excluded (not fees)
                        if metric_id in ['revenue', 'holders-revenue']:
                            excluded_metrics.append(label)
                
                # Check if we have any metrics to plot
                if not plotted_metrics:
                    print(f"No non-zero metrics to plot for {asset_slug}")
                    plt.close(fig)
                    continue
                
                # Build dynamic y-axis label based on plotted metrics
                if len(plotted_metrics) == 1:
                    metric_label = plotted_metrics[0]
                elif len(plotted_metrics) == 2:
                    metric_label = f"{plotted_metrics[0]} & {plotted_metrics[1]}"
                else:  # 3 or more
                    metric_label = ", ".join(plotted_metrics[:-1]) + f" & {plotted_metrics[-1]}"
                
                # Add "Annualized" prefix if applicable
                if annualize:
                    metric_label = f"Annualized {metric_label}"
                
                # Format y-axis for metrics (only add label in dual axis mode)
                if has_tvl:
                    plot_ax.set_ylabel(metric_label)
                
                # Get max from all available non-zero metrics for axis formatting
                available_cols = [f'{m}-{trailing_window}d' for m in line_metrics.keys() 
                                 if f'{m}-{trailing_window}d' in combined_df.columns 
                                 and combined_df[f'{m}-{trailing_window}d'].abs().max() > 1]
                if available_cols:
                    metrics_max = combined_df[available_cols].max().max()
                    plot_ax.yaxis.set_major_formatter(mtick.FuncFormatter(format_large_numbers(metrics_max)))
                
                plot_ax.set_ylim(0)
                
                # Set x-axis limits and formatting
                ax1.set_xlim(combined_df["date"].iloc[0], combined_df["date"].iloc[-1])
                ax1.margins(x=0)
                fig.subplots_adjust(left=0.07)
                
                # Format x-axis
                space_x_axis_dates(ax=ax1, df_name=combined_df)
                plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
                
                # Set z-order so lines appear on top of area (only needed in dual axis mode)
                if has_tvl:
                    ax1.set_zorder(1)
                    ax2.set_zorder(2)
                    ax1.patch.set_visible(False)
                    
                    # Combined legend for dual axis
                    h1, l1 = ax1.get_legend_handles_labels()
                    h2, l2 = ax2.get_legend_handles_labels()
                    ax1.legend(h1 + h2, l1 + l2, bbox_to_anchor=(0.85, -.2), ncol=2, loc="upper center")
                else:
                    # Single axis legend
                    ax1.legend(bbox_to_anchor=(0.85, -.2), ncol=2, loc="upper center")
                
                # Build dynamic title based on plotted metrics
                if len(plotted_metrics) == 1:
                    title_metrics = plotted_metrics[0]
                elif len(plotted_metrics) == 2:
                    title_metrics = f"{plotted_metrics[0]} & {plotted_metrics[1]}"
                else:  # 3 or more
                    title_metrics = ", ".join(plotted_metrics[:-1]) + f" & {plotted_metrics[-1]}"
                
                # Build title - include TVL only if it's being shown
                if has_tvl:
                    if annualize:
                        title = f"{asset_name}: TVL vs Annualized {title_metrics}"
                    else:
                        title = f"{asset_name}: TVL vs {title_metrics}"
                else:
                    if annualize:
                        title = f"{asset_name}: Annualized {title_metrics}"
                    else:
                        title = f"{asset_name}: {title_metrics}"
                ax1.set_title(title, fontsize=14)
                
                # Save & show
                plt.tight_layout()
                
                last_data_date = get_last_data_date(combined_df)
                
                # Build note with excluded metrics information
                base_note = f"Note: all data as of {last_data_date}. All figures are {trailing_window}-day rolling averages."
                
                # Add exclusion note if any metrics were excluded
                if excluded_metrics:
                    if len(excluded_metrics) == 1:
                        exclusion_note = f"\n{excluded_metrics[0]} not shown (no data or all zeros)."
                    else:
                        exclusion_note = f"\n{' & '.join(excluded_metrics)} not shown (no data or all zeros)."
                    base_note += exclusion_note
                
                base_note += "\nSource: DefiLlama"
                
                format_logo_and_note(
                    ax1,
                    base_note
                )
                
                # Save plot
                ensure_directory(plots_filepath)
                plt.savefig(f"{plots_filepath}/{asset_name}-{trailing_window}d-combined-metrics.png", dpi=300, bbox_inches="tight")
                
                # Show
                plt.show()
                
                print(combined_df.tail(5))
                
            except Exception as e:
                print(f'\nERROR\nProtocol: {asset_slug}\nError: {e}\n')
        
        return  # Exit after processing combined metrics
    
    # Original behavior when combine_metrics is False
    for metric in metrics:
        metric_id = 'fees'
        metric_name = 'Fees'
        if metric == 'dailyRevenue':
            metric_id = 'revenue'
            metric_name = 'Revenue'
        if metric == 'dailyHoldersRevenue':
            metric_id = 'holders-revenue'
            metric_name = 'Holders Revenue'
        if metric == 'dailyEarnings':
            metric_id = 'earnings'
            metric_name = 'Earnings'
        if metric == 'tvl':
            metric_id = 'tvl'
            metric_name = 'TVL'
        if metric == 'mcap':
            metric_id = 'mcap'
            metric_name = 'MCAP'
        if metric == 'fdv':
            metric_id = 'fdv'
            metric_name = 'FDV'

        for asset_slug, asset_name in assets.items():
            try:
                # Handle different file paths and structures for TVL vs mcap/fdv vs other metrics
                if metric == 'tvl':
                    with open(f'data/raw/defillama/tvl/{asset_slug}.json', "r") as f:
                        raw_data = json.load(f)
                    
                    df = pd.DataFrame(raw_data["tvl"])
                    df["date"] = pd.to_datetime(df["date"], unit="s")
                    df = df.rename(columns={"totalLiquidityUSD": metric_id})
                    df = df[["date", metric_id]]
                elif metric in ['mcap', 'fdv']:
                    with open(f'data/raw/messari/marketcap/{asset_slug}.json', "r") as f:
                        raw_data = json.load(f)
                    
                    # Extract the appropriate column index based on metric
                    # Index 1 = Circulating Marketcap (mcap), Index 3 = Fully Diluted Marketcap (fdv)
                    value_index = 1 if metric == 'mcap' else 3
                    
                    df = pd.DataFrame(raw_data["data"]["points"], columns=["timestamp", "mcap_raw", "dominance", "fdv_raw"])
                    df["date"] = pd.to_datetime(df["timestamp"], unit="s")
                    df[metric_id] = df["mcap_raw"] if metric == 'mcap' else df["fdv_raw"]
                    df = df[["date", metric_id]]
                else:
                    with open(f'data/raw/defillama/{metric_id}/{asset_slug}.json', "r") as f:
                        raw_data = json.load(f)
                    
                    df = pd.DataFrame(raw_data["totalDataChart"], columns=["timestamp", metric_id])
                    df["date"] = pd.to_datetime(df["timestamp"], unit="s")
                    df = df[["date", metric_id]]

                # Annualize, if relevant (but not for TVL, mcap, or fdv)
                if annualize == True:
                    if metric in ['tvl', 'mcap', 'fdv']:
                        print(f"did not annualize {metric_name}")
                    else:
                        df[metric_id] = df[metric_id] * 365

                # Calculate rolling average
                df[f'{metric_id}-{trailing_window}d'] = df[metric_id].rolling(window=trailing_window, min_periods=1).mean()

                # Filter by start_date or index from first date
                if index_from_first_date:
                    # Use all available data
                    df = df.dropna(subset=[f'{metric_id}-{trailing_window}d'])
                    first_value = df[f'{metric_id}-{trailing_window}d'].iloc[0]
                    df['pct_change'] = ((df[f'{metric_id}-{trailing_window}d'] - first_value) / first_value) * 100
                    df['days_from_start'] = (df['date'] - df['date'].iloc[0]).dt.days
                else:
                    df = df[df['date'] >= start_date]
                
                # Plot
                fig, ax = plt.subplots(figsize=(12, 6))
                
                if index_from_first_date:
                    # Line plot for indexed data
                    ax.plot(
                        df['days_from_start'], df['pct_change'],
                        color="#1f1f60", linewidth=2,
                        label=f"{metric_name} % Change"
                    )
                    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                    ax.set_xlabel("Days from First Date")
                    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
                    ax.grid(axis="y", linestyle="--", alpha=0.5)
                    ax.margins(x=0)  # No extra padding on x-axis
                else:
                    # Area: Circulating market cap
                    ax.fill_between(
                        df["date"], df[f"{metric_id}-{trailing_window}d"],
                        color="#1f1f60", alpha=1,
                        label=f"{metric_name} T{trailing_window}D"
                    )
                    data_max = df[f"{metric_id}-{trailing_window}d"].max()
                    ax.yaxis.set_major_formatter(mtick.FuncFormatter(format_large_numbers(data_max)))
                    ax.set_ylim(bottom=0)
                    ax.grid(axis="y", linestyle="--", alpha=0.5)
                    ax.margins(x=0) # no extra padding
                    space_x_axis_dates(ax = ax, df_name = df)
                    plt.xticks(rotation=45)
                
                # Title, grid & legend
                if index_from_first_date:
                    title = f"{asset_name}: {metric_name} Indexed from First Date"
                elif annualize == True and metric not in ['tvl', 'mcap', 'fdv']:
                    title = f"{asset_name}: Annualized {metric_name}"
                else:
                    title = f"{asset_name}: {metric_name}"
                ax.set_title(title, fontsize=14)
            
                # Ensure the layout adapts to fit all elements (legend, text, etc.)
                plt.tight_layout()

                # Update source based on metric data source
                if metric in ['mcap', 'fdv']:
                    source = "Messari"
                elif metric in ['rev-usd', 'transaction-fee-avg-usd', 'transaction-total']:
                    source = "Blockworks"
                else:
                    source = "DefiLlama"
                
                last_data_date = get_last_data_date(df)
                
                if index_from_first_date:
                    format_logo_and_note(
                        ax,
                        f"Note: all data as of {last_data_date}. Indexed to first available date with {trailing_window}-day rolling average.\nSource: {source}"
                    )
                else:
                    format_logo_and_note(
                        ax,
                        f"Note: all data as of {last_data_date}. All figures are {trailing_window}-day rolling averages.\nSource: {source}"
                    )
                
                # Save plot
                ensure_directory(plots_filepath)
                if index_from_first_date:
                    plt.savefig(f"{plots_filepath}/{asset_name}-{trailing_window}d-{metric_id}-indexed.png", dpi=300, bbox_inches='tight')
                else:
                    plt.savefig(f"{plots_filepath}/{asset_name}-{trailing_window}d-{metric_id}.png", dpi=300, bbox_inches='tight')
                
                # Show
                plt.show()

                print(df.tail(5))
            except Exception as e:
                print(f'\nERROR\nProtocol: {asset_slug}\nMetric: {metric}\nError: {error_code(e)}{type(e).__name__}: {e}\n')

# Function for plotting trailing average of a given metric across multiple assets
@report_errors()
def multi_asset_daily_metric(metrics, assets, start_date, trailing_window, plots_filepath, date, chart_title, stacked_100=False, annualize=False, index_from_first_date=False, additional_assets=None, top_n=None, log_scale=False, color_by_first_date=False):
    """
    Plot multiple assets on the same chart.

    Args:
        metrics: List of metrics to plot (e.g., ['dailyFees', 'rev-usd'])
        assets: Dict of {asset_slug: asset_name} - these all use the same metric from `metrics`
        additional_assets: Optional dict of {asset_slug: {'name': display_name, 'metric': metric_type}}
                          Allows adding assets with different metrics to the same chart.
                          Example: {'hyperliquid': {'name': 'Hyperliquid', 'metric': 'dailyFees'}}
        top_n: Optional number of top assets (by average value) to show individually;
               the rest are aggregated into an 'Other' grouping
        log_scale: Only for index_from_first_date charts - plots each asset as a
                   multiple of its first value (1x baseline) on a log y-axis
        color_by_first_date: Only for index_from_first_date charts - orders assets by
                             their first date and shades lines light (earliest) to dark
                             (newest), appending the first month to each legend label
        ... (other params)
    """

    def _get_metric_id(metric):
        """Helper to convert metric name to metric_id."""
        metric_mapping = {
            'dailyFees': 'fees',
            'dailyRevenue': 'revenue',
            'dailyHoldersRevenue': 'holders-revenue',
            'dailyEarnings': 'earnings',
            'tvl': 'tvl',
            'mcap': 'mcap',
            'fdv': 'fdv',
            'price': 'price',
            'rev-usd': 'rev-usd',
            'transaction-fee-avg-usd': 'transaction-fee-avg-usd',
            'transaction-total': 'transaction-total',
            'dex-spot-volume-total-usd': 'dex-spot-volume-total-usd'
        }
        return metric_mapping.get(metric, 'fees')

    def _load_asset_data(asset_slug, metric, metric_id, trailing_window, annualize, start_date, index_from_first_date):
        """Helper to load and process data for a single asset with a given metric."""
        if metric == 'tvl':
            with open(f'data/raw/defillama/tvl/{asset_slug}.json', "r") as f:
                raw_data = json.load(f)

            df = pd.DataFrame(raw_data["tvl"])
            df["date"] = pd.to_datetime(df["date"], unit="s")
            df = df.rename(columns={"totalLiquidityUSD": "value"})
            df = df[["date", "value"]]
        elif metric in ['mcap', 'fdv']:
            with open(f'data/raw/messari/marketcap/{asset_slug}.json', "r") as f:
                raw_data = json.load(f)

            df = pd.DataFrame(raw_data["data"]["points"], columns=["timestamp", "mcap_raw", "dominance", "fdv_raw"])
            df["date"] = pd.to_datetime(df["timestamp"], unit="s")
            df["value"] = df["mcap_raw"] if metric == 'mcap' else df["fdv_raw"]
            df = df[["date", "value"]]
        elif metric in ['rev-usd', 'transaction-fee-avg-usd', 'transaction-total', 'dex-spot-volume-total-usd']:
            with open(f'data/raw/blockworks/{metric}/{asset_slug}.json', "r") as f:
                raw_data = json.load(f)

            rows = raw_data.get(asset_slug) or []
            if not rows:
                return None
            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"])
            df = df.rename(columns={"value": "value"})
            df = df[["date", "value"]]
            df = df.sort_values('date').drop_duplicates(subset=['date'], keep='last')
        elif metric == 'price':
            with open(f'data/raw/coingecko/price/{asset_slug}.json', "r") as f:
                raw_data = json.load(f)

            rows = raw_data.get("prices") or []
            if not rows:
                return None
            df = pd.DataFrame(rows, columns=["timestamp", "value"])
            # ms timestamps; normalize to midnight so any intraday/live points
            # collapse to one daily close via the drop_duplicates below
            df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.normalize()
            df = df[["date", "value"]]
            # Drop broken feed ticks: CoinGecko sometimes reports near-zero prices
            # (e.g. 1e-8) on sparse trailing days after a market goes stale. Only
            # prices below 1/10,000th of the series median are removed.
            median_price = df["value"].median()
            if median_price > 0:
                df = df[df["value"] > median_price / 10000]
        else:
            # Default: DefiLlama fees/revenue/holders-revenue
            with open(f'data/raw/defillama/{metric_id}/{asset_slug}.json', "r") as f:
                raw_data = json.load(f)

            df = pd.DataFrame(raw_data["totalDataChart"], columns=["timestamp", "value"])
            df["date"] = pd.to_datetime(df["timestamp"], unit="s")
            df = df[["date", "value"]]

        # Annualize, if relevant (but not for TVL, mcap, fdv, or price)
        if annualize == True and metric not in ['tvl', 'mcap', 'fdv', 'price']:
            df["value"] = df["value"] * 365

        # Reindex to complete date range to ensure no gaps
        # Missing values stay as NaN - rolling average will use available data in window
        df = df.sort_values('date').drop_duplicates(subset=['date'], keep='last')
        df = df.set_index('date')
        full_date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
        df = df.reindex(full_date_range)
        df = df.reset_index().rename(columns={'index': 'date'})

        # Calculate rolling average (skips NaN values, so 2 missing days = 28/30 days used)
        df[f'value_{trailing_window}d'] = df["value"].rolling(window=trailing_window, min_periods=1).mean()

        # Filter by start_date or prepare for indexing
        if index_from_first_date:
            df = df.dropna(subset=[f'value_{trailing_window}d'])
        else:
            df = df[df['date'] >= start_date]

        return df[['date', f'value_{trailing_window}d']].set_index('date')

    for metric in metrics:
        metric_id = 'fees'
        metric_name = 'Fees'
        if metric == 'dailyRevenue':
            metric_id = 'revenue'
            metric_name = 'Revenue'
        if metric == 'dailyHoldersRevenue':
            metric_id = 'holders-revenue'
            metric_name = 'Holders Revenue'
        if metric == 'dailyEarnings':
            metric_id = 'earnings'
            metric_name = 'Earnings'
        if metric == 'tvl':
            metric_id = 'tvl'
            metric_name = 'TVL'
        if metric == 'mcap':
            metric_id = 'mcap'
            metric_name = 'MCAP'
        if metric == 'fdv':
            metric_id = 'fdv'
            metric_name = 'FDV'
        if metric == 'price':
            metric_id = 'price'
            metric_name = 'Price'
        if metric == 'rev-usd':
            metric_id = 'rev-usd'
            metric_name = 'Real Economic Value (USD)'
        if metric == 'transaction-fee-avg-usd':
            metric_id = 'transaction-fee-avg-usd'
            metric_name = 'Transaction Fee (Avg)'
        if metric == 'transaction-total':
            metric_id = 'transaction-total'
            metric_name = 'Transaction Total'
        if metric == 'dex-spot-volume-total-usd':
            metric_id = 'dex-spot-volume-total-usd'
            metric_name = 'DEX Spot Volume (USD)'

        # Print warning if trying to annualize TVL, mcap, fdv, or price
        if annualize == True and metric in ['tvl', 'mcap', 'fdv', 'price']:
            print(f"did not annualize {metric_name}")

        # Log scale only applies to indexed charts (stackplots can't be logged)
        use_log_scale = log_scale
        if log_scale and not index_from_first_date:
            print("log_scale is only supported for index_from_first_date charts; ignoring")
            use_log_scale = False

        # Collect data for all assets
        all_data = {}
        all_asset_names = []  # Track all asset names for later use
        data_sources = set()  # Track all data sources used
        asset_metrics = {}  # Track which metric each asset uses

        for asset_slug, asset_name in assets.items():
            df = _load_asset_data(asset_slug, metric, metric_id, trailing_window, annualize, start_date, index_from_first_date)
            if df is None or df.empty:
                print(f"skipping {asset_name}: no {metric_name} data")
                continue
            all_data[asset_name] = df
            all_asset_names.append(asset_name)
            asset_metrics[asset_name] = metric_name

            # Track data source
            if metric in ['mcap', 'fdv']:
                data_sources.add("Messari")
            elif metric in ['rev-usd', 'transaction-fee-avg-usd', 'transaction-total', 'dex-spot-volume-total-usd']:
                data_sources.add("Blockworks")
            elif metric == 'price':
                data_sources.add("CoinGecko")
            else:
                data_sources.add("DefiLlama")

        # Process additional_assets with their own metrics
        if additional_assets:
            for asset_slug, asset_config in additional_assets.items():
                asset_name = asset_config['name']
                asset_metric = asset_config['metric']
                asset_metric_id = _get_metric_id(asset_metric)

                # Get display name for the metric
                asset_metric_display = {
                    'dailyFees': 'Fees',
                    'dailyRevenue': 'Revenue',
                    'dailyHoldersRevenue': 'Holders Revenue',
                    'dailyEarnings': 'Earnings',
                    'tvl': 'TVL',
                    'mcap': 'MCAP',
                    'fdv': 'FDV',
                    'price': 'Price',
                    'rev-usd': 'Real Economic Value (USD)',
                    'transaction-fee-avg-usd': 'Transaction Fee (Avg)',
                    'transaction-total': 'Transaction Total',
                    'dex-spot-volume-total-usd': 'DEX Spot Volume (USD)'
                }.get(asset_metric, asset_metric)

                # Print warning if trying to annualize TVL, mcap, or fdv
                if annualize == True and asset_metric in ['tvl', 'mcap', 'fdv']:
                    print(f"did not annualize {asset_metric} for {asset_name}")

                df = _load_asset_data(asset_slug, asset_metric, asset_metric_id, trailing_window, annualize, start_date, index_from_first_date)
                if df is None or df.empty:
                    print(f"skipping {asset_name}: no {asset_metric_display} data")
                    continue
                all_data[asset_name] = df
                all_asset_names.append(asset_name)
                asset_metrics[asset_name] = asset_metric_display

                # Track data source for additional assets
                if asset_metric in ['mcap', 'fdv']:
                    data_sources.add("Messari")
                elif asset_metric in ['rev-usd', 'transaction-fee-avg-usd', 'transaction-total', 'dex-spot-volume-total-usd']:
                    data_sources.add("Blockworks")
                elif asset_metric == 'price':
                    data_sources.add("CoinGecko")
                else:
                    data_sources.add("DefiLlama")
        
        # Combine all dataframes
        combined_df = pd.concat(all_data.values(), axis=1, keys=all_data.keys())
        combined_df.columns = combined_df.columns.get_level_values(0)
        combined_df = combined_df.reset_index()

        # Roll up the long tail into an 'Other' grouping
        if top_n is not None and len(all_asset_names) > top_n:
            ranked = sorted(all_asset_names, key=lambda name: combined_df[name].mean(), reverse=True)
            top_assets, other_assets = ranked[:top_n], ranked[top_n:]
            combined_df['Other'] = combined_df[other_assets].sum(axis=1)
            combined_df = combined_df.drop(columns=other_assets)
            all_asset_names = top_assets + ['Other']
            asset_metrics['Other'] = metric_name
            print(f"Rolled {len(other_assets)} assets into 'Other': {', '.join(sorted(other_assets))}")

        # If indexing from first date, calculate percentage changes
        launch_dates = {}
        if index_from_first_date:
            # For each asset, calculate days from ITS OWN first date and percentage change
            for asset_name in all_asset_names:
                # Find the first non-zero value for this asset
                asset_data = combined_df[['date', asset_name]].copy()
                asset_data = asset_data[asset_data[asset_name] > 0]

                if len(asset_data) > 0:
                    first_date_for_asset = asset_data['date'].iloc[0]
                    first_value = asset_data[asset_name].iloc[0]
                    launch_dates[asset_name] = first_date_for_asset

                    # Calculate days from this asset's first date
                    combined_df[f'{asset_name}_days'] = (combined_df['date'] - first_date_for_asset).dt.days

                    # Calculate percentage change from first value
                    if first_value != 0:
                        combined_df[f'{asset_name}_pct'] = ((combined_df[asset_name] - first_value) / first_value) * 100
                    else:
                        combined_df[f'{asset_name}_pct'] = 0
                else:
                    combined_df[f'{asset_name}_days'] = 0
                    combined_df[f'{asset_name}_pct'] = 0

                # Multiple of first value, for log-scale plotting (always > 0)
                combined_df[f'{asset_name}_mult'] = 1 + combined_df[f'{asset_name}_pct'] / 100

        # Sort assets by average value (largest to smallest), keeping 'Other' last
        asset_averages = {asset_name: combined_df[asset_name].mean() for asset_name in all_asset_names}
        sorted_assets = sorted(asset_averages.keys(), key=lambda x: asset_averages[x], reverse=True)
        if top_n is not None and 'Other' in sorted_assets:
            sorted_assets.remove('Other')
            sorted_assets.append('Other')

        # Create color palette
        color_list = ['#1f1f60',
                      '#313EA5',
                      '#637BBF',
                      '#333333',
                      '#5F5F5F',
                      '#8A8A8A',
                      '#A0A0A0',
                      '#B6B6B6',
                      '#CCCCCC']

        # Assign colors to sorted assets, always graying out 'Other'
        colors = [color_list[i % len(color_list)] for i in range(len(sorted_assets))]
        if top_n is not None and 'Other' in sorted_assets:
            colors[-1] = '#CCCCCC'

        # When coloring by first date, order assets oldest-first and shade lines
        # from light (earliest first date) to dark (newest) using the house blues
        if index_from_first_date and color_by_first_date:
            sorted_assets = sorted(all_asset_names, key=lambda name: launch_dates.get(name, pd.Timestamp.max))
            house_blues = mcolors.LinearSegmentedColormap.from_list('house_blues', ['#CCCCCC', '#637BBF', '#313EA5', '#1f1f60'])
            colors = [mcolors.to_hex(house_blues(i / max(len(sorted_assets) - 1, 1))) for i in range(len(sorted_assets))]
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if index_from_first_date:
            # Line plot for indexed data - each asset uses its own timeline
            value_col = '_mult' if use_log_scale else '_pct'
            for i, asset_name in enumerate(sorted_assets):
                # Only plot data points where days >= 0 (i.e., from that asset's start date onwards)
                plot_df = combined_df[combined_df[f'{asset_name}_days'] >= 0].copy()
                label = asset_name
                if color_by_first_date and asset_name in launch_dates:
                    label = f"{asset_name} ({launch_dates[asset_name].strftime('%b')} '{launch_dates[asset_name].strftime('%y')})"
                ax.plot(
                    plot_df[f'{asset_name}_days'],
                    plot_df[f'{asset_name}{value_col}'],
                    color=colors[i],
                    linewidth=2,
                    label=label
                )
            ax.set_xlabel("Days from Each Asset's First Date")
            if use_log_scale:
                ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
                ax.set_yscale('log')
                # Per-value formatter: the log axis spans magnitudes, so each tick
                # gets decimals suited to its own size (0.05x, 0.50x, 5.0x)
                ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: format_multiple_value(x)))
                ax.grid(axis="y", linestyle="--", alpha=0.5, which='both')
            else:
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax.yaxis.set_major_formatter(mtick.PercentFormatter())
                ax.grid(axis="y", linestyle="--", alpha=0.5)
            ax.legend(loc='upper right', ncol=2, framealpha=0.9)
            ax.margins(x=0)  # No extra padding on x-axis
        else:
            # For stacked plots, fill NaN with 0
            combined_df = combined_df.fillna(0)

            # Prepare data for stacking
            if stacked_100:
                # Calculate percentages
                row_sums = combined_df[all_asset_names].sum(axis=1)
                plot_data = combined_df[all_asset_names].div(row_sums, axis=0) * 100
                plot_data['date'] = combined_df['date']
            else:
                plot_data = combined_df

            # Create stacked area chart with sorted order (largest on bottom) and custom colors
            ax.stackplot(
                plot_data['date'],
                *[plot_data[asset_name] for asset_name in sorted_assets],
                labels=sorted_assets,
                colors=colors,
                alpha=1
            )

            # Format y-axis
            if stacked_100:
                ax.set_ylim(0, 100)
                ax.yaxis.set_major_formatter(mtick.PercentFormatter())
            else:
                data_max = combined_df[all_asset_names].sum(axis=1).max()
                ax.yaxis.set_major_formatter(mtick.FuncFormatter(format_large_numbers(data_max)))
                ax.set_ylim(bottom=0)
            
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            ax.margins(x=0)
            space_x_axis_dates(ax=ax, df_name=combined_df)
            plt.xticks(rotation=45)
            ax.legend(bbox_to_anchor=(0.87, -0.18), loc='upper center', ncol=2, framealpha=0.9)
        
        # Title
        if index_from_first_date:
            title = f"{chart_title} (Indexed from First Date)"
            if use_log_scale:
                title = f"{title} (Log)"
        else:
            title = chart_title
        ax.set_title(title, fontsize=14)
        
        # Save & show
        plt.tight_layout()
        
        # Update source based on data sources used
        source = " & ".join(sorted(data_sources))

        last_data_date = get_last_data_date(combined_df)

        # Build metrics line only if additional_assets was used
        metrics_line = ""
        if additional_assets:
            # Group assets by metric
            metric_to_assets = {}
            for name, metric in asset_metrics.items():
                if metric not in metric_to_assets:
                    metric_to_assets[metric] = []
                metric_to_assets[metric].append(name)

            # Format as "Metric (Asset1, Asset2); Metric2 (Asset3, Asset4)"
            metrics_parts = [f"{metric} ({', '.join(assets_list)})" for metric, assets_list in metric_to_assets.items()]
            metrics_line = f"\nMetrics: {'; '.join(metrics_parts)}"

        if index_from_first_date:
            shading_line = " Lines shaded light to dark by first date." if color_by_first_date else ""
            format_logo_and_note(
                ax,
                f"Note: all data as of {last_data_date}. Indexed to first available date with {trailing_window}-day rolling average.{shading_line}{metrics_line}\nSource: {source}"
            )
        else:
            format_logo_and_note(
                ax,
                f"Note: all data as of {last_data_date}. All figures are {trailing_window}-day rolling averages.{metrics_line}\nSource: {source}"
            )
        
        # Save plot
        ensure_directory(plots_filepath)
        if index_from_first_date:
            scale_suffix = '-log' if use_log_scale else ''
            plt.savefig(f"{plots_filepath}/multi-asset-{trailing_window}d-{metric_id}-indexed{scale_suffix}.png", dpi=300, bbox_inches='tight')
        else:
            stacked_label = "100pct-stacked" if stacked_100 else "stacked"
            plt.savefig(f"{plots_filepath}/multi-asset-{trailing_window}d-{metric_id}-{stacked_label}.png", dpi=300, bbox_inches='tight')
        
        # Show
        plt.show()

        print(combined_df.tail(5))

# Function for plotting cumulative fees or revenue
@report_errors()
def single_asset_cumulitative_metric(metrics, assets, plots_filepath, date):
    for metric in metrics:
        metric_id = 'fees'
        metric_name = 'Fees'
        if metric == 'dailyRevenue':
            metric_id = 'revenue'
            metric_name = 'Revenue'
        if metric == 'dailyHoldersRevenue':
            metric_id = 'holders-revenue'
            metric_name = 'Holders Revenue'
        if metric == 'dailyEarnings':
            metric_id = 'earnings'
            metric_name = 'Earnings'

        for asset_slug, asset_name in assets.items():
            with open(f'data/raw/defillama/{metric_id}/{asset_slug}.json', "r") as f:
                raw_data = json.load(f)
            
            df = pd.DataFrame(raw_data["totalDataChart"], columns=["timestamp", metric_id])
            df["date"] = pd.to_datetime(df["timestamp"], unit="s")
            df = df[["date", metric_id]]
            
            # Calculate cumulative revenue
            df[f"cumulative-{metric_id}"] = df[metric_id].cumsum()

            # Find first non-zero cumulative metric date and start time series then
            nonzero_dates = df.loc[df[f"cumulative-{metric_id}"] > 0, "date"]
            if not nonzero_dates.empty:
                first_nonzero_date = nonzero_dates.iloc[0]
                df = df.loc[df["date"] >= first_nonzero_date]
            
            # Plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Area: Circulating market cap
            ax.fill_between(
                df["date"], df[f"cumulative-{metric_id}"],
                color="#1f1f60", alpha=1,
                label=f"Cumulative {metric_id}"
            )
            data_max = df[f"cumulative-{metric_id}"].max()
            ax.yaxis.set_major_formatter(mtick.FuncFormatter(format_large_numbers(data_max)))
            if df[f"cumulative-{metric_id}"].min() >= 0:     # cumulative earnings can go negative
                ax.set_ylim(bottom=0)
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            ax.margins(x=0) # no extra padding
            space_x_axis_dates(ax = ax, df_name = df)
            plt.xticks(rotation=45)
            
            # Title, grid & legend
            ax.set_title(f"{asset_name}: Cumulative {metric_name}", fontsize=14)
            
            # Ensure the layout adapts to fit all elements (legend, text, etc.)
            plt.tight_layout()
            last_data_date = get_last_data_date(df)
            format_logo_and_note(
                ax, 
                f"Note: all data as of {last_data_date}.\nSource: DefiLlama"
            )
            
            # Save plot
            ensure_directory(plots_filepath)
            plt.savefig(f"{plots_filepath}/{asset_name}-cumulative-{metric_id}.png", dpi=300, bbox_inches='tight')
            
            # Show
            plt.show()

            print(df.tail(5))
            
            last_val = df[f"cumulative-{metric_id}"].iloc[-1]
            last_metric = f'${last_val/1e6:,.1f}m'
            print(f"Latest {asset_name} Cumulative {metric_id}: {last_metric}")

# Plot top metric vs bottom metric for a given asset
@report_errors()
def asset_top_metric_vs_bottom_metric(asset_slug, asset_name, bottom_metric, top_metric, start_date, trailing_window, plots_filepath, date):
    """
    Plot two metrics against each other with trailing averages for a given asset.
    
    Parameters:
    - asset_slug: slug identifier for the asset (e.g., 'hyperliquid')
    - asset_name: display name for the asset (e.g., 'Hyperliquid', '$HYPE')
    - bottom_metric: metric for the area chart (e.g., 'fees', 'revenue', 'price', 'mcap', 'fdv')
    - top_metric: metric for the line chart (e.g., 'price', 'mcap', 'fdv', 'fees', 'revenue')
    - start_date: start date for filtering data
    - trailing_window: number of days for rolling average
    - plots_filepath: directory path to save plots
    - date: date string for the note
    """
    
    # Load top metric data
    top_metric_df, top_metric_name = load_metric_data(asset_slug, top_metric, trailing_window)
    
    # Load bottom metric data
    bottom_metric_df, bottom_metric_name = load_metric_data(asset_slug, bottom_metric, trailing_window)
    
    # Merge datasets
    df = pd.merge(top_metric_df, bottom_metric_df, on="date", how="inner")
    
    # Drop any rows without a full trailing average, plus the very last day
    plot_df = df[df[f"{bottom_metric}-{trailing_window}d"].notna()].iloc[:-1]
    plot_df = plot_df[plot_df[f'{top_metric}-{trailing_window}d'].notna()].iloc[:-1]
    
    # Filter by start_date
    plot_df = plot_df[plot_df['date'] >= start_date]
    
    # Plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Area: bottom metric
    ax1.fill_between(
        plot_df["date"], plot_df[f"{bottom_metric}-{trailing_window}d"],
        color="#a4a2a8", alpha=1,
        label=f"Daily {bottom_metric_name}"
    )
    ax1.set_ylabel(f"{asset_name} Daily {bottom_metric_name}")
    bottom_metric_max = plot_df[f"{bottom_metric}-{trailing_window}d"].max()
    ax1.yaxis.set_major_formatter(mtick.FuncFormatter(format_large_numbers(bottom_metric_max)))
    ax1.set_ylim(0)
    ax1.set_xlim(plot_df["date"].iloc[0], plot_df["date"].iloc[-1])
    ax1.margins(x=0)
    fig.subplots_adjust(left=0.07)
    
    # Format x-axis
    space_x_axis_dates(ax=ax1, df_name=plot_df)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
    
    # Line: top metric
    ax2 = ax1.twinx()
    ax2.plot(
        plot_df["date"], plot_df[f"{top_metric}-{trailing_window}d"],
        color="#1f1f60", linewidth=2,
        label=f"{asset_name} {top_metric_name}"
    )
    ax2.set_ylabel(f"{asset_name} {top_metric_name}")
    top_metric_max = plot_df[f"{top_metric}-{trailing_window}d"].max()
    ax2.yaxis.set_major_formatter(mtick.FuncFormatter(format_large_numbers(top_metric_max)))
    ax2.set_ylim(0)
    
    # Title, grid & legend
    ax1.set_title(f"{asset_name}: Daily {bottom_metric_name} vs {top_metric_name}", fontsize=14)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, bbox_to_anchor=(0.9, -.15), loc="upper center")
    
    # Save & show
    plt.tight_layout()

    last_data_date = get_last_data_date(df)

    format_logo_and_note(
        ax1, 
        f"Note: all data as of {last_data_date}. All figures are {trailing_window}-day rolling averages.\nSource: DefiLlama and Messari"
    )
    
    # Save plot
    ensure_directory(plots_filepath)
    plt.savefig(f"{plots_filepath}/{asset_slug}_{bottom_metric}_vs_{top_metric}_{trailing_window}d.png", dpi=300, bbox_inches="tight")
    
    # Show
    plt.show()
    
    print(plot_df.tail(5))

# Function to visualize annualized volatility from saved JSONs
@report_errors()
def annualized_volatility(assets, plots_filepath, title, date):
    df_master = None
    trailing_window = None
    data_filepath = 'data/processed/annualized-volatility'
    
    # Load data from JSON for each asset
    for asset_slug, asset_name in assets.items():
        with open(f"{data_filepath}/{asset_slug}.json", 'r') as f:
            json_data = json.load(f)
        
        trailing_window = json_data['trailing_window']
        df = pd.DataFrame(json_data['data'])
        df['date'] = pd.to_datetime(df['date'])
        df.rename(columns={'annualized_std': f'{asset_slug}_annualized_std'}, inplace=True)
        
        # Merge into master dataframe
        df_master = df if df_master is None else df_master.merge(df, on="date", how="outer")
        df_master = df_master.sort_values("date").reset_index(drop=True)
    
    df_master = df_master.ffill()
    
    # Create the line chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create color palette
    color_list = ['#1f1f60',
                  '#313EA5',
                  '#637BBF',
                  '#333333',
                  '#5F5F5F',
                  '#8A8A8A',
                  '#A0A0A0',
                  '#B6B6B6',
                  '#CCCCCC'
                 ]
    
    # Plot each asset
    for i, (asset_slug, asset_name) in enumerate(assets.items()):
        ax.plot(
            df_master['date'], 
            df_master[f'{asset_slug}_annualized_std'], 
            label=asset_name,
            color=color_list[i % len(color_list)],
            linewidth=2
        )
    
    # Setting title
    ax.set_title(title)
    ax.margins(x=0)
    
    # Format axes
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x*100:,.0f}%'))
    space_x_axis_dates(ax=ax, df_name=df_master)
    plt.xticks(rotation=45)
    
    # Add grid lines for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Position legend
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=2)
    
    plt.tight_layout()

    last_data_date = get_last_data_date(df_master)

    format_logo_and_note(
            ax, 
            f"Note: all data as of {last_data_date}. All figures are {trailing_window}-day rolling standard deviations.\nSource: Messari."
        )
    
    # Save plot
    ensure_directory(plots_filepath)
    plt.savefig(f"{plots_filepath}/annualized-volatility.png", dpi=300, bbox_inches='tight')
    
    # Show
    plt.show()
    
    print(df_master.tail(5))
    print('Done')

# FINANCIAL METRICS

# Function to compare gross margin by category
@report_errors()
def category_gross_margin_comparison(categories, start_date, plots_filepath, date):
    # Load JSON data
    with open('data/processed/financial-metrics/gross-margins-by-category.json', 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame by extracting category aggregates for each date
    rows = []
    for entry in data:
        row = {'date': entry['date']}
        for category in categories:
            if category in entry:
                row[f'{category}-gross-margin-30d'] = entry[category]['category_aggregate']
            else:
                row[f'{category}-gross-margin-30d'] = None
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] >= start_date]
    
    # Create the line chart
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create color palette
    color_list = ['#1f1f60',
                        '#313EA5',
                        '#637BBF',
                        '#333333',
                        '#5F5F5F',
                        '#8A8A8A',
                        '#A0A0A0',
                        '#B6B6B6',
                        '#CCCCCC'
               ]

    color_palette = {}
            
    # Prepare the values and colors in the right order
    dates   = df['date']
    series_list = [df[f'{category}-gross-margin-30d'] for category in categories]
    labels_list  = [category for category in categories]
    colors_list = [color for color in color_list]
    
    # Plotting the line chart with custom colors
    for i, (category) in enumerate(categories):
        ax.plot(
            dates, 
            series_list[i], 
            label = labels_list[i], 
            color = colors_list[i],
            linewidth=2
        )
    
    # Setting labels and title
    ax.set_title('Gross Margin by Defi Category')
    ax.margins(x=0)
    space_x_axis_dates(ax = ax, df_name = df)
    plt.xticks(rotation=45)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x*100:,.0f}%'))
    ax.set_ylim(bottom=0)

    # Add grid lines for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    # Center the legend below the chart (highest up)
    plt.legend(title=f'Select Categories', bbox_to_anchor=(0.825, -0.18), loc='upper center', ncol=2)
    
    # Ensure the layout adapts to fit all elements (legend, text, etc.)
    plt.tight_layout()

    last_data_date = get_last_data_date(df)

    format_logo_and_note(
            ax, 
            f"Note: all data as of {last_data_date}. All figures are 30-day rolling averages.\nSource: DefiLlama"
        )
    
    
    # Save plot
    ensure_directory(plots_filepath)
    plt.savefig(f"{plots_filepath}/category-gross-margins.png", dpi=300, bbox_inches='tight')
    
    # Show
    plt.show()

    print(df.tail(5))


# Function to compare fees to tvl by category
@report_errors()
def category_fees_to_tvl_comparison(categories, start_date, plots_filepath, date):
    # Load JSON data
    with open('data/processed/financial-metrics/fees-to-tvl-by-category.json', 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame by extracting category aggregates for each date
    rows = []
    for entry in data:
        row = {'date': entry['date']}
        for category in categories:
            if category in entry:
                row[f'{category}-fees-to-tvl-30d'] = entry[category]['category_aggregate']
            else:
                row[f'{category}-fees-to-tvl-30d'] = None
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] >= start_date]
    
    # Create the line chart
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create color palette
    color_list = ['#1f1f60',
                        '#313EA5',
                        '#637BBF',
                        '#333333',
                        '#5F5F5F',
                        '#8A8A8A',
                        '#A0A0A0',
                        '#B6B6B6',
                        '#CCCCCC'
               ]

    color_palette = {}
            
    # Prepare the values and colors in the right order
    dates   = df['date']
    series_list = [df[f'{category}-fees-to-tvl-30d'] for category in categories]
    labels_list  = [category for category in categories]
    colors_list = [color for color in color_list]
    
    # Plotting the line chart with custom colors
    for i, (category) in enumerate(categories):
        ax.plot(
            dates, 
            series_list[i], 
            label = labels_list[i], 
            color = colors_list[i],
            linewidth=2
        )
    
    # Setting labels and title
    ax.set_title('Annualized Fees vs TVL by Defi Category')
    ax.margins(x=0)
    space_x_axis_dates(ax = ax, df_name = df)
    plt.xticks(rotation=45)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x*100:,.0f}%'))
    ax.set_ylim(bottom=0)

    # Add grid lines for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    # Center the legend below the chart (highest up)
    plt.legend(title=f'Select Categories', bbox_to_anchor=(0.825, -0.18), loc='upper center', ncol=2)
    
    # Ensure the layout adapts to fit all elements (legend, text, etc.)
    plt.tight_layout()

    last_data_date = get_last_data_date(df)

    format_logo_and_note(
            ax, 
            f"Note: all data as of {last_data_date}. All figures are 30-day rolling averages.\nSource: DefiLlama"
        )
    
    # Save plot
    ensure_directory(plots_filepath)
    plt.savefig(f"{plots_filepath}/category-fees-to-tvl.png", dpi=300, bbox_inches='tight')
    
    # Show
    plt.show()

    print(df.tail(5))

# Function to compare fees to tvl by category
@report_errors()
def category_revenue_to_tvl_comparison(categories, start_date, plots_filepath, date):
    # Load JSON data
    with open('data/processed/financial-metrics/revenue-to-tvl-by-category.json', 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame by extracting category aggregates for each date
    rows = []
    for entry in data:
        row = {'date': entry['date']}
        for category in categories:
            if category in entry:
                row[f'{category}-revenue-to-tvl-30d'] = entry[category]['category_aggregate']
            else:
                row[f'{category}-revenue-to-tvl-30d'] = None
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] >= start_date]
    
    # Create the line chart
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create color palette
    color_list = ['#1f1f60',
                        '#313EA5',
                        '#637BBF',
                        '#333333',
                        '#5F5F5F',
                        '#8A8A8A',
                        '#A0A0A0',
                        '#B6B6B6',
                        '#CCCCCC'
               ]

    color_palette = {}
            
    # Prepare the values and colors in the right order
    dates   = df['date']
    series_list = [df[f'{category}-revenue-to-tvl-30d'] for category in categories]
    labels_list  = [category for category in categories]
    colors_list = [color for color in color_list]
    
    # Plotting the line chart with custom colors
    for i, (category) in enumerate(categories):
        ax.plot(
            dates, 
            series_list[i], 
            label = labels_list[i], 
            color = colors_list[i],
            linewidth=2
        )
    
    # Setting labels and title
    ax.set_title('Annualized Revenue vs TVL by Defi Category')
    ax.margins(x=0)
    space_x_axis_dates(ax = ax, df_name = df)
    plt.xticks(rotation=45)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x*100:,.0f}%'))
    ax.set_ylim(bottom=0)

    # Add grid lines for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    # Center the legend below the chart (highest up)
    plt.legend(title=f'Select Categories', bbox_to_anchor=(0.825, -0.18), loc='upper center', ncol=2)
    
    # Ensure the layout adapts to fit all elements (legend, text, etc.)
    plt.tight_layout()

    last_data_date = get_last_data_date(df)

    format_logo_and_note(
            ax, 
            f"Note: all data as of {last_data_date}. All figures are 30-day rolling averages.\nSource: DefiLlama"
        )
    
    # Save plot
    ensure_directory(plots_filepath)
    plt.savefig(f"{plots_filepath}/category-revenue-to-tvl.png", dpi=300, bbox_inches='tight')
    
    # Show
    plt.show()

    print(df.tail(5))


# DEXS

# Function to plot OI by dex
@report_errors()
def open_interest_by_dex(start_date, end_date, n, plots_filepath, date, trailing_window=7, 
                         percentage=False, chart_type='area', pct_change_from_start=False):
    """
    Plot Open Interest by DEX.
    
    This function:
    1. Loads raw data and calculates trailing averages
    2. Always calculates both % market share and % change from index
    3. Saves processed data to JSON
    4. Plots based on parameter selections
    
    Parameters:
    -----------
    start_date : str
        Start date for the analysis
    end_date : str
        End date for the analysis
    n : int
        Number of top DEXs to show (rest aggregated as 'Other')
    plots_filepath : str
        Directory to save plots
    date : str
        Date string for the note/source text
    trailing_window : int
        Rolling average window (default 7 days)
    percentage : bool
        If True, plot market share %; if False, plot absolute values
    chart_type : str
        'area' for stacked area chart, 'line' for line chart
    pct_change_from_start : bool
        If True and chart_type='line', plot values as % change from start_date
    """
    dex_colors = [
        '#1f1f60', '#313EA5', '#637BBF', '#94B9D9', '#333333',
        '#5F5F5F', '#8A8A8A', '#A0A0A0', '#B6B6B6', '#CCCCCC'
    ]
    
    start_date_dt = pd.to_datetime(start_date)
    end_date_dt = pd.to_datetime(end_date) - pd.Timedelta(days=1)

    # Load raw data
    with open("data/raw/defillama/open-interest/all-dexs.json", 'r') as f:
        data = json.load(f)
    
    # Flatten records from totalDataChartBreakdown
    records = []
    for entry in data['totalDataChartBreakdown']:
        timestamp = entry[0]
        df_date = pd.to_datetime(timestamp, unit='s')
        breakdown = entry[1]
        for dex, oi_value in breakdown.items():
            records.append({'date': df_date, 'dex': dex, 'openInterest': oi_value})
    
    # Create DataFrame and pivot
    df_all = pd.DataFrame(records)
    df_pivot = df_all.pivot_table(index='date', columns='dex', values='openInterest', aggfunc='sum')
    df_pivot = df_pivot.sort_index()
    
    # Fill NaN values: fillna(0) before start_date, ffill() after start_date
    before_start = df_pivot.index < start_date_dt
    after_start = df_pivot.index >= start_date_dt
    
    df_pivot.loc[before_start] = df_pivot.loc[before_start].fillna(0)
    df_pivot.loc[after_start] = df_pivot.loc[after_start].ffill()
    
    # Filter by date range
    df_pivot_filtered = df_pivot.loc[
        (df_pivot.index >= start_date_dt) & (df_pivot.index < end_date_dt)
    ].copy()
    
    # Fill any remaining NaN values with 0
    df_pivot_filtered = df_pivot_filtered.fillna(0)
    
    # Calculate trailing averages
    dex_columns = df_pivot_filtered.columns.tolist()
    df_pivot_filtered[dex_columns] = df_pivot_filtered[dex_columns].rolling(
        window=trailing_window, min_periods=1
    ).mean()
    
    # Determine top_n DEXs by the last value
    last_date = df_pivot_filtered.index.max()
    last_values = df_pivot_filtered.loc[last_date]
    sorted_dexs = last_values.sort_values(ascending=False)
    top_dexs = list(sorted_dexs.index[:n])
    other_dexs = list(sorted_dexs.index[n:])
    
    # Build final DataFrame with 'Other' aggregation
    df_top = df_pivot_filtered[top_dexs].copy()
    df_top['Other'] = df_pivot_filtered[other_dexs].sum(axis=1)
    df_master = df_top.reset_index()
    df_master = df_master.sort_values('date')
    
    # Get DEX names (columns excluding 'date')
    dexs = df_master.columns[1:].tolist()
    
    # -----------------------------------------------------------------
    # ALWAYS calculate both % market share and % change from index
    # -----------------------------------------------------------------
    
    # Calculate % market share
    df_pct_share = df_master[['date']].copy()
    row_totals = df_master[dexs].sum(axis=1)
    for dex in dexs:
        df_pct_share[dex] = (df_master[dex] / row_totals) * 100
    
    # Calculate % change from index (start date)
    df_pct_change = df_master[['date']].copy()
    start_values = df_master[dexs].iloc[0]
    for dex in dexs:
        if start_values[dex] != 0:
            df_pct_change[dex] = ((df_master[dex] - start_values[dex]) / start_values[dex]) * 100
        else:
            df_pct_change[dex] = 0  # Avoid division by zero
    
    # -----------------------------------------------------------------
    # Save processed data to JSON
    # -----------------------------------------------------------------
    output_dir = 'data/processed/open-interest'
    ensure_directory(output_dir)
    
    # Build JSON structure organized by date -> protocol -> metrics
    def build_organized_output(df_abs, df_pct, df_change, dexs):
        output = {}
        
        for idx in range(len(df_abs)):
            # Get date as string
            date_val = df_abs['date'].iloc[idx]
            if isinstance(date_val, (pd.Timestamp, np.datetime64)):
                date_str = pd.to_datetime(date_val).strftime('%Y-%m-%d')
            else:
                date_str = str(date_val)[:10]
            
            output[date_str] = {}
            
            for dex in dexs:
                # Get values and convert to native Python types
                abs_val = df_abs[dex].iloc[idx]
                pct_val = df_pct[dex].iloc[idx]
                change_val = df_change[dex].iloc[idx]
                
                # Convert numpy types to native Python
                if hasattr(abs_val, 'item'):
                    abs_val = abs_val.item()
                if hasattr(pct_val, 'item'):
                    pct_val = pct_val.item()
                if hasattr(change_val, 'item'):
                    change_val = change_val.item()
                
                output[date_str][dex] = {
                    'absolute_value': float(abs_val),
                    'market_share_pct': float(pct_val),
                    'pct_change_from_index': float(change_val)
                }
        
        return output
    
    processed_data = {
        'metadata': {
            'start_date': str(start_date),
            'end_date': str(end_date),
            'trailing_window': int(trailing_window),
            'top_n_dexs': int(n),
            'dexs': list(dexs),
            'generated_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        'data': build_organized_output(df_master, df_pct_share, df_pct_change, dexs)
    }
    
    json_filename = f"{output_dir}/oi-by-dex-processed.json"
    with open(json_filename, 'w') as f:
        json.dump(processed_data, f, indent=2)
    print(f"Saved processed data to: {json_filename}")
    
    # -----------------------------------------------------------------
    # Select data for plotting based on parameters
    # -----------------------------------------------------------------
    if pct_change_from_start and chart_type == 'line':
        df_plot = df_pct_change.copy()
        y_mode = 'pct_change'
    elif percentage:
        df_plot = df_pct_share.copy()
        y_mode = 'pct_share'
    else:
        df_plot = df_master.copy()
        y_mode = 'absolute'
    
    # -----------------------------------------------------------------
    # Create color mapping
    # -----------------------------------------------------------------
    colors = []
    color_index = 0
    for dex in dexs:
        if dex == 'Hyperliquid Perps':
            colors.append('#09b882')
        elif dex == 'Other':
            colors.append('#CCCCCC')
        else:
            colors.append(dex_colors[color_index % len(dex_colors)])
            color_index += 1
    
    # -----------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    dates = df_plot['date']
    
    if chart_type == 'line':
        for dex, color in zip(dexs, colors):
            ax.plot(dates, df_plot[dex], label=dex, color=color, linewidth=2)
    else:  # default to 'area'
        series = [df_plot[dex] for dex in dexs]
        ax.stackplot(dates, *series, labels=dexs, colors=colors)
    
    # Format y-axis based on mode
    if y_mode == 'pct_change':
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel('% Change from Index')
    elif y_mode == 'pct_share':
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_ylim(0, 100)
        ax.set_ylabel('Market Share')
    else:
        data_max = df_plot[dexs].sum(axis=1).max()
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(format_large_numbers(data_max)))
        ax.set_ylabel('Open Interest')
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(format_large_numbers(data_max)))
    
    # Format x-axis
    space_x_axis_dates(ax=ax, df_name=df_plot)
    plt.xticks(rotation=45)
    
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=2, framealpha=1)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    ax.margins(x=0)
    
    # Set title based on mode
    if y_mode == 'pct_change':
        title = f'Open Interest by DEX (% Change from {format_date_with_suffix(start_date_dt)})'
    elif y_mode == 'pct_share':
        title = 'Open Interest by DEX (%)'
    else:
        title = 'Open Interest by DEX'
    ax.set_title(title)
    
    plt.tight_layout()

    last_data_date = get_last_data_date(df_all)

    format_logo_and_note(
            ax, 
            f"Note: all data as of {last_data_date}. All figures are {trailing_window}-day rolling averages.\nSource: DefiLlama"
        )
    
    # Save plot with appropriate filename
    ensure_directory(plots_filepath)
    chart_suffix = '-line' if chart_type == 'line' else ''
    pct_suffix = '-pct' if percentage else ''
    pct_change_suffix = '-pct-change' if (pct_change_from_start and chart_type == 'line') else ''
    filename = f"oi-by-dex{pct_suffix}{chart_suffix}{pct_change_suffix}.png"
    plt.savefig(f"{plots_filepath}/{filename}", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(df_plot.tail(10))
    print('Done')

# STABLECOINS

# Function to plot stablecoin mcap by chain
@report_errors()
def stablecoin_mcap_by_chain(assets, start_date, end_date, n, plots_filepath, date):
    for asset in assets:
        chain_colors = ['#1f1f60',
                        '#313EA5',
                        '#637BBF',
                        '#94B9D9',
                        '#333333',
                        '#5F5F5F',
                        '#8A8A8A',
                        '#A0A0A0',
                        '#B6B6B6',
                        '#CCCCCC'
                    ]
        
        df_master = None
        
        counter = 0
        
        start_date_dt = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date) - pd.Timedelta(days=1)

        with open(f"data/raw/defillama/stablecoins/{asset}.json", 'r') as f:
            data = json.load(f)
        
        # Flatten records
        records = []
        for chain, chain_data in data.get('chainBalances', {}).items():
            for entry in chain_data.get('tokens', []):
                df_date = pd.to_datetime(entry.get('date'), unit='s')
                # Use circulating peggedUSD; if missing, fallback to bridgedTo peggedUSD
                value = entry.get('circulating', {}).get('peggedUSD')
                if value is None:
                    value = entry.get('bridgedTo', {}).get('peggedUSD', 0)
                records.append({'date': df_date, 'chain': chain, 'peggedUSD': value})
        
        # Create DataFrame and pivot
        df_all = pd.DataFrame(records)
        df_pivot = df_all.pivot_table(index='date', columns='chain', values='peggedUSD', aggfunc='sum')
        df_pivot = df_pivot.ffill().sort_index()
        
        # Determine top_n chains *by the last date* in the index
        last_date   = df_pivot.index.max()
        last_values = df_pivot.loc[last_date]
        sorted_chains = last_values.sort_values(ascending=False)
        top_chains  = list(sorted_chains.index[:n])
        other_chains = list(sorted_chains.index[n:])
        
        # Build final DataFrame with 'Other' aggregation
        df_top = df_pivot[top_chains].copy()
        df_top['Other'] = df_pivot[other_chains].sum(axis=1)
        df_master = df_top.reset_index()

        # Filter by appropriate dates
        df_master = df_master.loc[(df_master['date'] >= start_date_dt) &
        (df_master['date'] < end_date_dt)
        ]
            
        # Sort your master dataframe by date
        df_master = df_master.sort_values('date')

        # Grab chain names
        chains = df_master.columns[1:].tolist()

        # Calculate 7-day trailing averages for each chain column
        df_master[chains] = df_master[chains].rolling(window=7, min_periods=1).mean()
        
        # Prepare the values and colors in the right order
        dates   = df_master['date']
        series  = [df_master[chain] for chain in chains]
        
        # Create color mapping with special case for Hyperliquid L1
        colors = []
        color_index = 0
        for chain in chains:
            if chain == 'Hyperliquid L1':
                colors.append('#09b882')
            if chain == 'Other':
                colors.append('#CCCCCC')
            else:
                colors.append(chain_colors[color_index % len(chain_colors)])
                color_index += 1
        
        # Plot a single stacked‐area call
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.stackplot(
            dates,
            *series,
            labels=chains,
            colors=colors
        )
        
        data_max = df_master[chains].sum(axis=1).max()
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(format_large_numbers(data_max)))
        space_x_axis_dates(ax = ax, df_name = df_master)
        plt.xticks(rotation=45)
        
        ax.legend(loc='upper left', bbox_to_anchor=(0,1), ncol=2)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        ax.margins(x=0)
        ax.set_title(f'{asset} Market Cap by Chain')

        plt.tight_layout()

        last_data_date = get_last_data_date(df_master)

        format_logo_and_note(
            ax, 
            f"Note: all data as of {last_data_date}. All figures are 7-day rolling averages.\nSource: DefiLlama"
        )
        
        # Save plot
        ensure_directory(plots_filepath)
        plt.savefig(f"{plots_filepath}/{asset}-mcap-by-chain.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(df_master.tail(5))
    print('Done')


# Hyperliquid as a % of total usdc
@report_errors()
def hyperliquid_as_percent_of_total_usdc(dune_path, messari_path, start_date, plots_filepath, date):
    # Load dune data
    with open(dune_path, 'r') as f:
        dune_data = json.load(f)
    rows = dune_data['result']['rows']
    dune_df = pd.DataFrame(rows)

    # To datetime
    dune_df['date'] = pd.to_datetime(dune_df['date'], format='%Y-%m-%d %H:%M:%S.%f UTC').dt.normalize()
    
    # Load messari data
    with open(messari_path, 'r') as f:
        messari_data = json.load(f)

    points = messari_data['data']['points']
    messari_df = pd.DataFrame(points, columns=[
        'timestamp',
        'outstandingSupplyUsd',
        'mints24HourUsd',
        'stablecoin-redemptions-usd'
    ])

    messari_df["date"] = pd.to_datetime(messari_df["timestamp"], unit="s").dt.normalize().ffill()
    messari_df = messari_df[['date','outstandingSupplyUsd']]
    messari_df['USDC Supply'] = messari_df['outstandingSupplyUsd'].rolling(window=7, min_periods=1).mean()

    # Merge datasets for asset slug
    df = pd.merge(messari_df, dune_df, on="date", how="inner")

    # Cut off df to only be after start date
    df = df[df['date'] >= start_date]

    # Calculate Hyperliquid as % of total
    df['USDC on Hyperliquid (Percent of Total)'] = (df['cumulative_amount'] / df['USDC Supply']).rolling(window=7, min_periods=1).mean()


    # Plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax2 = ax1.twinx()

    # Set ax2 to be behind ax1, but make ax1 background transparent
    ax2.set_zorder(1)  # ax2 behind
    ax1.set_zorder(2)  # ax1 in front
    ax1.patch.set_visible(False)  # Make ax1 background transparent
    
    # Line: OI
    ax1.plot(
        df["date"], df["USDC Supply"],
        color="#1f1f60", linewidth=2,
        label="Total USDC Supply",
        zorder=2
    )
    ax1.set_ylabel("Total USDC Supply")
    ax1.yaxis.set_major_formatter(
            mtick.FuncFormatter(lambda x, _: f'${x/1e9:,.0f}b'))
    ax1.set_ylim(bottom=0)
    ax1.margins(x=0)                  # no extra padding

    space_x_axis_dates(ax = ax1, df_name = df)
  
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

    # Area: USDC on Hyperliquid (Percent of Total) 
    ax2.fill_between(
        df["date"], df["USDC on Hyperliquid (Percent of Total)"],
        color="#a4a2a8", alpha=1,
        label="USDC on Hyperliquid (Percent of Total)",
        zorder=1
    )
    ax2.yaxis.set_major_formatter(
            mtick.FuncFormatter(lambda x, _: f'{x*100:,.0f}%'))
    
    ax2.set_ylabel("Percent of Total USDC Supply")
    ax2.margins(x=0)                  # no extra padding
    ax2.set_ylim(bottom=0)

    # Monthly ticks, lighter crowding
    ax2.xaxis.set_major_locator(MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    
    # Title, grid & legend
    ax1.set_title(f"USDC: Hyperliquid vs Total Supply", fontsize=14)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")
    
    # Save & show
    plt.tight_layout()

    last_data_date = get_last_data_date(df)

    format_logo_and_note(
        ax1,
        f"Note: all data as of {last_data_date}. All figures are 7-day rolling averages.\nSource: Dune Analytics and Messari"
    )
    
    ensure_directory(plots_filepath)
    plt.savefig(f"{plots_filepath}/hyperliquid_usdc_supply.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    # Debug print
    print(df.tail(5))

# Function to plot m2 money supply vs an asset price
@report_errors()
def plot_m2_vs_price(asset_name, start_date, m2_filepath, 
                     asset_filepath, plots_filepath):
    """
    Plot M2 money supply versus asset price with dual y-axes.
    
    Parameters:
    -----------
    asset_name : str
        Name of the asset (e.g., 'Bitcoin')
    start_date : str
        Start date for filtering data (format: 'YYYY-MM-DD')
    m2_filepath : str
        Path to M2 money supply JSON file
    asset_filepath : str
        Path to asset price JSON file
    plots_filepath : str
        Directory to save the plot
    """
    
    # Load M2 money supply data
    with open(m2_filepath, 'r') as f:
        m2_data = json.load(f)
    
    m2_points = m2_data['observations']
    m2_df = pd.DataFrame(m2_points)
    m2_df['date'] = pd.to_datetime(m2_df['date'])
    m2_df['value'] = pd.to_numeric(m2_df['value'], errors='coerce')
    m2_df = m2_df[['date', 'value']].rename(columns={'value': 'm2'})
    
    # Calculate trailing 7-day average for M2
    m2_df['m2-7d'] = m2_df['m2'].rolling(window=7, min_periods=1).mean()
    
    # Load asset price data
    with open(asset_filepath, 'r') as f:
        price_data = json.load(f)
    
    points = price_data['data']['points']
    price_df = pd.DataFrame(points, columns=[
        'timestamp',
        'open',
        'high',
        'low',
        'close',
        'volume'
    ])
    price_df['date'] = pd.to_datetime(price_df['timestamp'], unit='s')
    price_df = price_df[['date', 'close']]
    
    # Calculate trailing 7-day average for price
    price_df['close-7d'] = price_df['close'].rolling(window=7).mean()
    
    # Merge datasets
    df = pd.merge(m2_df, price_df, on='date', how='inner')
    
    # Filter by start date
    df = df[df['date'] >= start_date]
    
    # Create plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Area: M2 Money Supply
    ax1.fill_between(
        df['date'], df['m2-7d'],
        color='#a4a2a8', alpha=1,
        label='M2 Money Supply'
    )
    ax1.yaxis.set_major_formatter(
        mtick.FuncFormatter(lambda x, p: f'${x/1e3:,.0f}t'))
    
    ax1.set_ylabel('M2 Money Supply (Billions USD)')
    ax1.margins(x=0)
    ax1.set_ylim(bottom=0)
    
    # Dynamic date formatting based on time series length
    space_x_axis_dates(ax = ax1, df_name = df)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Line: Asset Price
    ax2 = ax1.twinx()
    ax2.plot(
        df['date'], df['close-7d'],
        color='#1f1f60', linewidth=2,
        label=f'{asset_name} Price'
    )
    ax2.set_ylabel(f'{asset_name} Price (USD)')
    ax2.yaxis.set_major_formatter(
        mtick.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax2.set_ylim(bottom=0)
    ax2.margins(x=0)
    
    # Title, grid & legend
    ax1.set_title(f'M2 Money Supply vs {asset_name} Price', fontsize=14)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper left')
    
    # Save & show
    plt.tight_layout()

    last_data_date = get_last_data_date(df)

    format_logo_and_note(
        ax1,
        f'Note: All data as of {last_data_date}. All figures are 7-day rolling averages. \nSource: FRED and Messari'
    )

    ensure_directory(plots_filepath)
    plt.savefig(f'{plots_filepath}/m2_vs_{asset_name.lower()}_7d.png',
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Debug print
    print(df.tail(5))


# VALUATION MULTIPLES

@report_errors()
def protocol_multiples_by_category(metric, start_date, input_filepath, plots_filepath, tables_filepath, date):
    """
    Plot protocol valuation multiples by category.
    Creates separate charts for each category showing individual protocol lines.
    Also outputs a styled statistics table for each category.
    
    Parameters:
    -----------
    metric : str
        Metric to plot: e.g., 'FDV-to-annualized-fees-multiple', 
        'MCAP-to-annualized-revenue-multiple', etc.
    start_date : str
        Start date for filtering data (format: 'YYYY-MM-DD')
    input_filepath : str
        Path to the input JSON file
    plots_filepath : str
        Directory to save the plots
    tables_filepath : str
        Directory to save the tables
    date : str
        Data as-of date for chart annotation
    """
    
    # Load data
    with open(input_filepath, 'r') as f:
        data = json.load(f)
    
    # Reorganize data by category -> protocol -> date -> value
    category_data = {}
    
    for date_str, date_data in data.items():
        for category, protocols in date_data.items():
            if category not in category_data:
                category_data[category] = {}
            
            for protocol_name, protocol_values in protocols.items():
                if protocol_name not in category_data[category]:
                    category_data[category][protocol_name] = {}
                
                # Only add if metric exists in the data
                if metric in protocol_values and protocol_values[metric] is not None:
                    category_data[category][protocol_name][date_str] = protocol_values[metric]
    
    # Color palette for protocols
    protocol_colors = ['#1f1f60', '#313EA5', '#637BBF', '#94B9D9', 
                      '#333333', '#5F5F5F', '#8A8A8A', '#A0A0A0', 
                      '#B6B6B6', '#CCCCCC']
    
    # Create tables output directory
    ensure_directory(tables_filepath)
    
    # Plot each category
    for category, protocols_data in category_data.items():
        # Skip empty categories
        if not protocols_data:
            print(f"No data for category: {category}")
            continue
        
        # Clean up category name for title
        clean_category = category.replace('_', ' ').title()
        
        # Convert to DataFrame
        rows = []
        for protocol_name, protocol_dates in protocols_data.items():
            for date_str, value in protocol_dates.items():
                rows.append({
                    'date': date_str,
                    'protocol': protocol_name,
                    metric: value
                })
        
        if not rows:
            print(f"No data for category: {category}")
            continue
            
        df = pd.DataFrame(rows)
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter by start date
        df = df[df['date'] >= start_date]
        
        if df.empty:
            print(f"No data after {start_date} for category: {category}")
            continue
        
        # Pivot to get protocols as columns
        df_pivot = df.pivot_table(index='date', columns='protocol', values=metric)
        df_pivot = df_pivot.sort_index()
        
        # Forward fill null values
        df_pivot = df_pivot.ffill()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Get protocol names
        protocols = df_pivot.columns.tolist()
        
        # Plot each protocol
        for i, protocol in enumerate(protocols):
            color = protocol_colors[i % len(protocol_colors)]
            # Clean up protocol name for legend
            clean_protocol = protocol.replace('-', ' ').replace('_', ' ').title()
            ax.plot(
                df_pivot.index,
                df_pivot[protocol],
                label=clean_protocol,
                color=color,
                linewidth=2
            )
        
        # Extract trailing window from metric name (e.g., '30d' from 'FDV-to-annualized-fees-multiple-30d')
        window_match = re.search(r'-(\d+)d$', metric)
        trailing_window = window_match.group(1) if window_match else None
        
        # Clean up metric name for title - capitalize MCAP and FDV
        # Remove the trailing window from the title
        metric_title = metric
        if trailing_window:
            metric_title = metric_title.replace(f'-{trailing_window}d', '')
        metric_title = metric_title.replace('-', ' ').replace('_', ' ').title()
        metric_title = metric_title.replace('Mcap', 'MCAP').replace('Fdv', 'FDV').replace('Tvl', 'TVL')
        
        # Formatting
        ax.set_title(f'{clean_category}: {metric_title}')
        ax.margins(x=0)
        space_x_axis_dates(ax=ax, df_name=df_pivot.reset_index())
        plt.xticks(rotation=45)
        
        # Y-axis formatting
        data_max = df_pivot.max().max()  # Get max across all protocols
        ax.yaxis.set_major_formatter(format_multiples(data_max))
        ax.set_ylim(bottom=0)
        
        # Grid and legend
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=2, framealpha=1)
        
        # Format start date and end date for display
        formatted_start_date = format_date_with_suffix(start_date)
        last_data_date = get_last_data_date(df)
        
        # Build note text for chart
        chart_note = f'Note: all data from {formatted_start_date} to {last_data_date}.'
        if trailing_window:
            chart_note += f' All figures are {trailing_window}-day rolling averages.'
        chart_note += '\nSource: DefiLlama and Messari'
        
        # Add logo and notes
        plt.tight_layout()
        format_logo_and_note(ax, chart_note)
        
        # Save plot
        ensure_directory(plots_filepath)
        
        # Clean filename
        safe_metric_name = metric.replace(' ', '-').lower()
        plt.savefig(f"{plots_filepath}/{category}-{safe_metric_name}.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.show()
        
        # --- Generate statistics table ---
        stats_dict = {}
        for protocol in protocols:
            protocol_series = df_pivot[protocol].dropna()
            clean_protocol = protocol.replace('-', ' ').replace('_', ' ').title()
            stats_dict[clean_protocol] = [
                protocol_series.median(),
                protocol_series.quantile(0.8),
                protocol_series.quantile(0.2),
                protocol_series.iloc[-1] if len(protocol_series) > 0 else None
            ]
        
        stats_df = pd.DataFrame(stats_dict)
        stats_df.index = [
            'Median',
            '80th Percentile',
            '20th Percentile',
            'Most Recent'
        ]
        
        # Build protocol list for notes
        clean_protocols = [p.replace('-', ' ').replace('_', ' ').title() for p in protocols]
        if len(clean_protocols) > 1:
            protocol_list = ', '.join(clean_protocols[:-1]) + f", and {clean_protocols[-1]}"
        else:
            protocol_list = clean_protocols[0] if clean_protocols else ""
        
        # Build notes text for table
        notes_text = f"Note: All data from {formatted_start_date} to {last_data_date}."
        if trailing_window:
            notes_text += f" All figures are {trailing_window}-day rolling averages."
        notes_text += "\nSource: DefiLlama and Messari."
        
        # Determine format based on max value in stats_df (reusing format_multiples logic)
        stats_max = stats_df.max().max()
        formatter = format_multiples(stats_max)

        # Create a lambda that uses the formatter's internal function
        # The formatter.func is the function passed to FuncFormatter
        format_str = lambda v: formatter.func(v, None) if pd.notnull(v) else ""
        
        # Style the table for display in notebook
        styled = (
            stats_df
              .style
              .set_caption(f"{clean_category}: {metric_title}")
              .set_table_styles([
                  {"selector": "caption",
                   "props": [
                     ("font-weight", "bold"),
                     ("caption-side", "top"),
                     ("text-decoration", "underline")
                   ]},
                  {"selector": "thead th", "props": [("font-weight", "normal")]},
                  {"selector": "tbody th", "props": [("font-weight", "normal")]}
              ], overwrite=False)
              .format(format_str)
        )
        
        display(styled)
        print(notes_text)
        
        # --- Save table with notes using pure matplotlib ---
        num_cols = len(stats_df.columns)
        fig_width = max(12, 2 + num_cols * 2)
        fig_table, ax_table = plt.subplots(figsize=(fig_width, 4))
        ax_table.axis('off')
        
        # Format values for table using the same format logic
        table_data = [[format_str(v) for v in row] for row in stats_df.values]
        col_labels = stats_df.columns.tolist()
        row_labels = stats_df.index.tolist()
        
        # Create table with bbox to control position [left, bottom, width, height]
        table = ax_table.table(
            cellText=table_data,
            rowLabels=row_labels,
            colLabels=col_labels,
            cellLoc='center',
            rowLoc='right',
            loc='center',
            bbox=[0.1, 0.35, 0.88, 0.55]
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)
        
        # Auto-adjust column widths to fit content
        table.auto_set_column_width(col=list(range(-1, len(col_labels))))
        
        # Style header row
        for j in range(len(col_labels)):
            cell = table[(0, j)]
            cell.set_text_props(fontweight='bold')
            cell.set_facecolor('#f8f8f8')
            cell.set_edgecolor('#cccccc')
        
        # Style row labels
        for i in range(len(row_labels)):
            cell = table[(i + 1, -1)]
            cell.set_text_props(fontweight='bold')
            cell.set_facecolor('#f8f8f8')
            cell.set_edgecolor('#cccccc')
        
        # Style data cells
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                cell = table[(i + 1, j)]
                cell.set_edgecolor('#cccccc')
        
        # Add title
        ax_table.text(
            0.5, 0.95,
            f"{clean_category}: {metric_title}",
            transform=ax_table.transAxes,
            fontsize=14,
            fontweight='bold',
            ha='center',
            va='top'
        )
        
        plt.tight_layout()
        
        # Add logo and notes using the shared function
        format_logo_and_note(
            ax_table,
            notes_text,
            is_table=True
        )
        
        # Save table image
        table_filename = f"{tables_filepath}/{category}-{safe_metric_name}-table.png"
        plt.savefig(
            table_filename, 
            dpi=150, 
            bbox_inches='tight', 
            facecolor='white', 
            edgecolor='none',
            pad_inches=0.1
        )
        plt.close(fig_table)
        
        print()
        print(f"Table saved to: {table_filename}")

    print('Done')

@report_errors()
def borrow_lend_ratios_by_protocol(
    metrics,
    start_date,
    plots_filepath,
    tables_filepath,
    date,
    input_filepath='data/processed/borrow-lend/borrow-lend-ratios.json',
    trailing_window=30,
    annualize=False,
):
    """
    Plot Blockworks borrow/lend efficiency ratios across protocols, modeled on the
    valuation-multiple line charts (protocol_multiples_by_category): one line per protocol
    plus an accompanying statistics table for each requested ratio.

    Reads data/processed/borrow-lend/borrow-lend-ratios.json (produced by
    data_processing.calculate_borrow_lend_ratios). Because the saved ratios are raw daily
    values, a trailing rolling average is applied here for a comparable, smoothed look.

    Parameters:
    -----------
    metrics : str or list of str
        Ratio key(s) to plot, each producing its own chart and table. Valid keys:
        'interest_generated_to_total_borrows', 'net_interest_income_to_total_borrows',
        'revenue_to_total_borrows', 'interest_generated_to_total_deposits',
        'net_interest_income_to_total_deposits', 'revenue_to_total_deposits',
        'total_borrows_to_total_deposits'
    start_date : str
        Start date for filtering data (format: 'YYYY-MM-DD')
    plots_filepath : str
        Directory to save the plots
    tables_filepath : str
        Directory to save the tables
    date : str
        Data as-of date for chart annotation
    input_filepath : str
        Path to the borrow/lend ratios JSON file
    trailing_window : int
        Rolling-average window in days (default 30)
    annualize : bool
        If True, annualize the flow-based ratios (x365). The utilization ratio
        total_borrows_to_total_deposits is a stock/stock ratio and is never annualized.
    """

    # Allow a single metric or a list of metrics
    if isinstance(metrics, str):
        metrics = [metrics]

    # Human-readable names for each ratio metric
    metric_names = {
        'interest_generated_to_total_borrows': 'Interest Generated / Total Borrows',
        'net_interest_income_to_total_borrows': 'Net Interest Income / Total Borrows',
        'revenue_to_total_borrows': 'Revenue / Total Borrows',
        'interest_generated_to_total_deposits': 'Interest Generated / Total Deposits',
        'net_interest_income_to_total_deposits': 'Net Interest Income / Total Deposits',
        'revenue_to_total_deposits': 'Revenue / Total Deposits',
        'total_borrows_to_total_deposits': 'Total Borrows / Total Deposits',
    }

    # The utilization ratio is a stock/stock figure: it must not be annualized and is
    # expressed as a multiple (e.g. 0.4x) rather than a percentage.
    non_annualizable = {'total_borrows_to_total_deposits'}
    multiple_metrics = {'total_borrows_to_total_deposits'}

    # Load data
    with open(input_filepath, 'r') as f:
        data = json.load(f)

    # Color palette for protocols (matches protocol_multiples_by_category)
    protocol_colors = ['#1f1f60', '#313EA5', '#637BBF', '#94B9D9',
                       '#333333', '#5F5F5F', '#8A8A8A', '#A0A0A0',
                       '#B6B6B6', '#CCCCCC']

    category_title = 'Borrow/Lend Protocols'

    # Create tables output directory
    ensure_directory(tables_filepath)

    for metric in metrics:
        do_annualize = annualize and metric not in non_annualizable
        as_multiple = metric in multiple_metrics

        # Build a long dataframe of {date, protocol, value} for this ratio
        rows = []
        for protocol_slug, records in data.items():
            for record in records:
                value = record.get(metric)
                if value is None:
                    continue
                rows.append({
                    'date': record['date'],
                    'protocol': protocol_slug,
                    'value': value,
                })

        if not rows:
            print(f"No data for metric: {metric}")
            continue

        df = pd.DataFrame(rows)
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')

        if do_annualize:
            df['value'] = df['value'] * 365

        # Pivot to get protocols as columns, then smooth with a trailing average
        df_pivot = df.pivot_table(index='date', columns='protocol', values='value')
        df_pivot = df_pivot.sort_index()
        df_pivot = df_pivot.rolling(window=trailing_window, min_periods=1).mean()

        # Filter by start date
        df_pivot = df_pivot[df_pivot.index >= start_date]

        if df_pivot.empty:
            print(f"No data after {start_date} for metric: {metric}")
            continue

        # Build the metric title
        metric_title = metric_names.get(metric, metric.replace('_', ' ').title())
        if do_annualize:
            metric_title = f'Annualized {metric_title}'

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))

        protocols = df_pivot.columns.tolist()
        for i, protocol in enumerate(protocols):
            color = protocol_colors[i % len(protocol_colors)]
            clean_protocol = protocol.replace('-', ' ').replace('_', ' ').title()
            ax.plot(
                df_pivot.index,
                df_pivot[protocol],
                label=clean_protocol,
                color=color,
                linewidth=2
            )

        # Formatting
        ax.set_title(f'{category_title}: {metric_title}')
        ax.margins(x=0)
        space_x_axis_dates(ax=ax, df_name=df_pivot.reset_index())
        plt.xticks(rotation=45)

        # Y-axis: utilization ratio as a multiple (e.g. 0.4x), others as percentages.
        # Both use a magnitude-scaled helper so all ticks share consistent decimal places.
        data_max = df_pivot.max().max()
        if as_multiple:
            ax.yaxis.set_major_formatter(format_multiples(data_max))
        else:
            ax.yaxis.set_major_formatter(format_percentages(data_max))
        ax.set_ylim(bottom=0)

        # Grid and legend
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=2, framealpha=1)

        # Notes
        formatted_start_date = format_date_with_suffix(start_date)
        last_data_date = get_last_data_date(df_pivot.reset_index())
        chart_note = (
            f'Note: all data from {formatted_start_date} to {last_data_date}. '
            f'All figures are {trailing_window}-day rolling averages.'
        )
        chart_note += '\nSource: Blockworks'

        plt.tight_layout()
        format_logo_and_note(ax, chart_note)

        # Save plot
        ensure_directory(plots_filepath)
        safe_metric_name = metric.replace('_', '-')
        annualized_label = 'annualized-' if do_annualize else ''
        plot_filename = (
            f"{plots_filepath}/borrow-lend-{annualized_label}{safe_metric_name}-{trailing_window}d.png"
        )
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.show()

        # --- Generate statistics table ---
        stats_dict = {}
        for protocol in protocols:
            protocol_series = df_pivot[protocol].dropna()
            clean_protocol = protocol.replace('-', ' ').replace('_', ' ').title()
            stats_dict[clean_protocol] = [
                protocol_series.median(),
                protocol_series.quantile(0.8),
                protocol_series.quantile(0.2),
                protocol_series.iloc[-1] if len(protocol_series) > 0 else None
            ]

        stats_df = pd.DataFrame(stats_dict)
        stats_df.index = [
            'Median',
            '80th Percentile',
            '20th Percentile',
            'Most Recent'
        ]

        # Table formatter: multiple for the utilization ratio, percentage otherwise.
        # Reuse the same magnitude-scaled helpers for consistent decimal places.
        stats_max = stats_df.max().max()
        table_formatter = format_multiples(stats_max) if as_multiple else format_percentages(stats_max)
        format_str = lambda v: table_formatter.func(v, None) if pd.notnull(v) else ""

        # Build notes text for the table
        notes_text = (
            f"Note: All data from {formatted_start_date} to {last_data_date}. "
            f"All figures are {trailing_window}-day rolling averages."
        )
        notes_text += "\nSource: Blockworks."

        # Style the table for display in notebook
        styled = (
            stats_df
              .style
              .set_caption(f"{category_title}: {metric_title}")
              .set_table_styles([
                  {"selector": "caption",
                   "props": [
                     ("font-weight", "bold"),
                     ("caption-side", "top"),
                     ("text-decoration", "underline")
                   ]},
                  {"selector": "thead th", "props": [("font-weight", "normal")]},
                  {"selector": "tbody th", "props": [("font-weight", "normal")]}
              ], overwrite=False)
              .format(format_str)
        )

        display(styled)
        print(notes_text)

        # --- Save table with notes using pure matplotlib ---
        num_cols = len(stats_df.columns)
        fig_width = max(12, 2 + num_cols * 2)
        fig_table, ax_table = plt.subplots(figsize=(fig_width, 4))
        ax_table.axis('off')

        table_data = [[format_str(v) for v in row] for row in stats_df.values]
        col_labels = stats_df.columns.tolist()
        row_labels = stats_df.index.tolist()

        table = ax_table.table(
            cellText=table_data,
            rowLabels=row_labels,
            colLabels=col_labels,
            cellLoc='center',
            rowLoc='right',
            loc='center',
            bbox=[0.1, 0.35, 0.88, 0.55]
        )

        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)
        table.auto_set_column_width(col=list(range(-1, len(col_labels))))

        # Style header row
        for j in range(len(col_labels)):
            cell = table[(0, j)]
            cell.set_text_props(fontweight='bold')
            cell.set_facecolor('#f8f8f8')
            cell.set_edgecolor('#cccccc')

        # Style row labels
        for i in range(len(row_labels)):
            cell = table[(i + 1, -1)]
            cell.set_text_props(fontweight='bold')
            cell.set_facecolor('#f8f8f8')
            cell.set_edgecolor('#cccccc')

        # Style data cells
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                cell = table[(i + 1, j)]
                cell.set_edgecolor('#cccccc')

        ax_table.text(
            0.5, 0.95,
            f"{category_title}: {metric_title}",
            transform=ax_table.transAxes,
            fontsize=14,
            fontweight='bold',
            ha='center',
            va='top'
        )

        plt.tight_layout()
        format_logo_and_note(ax_table, notes_text, is_table=True)

        table_filename = f"{tables_filepath}/borrow-lend-{annualized_label}{safe_metric_name}-{trailing_window}d-table.png"
        plt.savefig(
            table_filename,
            dpi=150,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none',
            pad_inches=0.1
        )
        plt.close(fig_table)

        print()
        print(f"Table saved to: {table_filename}")

    print('Done')

@report_errors()
def weighted_multiples_by_category(metric, start_date, plots_filepath, tables_filepath, date, input_filepath='data/processed/valuation-multiples/weighted-multiples-by-category.json', log_scale=False):
    """
    Plot weighted valuation multiples by category.
    Creates one chart with all categories as separate lines.
    Also outputs a styled statistics table.
    
    Parameters:
    -----------
    metric : str
        Metric to plot: e.g., 'FDV-weighted-FDV-to-annualized-fees-multiple-30d'
    start_date : str
        Start date for filtering data (format: 'YYYY-MM-DD')
    input_filepath : str
        Path to the input JSON file
    plots_filepath : str
        Directory to save the plot
    tables_filepath : str
        Directory to save the tables
    date : str
        Data as-of date for chart annotation
    log_scale : bool, optional
        If True, use log scale for y-axis. Default is False (linear scale)
    """
    
    # Load data
    with open(input_filepath, 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame and track protocols
    rows = []
    category_protocols = {}  # Track protocols per category
    
    for date_str, date_data in data.items():
        for category, category_data in date_data.items():
            # Extract protocols-included list (using the most recent date's data)
            if category not in category_protocols and 'protocols-included' in category_data:
                protocols_included = category_data['protocols-included']
                # Clean up protocol names
                category_protocols[category] = [
                    p.replace('-', ' ').replace('_', ' ').title() 
                    for p in protocols_included
                ]
            
            # Extract from the "weighted-averages" nested dict
            if 'weighted-averages' in category_data:
                weighted_data = category_data['weighted-averages']
                if metric in weighted_data and weighted_data[metric] is not None:
                    rows.append({
                        'date': date_str,
                        'category': category,
                        metric: weighted_data[metric]
                    })
    
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter by start date
    df = df[df['date'] >= start_date]
    
    # Pivot to get categories as columns
    df_pivot = df.pivot_table(index='date', columns='category', values=metric)
    df_pivot = df_pivot.sort_index()
    
    # Forward fill null values
    df_pivot = df_pivot.ffill()
    
    # Create color palette
    color_list = ['#1f1f60', '#313EA5', '#637BBF', '#333333', 
                  '#5F5F5F', '#8A8A8A', '#A0A0A0', '#B6B6B6', '#CCCCCC']
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get category names
    categories = df_pivot.columns.tolist()
    
    # Plot each category
    for i, category in enumerate(categories):
        # Clean up category name for legend
        clean_category = category.replace('_', ' ').replace('-', ' ').title()
        color = color_list[i % len(color_list)]
        
        ax.plot(
            df_pivot.index,
            df_pivot[category],
            label=clean_category,
            color=color,
            linewidth=2
        )
    
    # Extract trailing window from metric name (e.g., '30d' from 'FDV-weighted-FDV-to-annualized-fees-multiple-30d')
    window_match = re.search(r'-(\d+)d$', metric)
    trailing_window = window_match.group(1) if window_match else None
    
    # Extract weighting method (FDV-weighted or MCAP-weighted)
    if metric.startswith('FDV-weighted'):
        weighting_method = 'FDV'
    elif metric.startswith('MCAP-weighted'):
        weighting_method = 'MCAP'
    else:
        weighting_method = None
    
    # Formatting - clean up metric title and capitalize MCAP and FDV
    # Remove the trailing window from the title
    metric_title = metric
    if trailing_window:
        metric_title = metric_title.replace(f'-{trailing_window}d', '')
    metric_title = metric_title.replace('FDV-weighted-', '').replace('MCAP-weighted-', '').replace('-', ' ').replace('_', ' ').title()
    metric_title = metric_title.replace('Mcap', 'MCAP').replace('Fdv', 'FDV').replace('Tvl', 'TVL')
    if log_scale == True:
        ax.set_title(f'Weighted {metric_title} by Category (Log)')
    else:
        ax.set_title(f'Weighted {metric_title} by Category')
    ax.margins(x=0)
    space_x_axis_dates(ax=ax, df_name=df_pivot.reset_index())
    plt.xticks(rotation=45)
    
    # Set scale based on parameter
    if log_scale:
        ax.set_yscale('log')
        # For log scale, don't set ylim(bottom=0) as log scale can't handle 0
        # Grid for log scale (both major and minor)
        ax.grid(axis='y', linestyle='--', alpha=0.4, which='both')
        # Y-axis formatting for log scale
        data_max = df_pivot.max().max()
        ax.yaxis.set_major_formatter(format_multiples(data_max))
    else:
        # Linear scale
        ax.set_ylim(bottom=0)
        # Grid for linear scale
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        # Y-axis formatting for linear scale
        data_max = df_pivot.max().max()
        ax.yaxis.set_major_formatter(format_multiples(data_max))
    
    # Legend
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=2, framealpha=1)
    
    # Format start date for display
    formatted_start_date = format_date_with_suffix(start_date)
    
    # Build note text for chart with protocol lists
    last_data_date = get_last_data_date(df)
    chart_note_lines = [f'Note: all data from {formatted_start_date} to {last_data_date}.']
    if trailing_window:
        chart_note_lines[0] += f' All figures are {trailing_window}-day rolling averages.'
    if weighting_method:
        chart_note_lines[0] += f' Metrics are weighted by {weighting_method}.'
    
    # Add protocol lists for each category
    for category in categories:
        clean_category = category.replace('_', ' ').replace('-', ' ').title()
        protocols = category_protocols.get(category, [])
        if len(protocols) > 1:
            protocol_list = ', '.join(protocols[:-1]) + f", and {protocols[-1]}"
        elif len(protocols) == 1:
            protocol_list = protocols[0]
        else:
            protocol_list = "N/A"
        chart_note_lines.append(f"{clean_category} included: {protocol_list}.")
    
    chart_note_lines.append('Source: DefiLlama and Messari')
    chart_note = '\n'.join(chart_note_lines)
    
    # Add logo and notes
    plt.tight_layout()
    format_logo_and_note(ax, chart_note)
    
    # Save plot
    ensure_directory(plots_filepath)
    
    # Clean filename - add log suffix if using log scale
    safe_metric_name = metric.replace(' ', '-').lower()
    scale_suffix = '-log' if log_scale else ''
    plt.savefig(f"{plots_filepath}/weighted-{safe_metric_name}-by-category{scale_suffix}.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    
    # --- Generate statistics table ---
    stats_dict = {}
    for category in categories:
        category_series = df_pivot[category].dropna()
        clean_category = category.replace('_', ' ').replace('-', ' ').title()
        stats_dict[clean_category] = [
            category_series.median(),
            category_series.quantile(0.8),
            category_series.quantile(0.2),
            category_series.iloc[-1] if len(category_series) > 0 else None
        ]
    
    stats_df = pd.DataFrame(stats_dict)
    stats_df.index = [
        'Median',
        '80th Percentile',
        '20th Percentile',
        'Most Recent'
    ]
    
    # Build notes with protocol lists for each category (for table)
    notes_lines = [f"Note: All data from {formatted_start_date} to {last_data_date}."]
    if trailing_window:
        notes_lines[0] += f" All figures are {trailing_window}-day rolling averages."
    if weighting_method:
        notes_lines[0] += f" Metrics are weighted by {weighting_method}."
    
    for category in categories:
        clean_category = category.replace('_', ' ').replace('-', ' ').title()
        protocols = category_protocols.get(category, [])
        if len(protocols) > 1:
            protocol_list = ', '.join(protocols[:-1]) + f", and {protocols[-1]}"
        elif len(protocols) == 1:
            protocol_list = protocols[0]
        else:
            protocol_list = "N/A"
        notes_lines.append(f"{clean_category} included: {protocol_list}.")
    notes_lines.append("Source: DefiLlama and Messari.")
    notes_text = '\n'.join(notes_lines)
    
    # Determine format for each column based on its max value
    format_funcs = {}
    for col in stats_df.columns:
        col_max = stats_df[col].max()
        formatter = format_multiples(col_max)
        format_funcs[col] = lambda v, f=formatter: f.func(v, None) if pd.notnull(v) else ""
    
    # Style the table for display in notebook
    styled = (
        stats_df
          .style
          .set_caption(f"Weighted {metric_title} by Category")
          .set_table_styles([
              {"selector": "caption",
               "props": [
                 ("font-weight", "bold"),
                 ("caption-side", "top"),
                 ("text-decoration", "underline")
               ]},
              {"selector": "thead th", "props": [("font-weight", "normal")]},
              {"selector": "tbody th", "props": [("font-weight", "normal")]}
          ], overwrite=False)
          .format(format_funcs)
    )
    
    display(styled)
    print(notes_text)
    
    # --- Save table with notes using pure matplotlib ---
    # Create tables output directory
    ensure_directory(tables_filepath)
    
    num_cols = len(stats_df.columns)
    fig_width = max(12, 2 + num_cols * 2)
    fig_table, ax_table = plt.subplots(figsize=(fig_width, 4))
    ax_table.axis('off')
    
    # Format values for table using column-specific formatting
    table_data = []
    for row_idx, row in enumerate(stats_df.values):
        formatted_row = []
        for col_idx, val in enumerate(row):
            col_name = stats_df.columns[col_idx]
            formatted_row.append(format_funcs[col_name](val))
        table_data.append(formatted_row)
    
    col_labels = stats_df.columns.tolist()
    row_labels = stats_df.index.tolist()
    
    # Create table with bbox to control position [left, bottom, width, height]
    table = ax_table.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc='center',
        rowLoc='right',
        loc='center',
        bbox=[0.1, 0.35, 0.88, 0.55]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Auto-adjust column widths to fit content
    table.auto_set_column_width(col=list(range(-1, len(col_labels))))
    
    # Style header row
    for j in range(len(col_labels)):
        cell = table[(0, j)]
        cell.set_text_props(fontweight='bold')
        cell.set_facecolor('#f8f8f8')
        cell.set_edgecolor('#cccccc')
    
    # Style row labels
    for i in range(len(row_labels)):
        cell = table[(i + 1, -1)]
        cell.set_text_props(fontweight='bold')
        cell.set_facecolor('#f8f8f8')
        cell.set_edgecolor('#cccccc')
    
    # Style data cells
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            cell = table[(i + 1, j)]
            cell.set_edgecolor('#cccccc')
    
    # Add title
    ax_table.text(
        0.5, 0.95,
        f"Weighted {metric_title} by Category",
        transform=ax_table.transAxes,
        fontsize=14,
        fontweight='bold',
        ha='center',
        va='top'
    )
    
    plt.tight_layout()
    
    # Add logo and notes using the shared function
    format_logo_and_note(
        ax_table,
        notes_text,
        is_table=True
    )
    
    # Save table image
    table_filename = f"{tables_filepath}/weighted-{safe_metric_name}-by-category-table.png"
    plt.savefig(
        table_filename, 
        dpi=150, 
        bbox_inches='tight', 
        facecolor='white', 
        edgecolor='none',
        pad_inches=0.1
    )
    plt.close(fig_table)
    
    print()
    print(f"Table saved to: {table_filename}")
    print('Done')

@report_errors()
def implied_valuation_from_category_multiples(
    protocol_slug,
    protocol_name,
    category,
    multiple_type,
    start_date,
    protocol_trailing_window=7,
    weighted_multiples_filepath='data/processed/valuation-multiples/weighted-multiples-by-category.json',
    protocol_multiples_filepath='data/processed/valuation-multiples/all-protocols-by-category.json',
    outlier_factor=100
):
    """
    Calculate implied valuations for a protocol based on category multiples.
    
    Takes the latest annualized metric for a protocol and multiplies it by
    the summary statistics (median, 80th percentile, 20th percentile, most recent)
    of a weighted multiple for a given category.
    
    If the protocol is included in the category, it will be excluded from
    the weighted average calculation to avoid circular comparison. Peers whose
    multiple is more than ``outlier_factor`` times above or below the category
    median on a given date are excluded for that date (see
    data_processing.weighted_average_multiple); the latest date's exclusions
    are reported and returned as ``latest_outliers_excluded``.

    Each implied FDV / MCAP is also converted to an implied token price by dividing
    by the supply behind the valuation type (latest raw FDV or MCAP divided by the
    close price on the same day from data/raw/messari): total supply for FDV,
    circulating supply for MCAP. Protocols without a Messari price file show N/A.

    A second table applies the same statistics (median, 80th / 20th percentile,
    most recent) to the protocol's own smoothed multiple over the same window, so
    the token is compared to its own history as well as to its peers. Returned as
    ``protocol_multiples``, ``protocol_implied_valuations``, ``protocol_implied_prices``
    and ``protocol_multiples_range``; None when the protocol has no multiple data
    after ``start_date``.

    Parameters:
    -----------
    protocol_slug : str
        Slug identifier for the protocol (e.g., 'hyperliquid')
    protocol_name : str
        Display name for the protocol (e.g., 'Hyperliquid')
    category : str
        Category to compare against (e.g., 'perps_dexs', 'liquid_staking')
    multiple_type : str
        Full multiple type string (e.g., 'FDV-weighted-FDV-to-annualized-fees-multiple-30d')
    start_date : str
        Start date for filtering multiples data (format: 'YYYY-MM-DD')
    protocol_trailing_window : int
        Trailing window for the protocol's metric (default 7 days)
    weighted_multiples_filepath : str
        Path to the weighted multiples JSON file
    protocol_multiples_filepath : str
        Path to the individual protocol multiples JSON file (output from process_asset_metrics_data)
    
    Returns:
    --------
    dict : Contains protocol_metric, category_multiples, implied_valuations,
        current_price, price_date, implied_supply and implied_prices
    """
    
    # Extract trailing window from multiple_type for the category multiples
    window_match = re.search(r'-(\d+)d$', multiple_type)
    multiples_trailing_window = int(window_match.group(1)) if window_match else 30
    
    # Extract weighting method (FDV or MCAP)
    if multiple_type.startswith('FDV-weighted'):
        weighting_method = 'FDV'
    elif multiple_type.startswith('MCAP-weighted'):
        weighting_method = 'MCAP'
    else:
        weighting_method = 'FDV'  # Default
    
    # Extract the underlying multiple metric (e.g., 'FDV-to-annualized-fees-multiple-30d')
    # Remove the weighting prefix to get the base multiple
    underlying_multiple = multiple_type.replace('FDV-weighted-', '').replace('MCAP-weighted-', '')
    
    # Extract metric from multiple_type
    # Pattern: ...-to-annualized-{metric}-multiple-... or ...-to-{metric}-multiple-...
    metric_match = re.search(r'-to-(?:annualized-)?(.+?)-multiple', multiple_type)
    if not metric_match:
        print(f"Could not extract metric from multiple_type: {multiple_type}")
        return None
    
    metric_id = metric_match.group(1)  # e.g., 'fees', 'revenue', 'holders-revenue', 'TVL'
    
    # Check if this is an annualized metric
    is_annualized = 'annualized' in multiple_type
    
    # Map metric_id to display name
    metric_display = {
        'fees': 'Fees',
        'revenue': 'Revenue',
        'holders-revenue': 'Holders Revenue',
        'earnings': 'Earnings',
        'TVL': 'TVL'
    }
    metric_name = metric_display.get(metric_id, metric_id.replace('-', ' ').title())
    
    # --- Load protocol metric data ---
    if metric_id == 'TVL':
        # Load TVL data
        with open(f'data/raw/defillama/tvl/{protocol_slug}.json', "r") as f:
            raw_data = json.load(f)
        
        df = pd.DataFrame(raw_data["tvl"])
        df["date"] = pd.to_datetime(df["date"], unit="s")
        df = df.rename(columns={"totalLiquidityUSD": metric_id})
        df = df[["date", metric_id]]
        
        # No annualization for TVL
        df[f'{metric_id}-{protocol_trailing_window}d'] = df[metric_id].rolling(window=protocol_trailing_window, min_periods=1).mean()
    else:
        # Load fees/revenue/holders-revenue data from DefiLlama
        with open(f'data/raw/defillama/{metric_id}/{protocol_slug}.json', "r") as f:
            raw_data = json.load(f)
        
        df = pd.DataFrame(raw_data["totalDataChart"], columns=["timestamp", metric_id])
        df["date"] = pd.to_datetime(df["timestamp"], unit="s")
        df = df[["date", metric_id]]
        
        # Annualize the metric
        df[metric_id] = df[metric_id] * 365
        
        # Calculate rolling average using protocol_trailing_window
        df[f'{metric_id}-{protocol_trailing_window}d'] = df[metric_id].rolling(window=protocol_trailing_window, min_periods=1).mean()
    
    # Get the latest value
    df = df.dropna(subset=[f'{metric_id}-{protocol_trailing_window}d'])
    latest_metric_value = df[f'{metric_id}-{protocol_trailing_window}d'].iloc[-1]
    latest_metric_date = df['date'].iloc[-1].strftime('%Y-%m-%d')
    
    # --- Load protocol-level multiples data for recalculation ---
    with open(protocol_multiples_filepath, 'r') as f:
        protocol_multiples_data = json.load(f)
    
    # --- Load weighted multiples data to get protocols included ---
    with open(weighted_multiples_filepath, 'r') as f:
        weighted_multiples_data = json.load(f)
    
    # --- Get protocol's smoothed multiple and valuation from pre-calculated data ---
    protocol_multiple = None
    latest_valuation = None
    
    # The protocol sits under its own category; look there first, then anywhere else it is filed
    # (e.g. comparing a perps DEX against the chains category).
    for date_str in sorted(protocol_multiples_data.keys(), reverse=True):
        date_data = protocol_multiples_data[date_str]
        asset_data = None
        if category in date_data and protocol_slug in date_data[category]:
            asset_data = date_data[category][protocol_slug]
        else:
            for cat_data in date_data.values():
                if protocol_slug in cat_data:
                    asset_data = cat_data[protocol_slug]
                    break
        if asset_data is not None:
            # Get the smoothed multiple (e.g., 'FDV-to-annualized-fees-multiple-7d')
            if underlying_multiple in asset_data and asset_data[underlying_multiple] is not None:
                protocol_multiple = asset_data[underlying_multiple]
            # Get the valuation (FDV or MCAP)
            if weighting_method in asset_data and asset_data[weighting_method] is not None:
                latest_valuation = asset_data[weighting_method]
            break
    
    # --- Load raw Messari marketcap data (valuation fallback and implied-price supply) ---
    with open(f'data/raw/messari/marketcap/{protocol_slug}.json', 'r') as f:
        marketcap_data = json.load(f)

    marketcap_df = pd.DataFrame(marketcap_data['data']['points'], columns=[
        'timestamp',
        'circulating-marketcap',
        'circulating-marketcap-dominance',
        'fully-diluted-marketcap'
    ])
    marketcap_df["date"] = pd.to_datetime(marketcap_df["timestamp"], unit="s")
    marketcap_df.rename(columns={
        'circulating-marketcap': 'MCAP',
        'fully-diluted-marketcap': 'FDV'
    }, inplace=True)
    marketcap_df = marketcap_df.sort_values('date')

    # Fallback: use the raw data if not found in pre-calculated data
    if latest_valuation is None:
        latest_valuation = marketcap_df[weighting_method].iloc[-1]

        # Calculate multiple from raw data as fallback
        if protocol_multiple is None and latest_metric_value > 0:
            protocol_multiple = latest_valuation / latest_metric_value

    # --- Derive the token supply behind the valuation type, for implied prices ---
    # FDV / price = total supply, MCAP / price = circulating supply. Use the latest day on
    # which both the raw valuation and the price are present so the pair is consistent.
    current_price = None
    price_date = None
    implied_supply = None
    try:
        with open(f'data/raw/messari/price/{protocol_slug}.json', 'r') as f:
            price_data = json.load(f)
        price_df = pd.DataFrame(price_data['data']['points'], columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume'
        ])
        price_df["date"] = pd.to_datetime(price_df["timestamp"], unit="s")
        pair_df = pd.merge(price_df[['date', 'close']], marketcap_df[['date', weighting_method]], on='date', how='inner')
        pair_df = pair_df[(pair_df['close'] > 0) & (pair_df[weighting_method] > 0)].dropna()
        if not pair_df.empty:
            latest_pair = pair_df.sort_values('date').iloc[-1]
            current_price = latest_pair['close']
            price_date = latest_pair['date'].strftime('%Y-%m-%d')
            implied_supply = latest_pair[weighting_method] / current_price
    except FileNotFoundError:
        print(f"No Messari price file for {protocol_slug}; implied prices will show N/A")

    # Get list of protocols in this category and check if target protocol is included
    original_category_protocols = []
    protocol_excluded = False
    
    for date_str, date_data in weighted_multiples_data.items():
        if category in date_data:
            category_data = date_data[category]
            if 'protocols-included' in category_data:
                original_category_protocols = category_data['protocols-included']
                # Check if our protocol is in the category
                if protocol_slug in original_category_protocols:
                    protocol_excluded = True
                break
    
    # Build clean protocol list (excluding target protocol for display)
    category_protocols = [
        p.replace('-', ' ').replace('_', ' ').title() 
        for p in original_category_protocols if p != protocol_slug
    ]
    
    # --- Recalculate weighted averages excluding target protocol ---
    # This follows the same logic as calculate_weighted_multiples
    category_values = []
    outliers_by_date = {}
    protocol_values = []  # the protocol's own smoothed multiple over the same window

    def own_multiple(asset_data):
        value = asset_data.get(underlying_multiple)
        if value is None or pd.isna(value) or value <= 0:
            return None
        return value

    for date_str in sorted(protocol_multiples_data.keys()):
        if date_str < start_date:
            continue

        if category not in protocol_multiples_data[date_str]:
            continue

        category_date_data = protocol_multiples_data[date_str][category]

        # Protocol's own multiple on this date (it sits under its category even when excluded from the average)
        if protocol_slug in category_date_data:
            value = own_multiple(category_date_data[protocol_slug])
            if value is not None:
                protocol_values.append({'date': date_str, 'multiple': value})

        # Determine which assets to include (exclude target protocol)
        candidate_slugs = [slug for slug in original_category_protocols if slug != protocol_slug]
        
        # Calculate weighted multiple using same logic as calculate_weighted_multiples
        valid_assets = []
        
        for slug in candidate_slugs:
            if slug not in category_date_data:
                continue
            
            asset_data = category_date_data[slug]
            top_col = weighting_method  # FDV or MCAP
            mult_col = underlying_multiple  # e.g., 'FDV-to-annualized-fees-multiple-30d'
            
            if top_col in asset_data and mult_col in asset_data:
                top_value = asset_data[top_col]
                multiple_value = asset_data[mult_col]
                
                # Skip invalid values
                if top_value is None or multiple_value is None:
                    continue
                if pd.isna(top_value) or pd.isna(multiple_value):
                    continue
                if top_value <= 0 or multiple_value <= 0:
                    continue
                
                valid_assets.append({
                    'slug': slug,
                    'top_value': top_value,
                    'multiple': multiple_value
                })
        
        # Calculate weighted average for this date (same helper as calculate_weighted_multiples)
        weighted_multiple, excluded = weighted_average_multiple(valid_assets, outlier_factor)
        if excluded:
            outliers_by_date[date_str] = excluded
        if weighted_multiple is not None:
            category_values.append({
                'date': date_str,
                'multiple': weighted_multiple
            })
    
    if not category_values:
        print(f"No data found for category '{category}' with multiple '{multiple_type}'")
        return None
    
    # Convert to DataFrame for statistics
    df_multiples = pd.DataFrame(category_values)
    df_multiples['date'] = pd.to_datetime(df_multiples['date'])
    df_multiples = df_multiples.sort_values('date')
    
    # Forward fill null values (same as weighted_multiples_by_category)
    df_multiples = df_multiples.ffill()
    
    if df_multiples.empty:
        print(f"No data found for category '{category}' after {start_date}")
        return None
    
    multiple_series = df_multiples['multiple']
    latest_date_str = df_multiples['date'].iloc[-1].strftime('%Y-%m-%d')
    latest_outliers_excluded = outliers_by_date.get(latest_date_str, [])
    
    # Calculate summary statistics (same as weighted_multiples_by_category)
    summary_stats = {
        'Median': multiple_series.median(),
        '80th Percentile': multiple_series.quantile(0.8),
        '20th Percentile': multiple_series.quantile(0.2),
        'Most Recent': multiple_series.iloc[-1]
    }
    
    # Calculate implied valuations
    implied_valuations = {
        stat_name: latest_metric_value * multiple_val
        for stat_name, multiple_val in summary_stats.items()
    }

    # Implied per-token prices: implied valuation / supply behind the valuation type
    def prices_from(valuations):
        return {
            stat_name: (implied_val / implied_supply if implied_supply else None)
            for stat_name, implied_val in valuations.items()
        }
    implied_prices = prices_from(implied_valuations)

    # --- Protocol vs its own history: same statistics on the protocol's own multiple ---
    if not protocol_values:
        # Fallback: the protocol may be filed under a different category than the one compared against
        for date_str in sorted(protocol_multiples_data.keys()):
            if date_str < start_date:
                continue
            for cat_data in protocol_multiples_data[date_str].values():
                if protocol_slug in cat_data:
                    value = own_multiple(cat_data[protocol_slug])
                    if value is not None:
                        protocol_values.append({'date': date_str, 'multiple': value})
                    break

    protocol_stats = None
    protocol_implied_valuations = None
    protocol_implied_prices = None
    protocol_multiples_range = None
    if protocol_values:
        df_protocol = pd.DataFrame(protocol_values).sort_values('date')
        protocol_series = df_protocol['multiple']
        protocol_stats = {
            'Median': protocol_series.median(),
            '80th Percentile': protocol_series.quantile(0.8),
            '20th Percentile': protocol_series.quantile(0.2),
            'Most Recent': protocol_series.iloc[-1]
        }
        protocol_implied_valuations = {
            stat_name: latest_metric_value * multiple_val
            for stat_name, multiple_val in protocol_stats.items()
        }
        protocol_implied_prices = prices_from(protocol_implied_valuations)
        protocol_multiples_range = {
            'start': df_protocol['date'].iloc[0],
            'end': df_protocol['date'].iloc[-1],
            'n_days': len(df_protocol)
        }

    # --- Print output in similar format to weighted_multiples_by_category ---
    
    # Clean up names for display
    clean_category = category.replace('_', ' ').replace('-', ' ').title()
    clean_multiple = multiple_type.replace('-', ' ').replace('_', ' ').title()
    clean_multiple = clean_multiple.replace('Mcap', 'MCAP').replace('Fdv', 'FDV').replace('Tvl', 'TVL')
    
    # Format dates for display
    formatted_start_date = format_date_with_suffix(start_date)
    formatted_end_date = format_date_with_suffix(latest_metric_date)
    
    # Get formatter for multiples
    multiples_max = max(summary_stats.values())
    if protocol_multiple is not None:
        multiples_max = max(multiples_max, protocol_multiple)
    if protocol_stats is not None:
        multiples_max = max(multiples_max, max(protocol_stats.values()))
    multiples_formatter = format_multiples(multiples_max)
    
    def format_multiple_val(value):
        return multiples_formatter.func(value, None)
    
    # Dynamic formatter for valuations (formats each value based on its own magnitude)
    def format_valuation(x):
        if x >= 1e12:
            return f'${x/1e12:.1f}t'
        elif x >= 1e9:
            return f'${x/1e9:.1f}b'
        elif x >= 1e6:
            return f'${x/1e6:.0f}m'
        elif x >= 1e3:
            return f'${x/1e3:.0f}k'
        elif x >= 1:
            return f'${x:.0f}'
        else:
            return f'${x:.2f}'

    # Per-token prices: cents above $1, three significant figures below it
    def format_price(x):
        if x is None or pd.isna(x):
            return 'N/A'
        if x >= 1:
            return f'${x:,.2f}'
        return f'${x:.3g}'

    # Plain token counts (no $) for the supply note
    def format_count(x):
        if x >= 1e12:
            return f'{x/1e12:.2f}t'
        elif x >= 1e9:
            return f'{x/1e9:.2f}b'
        elif x >= 1e6:
            return f'{x/1e6:.1f}m'
        elif x >= 1e3:
            return f'{x/1e3:.0f}k'
        return f'{x:.0f}'

    # Build protocol list string (already excludes target protocol)
    if len(category_protocols) > 1:
        protocol_list = ', '.join(category_protocols[:-1]) + f", and {category_protocols[-1]}"
    elif len(category_protocols) == 1:
        protocol_list = category_protocols[0]
    else:
        protocol_list = "N/A"
    
    # Print header
    line_spacing = 80
    print("=" * line_spacing)
    print(f"{protocol_name} Implied Valuation from {clean_category} Multiples")
    print("=" * line_spacing)
    print()
    
    # Print protocol metric
    print(f"Protocol: {protocol_name}")
    if is_annualized:
        print(f"Latest Annualized {metric_name} ({protocol_trailing_window}d avg): {format_valuation(latest_metric_value)}")
    else:
        print(f"Latest {metric_name} ({protocol_trailing_window}d avg): {format_valuation(latest_metric_value)}")
    print(f"Latest {weighting_method} ({multiples_trailing_window}d avg): {format_valuation(latest_valuation)}")
    print(f"Current Price: {format_price(current_price)}")
    if protocol_multiple is not None:
        print(f"Latest {weighting_method}/Annualized {metric_name} Multiple ({multiples_trailing_window}d avg): {format_multiple_val(protocol_multiple)}")
    else:
        print(f"Latest {weighting_method}/Annualized {metric_name} Multiple ({multiples_trailing_window}d avg): N/A")
    print(f"As of: {formatted_end_date}")
    print()
    
    # Shared table body: statistic, multiple, implied valuation, implied price
    def print_implied_table(multiple_col, stats, valuations, prices):
        multiple_width = max(17, len(multiple_col))
        print("-" * line_spacing)
        print(f"{'Statistic':<20} {multiple_col:>{multiple_width}} {f'Implied {weighting_method}':>14} {'Implied Price':>14}")
        print("-" * line_spacing)
        for stat_name in ['Median', '80th Percentile', '20th Percentile', 'Most Recent']:
            print(f"{stat_name:<20} {format_multiple_val(stats[stat_name]):>{multiple_width}} "
                  f"{format_valuation(valuations[stat_name]):>14} {format_price(prices[stat_name]):>14}")
        print("-" * line_spacing)
        print()

    # Table 1: protocol vs category multiples
    print(f"Category: {clean_category}")
    print(f"Multiple: {clean_multiple}")
    print_implied_table(f'{clean_category} Multiple', summary_stats, implied_valuations, implied_prices)

    # Table 2: protocol vs its own historical multiples
    if protocol_stats is not None:
        print(f"Protocol: {protocol_name}")
        print(f"Multiple: {weighting_method}/{'Annualized ' if is_annualized else ''}{metric_name} ({multiples_trailing_window}d avg)")
        print_implied_table(f'{protocol_name} Multiple', protocol_stats, protocol_implied_valuations, protocol_implied_prices)
    else:
        print(f"No historical {underlying_multiple} data for {protocol_name} after {start_date}")
        print()

    print(f"Note: Multiples are {multiples_trailing_window}-day rolling averages and weighted by {weighting_method}.")
    if implied_supply is not None:
        supply_kind = 'total' if weighting_method == 'FDV' else 'circulating'
        print(f"Implied price = implied {weighting_method} / {supply_kind} supply of ~{format_count(implied_supply)} tokens "
              f"(latest {weighting_method} / price on {format_date_with_suffix(price_date)}).")
    print(f"Category multiples from {formatted_start_date} to {formatted_end_date}.")
    if protocol_multiples_range is not None:
        print(f"{protocol_name} multiples from {format_date_with_suffix(protocol_multiples_range['start'])} "
              f"to {format_date_with_suffix(protocol_multiples_range['end'])} ({protocol_multiples_range['n_days']} days with data).")
    print(f"{clean_category} included: {protocol_list}.")
    if protocol_excluded:
        print(f"{protocol_name} was excluded from the {clean_category} weighted average to avoid circular comparison.")
    if latest_outliers_excluded:
        excluded_names = format_list_with_oxford_comma([p.replace('-', ' ').replace('_', ' ').title() for p in latest_outliers_excluded])
        print(f"Outliers excluded on {formatted_end_date} (multiple more than {outlier_factor}x above or below the category median): {excluded_names}.")
    print("Source: DefiLlama and Messari.")
    print()
    
    # Return data for programmatic use
    return {
        'protocol': protocol_name,
        'protocol_slug': protocol_slug,
        'metric': metric_name,
        'metric_value': latest_metric_value,
        'metric_date': latest_metric_date,
        'protocol_valuation': latest_valuation,
        'protocol_multiple': protocol_multiple,
        'valuation_type': weighting_method,
        'current_price': current_price,
        'price_date': price_date,
        'implied_supply': implied_supply,
        'implied_prices': implied_prices,
        'protocol_trailing_window': protocol_trailing_window,
        'category': category,
        'multiple_type': multiple_type,
        'multiples_trailing_window': multiples_trailing_window,
        'start_date': start_date,
        'category_multiples': summary_stats,
        'implied_valuations': implied_valuations,
        'protocol_multiples': protocol_stats,
        'protocol_implied_valuations': protocol_implied_valuations,
        'protocol_implied_prices': protocol_implied_prices,
        'protocol_multiples_range': protocol_multiples_range,
        'category_protocols': category_protocols,
        'protocol_excluded': protocol_excluded,
        'outlier_factor': outlier_factor,
        'latest_outliers_excluded': latest_outliers_excluded
    }

# Plot bottom metric vs valuation multiple for a given asset
@report_errors()
def asset_fees_vs_multiple(asset_slug, asset_name, bottom_metric, top_metric, start_date, trailing_window, multiples_filepath, plots_filepath, date):
    """
    Plot daily bottom metric (fees/revenue/holders-revenue) vs valuation multiple with trailing averages.
    
    Parameters:
    - asset_slug: slug identifier for the asset (e.g., 'hyperliquid')
    - asset_name: display name for the asset (e.g., 'Hyperliquid', '$HYPE')
    - bottom_metric: bottom metric to plot (e.g., 'fees', 'revenue', 'holders-revenue')
    - top_metric: top metric for multiple calculation (e.g., 'FDV', 'MCAP')
    - start_date: start date for filtering data
    - trailing_window: number of days for rolling average
    - multiples_filepath: path to the multiples JSON file
    - plots_filepath: directory path to save plots
    - date: date string for the note
    """
    
    # Determine metric display name
    metric_display = {
        'fees': 'Fees',
        'revenue': 'Revenue',
        'holders-revenue': 'Holders Revenue',
        'earnings': 'Earnings'
    }
    metric_name = metric_display.get(bottom_metric, bottom_metric.replace('-', ' ').title())
    
    # Load asset bottom metric data
    with open(f'data/raw/defillama/{bottom_metric}/{asset_slug}.json', "r") as f:
        bottom_data = json.load(f)
    
    bottom_df = pd.DataFrame(bottom_data["totalDataChart"], columns=["timestamp", bottom_metric])
    bottom_df["date"] = pd.to_datetime(bottom_df["timestamp"], unit="s")
    bottom_df["date"] = bottom_df["date"].dt.strftime('%Y-%m-%d')
    bottom_df = bottom_df[["date", bottom_metric]]
    
    # Annualize bototm metric
    bottom_df[bottom_metric] = bottom_df[bottom_metric] * 365
    
    # Load multiples data
    with open(multiples_filepath, 'r') as f:
        multiples_data = json.load(f)
    
    # Extract multiple data for this specific asset
    multiple_rows = []
    for date_str, date_data in multiples_data.items():
        for category, assets in date_data.items():
            for protocol_name, protocol_values in assets.items():
                # Match the asset slug
                if protocol_name == asset_slug:
                    # Build the multiple key
                    multiple_key = f"{top_metric}-to-annualized-{bottom_metric}-multiple-{trailing_window}d"
                    if multiple_key in protocol_values and protocol_values[multiple_key] is not None:
                        multiple_rows.append({
                            'date': date_str,
                            'multiple': protocol_values[multiple_key]
                        })
    
    if not multiple_rows:
        print(f"No multiple data found for {asset_slug} with metric {top_metric}-to-annualized-{bottom_metric}-multiple-{trailing_window}d")
        return
    
    multiple_df = pd.DataFrame(multiple_rows)
    
    # Merge datasets
    df = pd.merge(bottom_df, multiple_df, on="date", how="inner")
    
    # Convert date back to datetime for plotting
    df['date'] = pd.to_datetime(df['date'])
    
    # Calculate trailing average for bottom metric
    df = df.sort_values('date')
    df[f'{bottom_metric}-{trailing_window}d'] = df[bottom_metric].rolling(window=trailing_window).mean()
    
    # Drop any rows without a full trailing average, plus the very last day
    plot_df = df[df[f"{bottom_metric}-{trailing_window}d"].notna()].iloc[:-1]
    
    # Filter by start_date
    plot_df = plot_df[plot_df['date'] >= start_date]
    
    if plot_df.empty:
        print(f"No data after {start_date} for {asset_name}")
        return
    
    # Plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Area: Daily bottom metric
    ax1.fill_between(
        plot_df["date"], plot_df[f"{bottom_metric}-{trailing_window}d"],
        color="#a4a2a8", alpha=1,
        label=f"Annualized Daily {metric_name}"
    )
    
    ax1.set_ylabel(f"{asset_name} Annualized Daily {metric_name}")
    bottom_max = plot_df[f"{bottom_metric}-{trailing_window}d"].max()
    ax1.yaxis.set_major_formatter(mtick.FuncFormatter(format_large_numbers(bottom_max)))
    ax1.set_ylim(0)
    ax1.set_xlim(plot_df["date"].iloc[0], plot_df["date"].iloc[-1])
    ax1.margins(x=0)
    fig.subplots_adjust(left=0.07)
    
    # Format x-axis
    space_x_axis_dates(ax=ax1, df_name=plot_df)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
    
    # Line: Valuation Multiple (already smoothed from the data processing)
    ax2 = ax1.twinx()
    ax2.plot(
        plot_df["date"], plot_df["multiple"],
        color="#1f1f60", linewidth=2,
        label=f"{top_metric}/Annualized {metric_name}"
    )
    
    # Clean up metric name for labels
    multiple_label = f"{top_metric}/Annualized {metric_name}"
    ax2.set_ylabel(f"{multiple_label}")
    ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x:.0f}x'))
    ax2.set_ylim(0)
    
    # Title, grid & legend
    ax1.set_title(f"{asset_name}: Annualized Daily {metric_name} vs {multiple_label}", fontsize=14)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, bbox_to_anchor=(0.9, -.15), loc="upper center")
    
    # Save & show
    plt.tight_layout()

    last_data_date = get_last_data_date(df)

    format_logo_and_note(
        ax1,
        f"Note: all data as of {last_data_date}. All figures are {trailing_window}-day rolling averages.\nSource: DefiLlama and Messari"
    )
    
    # Save plot
    ensure_directory(plots_filepath)
    
    plt.savefig(f"{plots_filepath}/{asset_slug}_{bottom_metric}_vs_{top_metric}_multiple_{trailing_window}d.png", dpi=300, bbox_inches="tight")

    # Show
    plt.show()

    print(plot_df.tail(5))


@report_errors()
def asset_vs_comps_multiples(
    breakout_assets,
    liquid_comp_categories,
    equity_comps,
    multiple_type,
    start_date,
    plots_filepath,
    tables_filepath,
    date,
    log_scale=False,
    protocol_multiples_filepath='data/processed/valuation-multiples/all-protocols-by-category.json',
    weighted_multiples_filepath='data/processed/valuation-multiples/weighted-multiples-by-category.json',
    equity_data_dir='data/raw/yahoofinance/mcap-to-annualized-total-revenue',
    index_path=None
):
    """
    Compare breakout assets' valuation multiples against equity comps and liquid (token) comps.

    Creates a line chart with multiple lines:
    1. Each breakout asset's multiple over time
    2. Market-cap weighted average of equity comps (or equity category weighted averages)
    3. Market-cap weighted average for each liquid comp category (excluding breakout assets if present)

    Also creates a statistics table with median, 80th/20th percentile, and most recent values.

    Parameters:
    -----------
    breakout_assets : dict, list, or str
        Can be:
        - Dictionary mapping asset slugs to display names (e.g., {'hyperliquid': 'Hyperliquid'})
        - List of asset slugs (e.g., ['hyperliquid', 'dydx']) - display names auto-fetched from index
        - Single asset slug string (e.g., 'hyperliquid') - display name auto-fetched from index
    liquid_comp_categories : dict, list, or str
        Can be:
        - Dictionary mapping category slugs to display names (e.g., {'perps_dexs': 'Perps DEXs'})
        - List of category slugs (e.g., ['perps_dexs']) - names auto-formatted (title case, underscores to spaces)
        - Single category slug string (e.g., 'perps_dexs') - name auto-formatted
    equity_comps : dict, list, or str
        Can be:
        - Dictionary mapping ticker symbols to display names (e.g., {'COIN': 'Coinbase'})
        - List of ticker symbols (e.g., ['COIN', 'HOOD']) - display names auto-fetched from index
        - Single ticker symbol string (e.g., 'COIN') - display name auto-fetched from index
        - Dictionary mapping equity category slugs to display names (e.g., {'crypto_equities': 'Crypto Equities'})
          In this case, all equities in those categories will be loaded and grouped by category
    multiple_type : str
        Multiple type for the token (e.g., 'FDV-to-annualized-fees-multiple-7d')
        For equities, uses mcap-to-annualized-total-revenue (P/S equivalent)
    start_date : str
        Start date for filtering data (format: 'YYYY-MM-DD')
    plots_filepath : str
        Directory to save the plots
    tables_filepath : str
        Directory to save the tables
    date : str
        Data as-of date for chart annotation
    log_scale : bool, optional
        If True, use log scale for y-axis. Default is False (linear scale)
    protocol_multiples_filepath : str
        Path to the individual protocol multiples JSON file
    weighted_multiples_filepath : str
        Path to the weighted multiples JSON file (to get protocols included)
    equity_data_dir : str
        Directory containing equity comp JSON files
    index_path : str, optional
        Path to protocol_index.json file

    Returns:
    --------
    dict : Contains data for all series and summary statistics
    """
    # Import protocol_index for lookups
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data import protocol_index as pi

    # --- Normalize breakout_assets ---
    if isinstance(breakout_assets, str):
        breakout_assets = [breakout_assets]
    if isinstance(breakout_assets, list):
        breakout_assets = pi.get_display_names_dict(standard_names=breakout_assets, index_path=index_path)

    # --- Normalize liquid_comp_categories ---
    if isinstance(liquid_comp_categories, str):
        liquid_comp_categories = [liquid_comp_categories]
    if isinstance(liquid_comp_categories, list):
        # Auto-format category names: replace underscores with spaces, title case
        liquid_comp_categories = {
            cat: cat.replace('_', ' ').title()
            for cat in liquid_comp_categories
        }

    # --- Normalize equity_comps ---
    # Check if equity_comps is category-based (equity category slugs) or ticker-based
    equity_categories_mode = False
    equity_categories = {}

    if isinstance(equity_comps, str):
        equity_comps = [equity_comps]

    if isinstance(equity_comps, list):
        # Check if these are equity category slugs or individual tickers
        all_equity_categories = pi.get_all_equity_categories(index_path=index_path)
        if all(item in all_equity_categories for item in equity_comps):
            # These are category slugs
            equity_categories_mode = True
            equity_categories = {
                cat: cat.replace('_', ' ').title()
                for cat in equity_comps
            }
        else:
            # These are individual tickers
            equity_comps = pi.get_equity_display_names_dict(tickers=equity_comps, index_path=index_path)
    elif isinstance(equity_comps, dict):
        # Check if keys are equity categories
        all_equity_categories = pi.get_all_equity_categories(index_path=index_path)
        if all(key in all_equity_categories for key in equity_comps.keys()):
            equity_categories_mode = True
            equity_categories = equity_comps

    # Extract trailing window from multiple_type
    window_match = re.search(r'-(\d+)d$', multiple_type)
    trailing_window = int(window_match.group(1)) if window_match else 7

    # Extract weighting method from multiple_type
    if multiple_type.startswith('FDV'):
        weighting_method = 'FDV'
    elif multiple_type.startswith('MCAP'):
        weighting_method = 'MCAP'
    else:
        weighting_method = 'FDV'

    # Get list of breakout asset slugs for exclusion
    breakout_slugs = list(breakout_assets.keys())

    # --- 1. Load breakout assets' multiple data ---
    with open(protocol_multiples_filepath, 'r') as f:
        protocol_multiples_data = json.load(f)

    asset_dfs = []
    for asset_slug, asset_name in breakout_assets.items():
        asset_rows = []
        for date_str, date_data in protocol_multiples_data.items():
            if date_str < start_date:
                continue
            for category, protocols in date_data.items():
                if asset_slug in protocols:
                    asset_data = protocols[asset_slug]
                    if multiple_type in asset_data and asset_data[multiple_type] is not None:
                        asset_rows.append({
                            'date': date_str,
                            'value': asset_data[multiple_type]
                        })
                    break

        if not asset_rows:
            print(f"Warning: No data found for {asset_slug} with multiple {multiple_type}")
            continue

        asset_df = pd.DataFrame(asset_rows)
        asset_df['date'] = pd.to_datetime(asset_df['date'])
        asset_df = asset_df.sort_values('date').drop_duplicates(subset=['date'])
        asset_df = asset_df.set_index('date')
        asset_df = asset_df.rename(columns={'value': asset_name})
        asset_dfs.append(asset_df)

    if not asset_dfs:
        print("No data found for any breakout assets")
        return None

    # --- 2. Load and calculate equity comps weighted average(s) ---
    equity_result_dfs = []
    equity_constituents_map = {}  # category_display_name -> list of equity names (for footnotes)

    def calc_weighted_avg(group):
        total_mcap = group['mcap'].sum()
        if total_mcap > 0:
            return (group['mcap'] * group['multiple']).sum() / total_mcap
        return None

    if equity_categories_mode:
        # Handle equity categories (similar to category_comps_multiples)
        for category_slug, category_display_name in equity_categories.items():
            # Get all tickers in this equity category from the index
            tickers_dict = pi.get_equity_display_names_dict(category=category_slug, index_path=index_path)

            if not tickers_dict:
                print(f"Warning: No equities found for category {category_slug}")
                continue

            category_equity_dfs = []
            for ticker, display_name in tickers_dict.items():
                filepath = f"{equity_data_dir}/{ticker}.json"
                try:
                    with open(filepath, 'r') as f:
                        equity_data = json.load(f)
                except FileNotFoundError:
                    print(f"Warning: Equity data file not found for {ticker}")
                    continue

                rows = []
                for date_str, values in equity_data.items():
                    if date_str < start_date:
                        continue
                    mcap_key = f"{ticker}-market-cap"
                    multiple_key = f"{ticker}-7d-avg"

                    if mcap_key in values and multiple_key in values:
                        mcap = values[mcap_key]
                        multiple = values[multiple_key]
                        if mcap is not None and multiple is not None and mcap > 0 and multiple > 0:
                            rows.append({
                                'date': date_str,
                                'mcap': mcap,
                                'multiple': multiple,
                                'ticker': ticker
                            })

                if rows:
                    category_equity_dfs.append(pd.DataFrame(rows))

            if not category_equity_dfs:
                print(f"Warning: No equity data found for category {category_slug}")
                continue

            # Combine and calculate market-cap weighted average for this category
            equity_combined = pd.concat(category_equity_dfs, ignore_index=True)
            equity_combined['date'] = pd.to_datetime(equity_combined['date'])

            equity_weighted = equity_combined.groupby('date').apply(calc_weighted_avg, include_groups=False).reset_index()
            equity_weighted.columns = ['date', category_display_name]
            equity_weighted = equity_weighted.set_index('date')
            equity_result_dfs.append(equity_weighted)

            # Store constituent list for footnotes
            equity_constituents_map[category_display_name] = list(tickers_dict.values())
    else:
        # Handle individual tickers (original behavior)
        equity_dfs = []

        for ticker, display_name in equity_comps.items():
            filepath = f"{equity_data_dir}/{ticker}.json"
            try:
                with open(filepath, 'r') as f:
                    equity_data = json.load(f)
            except FileNotFoundError:
                print(f"Warning: Equity data file not found for {ticker}")
                continue

            rows = []
            for date_str, values in equity_data.items():
                if date_str < start_date:
                    continue
                mcap_key = f"{ticker}-market-cap"
                multiple_key = f"{ticker}-7d-avg"

                if mcap_key in values and multiple_key in values:
                    mcap = values[mcap_key]
                    multiple = values[multiple_key]
                    if mcap is not None and multiple is not None and mcap > 0 and multiple > 0:
                        rows.append({
                            'date': date_str,
                            'mcap': mcap,
                            'multiple': multiple,
                            'ticker': ticker
                        })

            if rows:
                equity_dfs.append(pd.DataFrame(rows))

        if equity_dfs:
            # Combine and calculate market-cap weighted average
            equity_combined = pd.concat(equity_dfs, ignore_index=True)
            equity_combined['date'] = pd.to_datetime(equity_combined['date'])

            equity_weighted = equity_combined.groupby('date').apply(calc_weighted_avg, include_groups=False).reset_index()
            equity_weighted.columns = ['date', 'Equity Comps']
            equity_weighted = equity_weighted.set_index('date')
            equity_result_dfs.append(equity_weighted)

    if not equity_result_dfs:
        print("No equity comp data found")
        return None

    # --- 3. Load and calculate liquid comps weighted averages for each category ---
    with open(weighted_multiples_filepath, 'r') as f:
        weighted_multiples_data = json.load(f)

    # Store DataFrames and protocol lists for each category
    liquid_dfs = []
    category_protocols_map = {}  # category_display_name -> list of protocol names
    category_exclusions_map = {}  # category_display_name -> list of excluded asset names

    for category_slug, category_display_name in liquid_comp_categories.items():
        # Get list of protocols in the category
        original_category_protocols = []
        for date_str, date_data in weighted_multiples_data.items():
            if category_slug in date_data:
                category_data = date_data[category_slug]
                if 'protocols-included' in category_data:
                    original_category_protocols = category_data['protocols-included']
                    break

        # Track which breakout assets are excluded from this category
        excluded_assets = []
        for asset_slug, asset_name in breakout_assets.items():
            if asset_slug in original_category_protocols:
                excluded_assets.append(asset_name)

        # Build weighted average excluding all breakout assets
        liquid_rows = []
        for date_str in sorted(protocol_multiples_data.keys()):
            if date_str < start_date:
                continue

            if category_slug not in protocol_multiples_data[date_str]:
                continue

            category_date_data = protocol_multiples_data[date_str][category_slug]

            # Exclude all breakout assets from this category
            candidate_slugs = [slug for slug in original_category_protocols if slug not in breakout_slugs]

            valid_assets = []
            for slug in candidate_slugs:
                if slug not in category_date_data:
                    continue

                asset_data = category_date_data[slug]
                top_col = weighting_method
                mult_col = multiple_type

                if top_col in asset_data and mult_col in asset_data:
                    top_value = asset_data[top_col]
                    multiple_value = asset_data[mult_col]

                    if top_value is None or multiple_value is None:
                        continue
                    if pd.isna(top_value) or pd.isna(multiple_value):
                        continue
                    if top_value <= 0 or multiple_value <= 0:
                        continue

                    valid_assets.append({
                        'slug': slug,
                        'top_value': top_value,
                        'multiple': multiple_value
                    })

            if valid_assets:
                total_top = sum(a['top_value'] for a in valid_assets)
                if total_top > 0:
                    weighted_multiple = sum(
                        (a['top_value'] / total_top) * a['multiple']
                        for a in valid_assets
                    )
                    liquid_rows.append({
                        'date': date_str,
                        'value': weighted_multiple
                    })

        if not liquid_rows:
            print(f"Warning: No liquid comp data found for category {category_slug}")
            continue

        # Use clean category name (no exclusion suffix)
        col_name = category_display_name

        liquid_df = pd.DataFrame(liquid_rows)
        liquid_df['date'] = pd.to_datetime(liquid_df['date'])
        liquid_df = liquid_df.sort_values('date')
        liquid_df = liquid_df.set_index('date')
        liquid_df = liquid_df.rename(columns={'value': col_name})
        liquid_dfs.append(liquid_df)

        # Store protocol list for footnotes (excluding breakout assets)
        liquid_protocols = [
            p.replace('-', ' ').replace('_', ' ').title()
            for p in original_category_protocols if p not in breakout_slugs
        ]
        category_protocols_map[col_name] = liquid_protocols

        # Store exclusion info
        if excluded_assets:
            category_exclusions_map[col_name] = excluded_assets

    if not liquid_dfs:
        print("No liquid comp data found for any category")
        return None

    # --- 4. Merge all series ---
    # Start with first breakout asset
    df_combined = asset_dfs[0]
    for asset_df in asset_dfs[1:]:
        df_combined = df_combined.join(asset_df, how='outer')

    for equity_df in equity_result_dfs:
        df_combined = df_combined.join(equity_df, how='outer')
    for liquid_df in liquid_dfs:
        df_combined = df_combined.join(liquid_df, how='outer')

    df_combined = df_combined.sort_index()
    # Forward fill gaps within each series (but not leading NaNs - let series start when data exists)
    df_combined = df_combined.ffill()
    # Only drop rows where ALL values are NaN
    df_combined = df_combined.dropna(how='all')

    if df_combined.empty:
        print("No overlapping data found for all series")
        return None

    # --- 5. Create line chart ---
    fig, ax = plt.subplots(figsize=(12, 6))

    # Color palette - extended for more categories
    colors = ['#1f1f60', '#313EA5', '#637BBF', '#94B9D9',
              '#333333', '#5F5F5F', '#8A8A8A', '#A0A0A0']

    for i, col in enumerate(df_combined.columns):
        ax.plot(
            df_combined.index,
            df_combined[col],
            label=col,
            color=colors[i % len(colors)],
            linewidth=2
        )

    # Clean up metric name for title
    metric_title = multiple_type
    if trailing_window:
        metric_title = metric_title.replace(f'-{trailing_window}d', '')
    metric_title = metric_title.replace('-', ' ').replace('_', ' ').title()
    metric_title = metric_title.replace('Mcap', 'MCAP').replace('Fdv', 'FDV').replace('Tvl', 'TVL')

    # Build title from breakout asset names
    breakout_names = list(breakout_assets.values())
    if len(breakout_names) > 1:
        title_assets = ', '.join(breakout_names[:-1]) + f" & {breakout_names[-1]}"
    else:
        title_assets = breakout_names[0]

    # Formatting
    ax.margins(x=0)
    space_x_axis_dates(ax=ax, df_name=df_combined.reset_index())
    plt.xticks(rotation=45)

    # Y-axis formatting
    data_max = df_combined.max().max()

    if log_scale:
        ax.set_yscale('log')
        # Use custom formatter to show 10x, 100x, 1000x instead of scientific notation
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x:.0f}x' if x >= 1 else f'{x:.1f}x'))
        ax.grid(axis='y', linestyle='--', alpha=0.4, which='both')
        ax.set_title(f'{title_assets} vs Comps: {metric_title} (Log)')
    else:
        ax.yaxis.set_major_formatter(format_multiples(data_max))
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.set_title(f'{title_assets} vs Comps: {metric_title}')

    # Legend
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=1, framealpha=1)

    # Format dates for display
    formatted_start_date = format_date_with_suffix(start_date)
    last_data_date = get_last_data_date(df_combined.reset_index())

    # Build note text
    weight_type = 'FDV' if multiple_type.startswith('FDV') else 'market cap'
    chart_note = f'Note: Data from {formatted_start_date} to {last_data_date}.\nAll figures are {trailing_window}-day rolling averages and weighted by {weight_type}.'

    # Add equity info to notes
    if equity_categories_mode:
        # Add each equity category's constituents to notes
        for cat_name, constituents in equity_constituents_map.items():
            if len(constituents) > 1:
                constituents_list = ', '.join(constituents[:-1]) + f", and {constituents[-1]}"
            else:
                constituents_list = constituents[0] if constituents else "N/A"
            chart_note += f'\n{cat_name}: {constituents_list}.'
    else:
        # Original behavior: single equity comps list
        equity_list = ', '.join(equity_comps.values())
        chart_note += f'\nEquity Comps: {equity_list}.'

    # Add each category's protocols to notes
    for col_name, protocols in category_protocols_map.items():
        if len(protocols) > 1:
            protocol_list = ', '.join(protocols[:-1]) + f", and {protocols[-1]}"
        else:
            protocol_list = protocols[0] if protocols else "N/A"
        chart_note += f'\n{col_name}: {protocol_list}.'

    # Add exclusion footnote
    if category_exclusions_map:
        exclusion_notes = []
        for cat_name, excluded in category_exclusions_map.items():
            if len(excluded) > 1:
                excluded_str = ', '.join(excluded[:-1]) + f", and {excluded[-1]}"
            else:
                excluded_str = excluded[0]
            exclusion_notes.append(f"{excluded_str} excluded from {cat_name}")
        chart_note += f'\n{"; ".join(exclusion_notes)} to avoid circular comparison.'

    chart_note += '\nSource: DefiLlama, Messari, and Yahoo Finance.'

    # Add logo and notes
    plt.tight_layout()
    format_logo_and_note(ax, chart_note)

    # Save plot
    ensure_directory(plots_filepath)

    safe_metric_name = multiple_type.replace(' ', '-').lower()
    # Use first breakout asset slug for filename
    first_slug = list(breakout_assets.keys())[0]
    plt.savefig(f"{plots_filepath}/{first_slug}-vs-comps-{safe_metric_name}.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()

    # --- 6. Generate statistics table ---
    stats_dict = {}
    for col in df_combined.columns:
        series = df_combined[col].dropna()
        stats_dict[col] = [
            series.median(),
            series.quantile(0.8),
            series.quantile(0.2),
            series.iloc[-1] if len(series) > 0 else None
        ]

    stats_df = pd.DataFrame(stats_dict)
    stats_df.index = [
        'Median',
        '80th Percentile',
        '20th Percentile',
        'Most Recent'
    ]

    # Build notes text for table
    notes_text = f"Note: Data from {formatted_start_date} to {last_data_date}. \nAll figures are {trailing_window}-day rolling averages and weighted by {weight_type}."

    # Add equity info to notes
    if equity_categories_mode:
        # Add each equity category's constituents to notes
        for cat_name, constituents in equity_constituents_map.items():
            if len(constituents) > 1:
                constituents_list = ', '.join(constituents[:-1]) + f", and {constituents[-1]}"
            else:
                constituents_list = constituents[0] if constituents else "N/A"
            notes_text += f"\n{cat_name}: {constituents_list}."
    else:
        # Original behavior: single equity comps list
        equity_list = ', '.join(equity_comps.values())
        notes_text += f"\nEquity Comps: {equity_list}."

    for col_name, protocols in category_protocols_map.items():
        if len(protocols) > 1:
            protocol_list = ', '.join(protocols[:-1]) + f", and {protocols[-1]}"
        else:
            protocol_list = protocols[0] if protocols else "N/A"
        notes_text += f"\n{col_name}: {protocol_list}."

    # Add exclusion footnote
    if category_exclusions_map:
        exclusion_notes = []
        for cat_name, excluded in category_exclusions_map.items():
            if len(excluded) > 1:
                excluded_str = ', '.join(excluded[:-1]) + f", and {excluded[-1]}"
            else:
                excluded_str = excluded[0]
            exclusion_notes.append(f"{excluded_str} excluded from {cat_name}")
        notes_text += f"\n{'; '.join(exclusion_notes)} to avoid circular comparison."

    notes_text += "\nSource: DefiLlama, Messari, and Yahoo Finance."

    # Style the table for display in notebook
    styled = (
        stats_df
          .style
          .set_caption(f"{title_assets} vs Comps: {metric_title}")
          .set_table_styles([
              {"selector": "caption",
               "props": [
                 ("font-weight", "bold"),
                 ("caption-side", "top"),
                 ("text-decoration", "underline")
               ]},
              {"selector": "thead th", "props": [("font-weight", "normal")]},
              {"selector": "tbody th", "props": [("font-weight", "normal")]}
          ], overwrite=False)
          .format(format_multiple_value)
    )

    display(styled)
    print(notes_text)

    # --- Save table with notes using pure matplotlib ---
    num_cols = len(stats_df.columns)
    fig_width = max(12, 2 + num_cols * 3)
    fig_table, ax_table = plt.subplots(figsize=(fig_width, 5))
    ax_table.axis('off')

    # Format values for table
    table_data = [[format_multiple_value(v) for v in row] for row in stats_df.values]
    col_labels = stats_df.columns.tolist()
    row_labels = stats_df.index.tolist()

    # Create table
    table = ax_table.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc='center',
        rowLoc='right',
        loc='center',
        bbox=[0.1, 0.35, 0.88, 0.55]
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Auto-adjust column widths
    table.auto_set_column_width(col=list(range(-1, len(col_labels))))

    # Style header row
    for j in range(len(col_labels)):
        cell = table[(0, j)]
        cell.set_text_props(fontweight='bold')
        cell.set_facecolor('#f8f8f8')
        cell.set_edgecolor('#cccccc')

    # Style row labels
    for i in range(len(row_labels)):
        cell = table[(i + 1, -1)]
        cell.set_text_props(fontweight='bold')
        cell.set_facecolor('#f8f8f8')
        cell.set_edgecolor('#cccccc')

    # Style data cells
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            cell = table[(i + 1, j)]
            cell.set_edgecolor('#cccccc')

    # Add title
    ax_table.text(
        0.5, 0.95,
        f"{title_assets} vs Comps: {metric_title}",
        transform=ax_table.transAxes,
        fontsize=14,
        fontweight='bold',
        ha='center',
        va='top'
    )

    plt.tight_layout()

    # Add logo and notes
    format_logo_and_note(
        ax_table,
        notes_text,
        is_table=True
    )

    # Save table image
    ensure_directory(tables_filepath)

    table_filename = f"{tables_filepath}/{first_slug}-vs-comps-{safe_metric_name}-table.png"
    plt.savefig(
        table_filename,
        dpi=150,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none',
        pad_inches=0.1
    )
    plt.close(fig_table)

    print()
    print(f"Table saved to: {table_filename}")

    # Return data for programmatic use
    result = {
        'breakout_assets': breakout_assets,
        'multiple_type': multiple_type,
        'trailing_window': trailing_window,
        'start_date': start_date,
        'liquid_comp_categories': liquid_comp_categories,
        'category_protocols': category_protocols_map,
        'category_exclusions': category_exclusions_map,
        'data': df_combined,
        'stats': stats_df
    }

    if equity_categories_mode:
        result['equity_categories'] = equity_categories
        result['equity_constituents'] = equity_constituents_map
    else:
        result['equity_comps'] = equity_comps

    return result


@report_errors()
def category_comps_multiples(
    protocol_categories,
    equity_categories,
    multiple_type,
    start_date,
    plots_filepath,
    tables_filepath,
    date,
    log_scale=False,
    protocol_multiples_filepath='data/processed/valuation-multiples/all-protocols-by-category.json',
    weighted_multiples_filepath='data/processed/valuation-multiples/weighted-multiples-by-category.json',
    equity_data_dir='data/raw/yahoofinance/mcap-to-annualized-total-revenue',
    index_path=None
):
    """
    Compare valuation multiples across different protocol and equity categories.

    Unlike asset_vs_comps_multiples which has breakout assets, this function
    compares category-level weighted averages directly against each other.

    Creates a line chart with multiple lines:
    1. Market-cap weighted average for each protocol category
    2. Market-cap weighted average for each equity category

    Also creates a statistics table with median, 80th/20th percentile, and most recent values.

    Parameters:
    -----------
    protocol_categories : dict
        Dictionary mapping category slugs to display names
        (e.g., {'perps_dexs': 'Perps DEXs', 'smart_contract_chains': 'L1s'})
    equity_categories : dict
        Dictionary mapping equity category slugs to display names
        (e.g., {'crypto_equities': 'Crypto Equities', 'gambling_equities': 'Gambling Equities'})
        Categories must be defined in the protocol_index.json
    multiple_type : str
        Multiple type for protocols (e.g., 'FDV-to-annualized-fees-multiple-7d')
        For equities, uses mcap-to-annualized-total-revenue (P/S equivalent)
    start_date : str
        Start date for filtering data (format: 'YYYY-MM-DD')
    plots_filepath : str
        Directory to save the plots
    tables_filepath : str
        Directory to save the tables
    date : str
        Data as-of date for chart annotation
    log_scale : bool, optional
        If True, use log scale for y-axis. Default is False (linear scale)
    protocol_multiples_filepath : str
        Path to the individual protocol multiples JSON file
    weighted_multiples_filepath : str
        Path to the weighted multiples JSON file
    equity_data_dir : str
        Directory containing equity comp JSON files
    index_path : str, optional
        Path to protocol_index.json file

    Returns:
    --------
    dict : Contains data for all series and summary statistics

    Example:
    --------
    >>> category_comps_multiples(
    ...     protocol_categories={'perps_dexs': 'Perps DEXs'},
    ...     equity_categories={
    ...         'crypto_equities': 'Crypto Equities',
    ...         'gambling_equities': 'Gambling Equities',
    ...         'saas_equities': 'SaaS Equities'
    ...     },
    ...     multiple_type='FDV-to-annualized-fees-multiple-7d',
    ...     start_date='2024-01-01',
    ...     plots_filepath='outputs/plots',
    ...     tables_filepath='outputs/tables',
    ...     date='Jan 25th, 2026'
    ... )
    """
    # Import protocol_index for equity lookups
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data import protocol_index as pi

    # Extract trailing window from multiple_type
    window_match = re.search(r'-(\d+)d$', multiple_type)
    trailing_window = int(window_match.group(1)) if window_match else 7

    # Extract weighting method from multiple_type
    if multiple_type.startswith('FDV'):
        weighting_method = 'FDV'
    elif multiple_type.startswith('MCAP'):
        weighting_method = 'MCAP'
    else:
        weighting_method = 'FDV'

    # --- 1. Load protocol categories weighted averages ---
    with open(protocol_multiples_filepath, 'r') as f:
        protocol_multiples_data = json.load(f)

    with open(weighted_multiples_filepath, 'r') as f:
        weighted_multiples_data = json.load(f)

    # Store DataFrames and protocol lists for each category
    protocol_dfs = []
    category_protocols_map = {}  # category_display_name -> list of protocol names

    for category_slug, category_display_name in protocol_categories.items():
        # Get list of protocols in the category
        category_protocols = []
        for date_str, date_data in weighted_multiples_data.items():
            if category_slug in date_data:
                category_data = date_data[category_slug]
                if 'protocols-included' in category_data:
                    category_protocols = category_data['protocols-included']
                    break

        # Build weighted average for all protocols in category
        liquid_rows = []
        for date_str in sorted(protocol_multiples_data.keys()):
            if date_str < start_date:
                continue

            if category_slug not in protocol_multiples_data[date_str]:
                continue

            category_date_data = protocol_multiples_data[date_str][category_slug]

            valid_assets = []
            for slug in category_protocols:
                if slug not in category_date_data:
                    continue

                asset_data = category_date_data[slug]
                top_col = weighting_method
                mult_col = multiple_type

                if top_col in asset_data and mult_col in asset_data:
                    top_value = asset_data[top_col]
                    multiple_value = asset_data[mult_col]

                    if top_value is None or multiple_value is None:
                        continue
                    if pd.isna(top_value) or pd.isna(multiple_value):
                        continue
                    if top_value <= 0 or multiple_value <= 0:
                        continue

                    valid_assets.append({
                        'slug': slug,
                        'top_value': top_value,
                        'multiple': multiple_value
                    })

            if valid_assets:
                total_top = sum(a['top_value'] for a in valid_assets)
                if total_top > 0:
                    weighted_multiple = sum(
                        (a['top_value'] / total_top) * a['multiple']
                        for a in valid_assets
                    )
                    liquid_rows.append({
                        'date': date_str,
                        'value': weighted_multiple
                    })

        if not liquid_rows:
            print(f"Warning: No protocol data found for category {category_slug}")
            continue

        liquid_df = pd.DataFrame(liquid_rows)
        liquid_df['date'] = pd.to_datetime(liquid_df['date'])
        liquid_df = liquid_df.sort_values('date')
        liquid_df = liquid_df.set_index('date')
        liquid_df = liquid_df.rename(columns={'value': category_display_name})
        protocol_dfs.append(liquid_df)

        # Store protocol list for footnotes
        liquid_protocols = [
            p.replace('-', ' ').replace('_', ' ').title()
            for p in category_protocols
        ]
        category_protocols_map[category_display_name] = liquid_protocols

    # --- 2. Load and calculate equity category weighted averages ---
    equity_dfs = []
    equity_constituents_map = {}  # category_display_name -> list of equity names

    for category_slug, category_display_name in equity_categories.items():
        # Get all tickers in this equity category from the index
        tickers_dict = pi.get_equity_display_names_dict(category=category_slug, index_path=index_path)

        if not tickers_dict:
            print(f"Warning: No equities found for category {category_slug}")
            continue

        category_equity_dfs = []
        for ticker, display_name in tickers_dict.items():
            filepath = f"{equity_data_dir}/{ticker}.json"
            try:
                with open(filepath, 'r') as f:
                    equity_data = json.load(f)
            except FileNotFoundError:
                print(f"Warning: Equity data file not found for {ticker}")
                continue

            rows = []
            for date_str, values in equity_data.items():
                if date_str < start_date:
                    continue
                mcap_key = f"{ticker}-market-cap"
                multiple_key = f"{ticker}-7d-avg"

                if mcap_key in values and multiple_key in values:
                    mcap = values[mcap_key]
                    multiple = values[multiple_key]
                    if mcap is not None and multiple is not None and mcap > 0 and multiple > 0:
                        rows.append({
                            'date': date_str,
                            'mcap': mcap,
                            'multiple': multiple,
                            'ticker': ticker
                        })

            if rows:
                category_equity_dfs.append(pd.DataFrame(rows))

        if not category_equity_dfs:
            print(f"Warning: No equity data found for category {category_slug}")
            continue

        # Combine and calculate market-cap weighted average for this category
        equity_combined = pd.concat(category_equity_dfs, ignore_index=True)
        equity_combined['date'] = pd.to_datetime(equity_combined['date'])

        # Group by date and calculate weighted average
        def calc_weighted_avg(group):
            total_mcap = group['mcap'].sum()
            if total_mcap > 0:
                return (group['mcap'] * group['multiple']).sum() / total_mcap
            return None

        equity_weighted = equity_combined.groupby('date').apply(calc_weighted_avg, include_groups=False).reset_index()
        equity_weighted.columns = ['date', category_display_name]
        equity_weighted = equity_weighted.set_index('date')
        equity_dfs.append(equity_weighted)

        # Store constituent list for footnotes
        equity_constituents_map[category_display_name] = list(tickers_dict.values())

    # --- 3. Merge all series ---
    all_dfs = protocol_dfs + equity_dfs

    if not all_dfs:
        print("No data found for any categories")
        return None

    df_combined = all_dfs[0]
    for df in all_dfs[1:]:
        df_combined = df_combined.join(df, how='outer')

    df_combined = df_combined.sort_index()
    # Forward fill gaps within each series
    df_combined = df_combined.ffill()
    # Only drop rows where ALL values are NaN
    df_combined = df_combined.dropna(how='all')

    if df_combined.empty:
        print("No overlapping data found for all series")
        return None

    # --- 4. Create line chart ---
    fig, ax = plt.subplots(figsize=(12, 6))

    # Color palette - extended for more categories
    colors = ['#1f1f60', '#313EA5', '#637BBF', '#94B9D9',
              '#333333', '#5F5F5F', '#8A8A8A', '#A0A0A0',
              '#2E8B57', '#3CB371', '#98FB98', '#9370DB']

    for i, col in enumerate(df_combined.columns):
        ax.plot(
            df_combined.index,
            df_combined[col],
            label=col,
            color=colors[i % len(colors)],
            linewidth=2
        )

    # Clean up metric name for title
    metric_title = multiple_type
    if trailing_window:
        metric_title = metric_title.replace(f'-{trailing_window}d', '')
    metric_title = metric_title.replace('-', ' ').replace('_', ' ').title()
    metric_title = metric_title.replace('Mcap', 'MCAP').replace('Fdv', 'FDV').replace('Tvl', 'TVL')

    # Formatting
    ax.margins(x=0)
    space_x_axis_dates(ax=ax, df_name=df_combined.reset_index())
    plt.xticks(rotation=45)

    # Y-axis formatting
    data_max = df_combined.max().max()

    if log_scale:
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x:.0f}x' if x >= 1 else f'{x:.1f}x'))
        ax.grid(axis='y', linestyle='--', alpha=0.4, which='both')
        ax.set_title(f'Category Comparison: {metric_title} (Log)')
    else:
        ax.yaxis.set_major_formatter(format_multiples(data_max))
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.set_title(f'Category Comparison: {metric_title}')

    # Legend
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=1, framealpha=1)

    # Format dates for display
    formatted_start_date = format_date_with_suffix(start_date)
    last_data_date = get_last_data_date(df_combined.reset_index())

    # Build note text
    weight_type = 'FDV' if multiple_type.startswith('FDV') else 'market cap'
    chart_note = f'Note: Data from {formatted_start_date} to {last_data_date}.\nAll figures are {trailing_window}-day rolling averages weighted by {weight_type}.'

    # Add each protocol category's constituents to notes
    for col_name, protocols in category_protocols_map.items():
        if len(protocols) > 1:
            protocol_list = ', '.join(protocols[:-1]) + f", and {protocols[-1]}"
        else:
            protocol_list = protocols[0] if protocols else "N/A"
        chart_note += f'\n{col_name}: {protocol_list}.'

    # Add each equity category's constituents to notes
    for col_name, equities in equity_constituents_map.items():
        if len(equities) > 1:
            equity_list = ', '.join(equities[:-1]) + f", and {equities[-1]}"
        else:
            equity_list = equities[0] if equities else "N/A"
        chart_note += f'\n{col_name}: {equity_list}.'

    chart_note += '\nSource: DefiLlama, Messari, and Yahoo Finance.'

    # Add logo and notes
    plt.tight_layout()
    format_logo_and_note(ax, chart_note)

    # Save plot
    ensure_directory(plots_filepath)

    safe_metric_name = multiple_type.replace(' ', '-').lower()
    # Generate filename from categories
    all_categories = list(protocol_categories.keys()) + list(equity_categories.keys())
    filename_base = '-vs-'.join(all_categories[:3])  # Limit to first 3 for filename length
    if len(all_categories) > 3:
        filename_base += '-etc'

    plt.savefig(f"{plots_filepath}/category-comps-{filename_base}-{safe_metric_name}.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()

    # --- 5. Generate statistics table ---
    stats_dict = {}
    for col in df_combined.columns:
        series = df_combined[col].dropna()
        stats_dict[col] = [
            series.median(),
            series.quantile(0.8),
            series.quantile(0.2),
            series.iloc[-1] if len(series) > 0 else None
        ]

    stats_df = pd.DataFrame(stats_dict)
    stats_df.index = [
        'Median',
        '80th Percentile',
        '20th Percentile',
        'Most Recent'
    ]

    # Build notes text for table
    notes_text = f"Note: Data from {formatted_start_date} to {last_data_date}.\nAll figures are {trailing_window}-day rolling averages weighted by {weight_type}."

    for col_name, protocols in category_protocols_map.items():
        if len(protocols) > 1:
            protocol_list = ', '.join(protocols[:-1]) + f", and {protocols[-1]}"
        else:
            protocol_list = protocols[0] if protocols else "N/A"
        notes_text += f"\n{col_name}: {protocol_list}."

    for col_name, equities in equity_constituents_map.items():
        if len(equities) > 1:
            equity_list = ', '.join(equities[:-1]) + f", and {equities[-1]}"
        else:
            equity_list = equities[0] if equities else "N/A"
        notes_text += f"\n{col_name}: {equity_list}."

    notes_text += "\nSource: DefiLlama, Messari, and Yahoo Finance."

    # Style the table for display in notebook
    styled = (
        stats_df
          .style
          .set_caption(f"Category Comparison: {metric_title}")
          .set_table_styles([
              {"selector": "caption",
               "props": [
                 ("font-weight", "bold"),
                 ("caption-side", "top"),
                 ("text-decoration", "underline")
               ]},
              {"selector": "thead th", "props": [("font-weight", "normal")]},
              {"selector": "tbody th", "props": [("font-weight", "normal")]}
          ], overwrite=False)
          .format(format_multiple_value)
    )

    display(styled)
    print(notes_text)

    # --- Save table with notes using pure matplotlib ---
    num_cols = len(stats_df.columns)
    fig_width = max(12, 2 + num_cols * 3)
    fig_table, ax_table = plt.subplots(figsize=(fig_width, 5))
    ax_table.axis('off')

    # Format values for table
    table_data = [[format_multiple_value(v) for v in row] for row in stats_df.values]
    col_labels = stats_df.columns.tolist()
    row_labels = stats_df.index.tolist()

    # Create table
    table = ax_table.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc='center',
        rowLoc='right',
        loc='center',
        bbox=[0.1, 0.35, 0.88, 0.55]
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Auto-adjust column widths
    table.auto_set_column_width(col=list(range(-1, len(col_labels))))

    # Style header row
    for j in range(len(col_labels)):
        cell = table[(0, j)]
        cell.set_text_props(fontweight='bold')
        cell.set_facecolor('#f8f8f8')
        cell.set_edgecolor('#cccccc')

    # Style row labels
    for i in range(len(row_labels)):
        cell = table[(i + 1, -1)]
        cell.set_text_props(fontweight='bold')
        cell.set_facecolor('#f8f8f8')
        cell.set_edgecolor('#cccccc')

    # Style data cells
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            cell = table[(i + 1, j)]
            cell.set_edgecolor('#cccccc')

    # Add title
    ax_table.text(
        0.5, 0.95,
        f"Category Comparison: {metric_title}",
        transform=ax_table.transAxes,
        fontsize=14,
        fontweight='bold',
        ha='center',
        va='top'
    )

    plt.tight_layout()

    # Add logo and notes
    format_logo_and_note(
        ax_table,
        notes_text,
        is_table=True
    )

    # Save table image
    ensure_directory(tables_filepath)

    table_filename = f"{tables_filepath}/category-comps-{filename_base}-{safe_metric_name}-table.png"
    plt.savefig(
        table_filename,
        dpi=150,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none',
        pad_inches=0.1
    )
    plt.close(fig_table)

    print()
    print(f"Table saved to: {table_filename}")

    # Return data for programmatic use
    return {
        'protocol_categories': protocol_categories,
        'equity_categories': equity_categories,
        'multiple_type': multiple_type,
        'trailing_window': trailing_window,
        'start_date': start_date,
        'category_protocols': category_protocols_map,
        'equity_constituents': equity_constituents_map,
        'data': df_combined,
        'stats': stats_df
    }


def _pearson_pvalue(r, n):
    """
    Two-tailed p-value for an unweighted Pearson r via Fisher z-transformation.
    Returns nan if inputs aren't usable. No scipy dependency.
    """
    import math
    if n is None or n < 4 or r is None or not np.isfinite(r):
        return float('nan')
    if abs(r) >= 1:
        return 0.0
    z = 0.5 * math.log((1.0 + r) / (1.0 - r))
    se = 1.0 / math.sqrt(n - 3)
    z_score = abs(z) / se
    # Two-tailed normal: 2 * (1 - Phi(z_score))
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z_score / math.sqrt(2.0))))


@report_errors()
def growth_vs_margin_correlation(
    categories=None,
    trailing_window=30,
    min_annualized_fees=10000000,
    growth_outlier_pct=0.01,
    plots_filepath='outputs/plots',
    tables_filepath='outputs/tables',
    date=None,
    chart_title=None,
    log_x=False,
    fees_summary_path='data/raw/defillama/fees/all-protocols.json',
    revenue_summary_path='data/raw/defillama/revenue/all-protocols.json',
):
    """
    Scatter showing the correlation between trailing fee growth and gross margin
    across all DefiLlama-tracked protocols in a given set of categories. Each
    protocol is one bubble:
        X = % change in fees (DefiLlama's pre-computed N-over-N growth)
        Y = gross margin = revenue / fees over the trailing N days
        size = annualized fees over the trailing N days
        color = DefiLlama category

    Data source: the `all-protocols.json` summary files for fees and revenue,
    which carry per-protocol totals and pre-computed growth fields. Pass
    `categories` as DefiLlama category strings, e.g. ['Derivatives','Lending',
    'Liquid Staking','DEX Aggregator','Dexs','Chain'].

    A regression line is fit across all points (weighted by annualized fees)
    and the Pearson correlation coefficient is annotated on the chart.

    Parameters
    ----------
    categories : list[str], optional
        DefiLlama category names. If None, includes every category present in
        the data (filtered by `min_annualized_fees`).
    trailing_window : int
        Window for both growth and gross margin. Must be one of {1, 7, 30}
        (DefiLlama only pre-computes these N-over-N growth fields). Default 30.
    min_annualized_fees : float
        Minimum annualized fees (USD) to include a protocol. Default $10m,
        which keeps the chart legible and filters out long-tail noise.
    growth_outlier_pct : float
        Drop the top and bottom this fraction of protocols by growth (e.g.
        0.01 = drop top and bottom 1%). Set to 0 to disable. Default 0.01.
        Removes pathological growers (new protocols going from ~$0 to large
        fees) and pathological losers (dying protocols) that distort both
        the chart layout and the correlation.
    log_x : bool
        If True, X axis is symlog so very high growers don't compress the plot.
    fees_summary_path, revenue_summary_path : str
        Paths to the all-protocols summary JSONs.
    """
    window_to_fields = {
        1:  {'total': 'total24h', 'growth': 'change_1d'},
        7:  {'total': 'total7d',  'growth': 'change_7dover7d'},
        30: {'total': 'total30d', 'growth': 'change_30dover30d'},
    }
    if trailing_window not in window_to_fields:
        raise ValueError(
            f"trailing_window must be one of {sorted(window_to_fields)}; got {trailing_window}."
        )
    total_field = window_to_fields[trailing_window]['total']
    growth_field = window_to_fields[trailing_window]['growth']

    # --- Load both summary files ---
    with open(fees_summary_path, 'r') as f:
        fees_data = json.load(f)
    with open(revenue_summary_path, 'r') as f:
        revenue_data = json.load(f)

    revenue_by_slug = {p['slug']: p for p in revenue_data.get('protocols', [])}

    # --- Build per-protocol rows ---
    rows = []
    skipped_counts = {'no_revenue': 0, 'below_min_fees': 0, 'missing_growth': 0,
                      'invalid_margin': 0, 'wrong_category': 0}
    for p in fees_data.get('protocols', []):
        cat = p.get('category')
        if categories is not None and cat not in categories:
            skipped_counts['wrong_category'] += 1
            continue

        fees_total = p.get(total_field)
        growth_pct = p.get(growth_field)
        if fees_total is None or growth_pct is None or fees_total <= 0:
            skipped_counts['missing_growth'] += 1
            continue

        annualized_fees = fees_total * (365.0 / trailing_window)
        if annualized_fees < min_annualized_fees:
            skipped_counts['below_min_fees'] += 1
            continue

        rev_entry = revenue_by_slug.get(p['slug'])
        if rev_entry is None:
            skipped_counts['no_revenue'] += 1
            continue
        rev_total = rev_entry.get(total_field)
        if rev_total is None:
            skipped_counts['no_revenue'] += 1
            continue

        gross_margin = rev_total / fees_total
        # Drop obvious data anomalies — revenue can equal or slightly exceed
        # fees for some protocols (all fees go to revenue), but margins outside
        # [0, 1.5] indicate bad data.
        if not np.isfinite(gross_margin) or gross_margin < 0 or gross_margin > 1.5:
            skipped_counts['invalid_margin'] += 1
            continue

        rows.append({
            'protocol': p.get('displayName') or p.get('name') or p['slug'],
            'category': cat or 'Uncategorized',
            'annualized_fees': annualized_fees,
            'growth': growth_pct / 100.0,  # DefiLlama gives %, convert to fraction
            'gross_margin': gross_margin,
            'slug': p['slug'],
        })

    if not rows:
        raise RuntimeError("No protocols passed the filters. Loosen `min_annualized_fees` or check `categories`.")

    df_plot = pd.DataFrame(rows).sort_values('annualized_fees', ascending=False).reset_index(drop=True)
    skipped = [(reason, n) for reason, n in skipped_counts.items() if n > 0]

    # Drop the top and bottom `growth_outlier_pct` of protocols by growth to
    # remove pathological growers/losers that distort both the chart and the
    # correlation. Requires enough samples that quantile filtering is meaningful.
    outliers_dropped = pd.DataFrame()
    if growth_outlier_pct and growth_outlier_pct > 0 and len(df_plot) >= 10:
        lo = df_plot['growth'].quantile(growth_outlier_pct)
        hi = df_plot['growth'].quantile(1 - growth_outlier_pct)
        outlier_mask = (df_plot['growth'] < lo) | (df_plot['growth'] > hi)
        outliers_dropped = df_plot[outlier_mask].copy()
        df_plot = df_plot[~outlier_mask].reset_index(drop=True)

    # Attach a 'date' column so get_last_data_date() works; all-protocols.json
    # is a snapshot, so use today.
    df_plot['date'] = pd.Timestamp.today().normalize()

    # --- Correlation stats ---
    x_vals = df_plot['growth'].values
    y_vals = df_plot['gross_margin'].values
    w_vals = df_plot['annualized_fees'].values

    pearson_r = float(np.corrcoef(x_vals, y_vals)[0, 1]) if len(x_vals) > 1 else np.nan
    pearson_p = _pearson_pvalue(pearson_r, len(x_vals))

    # Weighted regression line + weighted Pearson r (consistent moments)
    if len(x_vals) > 1 and np.sum(w_vals) > 0:
        wm_x = np.average(x_vals, weights=w_vals)
        wm_y = np.average(y_vals, weights=w_vals)
        cov_xy = np.average((x_vals - wm_x) * (y_vals - wm_y), weights=w_vals)
        var_x = np.average((x_vals - wm_x) ** 2, weights=w_vals)
        var_y = np.average((y_vals - wm_y) ** 2, weights=w_vals)
        if var_x > 0:
            slope = cov_xy / var_x
            intercept = wm_y - slope * wm_x
        else:
            slope, intercept = np.nan, np.nan
        weighted_r = float(cov_xy / np.sqrt(var_x * var_y)) if var_x > 0 and var_y > 0 else np.nan
    else:
        slope, intercept = np.nan, np.nan
        weighted_r = np.nan

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12, 7))

    # Assign colors dynamically from a palette of visually-distinct hues, in
    # order of unique categories present. Stable across runs (same category →
    # same color) without hardcoding category names that may change over time.
    distinct_palette = [
        '#1f1f60',  # deep navy
        '#2E7D5B',  # forest green
        '#C77B30',  # burnt orange
        '#7A2A4D',  # mulberry
        '#637BBF',  # mid blue
        '#8C5E25',  # rust
        '#4F7530',  # olive
        '#4A6A82',  # slate
        '#6B2C2C',  # wine
        '#333333',  # charcoal
    ]
    unique_categories = sorted(df_plot['category'].unique())
    category_color_map = {
        cat: distinct_palette[i % len(distinct_palette)]
        for i, cat in enumerate(unique_categories)
    }

    max_fees = df_plot['annualized_fees'].max()
    min_size, max_size = 120, 2200
    sizes = min_size + (df_plot['annualized_fees'] / max_fees) * (max_size - min_size)

    for cat in unique_categories:
        sub = df_plot[df_plot['category'] == cat]
        ax.scatter(
            sub['growth'],
            sub['gross_margin'],
            s=sizes[sub.index],
            color=category_color_map[cat],
            alpha=0.65,
            edgecolor='white',
            linewidth=1.2,
            label=cat,  # DefiLlama categories are already nicely cased
            zorder=3,
        )

    # Regression line (uses weighted moments above so it's consistent with weighted r)
    if np.isfinite(slope):
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color='#B23A48', linewidth=2, linestyle='--',
                label='Weighted fit', zorder=2)

    # Bubble labels — only label the top N by annualized fees to avoid clutter
    # when plotting hundreds of protocols.
    top_n_to_label = min(15, len(df_plot))
    for _, row in df_plot.head(top_n_to_label).iterrows():
        ax.annotate(
            row['protocol'],
            xy=(row['growth'], row['gross_margin']),
            xytext=(6, 6),
            textcoords='offset points',
            fontsize=8,
            color='#222',
            zorder=4,
        )

    ax.axhline(0, color='#888', linewidth=0.6)
    ax.axvline(0, color='#888', linewidth=0.6)

    ax.set_xlabel(f'{trailing_window}d Fees Growth (vs prior {trailing_window}d)')
    ax.set_ylabel(f'Gross Margin (Revenue / Fees, {trailing_window}d)')
    if chart_title is None:
        chart_title = 'Fee Growth vs Gross Margin'
    ax.set_title(chart_title)

    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x*100:,.0f}%'))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x*100:,.0f}%'))
    ax.grid(axis='both', linestyle='--', alpha=0.4)
    if log_x:
        ax.set_xscale('symlog', linthresh=0.5)

    # Correlation annotation in upper-left of plot area
    sig_label = '**' if pearson_p < 0.05 else ('*' if pearson_p < 0.10 else 'n.s.')
    r_text = (
        f"Pearson r = {pearson_r:0.2f}  (p = {pearson_p:.2f}, {sig_label})\n"
        f"Weighted r = {weighted_r:0.2f}\n"
        f"n = {len(df_plot)}"
    )
    ax.text(
        0.02, 0.98, r_text,
        transform=ax.transAxes,
        ha='left', va='top',
        fontsize=10,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#888', alpha=0.9),
        zorder=5,
    )

    # Build legend with uniform marker sizes (proxy artists so legend bubbles
    # don't inherit the data point sizes)
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D(
            [0], [0],
            marker='o',
            linestyle='None',
            color='w',
            markerfacecolor=category_color_map[cat],
            markeredgecolor='white',
            markersize=10,
            label=cat,  # DefiLlama categories already nicely cased
        )
        for cat in unique_categories
    ]
    if np.isfinite(slope):
        legend_handles.append(
            Line2D([0], [0], color='#B23A48', linewidth=2, linestyle='--', label='Weighted fit')
        )
    ax.legend(
        handles=legend_handles,
        loc='upper center',
        bbox_to_anchor=(0.825, -0.12),
        ncol=2,
    )
    plt.tight_layout()

    last_data_date = get_last_data_date(df_plot)
    min_fees_formatter = format_large_numbers(min_annualized_fees)
    min_fees_str = min_fees_formatter(min_annualized_fees, None)
    outlier_note = (
        f" Top/bottom {growth_outlier_pct*100:.1f}% by growth excluded as outliers."
        if growth_outlier_pct and growth_outlier_pct > 0 and not outliers_dropped.empty
        else ""
    )
    note = (
        f"Note: data as of {last_data_date}. Each bubble = one protocol; size scales with annualized fees.\n"
        f"Growth and gross margin both computed over the trailing {trailing_window}d. "
        f"Filtered to protocols with at least {min_fees_str} in annualized fees.{outlier_note}\n"
        f"Regression line is weighted by annualized fees.\nSource: DefiLlama."
    )
    format_logo_and_note(ax, note)

    ensure_directory(plots_filepath)
    safe_title = re.sub(r'[^a-z0-9]+', '-', chart_title.lower()).strip('-')
    plot_filename = f"{plots_filepath}/{safe_title}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()

    if skipped:
        print("\nProtocols dropped during filtering:")
        for reason, n in skipped:
            print(f"  - {reason}: {n}")

    if not outliers_dropped.empty:
        print(f"\nGrowth outliers dropped (top/bottom {growth_outlier_pct*100:.1f}% by growth):")
        for _, row in outliers_dropped.sort_values('growth').iterrows():
            print(f"  - {row['protocol']:<35} {row['category']:<22} growth={row['growth']*100:>+8.0f}%")

    print(f"\nProtocol data (showing top {min(20, len(df_plot))} of {len(df_plot)} by annualized fees):")
    display_df = df_plot[['protocol', 'category', 'annualized_fees', 'growth', 'gross_margin']].head(20).copy()
    fees_formatter = format_large_numbers(df_plot['annualized_fees'].max())
    display_df['annualized_fees'] = display_df['annualized_fees'].apply(lambda v: fees_formatter(v, None))
    display_df['growth'] = display_df['growth'].apply(lambda v: f"{v*100:,.0f}%")
    display_df['gross_margin'] = display_df['gross_margin'].apply(lambda v: f"{v*100:,.0f}%")
    print(display_df.to_string(index=False))

    print(f"\nPearson r: {pearson_r:0.3f}    p-value: {pearson_p:0.3f}    n: {len(df_plot)}")
    print(f"Weighted Pearson r: {weighted_r:0.3f}    (no closed-form p-value; rely on unweighted p for significance)")
    print(f"Saved plot: {plot_filename}")

    return {
        'data': df_plot,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'weighted_pearson_r': weighted_r,
        'slope': slope,
        'intercept': intercept,
        'skipped': skipped,
    }


# Column formats for ranking_table: name -> callable(value) -> str
_RANKING_FORMATS = {
    'usd': lambda v: '-' if v is None or pd.isna(v) else ('-' if v < 0 else '') + format_large_numbers(abs(v)).func(abs(v), None),
    'usd0': lambda v: '-' if v is None or pd.isna(v) else ('-' if v < 0 else '') + f"${abs(v):,.0f}",
    'pct1': lambda v: '-' if v is None or pd.isna(v) else f"{v * 100:,.1f}%",
    'pct3': lambda v: '-' if v is None or pd.isna(v) else f"{v * 100:,.3f}%",
    'mult': lambda v: '-' if v is None or pd.isna(v) else f"{v:,.1f}x",
    'int': lambda v: '-' if v is None or pd.isna(v) else f"{int(v):,}",
    None: lambda v: '-' if v is None or (isinstance(v, float) and pd.isna(v)) else str(v),
}


def _table_png(show, fmt, title, note, tables_filepath, filename, bold_rows=()):
    """Render a formatted DataFrame as the house-style table PNG (shared by the ranking and MetaDAO tables)."""
    table_data = [[fmt[h](v) for h, v in zip(show.columns, r)] for r in show.values]
    # format_logo_and_note(is_table=True) draws the logo at y=0.30 and the note at y=0.20 (axes
    # fraction), so the table must end at y=0.36; the figure grows with the row count instead.
    fig, ax = plt.subplots(figsize=(max(12, 2 + len(show.columns) * 2), max(5, 2.5 + 0.6 * len(show))))
    ax.axis('off')
    tbl = ax.table(cellText=table_data, rowLabels=[str(i) for i in show.index], colLabels=show.columns.tolist(),
                   cellLoc='center', rowLoc='right', loc='center', bbox=[0.1, 0.36, 0.88, 0.58])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.auto_set_column_width(col=list(range(-1, len(show.columns))))
    for j in range(len(show.columns)):
        tbl[(0, j)].set_text_props(fontweight='bold'); tbl[(0, j)].set_facecolor('#f8f8f8'); tbl[(0, j)].set_edgecolor('#cccccc')
    for i, label in enumerate(show.index):
        tbl[(i + 1, -1)].set_text_props(fontweight='bold'); tbl[(i + 1, -1)].set_facecolor('#f8f8f8'); tbl[(i + 1, -1)].set_edgecolor('#cccccc')
        for j in range(len(show.columns)):
            tbl[(i + 1, j)].set_edgecolor('#cccccc')
            if label in bold_rows:
                tbl[(i + 1, j)].set_text_props(fontweight='bold'); tbl[(i + 1, j)].set_facecolor('#eef0f8')
    ax.text(0.5, 0.97, title, transform=ax.transAxes, fontsize=14, fontweight='bold', ha='center', va='top')
    plt.tight_layout()
    format_logo_and_note(ax, note, is_table=True)
    os.makedirs(tables_filepath, exist_ok=True)
    path = f"{tables_filepath}/{filename}.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.1)
    plt.close(fig)
    print(f"Table saved to: {path}")


@report_errors()
def ranking_table(df, sort_col, n, title, columns, tables_filepath, date, filename,
                  ascending=False, note=None, require=None, flag_col=None, flag_note=None,
                  exclude_flagged=False):
    """Display and save a top-N table ranked on ``sort_col``.

    ``df`` is any DataFrame (e.g. data_processing.load_universe_earnings()); ``columns`` maps
    source column -> (header, format) with format one of 'usd', 'pct1', 'mult', 'int' or
    None (as-is). ``require`` names columns that must be non-null for a row to qualify (rows
    with a null ``sort_col`` never qualify). ``flag_col`` names a boolean column: flagged
    rows keep their place but their first display column is marked with ``*`` and
    ``flag_note`` is appended to the note; ``exclude_flagged=True`` drops them instead. The
    table is shown as a styled DataFrame (caption + note, the notebook's usual look) and
    saved as ``{tables_filepath}/{filename}.png`` with the logo and note. Returns the
    ranked, unformatted DataFrame (rank 1..N as index).
    """
    keep = df.dropna(subset=[sort_col] + list(require or []))
    if flag_col and exclude_flagged:
        keep = keep[~keep[flag_col].fillna(False).astype(bool)]
    ranked = keep.sort_values(sort_col, ascending=ascending).head(n).copy()
    ranked.index = range(1, len(ranked) + 1)
    ranked.index.name = '#'
    show = ranked[list(columns)].rename(columns={c: h for c, (h, _) in columns.items()})
    fmt = {h: _RANKING_FORMATS[f] for c, (h, f) in columns.items()}
    note = note or f"Note: data as of {date}. Source: DefiLlama."
    if flag_col and not exclude_flagged:
        flagged = ranked[flag_col].fillna(False).astype(bool)
        if flagged.any():
            first = show.columns[0]
            show[first] = [f"{v} *" if f else v for v, f in zip(show[first], flagged)]
            note = f"{note}\n* {flag_note or 'flagged'}"

    styled = (show.style.set_caption(title)
              .set_table_styles([
                  {"selector": "caption", "props": [("font-weight", "bold"), ("caption-side", "top"),
                                                    ("text-decoration", "underline")]},
                  {"selector": "thead th", "props": [("font-weight", "normal")]},
                  {"selector": "tbody th", "props": [("font-weight", "normal")]}], overwrite=False)
              .format(fmt))
    display(styled)
    print(f"{len(keep)} protocols qualified; showing top {len(ranked)}.")
    print(note)

    _table_png(show, fmt, title, note, tables_filepath, filename)
    return ranked


# MetaDAO launchpad: first n days of trading vs ICO price, hourly VWAP
@report_errors()
def metadao_launch_performance(processed_path, plots_filepath, date, chart_title, n_days=None,
                               log_scale=False, color_by_launch_date=True, min_fundraise_usd=0):
    """Plot every launch in a metadao.build_launch_prices() file against its ICO price.

    One line per launch from its hourly VWAP points (USDC volume / token volume per hour
    on the MetaDAO spot AMM), starting at the ICO price at day 0. The x-axis is days
    from the first trade (hour k plots at (k + 1) / 24); y is the % change from the ICO
    price, or the multiple of it on a log axis when ``log_scale=True``. Hours with no
    trades are skipped so lines stay connected. ``n_days`` defaults to the file's window
    and can only shorten it. Lines are shaded light (earliest launch) to dark (newest)
    with the launch month in the legend when ``color_by_launch_date`` is set. Launches that
    raised less than ``min_fundraise_usd`` (USDC accepted) are left off; the cutoff is stated
    in the note and appended to the filename so different cutoffs do not overwrite each other.
    """
    cutoff_label = format_large_numbers(min_fundraise_usd).func(min_fundraise_usd, None) if min_fundraise_usd > 0 else None
    with open(processed_path) as f:
        data = json.load(f)
    file_days = data['_metadata']['n_days']
    n_days = min(n_days or file_days, file_days)

    launches = []
    below_cutoff = []
    for mint, launch in data['launches'].items():
        if (launch.get('usdc_accepted') or 0) < min_fundraise_usd:
            below_cutoff.append(launch.get('ticker') or mint)
            continue
        points = [(h['hour'], h['multiple']) for h in launch['hours'][:n_days * 24] if h['multiple'] is not None]
        if not points:
            print(f"skipping {launch.get('ticker') or mint}: no trades in the first {n_days} days")
            continue
        launches.append((launch, points))
    if below_cutoff:
        print(f"below the {cutoff_label} fundraise cutoff, not plotted: {', '.join(below_cutoff)}")
    if not launches:
        print("no launches to plot")
        return
    launches.sort(key=lambda item: item[0]['trading_start'])

    color_list = ['#1f1f60', '#313EA5', '#637BBF', '#333333', '#5F5F5F', '#8A8A8A', '#A0A0A0', '#B6B6B6', '#CCCCCC']
    colors = [color_list[i % len(color_list)] for i in range(len(launches))]
    if color_by_launch_date:
        house_blues = mcolors.LinearSegmentedColormap.from_list('house_blues', ['#CCCCCC', '#637BBF', '#313EA5', '#1f1f60'])
        colors = [mcolors.to_hex(house_blues(i / max(len(launches) - 1, 1))) for i in range(len(launches))]

    fig, ax = plt.subplots(figsize=(12, 6))
    for (launch, points), color in zip(launches, colors):
        xs = [0] + [(hour + 1) / 24 for hour, _ in points]
        mults = [1.0] + [mult for _, mult in points]
        ys = mults if log_scale else [(m - 1) * 100 for m in mults]
        label = launch['ticker'] or launch['name']
        if color_by_launch_date:
            start = datetime.strptime(launch['trading_start'][:10], '%Y-%m-%d')
            label = f"{label} ({start.strftime('%b')} '{start.strftime('%y')})"
        ax.plot(xs, ys, color=color, linewidth=1.5, label=label)

    ax.set_xlim(0, n_days)
    ax.set_xticks(range(n_days + 1))
    ax.set_xlabel("Days from First Trade (hourly VWAP)")
    if log_scale:
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: format_multiple_value(x)))
        ax.grid(axis="y", linestyle="--", alpha=0.5, which='both')
        ax.set_ylabel("Multiple of ICO Price")
    else:
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.set_ylabel("Change from ICO Price")
    ax.legend(loc='upper right', ncol=2, framealpha=0.9)
    ax.margins(x=0)

    title = f"{chart_title} (First {n_days} Days, Indexed to ICO Price)"
    if log_scale:
        title = f"{title} (Log)"
    ax.set_title(title, fontsize=14)
    plt.tight_layout()

    shading_line = " Lines shaded light to dark by launch date." if color_by_launch_date else ""
    cutoff_line = f" Launches that raised at least {cutoff_label}." if cutoff_label else ""
    format_logo_and_note(
        ax,
        f"Note: data as of {date}. Indexed to ICO price (USDC accepted / 10M tokens) at day 0; hourly "
        f"volume-weighted average prices on the MetaDAO spot AMM from the first trade.{cutoff_line}{shading_line}\nSource: MetaDAO"
    )

    ensure_directory(plots_filepath)
    scale_suffix = '-log' if log_scale else ''
    cutoff_suffix = f"-min{min_fundraise_usd / 1e6:g}m" if cutoff_label else ''
    plt.savefig(f"{plots_filepath}/metadao-launch-first-{n_days}d-hourly{cutoff_suffix}{scale_suffix}.png", dpi=300, bbox_inches='tight')
    plt.show()


_HOUSE_BLUE = '#313EA5'
_HOUSE_GREY = '#A0A0A0'


@report_errors()
def metadao_horizon_table(returns_df, summary_df, tables_filepath, date, horizons=(1, 7, 30, 90),
                          headline=30, filename='metadao-ico-returns', title=None, note=None):
    """The headline table: per launch, fill rate and multiple of ICO price at each horizon and today, the return
    per $1 committed at the headline horizon and today, then the cohort summary rows in bold.

    ``returns_df`` / ``summary_df`` come from metadao.horizon_returns() / metadao.cohort_summary().
    """
    cols = {'sale_closed': ('Sale close', None), 'usdc_committed': ('Committed', 'usd'),
            'usdc_accepted': ('Accepted', 'usd'), 'fill_rate': ('Fill', 'pct1')}
    for h in horizons:
        cols[f'mult_{h}d'] = (f'{h}d', 'mult')
    cols['mult_latest'] = ('Latest', 'mult')
    cols['latest_date'] = ('Latest date', None)
    cols[f'ret_{headline}d'] = (f'Return / $1 committed, {headline}d', 'pct1')
    cols['ret_latest'] = ('Return / $1 committed, latest', 'pct1')

    launches = returns_df[list(cols)].copy()
    summary = pd.DataFrame(index=summary_df.index, columns=list(cols), dtype=object)
    for h in horizons:
        summary[f'mult_{h}d'] = summary_df[f'{h}d']
    summary['mult_latest'] = summary_df['latest']
    # the return row is a return, the other rows are multiples / counts / rates
    ret_row = 'return per $1 committed to every raise'
    summary.loc[ret_row, f'mult_{headline}d'] = None
    summary.loc[ret_row, 'mult_latest'] = None
    summary.loc[ret_row, f'ret_{headline}d'] = summary_df.loc[ret_row, f'{headline}d']
    summary.loc[ret_row, 'ret_latest'] = summary_df.loc[ret_row, 'latest']
    for h in horizons:
        summary.loc[ret_row, f'mult_{h}d'] = None
    show = pd.concat([launches, summary])
    show.index.name = ''

    def cell_fmt(row_label, col, v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return '-'
        if row_label == 'launches with data':
            return f'{int(v)}'
        if row_label == 'hit rate (above ICO price)':
            return f'{v * 100:,.0f}%'
        return _RANKING_FORMATS[cols[col][1]](v)

    rendered = show.copy()
    for col in show.columns:
        rendered[col] = [cell_fmt(idx, col, v) for idx, v in zip(show.index, show[col])]
    rendered.columns = [cols[c][0] for c in show.columns]
    styled = (rendered.style.set_caption(title)
              .set_table_styles([
                  {"selector": "caption", "props": [("font-weight", "bold"), ("caption-side", "top"), ("text-decoration", "underline")]},
                  {"selector": "thead th", "props": [("font-weight", "normal")]},
                  {"selector": "tbody th", "props": [("font-weight", "normal")]}], overwrite=False))
    display(styled)
    print(note)
    fmt = {h: (lambda v: v) for h in rendered.columns}
    _table_png(rendered, fmt, title, note, tables_filepath, filename, bold_rows=set(summary_df.index))


@report_errors()
def metadao_cohort_curve(daily_df, plots_filepath, date, chart_title, max_days=90, min_launches=5,
                         log_scale=True, filename='metadao-ico-cohort-curve', note_extra=''):
    """Median multiple of the ICO price across launches by day since first trade, with the interquartile band and
    each launch as a thin grey line. Days with fewer than ``min_launches`` launches trading are not summarised."""
    df = daily_df.loc[:max_days]
    counts = df.notna().sum(axis=1)
    ok = counts >= min_launches
    med = df.median(axis=1).where(ok)
    q1 = df.quantile(0.25, axis=1).where(ok)
    q3 = df.quantile(0.75, axis=1).where(ok)

    fig, ax = plt.subplots(figsize=(12, 6))
    for ticker in df.columns:
        ax.plot(df.index, df[ticker], color='#CCCCCC', linewidth=0.8, alpha=0.9, zorder=1)
    ax.fill_between(df.index, q1, q3, color=_HOUSE_BLUE, alpha=0.15, linewidth=0, zorder=2, label='Interquartile range')
    ax.plot(df.index, med, color=_HOUSE_BLUE, linewidth=2.5, zorder=3, label=f'Median of {int(counts.max())} launches')
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.6, zorder=0)
    last_day = int(ok[ok].index.max()) if ok.any() else max_days
    if last_day < max_days:
        ax.axvline(x=last_day, color='gray', linestyle=':', alpha=0.6)
        ax.text(last_day + 1, ax.get_ylim()[1] if not log_scale else med.max(), f'fewer than {min_launches}\nlaunches beyond day {last_day}', fontsize=8, color='gray', va='top')
    ax.set_xlim(0, max_days)
    ax.set_xlabel('Days from First Trade')
    if log_scale:
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: format_multiple_value(x)))
        ax.grid(axis='y', linestyle='--', alpha=0.5, which='both')
    else:
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: format_multiple_value(x)))
        ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_ylabel('Multiple of ICO Price (daily VWAP)')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.margins(x=0)
    ax.set_title(f'{chart_title} (Cohort Median, First {max_days} Days)' + (' (Log)' if log_scale else ''), fontsize=14)
    plt.tight_layout()
    format_logo_and_note(ax, f"Note: data as of {date}. Each grey line is one launch's daily VWAP on the MetaDAO spot AMM divided by its ICO price "
                             f"(USDC accepted / 10M tokens); the blue line is the median across launches trading that day, the band the 25th-75th "
                             f"percentile. Every completed raise is included, listed or not.{note_extra}\nSource: MetaDAO")
    ensure_directory(plots_filepath)
    plt.savefig(f"{plots_filepath}/{filename}{'-log' if log_scale else ''}.png", dpi=300, bbox_inches='tight')
    plt.show()


@report_errors()
def metadao_fill_adjusted_returns(returns_df, plots_filepath, date, chart_title, horizon=30,
                                  filename='metadao-ico-return-per-dollar'):
    """Return per $1 committed to each raise at ``horizon`` days (fill rate x (multiple - 1)), one bar per launch in
    launch order, with the equal-weight strategy average as a line and each launch's fill rate under its name."""
    col, mcol = f'ret_{horizon}d', f'mult_{horizon}d'
    df = returns_df[returns_df[col].notna()]
    too_young = [t for t in returns_df.index if t not in df.index]
    if df.empty:
        print('no launch is old enough for that horizon')
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(df))
    colors = [_HOUSE_BLUE if v >= 0 else _HOUSE_GREY for v in df[col]]
    ax.bar(x, df[col] * 100, color=colors, width=0.7, zorder=2)
    avg = df[col].mean() * 100
    ax.axhline(y=avg, color='#1f1f60', linestyle='--', linewidth=1.5, zorder=3,
               label=f'Equal amount committed to every raise: {avg:+.1f}%')
    ax.axhline(y=0, color='gray', linewidth=1, zorder=1)
    for i, (t, r) in enumerate(df.iterrows()):
        y = r[col] * 100
        ax.text(i, y + (1.0 if y >= 0 else -1.0), f"{r[mcol]:.1f}x", ha='center', va='bottom' if y >= 0 else 'top', fontsize=8, color='#333333')
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{t}\nfill {r['fill_rate'] * 100:.0f}%" for t, r in df.iterrows()], fontsize=8)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylabel(f'Return per $1 committed, day {horizon}')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_title(f'{chart_title} (Return per $1 Committed at Day {horizon})', fontsize=14)
    plt.tight_layout()
    young = f" Not yet {horizon} days old: {', '.join(too_young)}." if too_young else ''
    format_logo_and_note(ax, f"Note: data as of {date}. Return = fill rate x (day-{horizon} VWAP / ICO price - 1); fill rate = USDC accepted / "
                             f"USDC committed, the pro-rata share of a commitment that bought tokens (the rest was refunded). Labels above bars "
                             f"are the multiple of the ICO price.{young}\nSource: MetaDAO")
    ensure_directory(plots_filepath)
    plt.savefig(f"{plots_filepath}/{filename}-{horizon}d.png", dpi=300, bbox_inches='tight')
    plt.show()


# ---------------------------------------------------------------------------
# Hyperliquid AQAv2 reserve yield (data/hyperliquid.py)
# ---------------------------------------------------------------------------

_AQAV2_SOURCE = 'Source: Hyperliquid API (aqav2InterestHistory, userNonFundingLedgerUpdates); HyperEVM archive RPC; FRED (EFFR)'


def _aqav2_styled(rendered, title, note):
    styled = (rendered.style.set_caption(title)
              .set_table_styles([
                  {"selector": "caption", "props": [("font-weight", "bold"), ("caption-side", "top"), ("text-decoration", "underline")]},
                  {"selector": "thead th", "props": [("font-weight", "normal")]},
                  {"selector": "tbody th", "props": [("font-weight", "normal")]}], overwrite=False))
    display(styled)
    print(note)


def _iso(ts):
    return '-' if ts is None or pd.isna(ts) else pd.Timestamp(ts).strftime('%Y-%m-%d')


def _mpl_safe(text):
    """Escape dollar signs so matplotlib does not read a pair of them as math markup."""
    return text.replace('$', r'\$')


@report_errors()
def hyperliquid_aqav2_period_table(periods, tables_filepath, date, title=None, note=None,
                                   filename='hyperliquid-aqav2-periods', bold_periods=(1,)):
    """One row per 30-day AQAv2 period from hyperliquid.aqav2_periods(): dates, days sampled, the
    accrued interest (sum of the official daily interest), average charged balance and rate, and
    the actual payment once the ledger shows it. Shown styled and saved as a PNG."""
    title = title or 'Hyperliquid AQAv2 reserve yield: accrued interest by 30-day period'
    note = note or (f"Note: data as of {date}. Interest = treasury balance x AQA rate / 365.25 per UTC date, summed over each "
                    f"30-day interval from activation (2026-08-27); paid to the Assistance Fund 8 days after the interval's last "
                    f"date. The charged balance is the treasury deployer's native-USDC balance on HyperEVM at 00:00 UTC, 90% of the "
                    f"USDC backing HyperCore.\n{_AQAV2_SOURCE}")
    f = _RANKING_FORMATS
    rendered = pd.DataFrame({
        'Start': [_iso(v) for v in periods['start']],
        'End': [_iso(v) for v in periods['end']],
        'Paid on': [_iso(v) for v in periods['payment_date']],
        'Days': [f"{int(v)} / 30" for v in periods['days_sampled']],
        'Status': list(periods['status']),
        'Accrued (USDC)': [f['usd0'](v) for v in periods['interest_accrued']],
        'Avg balance': [f['usd'](v) for v in periods['avg_supply']],
        'Avg rate': [f['pct3'](v) for v in periods['avg_rate']],
        'Paid (USDC)': [f['usd0'](v) for v in periods['actual_paid']],
        'Paid - accrued': [f['usd0'](v) for v in periods['difference']],
    }, index=[f'Period {k}' for k in periods.index])
    rendered.index.name = ''
    _aqav2_styled(rendered, title, note)
    fmt = {h: (lambda v: v) for h in rendered.columns}
    _table_png(rendered, fmt, title, _mpl_safe(note), tables_filepath, filename, bold_rows={f'Period {k}' for k in bold_periods})


@report_errors()
def hyperliquid_aqav2_daily_chart(daily, periods, plots_filepath, date, chart_title='Hyperliquid AQAv2 Reserve Yield (USDC)',
                                  filename='hyperliquid-aqav2-daily'):
    """The charged balance (line) and each day's booked interest (bars) from hyperliquid.aqav2_daily(),
    with the 30-day periods shaded and every payment date marked."""
    df = daily.copy()
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    ax2.set_zorder(1); ax1.set_zorder(2); ax1.patch.set_visible(False)

    last_pay = periods['payment_date'].max()
    x_end = max(df['date'].max(), last_pay) + pd.Timedelta(days=1)
    pre = df[~df['active']]
    if not pre.empty:
        ax1.axvspan(pre['date'].min(), pre['date'].max() + pd.Timedelta(days=1), color='#EEEEEE', zorder=0)
        ax1.text(pre['date'].min() + pd.Timedelta(hours=12), 0.98, 'pre-activation', transform=ax1.get_xaxis_transform(),
                 fontsize=8, color='gray', va='top')
    for k, r in periods.iterrows():
        if k % 2 == 1:
            ax1.axvspan(r['start'], r['end'] + pd.Timedelta(days=1), color=_HOUSE_BLUE, alpha=0.06, zorder=0)
        ax1.text(r['start'] + pd.Timedelta(hours=12), 0.98, f"P{k}", transform=ax1.get_xaxis_transform(), fontsize=9,
                 color=_HOUSE_BLUE, va='top', fontweight='bold')
        label = 'paid' if r['status'] == 'paid' else 'due'
        ax1.axvline(r['payment_date'], color='gray', linestyle=':', linewidth=1, zorder=1)
        ax1.text(r['payment_date'] + pd.Timedelta(hours=6), 0.02, f"P{k} {label}\n{_iso(r['payment_date'])}",
                 transform=ax1.get_xaxis_transform(), fontsize=7, color='gray', va='bottom')

    ax2.bar(df['date'], df['interest'], width=0.8, color=_HOUSE_GREY, alpha=0.8, label='Daily interest booked (USDC)', zorder=1)
    ax2.set_ylabel('Daily interest (USDC)')
    ax2.yaxis.set_major_formatter(format_large_numbers(df['interest'].max()))
    ax2.set_ylim(bottom=0)
    ax1.plot(df['date'], df['usdc_supply'], color='#1f1f60', linewidth=2, label='Charged balance (treasury USDC on HyperEVM)', zorder=3)
    ax1.set_ylabel('Charged balance')
    ax1.yaxis.set_major_formatter(format_large_numbers(df['usdc_supply'].max()))
    ax1.set_ylim(bottom=0, top=df['usdc_supply'].max() * 1.15)
    ax1.set_xlim(df['date'].min() - pd.Timedelta(days=1), x_end)
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=MO, interval=1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper right', framealpha=0.9)
    ax1.set_title(f'{chart_title}: Daily Balance and Interest', fontsize=14)
    plt.tight_layout()
    latest = df[df['active']].iloc[-1]
    format_logo_and_note(ax1, _mpl_safe(f"Note: data as of {date}. Each bar is one UTC day's interest = balance at 00:00 UTC x AQA rate / 365.25; "
                              f"shaded bands are the 30-day periods, dotted lines the payment dates (8 days after each period). "
                              f"Latest: {format_large_numbers(latest['usdc_supply']).func(latest['usdc_supply'], None)} at "
                              f"{latest['effective_rate'] * 100:.3f}% = ${latest['interest']:,.0f}/day.\n{_AQAV2_SOURCE}"))
    ensure_directory(plots_filepath)
    plt.savefig(f"{plots_filepath}/{filename}.png", dpi=300, bbox_inches='tight')
    plt.show()


@report_errors()
def hyperliquid_aqav2_rate_chart(daily, ref, calibration, plots_filepath, date, chart_title='Hyperliquid AQAv2 Reserve Yield (USDC)',
                                 filename='hyperliquid-aqav2-rate'):
    """The published AQA rate against the reference (EFFR) and the fitted model (trailing mean of the
    reference minus the fixed cost), so the pass-through lag after a Fed move is visible."""
    from data import hyperliquid
    df = daily[['date', 'effective_rate']].copy()
    start, end = df['date'].min(), df['date'].max()
    ref_w = ref.loc[start - pd.Timedelta(days=1):end]
    model = hyperliquid.project_rate(ref, calibration['cost_bp'], calibration['passthrough_days'])
    model = model.loc[start:end]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.step(ref_w.index, ref_w.values, where='post', color=_HOUSE_GREY, linewidth=1.5, label='Effective Fed funds rate (EFFR)')
    ax.plot(df['date'], df['effective_rate'], color='#1f1f60', linewidth=2.5, label='AQA effective rate (published)')
    ax.plot(model.index, model.values, color=_HOUSE_BLUE, linewidth=1.5, linestyle='--',
            label=f"Model: EFFR trailing {calibration['passthrough_days']}d mean - {calibration['cost_bp']}bp")
    ax.axhline(calibration['steady_state_rate'], color=_HOUSE_BLUE, linestyle=':', linewidth=1,
               label=f"Steady state at current EFFR: {calibration['steady_state_rate'] * 100:.3f}%")
    lo = min(df['effective_rate'].min(), model.min()) - 0.001
    hi = max(ref_w.max(), calibration['steady_state_rate']) + 0.001
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x * 100:.2f}%'))
    ax.set_ylabel('Annual rate')
    ax.set_xlim(start - pd.Timedelta(days=1), end + pd.Timedelta(days=1))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=MO, interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
    ax.set_title(f'{chart_title}: AQA Rate vs Fed Funds', fontsize=14)
    plt.tight_layout()
    imp = f"{calibration['implied_cost_bp']:.1f}bp" if calibration.get('implied_cost_bp') is not None else 'n/a'
    format_logo_and_note(ax, _mpl_safe(f"Note: data as of {date}. The AQA rate is the cost-adjusted reserve yield validators publish daily; it "
                             f"tracks a trailing mean of Fed funds as reserves roll into new bills. Fixed cost {calibration['cost_bp']}bp "
                             f"assumed (observed before the Sep 17 move: {imp}); lag fitted at {calibration['passthrough_days']} days, "
                             f"RMSE {calibration['rmse_bp']:.1f}bp.\n{_AQAV2_SOURCE}"))
    ensure_directory(plots_filepath)
    plt.savefig(f"{plots_filepath}/{filename}.png", dpi=300, bbox_inches='tight')
    plt.show()


def _fed_path_label(path):
    if not path:
        return 'no Fed moves'
    parts = [f"{bp:+d}bp {pd.Timestamp(d).strftime('%b %d %Y')}" for d, bp in sorted(path.items())]
    return ', '.join(parts)


@report_errors()
def hyperliquid_aqav2_estimate_table(est, tables_filepath, date, title=None, note=None, filename='hyperliquid-aqav2-estimate'):
    """Estimated interest per period and scenario from hyperliquid.estimate_aqav2_payments(): one row per
    period (dates, payment date, actual vs projected days), the total and the assumed average rate
    under each scenario, and a total row."""
    title = title or 'Hyperliquid AQAv2 reserve yield: estimated payments to the Assistance Fund'
    a = est.attrs.get('assumptions', {})
    scen = list(dict.fromkeys(est['scenario']))
    f = _RANKING_FORMATS
    first = est[est['scenario'] == scen[0]].set_index('period')
    rendered = pd.DataFrame(index=[f"P{k}: {_iso(r['start'])} to {_iso(r['end'])}, paid {_iso(r['payment_date'])}" for k, r in first.iterrows()])
    rendered['Days actual / projected'] = [f"{int(r['days_actual'])} / {int(r['days_projected'])}" for _, r in first.iterrows()]
    totals = {}
    for s in scen:
        block = est[est['scenario'] == s].set_index('period')
        rendered[f'{s} (USDC)'] = [f['usd0'](v) for v in block['interest_total']]
        rendered[f'{s} avg rate'] = [f['pct3'](v) for v in block['avg_rate_assumed']]
        totals[s] = float(block['interest_total'].sum())
    total_row = {'Days actual / projected': ''}
    for s in scen:
        total_row[f'{s} (USDC)'] = f['usd0'](totals[s]); total_row[f'{s} avg rate'] = '-'
    label = f"Total, {len(first)} periods"
    rendered.loc[label] = total_row
    rendered.index.name = ''
    if note is None:
        lines = [f"Note: data as of {date}. Days already published use the node's actual interest; remaining days are balance x rate / 365.25 "
                 f"with the balance held at {format_large_numbers(a.get('latest_base', 0)).func(a.get('latest_base', 0), None)} "
                 f"({a.get('latest_date')}) and the rate = EFFR trailing {a.get('passthrough_days')}d mean - {a.get('cost_bp')}bp, "
                 f"EFFR starting at {a.get('latest_ref', 0) * 100:.2f}%."]
        for s in scen:
            sp = a.get('scenarios', {}).get(s, {})
            extra = f"; balance CAGR {sp.get('treasury_cagr', 0) * 100:.0f}%" if sp.get('treasury_cagr') else ''
            fixed = f"; flat rate {sp['fixed_rate'] * 100:.2f}%" if sp.get('fixed_rate') is not None else ''
            lines.append(f"{s}: {_fed_path_label(sp.get('fed_path'))}{extra}{fixed}.")
        lines.append("Each Fed move applies the day after the decision and passes through over the lag. 365.25-day count as the node uses "
                     f"(hl.eco's estimator uses 365).\n{_AQAV2_SOURCE}")
        note = ' '.join(lines[:1]) + '\n' + ' '.join(lines[1:-1]) + '\n' + lines[-1]
    _aqav2_styled(rendered, title, note)
    fmt = {h: (lambda v: v) for h in rendered.columns}
    _table_png(rendered, fmt, title, _mpl_safe(note), tables_filepath, filename, bold_rows={label})


@report_errors()
def hyperliquid_aqav2_payments_chart(periods, est, plots_filepath, date, chart_title='Hyperliquid AQAv2 Reserve Yield (USDC)',
                                     filename='hyperliquid-aqav2-payments'):
    """Payments by period: completed periods as solid bars of the accrued (or paid) amount, the
    in-progress period as actual (solid) plus projected (hatched), and future periods as one hatched
    bar per scenario."""
    scen = list(dict.fromkeys(est['scenario']))
    n = len(scen)
    house_blues = mcolors.LinearSegmentedColormap.from_list('house_blues', ['#637BBF', '#313EA5', '#1f1f60'])
    colors = [mcolors.to_hex(house_blues(i / max(n - 1, 1))) for i in range(n)]
    est_periods = sorted(set(est['period']))
    done = periods[~periods.index.isin(est_periods)]
    all_k = sorted(set(done.index) | set(est_periods))
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(12, 6))
    for k, r in done.iterrows():
        val = r['actual_paid'] if r['status'] == 'paid' else r['interest_accrued']
        ax.bar(k, val, width=0.8, color=_HOUSE_BLUE, zorder=2, label='Completed period (official accrual)' if k == done.index.min() else None)
        ax.text(k, val, _mpl_safe(f"${val / 1e6:.2f}m"), ha='center', va='bottom', fontsize=8, color='#333333')
    for i, s in enumerate(scen):
        block = est[est['scenario'] == s].set_index('period')
        xs = [k + (i - (n - 1) / 2) * width for k in block.index]
        ax.bar(xs, block['interest_actual'], width=width, color=colors[i], zorder=2)
        ax.bar(xs, block['interest_projected'], bottom=block['interest_actual'], width=width, color=colors[i], alpha=0.45,
               hatch='//', edgecolor='white', linewidth=0.5, zorder=2, label=f'{s} (projected)')
    ax.set_xticks(all_k)
    labels = []
    for k in all_k:
        pay = periods.loc[k, 'payment_date'] if k in periods.index else est[est['period'] == k]['payment_date'].iloc[0]
        labels.append(f"P{k}\n{pd.Timestamp(pay).strftime('%b %d')}")
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel('Period (payment date)')
    ax.yaxis.set_major_formatter(format_large_numbers(max(est['interest_total'].max(), done['interest_accrued'].max() if len(done) else 0)))
    ax.set_ylabel('Interest per 30-day period (USDC)')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
    ax.set_title(f'{chart_title}: Payments by Period, Actual and Estimated', fontsize=14)
    plt.tight_layout()
    a = est.attrs.get('assumptions', {})
    format_logo_and_note(ax, _mpl_safe(f"Note: data as of {date}. Solid = interest already booked by the node; hatched = projected at the balance held flat "
                             f"and the rate model (EFFR trailing {a.get('passthrough_days')}d mean - {a.get('cost_bp')}bp) under each scenario's Fed path. "
                             f"Bars sit at their payment date, 8 days after the period ends.\n{_AQAV2_SOURCE}"))
    ensure_directory(plots_filepath)
    plt.savefig(f"{plots_filepath}/{filename}.png", dpi=300, bbox_inches='tight')
    plt.show()
