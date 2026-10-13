import json
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.helper_functions import fetch_and_save_json, pull_progress, report_pull, report_errors, is_fresh, run_parallel

# Default freshness window (days) and request concurrency for api.messari.io
DEFAULT_MAX_STALE_DAYS = 3
DEFAULT_MAX_WORKERS = 4


def last_point_timestamp(data):
    """Unix timestamp of the newest point in a Messari time-series response."""
    return data['data']['points'][-1][0]


def _pull_series(entities, metrics, url_for, headers, messari_output_dir, desc, max_stale_days, max_workers):
    """Pull one file per (entity, metric) in parallel, skipping files already fresh on disk.

    ``entities`` is a {slug: name} dict or an iterable of slugs (name == slug).
    Returns the list of (metric, name) failures.
    """
    items = entities.items() if isinstance(entities, dict) else [(e, e) for e in entities]
    tasks = [dict(slug=slug, name=name, metric=metric, url=url_for(slug, metric),
                  output_path=f"{messari_output_dir}/{metric}/{name}.json")
             for slug, name in items for metric in metrics]
    to_pull = [t for t in tasks if not is_fresh(t['output_path'], last_point_timestamp, max_stale_days)]
    if len(to_pull) < len(tasks):
        print(f"{desc}: {len(tasks) - len(to_pull)} fresh file(s) skipped, {len(to_pull)} to pull")
    results = run_parallel(to_pull, lambda t: fetch_and_save_json(t['url'], t['output_path'], headers=headers),
                           max_workers, desc)
    return [(t['metric'], t['name']) for t, data in zip(to_pull, results) if data is None]


# Function to pull all assets and save down
@report_errors()
def pull_all_assets(messari_api_key, messari_output_dir):
    url = 'https://api.messari.io/metrics/v2/assets'
    headers = {
        "accept": "application/json",
        "x-messari-api-key": messari_api_key
    }
    output_path = f"{messari_output_dir}/all-assets.json"

    failures = []
    if fetch_and_save_json(url, output_path, headers=headers) is None:
        failures.append((None, "all-assets"))

    report_pull(messari_output_dir, [], failures)


# Function to pull data for a list of assets
@report_errors()
def pull_asset_data(assets, metrics, start_date, end_date, messari_api_key, messari_output_dir,
                    max_stale_days=DEFAULT_MAX_STALE_DAYS, max_workers=DEFAULT_MAX_WORKERS):
    headers = {
        "accept": "application/json",
        "x-messari-api-key": messari_api_key
    }

    def url_for(asset_slug, metric):
        return (
            f"https://api.messari.io/metrics/v2/assets/{asset_slug}/metrics/{metric}"
            f"/time-series/1d?start={start_date}T00%3A00%3A00Z&end={end_date}T00%3A00%3A00Z"
        )

    failures = _pull_series(assets, metrics, url_for, headers, messari_output_dir,
                            "Messari asset data", max_stale_days, max_workers)
    report_pull(messari_output_dir, metrics, failures)


# Function to save down all networks tracked by Messari's API, returning a
# {slug: name} dict of the top N for use by pull_network_data
@report_errors(default={})
def fetch_and_save_top_n_networks(metric_category, metric_extract, excluded_slugs, n, messari_api_key, messari_output_dir):
    url = "https://api.messari.io/metrics/v2/networks"
    headers = {
        "accept": "application/json",
        "x-messari-api-key": messari_api_key
    }

    output_path = f"{messari_output_dir}/network-data/all-networks.json"
    data = fetch_and_save_json(url, output_path, headers=headers)
    if data is None:
        print("Warning: could not pull Messari networks list; returning no networks")
        return {}

    # Extract top n list of networks by metric
    filtered = []
    datalist = data['data']['data']

    for entry in datalist:  # Iterate directly over the entries
        try:
            if entry["slug"] in excluded_slugs:
                continue
            metric_value = entry["metrics"][metric_category][metric_extract]
            if metric_value is not None:  # Only include entries with valid metric values
                filtered.append((entry["slug"], entry["name"], metric_value))
        except (KeyError, TypeError):
            continue

    sorted_filtered = sorted(filtered, key=lambda x: x[2], reverse=True)
    top_dict = {slug: name for slug, name, _ in sorted_filtered[:n]}
    return top_dict


# Function to pull data for a list of networks
@report_errors()
def pull_network_data(networks, metrics, start_date, end_date, messari_api_key, messari_output_dir,
                      max_stale_days=DEFAULT_MAX_STALE_DAYS, max_workers=DEFAULT_MAX_WORKERS):
    headers = {
        "accept": "application/json",
        "x-messari-api-key": messari_api_key
    }

    def url_for(network, metric):
        return (
            f"https://api.messari.io/metrics/v2/networks/{network}/metrics/{metric}"
            f"/time-series/1d?start={start_date}T00%3A00%3A00Z&end={end_date}T00%3A00%3A00Z"
        )

    failures = _pull_series(networks, metrics, url_for, headers, messari_output_dir,
                            "Messari network data", max_stale_days, max_workers)
    report_pull(messari_output_dir, metrics, failures)


# Function to pull data for a list of stablecoins
@report_errors()
def pull_stablecoin_data(stablecoins, metrics, start_date, end_date, messari_api_key, messari_output_dir):
    headers = {
        "accept": "application/json",
        "x-messari-api-key": messari_api_key
    }

    failures = []
    for stablecoin in pull_progress(stablecoins, "Messari stablecoin data"):
        for metric in metrics:
            url = (
                f"https://api.messari.io/metrics/v2/stablecoins/{stablecoin}/metrics/{metric}"
                f"/time-series/?start={start_date}T00%3A00%3A00Z&end={end_date}T00%3A00%3A00Z"
            )
            output_path = f"{messari_output_dir}/{metric}/{stablecoin}.json"

            if fetch_and_save_json(url, output_path, headers=headers) is None:
                failures.append((metric, stablecoin))

    report_pull(messari_output_dir, metrics, failures)


# Function to pull data for a list of exchanges
@report_errors()
def pull_exchange_data(exchanges, metrics, start_date, end_date, messari_api_key, messari_output_dir):
    headers = {
        "accept": "application/json",
        "x-messari-api-key": messari_api_key
    }

    failures = []
    for exchange in pull_progress(exchanges, "Messari exchange data"):
        for metric in metrics:
            url = (
                f"https://api.messari.io/metrics/v1/exchanges/{exchange}/metrics/{metric}"
                f"/time-series/1d?start={start_date}T00%3A00%3A00Z&end={end_date}T00%3A00%3A00Z"
            )
            output_path = f"{messari_output_dir}/{metric}/{exchange}.json"

            if fetch_and_save_json(url, output_path, headers=headers) is None:
                failures.append((metric, exchange))

    report_pull(messari_output_dir, metrics, failures)
