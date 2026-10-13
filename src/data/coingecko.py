import os
import sys
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.helper_functions import fetch_and_save_json, pull_progress, report_pull, report_errors

BASE_URL = 'https://api.coingecko.com/api/v3'

# Demo (free) API tier allows 30 calls/min, so space sequential calls out
SECONDS_BETWEEN_CALLS = 2.1


def _build_headers(coingecko_api_key):
    return {
        "accept": "application/json",
        "x-cg-demo-api-key": coingecko_api_key
    }


@report_errors(default={})
def get_category_coins(category, coingecko_api_key, coingecko_output_dir):
    """Return {coingecko_id: name} for every coin in a CoinGecko category.

    Discovers the list dynamically (GET /coins/markets?category=...), so newly
    launched tokens are picked up on each run. Saves the raw response to
    {output_dir}/categories/{category}.json for use by analysis notebooks.
    """
    url = f'{BASE_URL}/coins/markets'
    params = {'vs_currency': 'usd', 'category': category, 'per_page': 250, 'page': 1}
    output_path = f"{coingecko_output_dir}/categories/{category}.json"

    data = fetch_and_save_json(url, output_path, headers=_build_headers(coingecko_api_key), params=params)
    if not data:
        print(f"Warning: no coins found for CoinGecko category '{category}'")
        return {}
    return {coin['id']: coin['name'] for coin in data}


# Function for pulling price history for one or more coins
@report_errors()
def pull_asset_data(assets, coingecko_api_key, coingecko_output_dir, days=365, vs_currency='usd'):
    """Pull daily price history for one or more coins (GET /coins/{id}/market_chart).

    Saves one raw market_chart response per coin ({output_dir}/price/{id}.json).
    Demo tier allows at most 365 days of history; omitting `interval` returns
    daily granularity automatically for days > 90.
    """
    headers = _build_headers(coingecko_api_key)

    failures = []
    for coingecko_id in pull_progress(assets, "CoinGecko price data"):
        url = f'{BASE_URL}/coins/{coingecko_id}/market_chart'
        params = {'vs_currency': vs_currency, 'days': days}
        output_path = f"{coingecko_output_dir}/price/{coingecko_id}.json"

        if fetch_and_save_json(url, output_path, headers=headers, params=params) is None:
            failures.append(('price', coingecko_id))
        time.sleep(SECONDS_BETWEEN_CALLS)

    report_pull(coingecko_output_dir, ['price'], failures)
