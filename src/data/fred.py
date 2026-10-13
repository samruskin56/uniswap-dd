# Load libraries
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.helper_functions import fetch_and_save_json, report_pull, report_errors


# Function to pull m2 money supply
@report_errors()
def pull_m2_money_supply(series_id, fred_api_key, fred_output_dir):
    """
    Pull M2 money supply data from the FRED API.

    Args:
        series_id: FRED series ID for M2 data. Options:
            - 'M2SL': M2 Money Stock, Seasonally Adjusted, Monthly (recommended)
            - 'M2NS': M2 Money Stock, Not Seasonally Adjusted, Monthly
            - 'WM2NS': M2 Money Stock, Not Seasonally Adjusted, Weekly
            - 'M2REAL': Real M2 Money Stock (inflation-adjusted), Monthly
        fred_api_key: Your FRED API key.
        fred_output_dir: Directory to save the output JSON file.

    Returns:
        None. Saves data to a JSON file.
    """
    url = (
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&api_key={fred_api_key}&file_type=json"
    )
    output_path = f"{fred_output_dir}/m2-money-supply-{series_id}.json"

    failures = []
    if fetch_and_save_json(url, output_path) is None:
        failures.append((None, f"m2-money-supply-{series_id}"))

    report_pull(fred_output_dir, [], failures)


@report_errors()
def pull_series(series_id, fred_api_key, fred_output_dir, observation_start=None, max_stale_days=3):
    """Save any FRED series' observations as returned, to ``{fred_output_dir}/{series_id}.json``.

    Used for daily rate series such as ``EFFR`` (effective Fed funds), the reference the
    Hyperliquid AQAv2 estimate is linked to. The file is kept when its last observation is
    within ``max_stale_days`` of today (daily series print with a one-business-day lag, so 3
    covers weekends); ``max_stale_days=None`` forces a repull. Returns the saved dict or None.
    """
    from utils.helper_functions import is_fresh
    import json
    from datetime import datetime, timezone
    output_path = f"{fred_output_dir}/{series_id}.json"

    def last_obs_ts(data):
        obs = [o for o in data.get('observations', []) if o.get('value') not in (None, '.', '')]
        return int(datetime.fromisoformat(obs[-1]['date']).replace(tzinfo=timezone.utc).timestamp()) if obs else None

    if is_fresh(output_path, last_obs_ts, max_stale_days):
        with open(output_path) as f:
            data = json.load(f)
        print(f"FRED {series_id}: fresh through {data['observations'][-1]['date']}, kept")
        return data
    url = (f"https://api.stlouisfed.org/fred/series/observations"
           f"?series_id={series_id}&api_key={fred_api_key}&file_type=json")
    if observation_start:
        url += f"&observation_start={observation_start}"
    data = fetch_and_save_json(url, output_path)
    if data is None:
        report_pull(fred_output_dir, [], [(None, series_id)])
        return None
    obs = [o for o in data.get('observations', []) if o.get('value') not in (None, '.', '')]
    print(f"FRED {series_id}: pulled {len(obs)} observations, {obs[0]['date']}..{obs[-1]['date']}" if obs else f"FRED {series_id}: no observations")
    return data
