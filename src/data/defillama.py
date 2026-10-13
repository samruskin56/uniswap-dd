# Load libraries
import json
import time
import os
import sys
import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.helper_functions import (fetch_and_save_json, ensure_directory, pull_progress, report_pull,
                                    report_errors, is_fresh, run_parallel, llama_url, has_pro_key,
                                    get_session, save_json, error_code, redact_url)


FLOW_METRICS = ('dailyFees', 'dailyRevenue', 'dailyHoldersRevenue')
# Pro-only per-protocol series. dailyIncentives is the USD value of tokens emitted
# each day in the incentive categories (farming / staking rewards), read from the
# emission endpoint's unlockUsdChart; its monthly sums tie exactly to the Incentives
# line of DefiLlama's financial statement (verified on 26 protocols). Earnings =
# revenue - incentives is DERIVED from these files by data_processing.build_earnings_series.
PRO_METRICS = ('dailyIncentives',)

# Default freshness window (days) and request concurrency. Every URL is built by
# helper_functions.llama_url: with DEFILLAMA_API_KEY set the pulls run on
# pro-api.llama.fi (1,000 req/min), so the pool can use the whole shared session
# (16 connections); the free api.llama.fi host throttles well below that.
DEFAULT_MAX_STALE_DAYS = 3
DEFAULT_MAX_WORKERS = 16 if has_pro_key() else 6


def last_flow_timestamp(data):
    """Unix timestamp of the newest point in a /summary/fees or /overview/fees response."""
    return data['totalDataChart'][-1][0]


def last_tvl_timestamp(data):
    """Unix timestamp of the newest point in a /protocol/{slug} response."""
    return max(p['date'] for p in data['tvl'])


def last_incentives_timestamp(data):
    """Unix timestamp of the newest point in a saved incentives file. A protocol with no
    incentives tracked has an empty chart, so the pull's ``asOf`` day stands in for it
    (otherwise the file would look missing and be repulled on every run)."""
    chart = data.get('totalDataChart') or []
    return chart[-1][0] if chart else data.get('asOf')


def metric_url(metric_slug, asset_slug=None):
    """API URL for a metric, per-protocol when ``asset_slug`` is given, else all-protocols. None if unrecognized."""
    if metric_slug in FLOW_METRICS:
        if asset_slug is None:
            return llama_url(f"/overview/fees?excludeTotalDataChart=false&excludeTotalDataChartBreakdown=false&dataType={metric_slug}")
        return llama_url(f"/summary/fees/{asset_slug}?dataType={metric_slug}")
    if metric_slug == 'tvl':
        return llama_url("/protocols") if asset_slug is None else llama_url(f"/protocol/{asset_slug}")
    if metric_slug in PRO_METRICS:
        # per-protocol only (no all-protocols endpoint); None without a Pro key
        return None if asset_slug is None else llama_url(f"/emission/{asset_slug}", pro_only=True)
    return None


def _skip_fresh(tasks, extract_for, max_stale_days, label):
    """Split tasks into those whose output file is already fresh and those to pull, printing a one-line summary."""
    to_pull = [t for t in tasks if not is_fresh(t['output_path'], extract_for(t['metric_slug']), max_stale_days)]
    skipped = len(tasks) - len(to_pull)
    if skipped:
        print(f"{label}: {skipped} fresh file(s) skipped, {len(to_pull)} to pull")
    return to_pull


# Pull down fees/revenue data for all protocols
@report_errors()
def pull_all_protocols_metric(metrics: dict, defillama_output_dir: str,
                              max_stale_days: int = DEFAULT_MAX_STALE_DAYS, max_workers: int = 4):
    """Pull fees/revenue/holders-revenue/tvl data for all protocols.

    The requests run in parallel. Flow-metric files whose newest point is
    within ``max_stale_days`` are not repulled (``None`` forces a full pull);
    the tvl ``/protocols`` list carries no series, so it is always repulled.
    """

    failures = []
    tasks = []
    for metric_slug, metric_name in metrics.items():
        if metric_slug in PRO_METRICS:
            print(f"DefiLlama all-protocols: no all-protocols endpoint for {metric_slug}; skipped (pull it per asset)")
            continue
        url = metric_url(metric_slug)
        if url is None:
            failures.append((metric_name, "unrecognized metric"))
            continue
        tasks.append(dict(metric_slug=metric_slug, metric_name=metric_name, url=url,
                          output_path=f"{defillama_output_dir}/{metric_name}/all-protocols.json"))

    # tvl all-protocols is a snapshot list, never fresh by timestamp; always pull it
    extract_for = lambda m: last_flow_timestamp if m in FLOW_METRICS else (lambda d: None)
    tasks = _skip_fresh(tasks, extract_for, max_stale_days, "DefiLlama all-protocols")

    results = run_parallel(tasks, lambda t: fetch_and_save_json(t['url'], t['output_path']),
                           max_workers, "DefiLlama all-protocol metrics")
    for task, data in zip(tasks, results):
        if data is None:
            failures.append((task['metric_name'], "all-protocols"))

    report_pull(defillama_output_dir, metrics.values(), failures)


def drop_partial_current_day(data: dict) -> bool:
    """Remove the in-progress current UTC day from a DefiLlama /summary/fees response.

    Daily fee/revenue points are stamped at 00:00 UTC and the point for the
    current day only fills in as adapters run, so a fresh pull carries a
    partial (often near-zero) value for today. Trailing averages, annualized
    figures and multiples built on that point are wrong until the day closes,
    so it is dropped at pull time. Trims ``totalDataChart`` and
    ``totalDataChartBreakdown`` in place; ``tvl`` responses are untouched
    because TVL is a point-in-time stock, not a daily flow.

    Returns:
        True if any points were removed.
    """
    today_utc = datetime.datetime.now(datetime.timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    cutoff = int(today_utc.timestamp())

    dropped = False
    for key in ('totalDataChart', 'totalDataChartBreakdown'):
        points = data.get(key)
        if not isinstance(points, list):
            continue
        kept = [p for p in points if not (isinstance(p, list) and p and isinstance(p[0], (int, float)) and p[0] >= cutoff)]
        if len(kept) != len(points):
            data[key] = kept
            dropped = True
    return dropped


def _trim_partial_day(data):
    """fetch_and_save_json transform: drop the partial current day in place and return the data."""
    drop_partial_current_day(data)
    return data


def _today_utc_ts():
    return int(datetime.datetime.now(datetime.timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp())


def incentives_chart(body, emission_slug):
    """Project a Pro /emission/{slug} body onto the standard per-protocol file shape.

    ``totalDataChart`` is the body's ``unlockUsdChart`` (daily USD of incentive-category
    emissions) with the partial current UTC day dropped, values untouched. The body's
    ``categories`` (which labels count as incentives) and ``supplyMetrics`` ride along
    for provenance. An empty chart means DefiLlama tracks no incentives for the
    protocol; ``status`` says so and ``asOf`` lets the freshness check accept the file.
    """
    out = {
        'totalDataChart': [[ts, v] for ts, v in (body.get('unlockUsdChart') or []) if isinstance(v, (int, float))],
        'incentiveCategories': body.get('categories') or {},
        'supplyMetrics': body.get('supplyMetrics'),
        'emissionSlug': emission_slug,
        'source': 'pro-api.llama.fi /api/emission/{slug} -> body.unlockUsdChart',
        'asOf': _today_utc_ts(),
    }
    drop_partial_current_day(out)
    out['status'] = 'ok' if out['totalDataChart'] else 'no-incentives-tracked'
    return out


def _fetch_emission_body(slug):
    """Parsed emission body for ``slug`` (Pro), or None when the endpoint has no entry."""
    url = llama_url(f"/emission/{slug}", pro_only=True)
    if url is None:
        return None
    r = get_session().get(url, timeout=60)
    if r.status_code != 200:
        return None
    body = r.json().get('body')
    if isinstance(body, str):
        body = json.loads(body)
    return body or None


def pull_incentives(asset_slug, asset_name, defillama_output_dir):
    """Pull one protocol's daily incentives file (see incentives_chart). Returns the saved
    dict, or None on a request failure.

    Uses the protocol's own slug only. DefiLlama attributes incentives to the protocol
    that owns the token schedule: a parent like ``curve-finance`` has an emission entry
    and an Incentives line in its financial statement, while a child like ``curve-dex``
    has neither, so its earnings equal its revenue on DefiLlama. Falling back to the
    parent's chart would charge the whole group's incentives to one child and break the
    tie-out, so it is deliberately not done; pull the parent slug for group-level
    earnings. A slug with no emission entry still gets a file, with an empty chart and
    ``status='no-emissions-data'``, so earnings = revenue downstream rather than a
    permanent pull failure.
    """
    output_path = f"{defillama_output_dir}/incentives/{asset_name}.json"
    try:
        body = _fetch_emission_body(asset_slug)
        if body is not None:
            data = incentives_chart(body, asset_slug)
        else:
            data = {'totalDataChart': [], 'incentiveCategories': {}, 'supplyMetrics': None,
                    'emissionSlug': None, 'source': 'pro-api.llama.fi /api/emission/{slug} -> body.unlockUsdChart',
                    'asOf': _today_utc_ts(), 'status': 'no-emissions-data'}
        save_json(data, output_path)
        return data
    except Exception as e:
        print(f"  request failed: {error_code(e)}{type(e).__name__}: {redact_url(llama_url(f'/emission/{asset_slug}', pro_only=True))}")
        return None


# Function to pull both fees and revenue data for a group of assets
@report_errors(default=[])
def pull_metric_data_by_asset(metrics: dict, assets: dict, defillama_output_dir: str, report: bool = True,
                              max_stale_days: int = DEFAULT_MAX_STALE_DAYS, max_workers: int = DEFAULT_MAX_WORKERS):
    """Pull fees/revenue/holders-revenue/tvl data for a specific asset.

    Every (asset, metric) pair is one request; the requests run on a thread
    pool of ``max_workers`` (Pro host when DEFILLAMA_API_KEY is set, else the
    free host) with retry/backoff on 429.
    A pair whose saved file already has a data point within ``max_stale_days``
    of today is skipped (pass ``None`` to force a full repull).

    Daily flow metrics (fees/revenue/holders-revenue) have the partial current
    UTC day removed before saving; see drop_partial_current_day(). The Pro-only
    ``dailyIncentives`` metric needs DEFILLAMA_API_KEY and is saved via
    pull_incentives() (own slug only; empty chart when DefiLlama tracks none).

    Args:
        report: When True (default), print the standardized pull summary. Set
            False to suppress it and report via the returned failures instead.
        max_stale_days: Freshness window for skipping files already on disk.
        max_workers: Number of requests in flight at once.

    Returns:
        List of (metric_name, asset_name) tuples for pulls that failed.
    """

    failures = []
    tasks = []
    for asset_slug, asset_name in assets.items():
        for metric_slug, metric_name in metrics.items():
            if metric_slug in PRO_METRICS and not has_pro_key():
                failures.append((metric_name, f"{asset_name} (needs DEFILLAMA_API_KEY)"))
                continue
            url = metric_url(metric_slug, asset_slug)
            if url is None:
                failures.append((metric_name, f"{asset_name} (unrecognized metric)"))
                continue
            tasks.append(dict(asset_slug=asset_slug, asset_name=asset_name, metric_slug=metric_slug,
                              metric_name=metric_name, url=url,
                              output_path=f"{defillama_output_dir}/{metric_name}/{asset_name}.json"))

    extract_for = lambda m: (last_tvl_timestamp if m == 'tvl'
                             else last_incentives_timestamp if m in PRO_METRICS
                             else last_flow_timestamp)
    tasks = _skip_fresh(tasks, extract_for, max_stale_days, "DefiLlama metric data")

    def pull(task):
        if task['metric_slug'] in PRO_METRICS:
            return pull_incentives(task['asset_slug'], task['asset_name'], defillama_output_dir)
        transform = None if task['metric_slug'] == 'tvl' else _trim_partial_day
        return fetch_and_save_json(task['url'], task['output_path'], transform=transform)

    results = run_parallel(tasks, pull, max_workers, "DefiLlama metric data")
    for task, data in zip(tasks, results):
        if data is None:
            failures.append((task['metric_name'], task['asset_name']))

    if report:
        report_pull(defillama_output_dir, metrics.values(), failures)
    return failures

# Pull down data for all stablecoins
@report_errors()
def pull_all_stablecoins(defillama_output_dir):
    """Pull data for all stablecoins."""
    url = llama_url("/stablecoins?includePrices=true", section='stablecoins')
    output_path = f"{defillama_output_dir}/stablecoins/all-stablecoins.json"

    failures = [] if fetch_and_save_json(url, output_path) is not None else [(None, "all-stablecoins")]
    report_pull(defillama_output_dir, ["stablecoins"], failures)


# Function to pull information for specific stablecoins
@report_errors()
def pull_specific_stablecoins(assets, defillama_output_dir):
    """Pull data for specific stablecoins by symbol."""
    with open(f"{defillama_output_dir}/stablecoins/all-stablecoins.json", 'r') as f:
        data = json.load(f)
    
    # Build symbol to ID mapping
    symbol_to_id = {
        asset.get('symbol'): asset.get('id')
        for asset in data.get('peggedAssets', [])
        if asset.get('symbol') in assets
    }
    
    # Fetch data for each asset
    failures = []
    for symbol, asset_id in pull_progress(symbol_to_id.items(), "DefiLlama stablecoins"):
        url = llama_url(f"/stablecoin/{asset_id}", section='stablecoins')
        output_path = f"{defillama_output_dir}/stablecoins/{symbol}.json"

        if fetch_and_save_json(url, output_path) is None:
            failures.append((None, symbol))

    # Surface any requested stablecoins not found in all-stablecoins.json
    for symbol in assets:
        if symbol not in symbol_to_id:
            failures.append((None, f"{symbol} (not found in all-stablecoins)"))

    report_pull(defillama_output_dir, ["stablecoins"], failures)


# Function to pull stablecoin supply on a given network
@report_errors()
def stablecoin_supply_by_network(networks, defillama_output_dir):
    failures = []
    for network in pull_progress(networks, "DefiLlama stablecoin supply by network"):
        url = llama_url(f'/stablecoincharts/{network}', section='stablecoins')
        output_path = f'{defillama_output_dir}/stablecoins/{network}.json'

        if fetch_and_save_json(url, output_path) is None:
            failures.append((None, network))

    report_pull(defillama_output_dir, ["stablecoins"], failures)


# Function to find and return a dict of top N stablecoins as {slug: name}
def top_n_stablecoins_by_mcap(n, pegMechanism, excluded_assets, defillama_output_dir):
    with open(f'{defillama_output_dir}/stablecoins/all-stablecoins.json', 'r') as f:
        data = json.load(f)

    assets = data['peggedAssets']
    
    filtered = []
    for entry in assets:
        try:
            if entry["name"] in excluded_assets:
                continue
            if entry['pegMechanism'] != pegMechanism:
                continue
            mcap = entry['circulating']['peggedUSD']
            filtered.append((entry["symbol"], entry["name"], mcap))
        except KeyError:
            continue

    sorted_filtered = sorted(filtered, key=lambda x: x[2], reverse=True)
    top_dict = {symbol: name for symbol, name, _ in sorted_filtered[:n]}
    top_names = list(top_dict.values())
    
    # Save to disk
    with open(f'{defillama_output_dir}/stablecoins/top-{pegMechanism}-stablecoins-by-mcap.json', 'w') as f:
        json.dump(top_names, f, indent=2)
    
    print(f"Data saved to {defillama_output_dir}/stablecoins/top-{pegMechanism}-stablecoins-by-mcap.json")
    print('Done')
    
    return top_names, top_dict


# Function to find and return a dict of top N HyperEVM protocols as {slug: name}
@report_errors(default={})
def fetch_and_save_top_hyperevm_protocols(input_dir, n, defillama_output_dir, excluded_slugs):
    """
    Find top N HyperEVM protocols, save the list, and fetch historical TVL data for each.
    
    Args:
        input_dir: Path to the input JSON file with protocol data
        n: Number of top protocols to fetch
        defillama_output_dir: Directory to save output files
        excluded_slugs: List of protocol slugs to exclude
    """
    # Load and filter data
    with open(input_dir, 'r') as f:
        data = json.load(f)

    filtered = []
    for entry in data:
        try:
            if entry["slug"] in excluded_slugs:
                continue
            tvl = entry['chainTvls']['Hyperliquid L1']
            filtered.append((entry["slug"], entry["name"], tvl))
        except KeyError:
            continue

    # Get top N protocols
    sorted_filtered = sorted(filtered, key=lambda x: x[2], reverse=True)
    top_protocols = {slug: name for slug, name, _ in sorted_filtered[:n]}
    
    print(f"Top {n} HyperEVM protocols found:")
    print(top_protocols)
    print(f"Length: {len(top_protocols)}")

    # Save the top protocols list
    top_protocols_path = f'{defillama_output_dir}/tvl/top_hyperevm_protocols.json'
    ensure_directory(os.path.dirname(top_protocols_path))
    with open(top_protocols_path, 'w') as f:
        json.dump(top_protocols, f, indent=2)
    print(f"\nProtocol list saved to {top_protocols_path}")

    # Fetch historical TVL data for each protocol
    failures = []
    for slug, name in pull_progress(top_protocols.items(), "Fetching historical TVL data"):
        url = llama_url(f"/protocol/{slug}")
        output_path = f"{defillama_output_dir}/tvl/{slug}.json"

        if fetch_and_save_json(url, output_path) is None:
            failures.append((None, name))

        if not has_pro_key():
            time.sleep(1)  # free-host rate limiting

    report_pull(defillama_output_dir, ["tvl"], failures)
    return top_protocols

# Pull down OI data for all dexs, replace null values with ffill
@report_errors()
def pull_all_open_interest(defillama_output_dir):
    """Pull data for all dexs."""
    url = llama_url("/overview/open-interest?excludeTotalDataChart=false&excludeTotalDataChartBreakdown=false")
    output_path = f"{defillama_output_dir}/open-interest/all-dexs.json"

    failures = [] if fetch_and_save_json(url, output_path) is not None else [(None, "all-dexs")]
    report_pull(defillama_output_dir, ["open-interest"], failures)


# Pull the universe-wide token-incentives snapshot (Pro)
@report_errors()
def pull_emissions_breakdown(defillama_output_dir):
    """Save the two universe-level emissions snapshots (Pro) under ``incentives/``.

    ``all-protocols.json`` = /emissionsBreakdownAggregated: per protocol the USD value of
    incentive-category emissions over trailing windows (``emission24h``, ``emission7d``,
    ``emission30d``, ``emissions1y``, ...), the universe companion of the per-protocol
    ``dailyIncentives`` pull (same source, the emission endpoint's unlockUsdChart; the 30d
    and 1y windows tie to its sums). ``all-emissions.json`` = /emissions: one entry per
    token with ``protocolSlug`` / ``protocolId``, ``mcap``, ``circSupply``, ``maxSupply``,
    the unlock events; build_universe_earnings uses its ``mcap`` for protocols whose
    /protocols children report none (Hyperliquid, Uniswap, Pump, ...). Both saved as
    reported; snapshots, so always repulled. Needs DEFILLAMA_API_KEY; without it the pull
    is reported as a failure and nothing is written.
    """
    targets = {"all-protocols": "/emissionsBreakdownAggregated", "all-emissions": "/emissions"}
    if not has_pro_key():
        report_pull(defillama_output_dir, ["incentives"],
                    [(None, f"{name} (needs DEFILLAMA_API_KEY)") for name in targets])
        return
    failures = []
    for name, path in targets.items():
        if fetch_and_save_json(llama_url(path, pro_only=True), f"{defillama_output_dir}/incentives/{name}.json") is None:
            failures.append((None, name))
    report_pull(defillama_output_dir, ["incentives"], failures)
