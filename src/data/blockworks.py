"""Blockworks Research API pulls (https://api.blockworks.com/v1/metrics, header x-api-key).

Rate limits and quota
---------------------
The API enforces a per-minute client limit (``x-ratelimit-*-client-minute`` headers) and an
account-level monthly usage cap. When the cap is hit EVERY request answers
``429 {"message": "Monthly usage limit reached."}`` until the month resets, so retrying is
pointless: ``_get`` raises BlockworksQuotaExhausted on the first such response and the pull
functions stop immediately, report the un-pulled pairs, and return. A per-minute 429 is
waited out once (``x-ratelimit-reset-client-minute``, at most 60 s). 5xx responses retry
with backoff. Requests deliberately do not use helper_functions.get_session, whose retry
policy treats every 429 as transient (~15 s of backoff per request).

Note: this is Blockworks' legacy Research API. Their current Data API
(docs.blockworks.com, ``/query/timeseries/...``, header ``X-Blockworks-API-Key``) carries the
same metrics under a paid plan with the ``blockchains_unified`` /
``projects_protocols_unified`` permissions; the Research API key is not accepted there.
"""
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.helper_functions import pull_progress, report_pull, report_errors, is_fresh, save_json

BASE_URL = 'https://api.blockworks.com/v1/metrics'

# Number of project slugs sent per request when batch-pulling a metric
PROJECTS_PER_REQUEST = 20
# Files whose newest point is within this many days are not repulled (None forces a repull)
DEFAULT_MAX_STALE_DAYS = 3


class BlockworksQuotaExhausted(Exception):
    """The account's monthly usage cap is reached; every further request would 429."""

    def __init__(self, message, reset_seconds=None):
        super().__init__(message)
        self.reset_seconds = reset_seconds

    def describe(self):
        text = f"monthly usage limit reached ({self})"
        if self.reset_seconds:
            text += f"; resets in ~{max(1, round(self.reset_seconds / 86400))} day(s)"
        return text


def _build_headers(blockworks_api_key):
    return {
        "x-api-key": blockworks_api_key,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache"
    }


_SESSION = None
_SESSION_LOCK = threading.Lock()


def _session():
    """Shared session retrying 5xx with backoff but never 429 (handled in _get)."""
    global _SESSION
    if _SESSION is None:
        with _SESSION_LOCK:
            if _SESSION is None:
                retry = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504],
                              allowed_methods=["GET"], raise_on_status=False)
                s = requests.Session()
                s.mount("https://", HTTPAdapter(max_retries=retry))
                _SESSION = s
    return _SESSION


def _int_header(response, name):
    try:
        return int(response.headers.get(name))
    except (TypeError, ValueError):
        return None


def _error_message(response):
    try:
        return response.json().get('message') or response.text[:120]
    except ValueError:
        return response.text[:120]


def _get(url, headers, params=None, timeout=60):
    """GET and parse JSON, interpreting Blockworks' two kinds of 429 (see module docstring)."""
    for attempt in (1, 2):
        response = _session().get(url, headers=headers, params=params, timeout=timeout)
        if response.status_code != 429:
            response.raise_for_status()
            return response.json()
        message = _error_message(response)
        if 'monthly usage limit' in message.lower():
            raise BlockworksQuotaExhausted(message, _int_header(response, 'x-ratelimit-reset-client-month'))
        if attempt == 1:
            wait = min(_int_header(response, 'x-ratelimit-reset-client-minute') or 60, 60)
            print(f"  Blockworks per-minute limit hit; waiting {wait}s")
            time.sleep(wait)
    response.raise_for_status()


def last_point_timestamp(data):
    """Unix timestamp of the newest point in a saved {slug: [{date, value}, ...]} file."""
    series = next(iter(data.values()))
    latest = max(str(p['date']) for p in series)
    dt = datetime.fromisoformat(latest.replace('Z', '+00:00'))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _stale(slugs, metric, output_dir, max_stale_days):
    return [s for s in slugs if not is_fresh(f"{output_dir}/{metric}/{s}.json", last_point_timestamp, max_stale_days)]


def _pull_metric_batches(metric, slugs, start_date, end_date, headers, output_dir, desc):
    """Pull ``metric`` for ``slugs`` in comma-separated batches, saving one {slug: series}
    file per slug. Returns (pulled, failures); a quota 429 propagates immediately.

    One broken project can 500 a whole batch (a slug in the catalog with no working data
    endpoint), so a failed batch is retried slug by slug and only that slug fails.
    """
    chunks = [slugs[i:i + PROJECTS_PER_REQUEST] for i in range(0, len(slugs), PROJECTS_PER_REQUEST)]

    def fetch(batch):
        return _get(f'{BASE_URL}/{metric}', headers,
                    {'project': ','.join(batch), 'start_date': start_date, 'end_date': end_date})

    pulled, failures = [], []
    for chunk in pull_progress(chunks, desc):
        try:
            data = fetch(chunk)
        except BlockworksQuotaExhausted:
            raise
        except requests.RequestException:
            data = {}
            for slug in chunk:
                try:
                    data.update(fetch([slug]))
                except BlockworksQuotaExhausted:
                    raise
                except requests.RequestException as e:
                    print(f"  Error fetching {metric} for {slug}: {e}")
        for slug in chunk:
            if slug not in data:
                failures.append((metric, slug))
                continue
            save_json({slug: data[slug]}, f"{output_dir}/{metric}/{slug}.json")
            pulled.append(slug)
    return pulled, failures


# Function for pulling one or more metrics for one or more projects
@report_errors()
def pull_asset_data(assets, metrics, start_date, end_date, blockworks_api_key, blockworks_output_dir,
                    max_stale_days=DEFAULT_MAX_STALE_DAYS):
    """Pull multiple metrics for one or more projects (slugs, or a {slug: name} dict).

    One request per metric carries every stale project (comma-separated), so a run costs
    len(metrics) requests instead of len(assets) x len(metrics). Files whose newest point
    is within ``max_stale_days`` are skipped (``None`` forces a repull). If the account's
    monthly usage cap is reached the pull stops at once and the remaining pairs are
    reported as ``quota exhausted``.
    """
    headers = _build_headers(blockworks_api_key)
    slugs = list(assets.keys() if isinstance(assets, dict) else assets)

    todo = {m: _stale(slugs, m, blockworks_output_dir, max_stale_days) for m in metrics}
    skipped = sum(len(slugs) - len(v) for v in todo.values())
    if skipped:
        print(f"Blockworks asset data: {skipped} fresh file(s) skipped, {sum(map(len, todo.values()))} to pull")

    failures = []
    try:
        for metric, stale in todo.items():
            if not stale:
                continue
            _, f = _pull_metric_batches(metric, stale, start_date, end_date, headers,
                                        blockworks_output_dir, f"Blockworks {metric}")
            failures += f
            todo[metric] = []
    except BlockworksQuotaExhausted as e:
        remaining = [(m, f"{s} (quota exhausted)") for m, stale in todo.items() for s in stale
                     if not os.path.exists(f"{blockworks_output_dir}/{m}/{s}.json")]
        print(f"Blockworks: {e.describe()}. Skipping the remaining {len(remaining)} pull(s).")
        failures += remaining

    report_pull(blockworks_output_dir, metrics, failures)


def get_metric_projects(metric, blockworks_api_key):
    """Return all project slugs that track a metric, from the Blockworks metrics catalog.
    Raises BlockworksQuotaExhausted when the monthly cap is reached."""
    headers = _build_headers(blockworks_api_key)
    projects = []
    page = 1
    while True:
        payload = _get(BASE_URL, headers, {'identifier': metric, 'limit': 1000, 'page': page})
        entries = payload.get('data', [])
        projects.extend(entry['project'] for entry in entries)
        if len(projects) >= payload.get('total', 0) or not entries:
            break
        page += 1

    if not projects:
        print(f"Warning: no projects found for metric '{metric}' in the Blockworks catalog")
    return sorted(projects)


@report_errors(default=[])
def pull_metric_data(metric, start_date, end_date, blockworks_api_key, blockworks_output_dir, projects=None,
                     max_stale_days=DEFAULT_MAX_STALE_DAYS):
    """Pull one metric for every project that tracks it (or a provided list of projects).

    Saves one file per project ({output_dir}/{metric}/{slug}.json, shaped {slug: series}),
    matching the layout written by pull_asset_data; fresh files are skipped. Returns the
    list of projects pulled. Stops at once, reporting the rest, when the monthly cap is hit.
    """
    headers = _build_headers(blockworks_api_key)
    pulled, failures = [], []
    stale = []
    try:
        if projects is None:
            projects = get_metric_projects(metric, blockworks_api_key)
            print(f"Found {len(projects)} projects tracking '{metric}'")
        if not projects:
            return []
        stale = _stale(list(projects), metric, blockworks_output_dir, max_stale_days)
        if len(stale) < len(projects):
            print(f"Blockworks {metric}: {len(projects) - len(stale)} fresh file(s) skipped, {len(stale)} to pull")
        pulled, failures = _pull_metric_batches(metric, stale, start_date, end_date, headers,
                                                blockworks_output_dir, f"Blockworks {metric}")
    except BlockworksQuotaExhausted as e:
        remaining = [(metric, f"{s} (quota exhausted)") for s in stale
                     if not os.path.exists(f"{blockworks_output_dir}/{metric}/{s}.json")]
        print(f"Blockworks: {e.describe()}. Skipping the remaining {len(remaining) or 'catalog'} pull(s).")
        failures += remaining

    report_pull(blockworks_output_dir, [metric], failures)
    return pulled
