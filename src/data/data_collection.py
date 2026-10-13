"""
Per-protocol data collection for token briefs.

Resolve a protocol name against the protocol index, work out exactly which raw
files a set of outputs needs, check that each one is fresh (not just present),
repull only the stale (protocol, metric) pairs, and rebuild the processed
valuation multiples when their inputs changed.

Used by example-token-brief.ipynb, which calls these the same way it calls
src/visualization. All paths are relative to the repo root, matching the rest
of src/.
"""
import difflib
import json
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from data import defillama, messari, blockworks, data_processing
from data import protocol_index as pi
from utils.helper_functions import report_errors, last_timestamp

# DefiLlama API metric names <-> data/raw/defillama subdirectories
API_TO_DIR = {
    'dailyFees': 'fees',
    'dailyRevenue': 'revenue',
    'dailyHoldersRevenue': 'holders-revenue',
    'dailyIncentives': 'incentives',     # Pro only: emission endpoint's unlockUsdChart
    'dailyEarnings': 'earnings',         # derived: revenue - incentives (data_processing.build_earnings_series)
    'tvl': 'tvl',
}
DIR_TO_API = {v: k for k, v in API_TO_DIR.items()}
# Derived DefiLlama dirs -> the raw dirs they are built from. A derived metric is never
# pulled; repull_stale refreshes its inputs and rebuilds it.
DERIVED_INPUTS = {'earnings': ('revenue', 'incentives')}

DEFILLAMA_OUTPUT_DIR = 'data/raw/defillama'
MESSARI_OUTPUT_DIR = 'data/raw/messari'
ALL_PROTOCOLS_MULTIPLES = 'data/processed/valuation-multiples/all-protocols-by-category.json'
WEIGHTED_MULTIPLES = 'data/processed/valuation-multiples/weighted-multiples-by-category.json'

STATUS_FRESH = 'fresh'
STATUS_STALE = 'stale'
STATUS_MISSING = 'missing'
STATUS_UPSTREAM = 'stale (upstream)'   # repulled and still stale: the source has no newer complete data


class ProtocolNotFoundError(ValueError):
    """Raised when a name cannot be matched to an entry in config/protocol_index.json."""


# --------------------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------------------
def resolve_protocol(query: str, index_path: str = None) -> Tuple[str, Dict]:
    """Return (standard_name, index_entry) for a case-insensitive match on the index key,
    display name, DefiLlama slug or Messari slug.

    Raises:
        ProtocolNotFoundError with close-match suggestions when nothing matches.
    """
    q = query.strip().lower()
    protocols = pi.load_index(index_path)['protocols']

    if q in protocols:
        return q, protocols[q]
    for std, entry in protocols.items():
        aliases = {entry.get('display_name'), entry.get('defillama_slug'), entry.get('messari_slug')}
        if q in {a.lower() for a in aliases if a}:
            return std, entry

    universe = list(protocols) + [e.get('display_name', '') for e in protocols.values()]
    suggestions = difflib.get_close_matches(query, universe, n=5, cutoff=0.5)
    msg = f"'{query}' is not in config/protocol_index.json."
    if suggestions:
        msg += f" Did you mean: {', '.join(suggestions)}?"
    else:
        msg += " Add it with protocol_index.add_protocol(...) in example-pull-raw-data.ipynb first."
    raise ProtocolNotFoundError(msg)


# --------------------------------------------------------------------------------------
# Required data, derived from the outputs a brief presents
# --------------------------------------------------------------------------------------
def bottom_metric_dir(multiple: str) -> Optional[str]:
    """'MCAP-to-annualized-fees-multiple-7d' -> 'fees'; 'FDV-to-TVL-multiple-30d' -> 'tvl'."""
    m = re.search(r'-to-(?:annualized-)?(.+?)-multiple', multiple)
    return m.group(1).lower() if m else None


def required_data(single_asset_metrics: Iterable[str], cohort_metric: str,
                  multiples: Iterable[str]) -> Dict[str, List[Tuple[str, str]]]:
    """Build the (source, metric) pairs a brief needs for the target and for each peer.

    Target: DefiLlama tvl (description) + every single-asset metric + the cohort metric
    + the bottom metric of each multiple, plus Messari marketcap (FDV / MCAP).
    Peer: the cohort metric + the bottom metric of each multiple, plus Messari marketcap.
    Nothing else is checked or pulled.

    Raises:
        ValueError if a multiple is built on a metric this pipeline does not pull.
    """
    multiple_dirs = {bottom_metric_dir(m) for m in multiples}
    unsupported = multiple_dirs - set(DIR_TO_API)
    if unsupported:
        raise ValueError(f"Multiples on {sorted(unsupported)} need data this pipeline does not pull "
                         f"(DefiLlama {sorted(DIR_TO_API)} only).")

    target_dl = {'tvl'} | {API_TO_DIR[m] for m in single_asset_metrics} | {API_TO_DIR[cohort_metric]} | multiple_dirs
    peer_dl = {API_TO_DIR[cohort_metric]} | multiple_dirs
    # a derived metric needs its inputs checked (and repulled) too
    for scope_set in (target_dl, peer_dl):
        for d in list(scope_set):
            scope_set.update(DERIVED_INPUTS.get(d, ()))
    return {
        'target': [('defillama', d) for d in sorted(target_dl)] + [('messari', 'marketcap')],
        'peer': [('defillama', d) for d in sorted(peer_dl)] + [('messari', 'marketcap')],
    }


# --------------------------------------------------------------------------------------
# Freshness
# --------------------------------------------------------------------------------------
def data_path(source: str, metric: str, standard_name: str) -> str:
    return f'data/raw/{source}/{metric}/{standard_name}.json'


def last_data_date(path: str, source: str, metric: str) -> Optional[pd.Timestamp]:
    """Most recent data point in a raw file (normalized to midnight), or None if missing/unreadable."""
    if not os.path.exists(path):
        return None
    # Same extractors the pull functions use for their skip-if-fresh check
    if source == 'defillama' and metric == 'tvl':
        extract = defillama.last_tvl_timestamp
    elif source == 'defillama' and metric == 'incentives':
        extract = defillama.last_incentives_timestamp
    elif source == 'defillama':
        extract = defillama.last_flow_timestamp
    elif source == 'messari':
        extract = messari.last_point_timestamp
    elif source == 'blockworks':
        extract = blockworks.last_point_timestamp
    else:
        return None
    ts = last_timestamp(path, extract)
    if ts is None:
        return None
    return pd.to_datetime(ts, unit='s').normalize()


def processed_last_date(path: str) -> Optional[pd.Timestamp]:
    """Latest date key in a processed {date: ...} JSON file, or None."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return pd.to_datetime(max(json.load(f).keys())).normalize()
    except Exception:
        return None


def _status(d: Optional[pd.Timestamp], max_stale_days: int) -> str:
    if d is None:
        return STATUS_MISSING
    return STATUS_FRESH if (pd.Timestamp.today().normalize() - d).days <= max_stale_days else STATUS_STALE


def check_data(standard_name: str, category: str, required: Dict[str, List[Tuple[str, str]]],
               max_stale_days: int = 3,
               processed_paths: Dict[str, str] = None,
               index_path: str = None) -> pd.DataFrame:
    """Freshness table for the target, its category peers and the processed multiples files.

    Columns: scope (target / peer / processed), protocol, source, metric, last_date, status, path.
    A file is fresh when its last data point is within ``max_stale_days`` of today.
    """
    if processed_paths is None:
        processed_paths = {'all-protocols multiples': ALL_PROTOCOLS_MULTIPLES,
                           'weighted multiples': WEIGHTED_MULTIPLES}
    cohort = pi.get_protocols_by_category(category, index_path)
    rows = []
    for scope, protos in (('target', [standard_name]), ('peer', [p for p in cohort if p != standard_name])):
        for p in protos:
            for source, metric in required[scope]:
                path = data_path(source, metric, p)
                d = last_data_date(path, source, metric)
                rows.append(dict(scope=scope, protocol=p, source=source, metric=metric,
                                 last_date=None if d is None else d.date(),
                                 status=_status(d, max_stale_days), path=path))
    for label, path in processed_paths.items():
        d = processed_last_date(path)
        rows.append(dict(scope='processed', protocol='-', source='processed', metric=label,
                         last_date=None if d is None else d.date(),
                         status=_status(d, max_stale_days), path=path))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Repull
# --------------------------------------------------------------------------------------
@report_errors(reraise=True)
def repull_stale(checks: pd.DataFrame, messari_api_key: str = None,
                 defillama_output_dir: str = DEFILLAMA_OUTPUT_DIR,
                 messari_output_dir: str = MESSARI_OUTPUT_DIR,
                 messari_start_date: str = '2021-01-01',
                 max_stale_days: int = 3,
                 index_path: str = None) -> Dict:
    """Repull only the stale/missing raw (protocol, metric) pairs in a check_data() table.

    DefiLlama pulls go one metric at a time, carrying only the protocols stale for that
    metric. Messari pulls marketcap only. Processed rows are ignored here; see
    rebuild_valuation_multiples().

    Returns:
        dict with
          repulled  - True if any pull ran
          failures  - list of (metric, protocol) pulls that failed
          checks    - the stale rows re-checked after the pull; rows still stale are marked
                      'stale (upstream)' because the source itself has no newer complete data
    """
    stale = checks[(checks.status != STATUS_FRESH) & (checks.scope != 'processed')]
    result = {'repulled': False, 'failures': [], 'checks': None}
    if stale.empty:
        return result

    end_date = pd.Timestamp.today().strftime('%Y-%m-%d')

    dl_metrics = sorted(stale[stale.source == 'defillama'].metric.unique())
    for metric in dl_metrics:
        if metric in DERIVED_INPUTS:
            continue                      # rebuilt below, after its inputs are fresh
        protos = sorted(stale[(stale.source == 'defillama') & (stale.metric == metric)].protocol.unique())
        assets = {pi.get_defillama_slug(p, index_path): p for p in protos if pi.get_defillama_slug(p, index_path)}
        if not assets:
            continue
        print(f"DefiLlama {metric}: {', '.join(assets.values())}")
        result['failures'] += defillama.pull_metric_data_by_asset(
            {DIR_TO_API[metric]: metric}, assets, defillama_output_dir, report=False)
        result['repulled'] = True

    # Derived metrics: rebuild from the (now fresh) inputs. Inputs already fresh were not
    # in the stale table and are simply read back; build_earnings_series pulls any that
    # are missing outright.
    for metric in dl_metrics:
        if metric not in DERIVED_INPUTS:
            continue
        protos = sorted(stale[(stale.source == 'defillama') & (stale.metric == metric)].protocol.unique())
        assets = {pi.get_defillama_slug(p, index_path): p for p in protos if pi.get_defillama_slug(p, index_path)}
        if not assets:
            continue
        print(f"DefiLlama {metric} (derived from {' - '.join(DERIVED_INPUTS[metric])}): {', '.join(assets.values())}")
        data_processing.build_earnings_series(assets, defillama_output_dir)
        result['repulled'] = True

    protos = sorted(stale[(stale.source == 'messari') & (stale.metric == 'marketcap')].protocol.unique())
    assets = {pi.get_messari_slug(p, index_path): p for p in protos if pi.get_messari_slug(p, index_path)}
    if assets:
        if messari_api_key:
            print(f"Messari marketcap: {', '.join(assets.values())}")
            messari.pull_asset_data(assets, ['marketcap'], messari_start_date, end_date,
                                    messari_api_key, messari_output_dir)
            result['repulled'] = True
        else:
            print('MESSARI_API_KEY not set; skipping marketcap repull for', ', '.join(assets.values()))

    if result['failures']:
        print('Failed pulls:', result['failures'])

    # Re-check what was pulled; anything still stale is an upstream gap, not a cache problem
    rechecked = stale.copy()
    for i, row in rechecked.iterrows():
        d = last_data_date(row['path'], row['source'], row['metric'])
        status = _status(d, max_stale_days)
        rechecked.at[i, 'last_date'] = None if d is None else d.date()
        rechecked.at[i, 'status'] = STATUS_UPSTREAM if status == STATUS_STALE else status
    upstream = rechecked[rechecked.status == STATUS_UPSTREAM]
    for _, row in upstream.iterrows():
        print(f"Note: {row['source']} has no {row['metric']} data for {row['protocol']} after "
              f"{row['last_date']} (the partial current day is dropped at pull time).")
    result['checks'] = rechecked
    return result


# --------------------------------------------------------------------------------------
# Processed multiples
# --------------------------------------------------------------------------------------
def rebuild_valuation_multiples(all_protocols_path: str = ALL_PROTOCOLS_MULTIPLES,
                                weighted_path: str = WEIGHTED_MULTIPLES,
                                ensure_category: str = None,
                                start_date: str = '2021-01-01',
                                index_path: str = None) -> List[str]:
    """Rebuild both processed valuation-multiples files from the cached raw data.

    Mirrors the valuation cells of example-pull-raw-data.ipynb. Every category already in
    the all-protocols file is kept (plus ``ensure_category``) so the overwrite drops nothing.
    Local recomputation only; no network calls.

    Returns:
        The list of categories processed.
    """
    if os.path.exists(all_protocols_path):
        with open(all_protocols_path) as f:
            existing = json.load(f)
        categories = list(existing[max(existing)].keys())
    else:
        categories = pi.get_all_categories(index_path)
    if ensure_category and ensure_category not in categories:
        categories.append(ensure_category)

    print(f"Rebuilding valuation multiples for categories: {', '.join(categories)}")
    cat_dict = pi.get_standard_names_dict(categories=categories, index_path=index_path)
    data_processing.process_asset_metrics_data(categories=cat_dict, output_filepath=all_protocols_path)
    data_processing.calculate_weighted_multiples(
        input_filepath=all_protocols_path,
        categories=cat_dict,
        breakout_assets={c: None for c in categories},
        start_date=start_date,
        output_filepath=weighted_path,
    )
    return categories


def multiples_availability(standard_name: str, category: str,
                           all_protocols_path: str = ALL_PROTOCOLS_MULTIPLES,
                           weighted_path: str = WEIGHTED_MULTIPLES) -> Tuple[bool, bool]:
    """(protocol is in the all-protocols file under category, category is in the weighted file)."""
    has_multiples = has_weighted = False
    if os.path.exists(all_protocols_path):
        with open(all_protocols_path) as f:
            m = json.load(f)
        has_multiples = standard_name in m[max(m)].get(category, {})
    if os.path.exists(weighted_path):
        with open(weighted_path) as f:
            w = json.load(f)
        has_weighted = category in w[max(w)]
    return has_multiples, has_weighted


def filter_multiples_to_category(category: str, output_path: str,
                                 all_protocols_path: str = ALL_PROTOCOLS_MULTIPLES) -> str:
    """Write a copy of the all-protocols multiples file containing only one category.

    visualization.protocol_multiples_by_category plots every category in its input, so a
    single-category brief hands it this derived (gitignored, safe-to-delete) file instead.
    """
    with open(all_protocols_path) as f:
        all_multiples = json.load(f)
    filtered = {d: {category: v[category]} for d, v in all_multiples.items() if category in v}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(filtered, f)
    return output_path


# --------------------------------------------------------------------------------------
# Cohort and metadata
# --------------------------------------------------------------------------------------
def cohort_assets(category: str, metric_dir: str, index_path: str = None) -> Tuple[Dict[str, str], List[str]]:
    """({standard_name: display_name} for category members with a DefiLlama ``metric_dir`` file,
    [members dropped for lacking one]). Includes the target; ready for multi_asset_daily_metric."""
    cohort = pi.get_protocols_by_category(category, index_path)
    peers = {p: pi.get_display_name(p, index_path) for p in cohort
             if os.path.exists(data_path('defillama', metric_dir, p))}
    dropped = [p for p in cohort if p not in peers]
    return peers, dropped


def has_raw(source: str, metric: str, standard_name: str) -> bool:
    return os.path.exists(data_path(source, metric, standard_name))


def load_defillama_metadata(standard_name: str) -> Dict:
    """The DefiLlama /protocol response cached at data/raw/defillama/tvl/{name}.json
    (description, url, twitter, chains, currentChainTvls, raises, hacks, mcap, tvl history)."""
    with open(data_path('defillama', 'tvl', standard_name)) as f:
        return json.load(f)


PARENT_PROTOCOLS = 'data/processed/parent_protocols.json'


def parent_components(defillama_slug: Optional[str],
                      parent_protocols_path: str = PARENT_PROTOCOLS) -> List[Dict]:
    """The DefiLlama sub-protocols a parent slug aggregates, largest fees first.

    Looks the slug up in parent_protocols.json (built by
    data_processing.establish_parent_protocols). Pass the entry's defillama_slug,
    not the index key (bnb -> bsc, ether-fi -> ether.fi). Returns [] when the file
    or the slug is absent, so callers can print the list only when it exists.
    A parent that is a single standalone protocol returns a one-item list.
    """
    if not defillama_slug or not os.path.exists(parent_protocols_path):
        return []
    with open(parent_protocols_path) as f:
        parents = json.load(f)
    subs = (parents.get(defillama_slug) or {}).get('sub_protocols') or []
    return sorted(subs, key=lambda s: s.get('fees') or 0, reverse=True)
