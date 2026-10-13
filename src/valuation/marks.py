"""Current marks for a liquid token.

Live pull from DeFiLlama with per-field provenance, an on-disk snapshot cache,
and an optional caller-supplied fallback dict.

USE_LIVE_DATA (passed to get_marks) controls the source:
  True  -> pull live, auto-save a snapshot, return live marks.
  False -> load the most recent saved snapshot; if none exists, fall back to the
           notebook-supplied `fallback` dict, or raise if none was given.

`units` is the protocol's traction quantity behind the model's units row.
units_metric selects its DeFiLlama source: 'borrowed' (lending-style active
loans), 'tvl' (the protocol's total TVL), or None (the notebook sets
marks['units'] itself, e.g. an annualized volume for a DEX). Revenue /
holders-revenue from /summary/fees. FDV (max) supply + forward emissions from
the Pro emission endpoint. Treasury split from the Pro treasuries endpoint.
Category TAM history from the Pro categories endpoint. The income statement's
core lines are calendar-year / trailing sums of the daily series; the Pro
financial-statement endpoint adds the incentives and earnings lines.

Every URL is built by helper_functions.llama_url: with DEFILLAMA_API_KEY set
(API_KEYS.env) all pulls, including the free-mirrored endpoints, run on
pro-api.llama.fi; without it the free hosts serve and the Pro-only pulls are
skipped quietly.
"""

import os
import json
import time
from datetime import datetime, timezone
from collections import defaultdict
from utils.helper_functions import (report_errors, error_code, redact_url, get_session,
                                    llama_url, has_pro_key)

# Trailing windows DeFiLlama exposes on /summary/fees -> which total to annualize.
WINDOW_KEY = {7: 'total7d', 30: 'total30d', 365: 'total1y'}

# `units` sources pull_marks understands (see module docstring).
UNITS_METRICS = ('borrowed', 'tvl', None)

# Snapshot cache (gitignored via *.json). Auto-written on every live pull.
DEFAULT_SNAPSHOT = os.path.join('data', 'marks_snapshot.json')
DEFAULT_INCOME_CACHE = os.path.join('data', 'defillama_income.json')
# Per-period average TAM / units values ($mm) for the historical income-statement
# columns: {period: {tam, units}}. Filled by ensure_units_history from the Pro
# categories endpoint (TAM) and the protocol's own series (units); a category's
# TAM history is not on the free API, so without a key the file is hand-entered
# (or generated out-of-band, e.g. via the DeFiLlama MCP). Existing values are
# never overwritten.
DEFAULT_UNITS_CACHE = os.path.join('data', 'units_history.json')
# Financial-statement cache: one file per protocol slug. Auto-written on live pull;
# the dir is created on demand (see financial_statement) so a fresh fork need not
# pre-create it.
DEFAULT_FINSTMT_DIR = os.path.join('data', 'financial_statements')

# DeFiLlama income-statement lines (sum of the daily series over a calendar year).
# Identity: fees - supply_side = revenue = protocol_rev + holders_rev.
_INCOME_TYPES = [('fees', 'dailyFees'), ('supply_side', 'dailySupplySideRevenue'),
                 ('revenue', 'dailyRevenue'), ('protocol_rev', 'dailyProtocolRevenue'),
                 ('holders_rev', 'dailyHoldersRevenue')]
# The Pro financial statement reports the same core lines (Gross Protocol Revenue =
# fees, Cost Of Revenue = supply_side, Gross Profit = revenue, Token Holder Net
# Income = holders_rev; verified to tie to the daily sums), plus two lines the daily
# series cannot give: Incentives (token emissions paid out) and Earnings
# (= Gross Profit - Incentives). Only those two are taken from it, so the core
# lines stay exactly as the daily series report them on either host.
_FINSTMT_INCENTIVES, _FINSTMT_EARNINGS = 'Incentives', 'Earnings'

# In-process memo for payloads several functions slice differently (the daily
# fee series behind every calendar-year / TTM sum, the financial statement, the
# categories chart). Entries expire so a kernel left open for days does not
# keep serving stale data.
_MEMO = {}
_MEMO_TTL = 3600


def _memo(key, fetch):
    hit = _MEMO.get(key)
    if hit and time.time() - hit[0] < _MEMO_TTL:
        return hit[1]
    data = fetch()
    _MEMO[key] = (time.time(), data)
    return data


def _get(url, timeout=30):
    """GET ``url`` on the shared retrying session and parse JSON. Prints the HTTP
    status code (when available) and URL on failure, then re-raises so callers'
    cache fallbacks still apply."""
    try:
        r = get_session().get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  request failed: {error_code(e)}{type(e).__name__}: {redact_url(url)}")
        raise


def _emission_body(slug):
    """Raw parsed body from the Pro emission endpoint, or None (no key / failure).
    Memoized: the marks pull and the income statement both read it."""
    url = llama_url(f'/emission/{slug}', pro_only=True)
    if url is None:
        return None

    def fetch():
        try:
            body = _get(url).get('body')
            if isinstance(body, str):
                body = json.loads(body)
            return body
        except Exception:
            return None
    return _memo(('emission', slug), fetch)


def _incentives_chart(slug):
    """Daily incentives ([[ts, usd], ...]) = the emission body's unlockUsdChart, or None when
    the protocol has no emission entry. An empty list means no incentives are tracked.
    Monthly sums of this chart tie exactly to the financial statement's Incentives line
    (verified on 26 protocols), so it is the daily basis for earnings."""
    body = _emission_body(slug)
    if body is None:
        return None
    return [(ts, v) for ts, v in (body.get('unlockUsdChart') or []) if isinstance(v, (int, float))]


def _curve_from_body(body):
    """(sorted [(ts, cumulative unlocked tokens)], max_supply) from a parsed body."""
    if not body:
        return None, None
    acc = defaultdict(float)
    for seg in (body.get('documentedData') or {}).get('data') or []:
        for pt in seg.get('data') or []:
            ts, v = pt.get('timestamp'), pt.get('unlocked')
            if ts is not None and isinstance(v, (int, float)):
                acc[float(ts)] += float(v)
    ser = sorted(acc.items())
    return (ser or None), (body.get('supplyMetrics') or {}).get('maxSupply')


def _emission_curve(slug):
    """(sorted [(ts, cumulative unlocked tokens)], max_supply) from the Pro emission
    endpoint, or (None, None). Shared by the marks pull and the comps tab."""
    return _curve_from_body(_emission_body(slug))


def _next_unlock(body, ser, now):
    """The next forward unlock as (ts, tokens, kind), or None if nothing remains.

    Prefers the next discrete cliff in ``metadata.events`` (tokens summed across all
    cliff events landing on the earliest future timestamp). If the remaining schedule
    is purely linear (no future cliff), falls back to the linear completion date and
    the tokens still unlocking between now and then ('linear')."""
    events = ((body or {}).get('metadata') or {}).get('events') or []
    cliff_ts = sorted(e['timestamp'] for e in events
                      if e.get('unlockType') == 'cliff' and e.get('timestamp', 0) > now)
    if cliff_ts:
        ts = cliff_ts[0]
        tok = sum(sum(e.get('noOfTokens') or [0]) for e in events
                  if e.get('unlockType') == 'cliff' and e.get('timestamp') == ts)
        return ts, tok, 'cliff'
    if ser and ser[-1][0] > now:
        remaining = _supply_at(ser, ser[-1][0]) - _supply_at(ser, now)
        if remaining > 1e3:                       # ignore interpolation dust
            return ser[-1][0], remaining, 'linear'
    return None


def _supply_at(ser, ts):
    """Interpolate cumulative unlocked supply at timestamp ts (clamped at the ends)."""
    if not ser:
        return None
    if ts <= ser[0][0]:
        return ser[0][1]
    if ts >= ser[-1][0]:
        return ser[-1][1]
    p = ser[0]
    for c in ser[1:]:
        if c[0] >= ts:
            f = (ts - p[0]) / (c[0] - p[0]) if c[0] > p[0] else 0
            return p[1] + f * (c[1] - p[1])
        p = c
    return ser[-1][1]


@report_errors(reraise=True)
def pull_marks(slug, gecko, window_days, hold_yrs, units_metric='borrowed', fallback=None):
    """Latest marks from DeFiLlama; falls back per-field on any error.

    `fallback` seeds the marks dict (each seeded field carries 'fallback'
    provenance until a live value replaces it); with no fallback, a field that
    fails to pull is simply absent. units_metric: 'borrowed' | 'tvl' | None
    (None = skip the units mark; the notebook supplies it).

    Returns (marks, provenance) where provenance[field] in {'live', 'fallback'}.
    """
    assert window_days in WINDOW_KEY, \
        f'window_days must be one of {sorted(WINDOW_KEY)} (got {window_days})'
    assert units_metric in UNITS_METRICS, \
        f'units_metric must be one of {UNITS_METRICS} (got {units_metric!r})'
    fallback = fallback or {}
    m, src = dict(fallback), {k: 'fallback' for k in fallback}
    wkey = WINDOW_KEY[window_days]

    # Lazy CoinGecko markets row for the fields DeFiLlama can miss (parent slugs
    # report mcap=None; FDV needs the Pro emission endpoint otherwise). Routed
    # through comps.pull_market_technicals for its retry + last-good on-disk
    # cache (the free CoinGecko tier hard-429s for minutes at a time).
    _cg_row = {}
    def _cg():
        nonlocal _cg_row
        if not _cg_row:
            try:
                from . import comps as _comps   # lazy: comps imports marks at module load
                _cg_row = _comps.pull_market_technicals([gecko]).get(gecko) or {'err': True}
            except Exception:
                _cg_row = {'err': True}
        return _cg_row

    # --- units, tvl, price, mcap (free-mirrored endpoints; fields guarded separately) ---
    p = None
    try:
        p = _get(llama_url(f'/protocol/{slug}'))
    except Exception as e:
        print(f'[marks] protocol pull failed -> fallback ({type(e).__name__})')
    if p:
        try:
            m['tvl'] = p['tvl'][-1]['totalLiquidityUSD'] / 1e6; src['tvl'] = 'live'
        except Exception:
            pass
        try:
            if units_metric == 'borrowed':
                m['units'] = p['currentChainTvls']['borrowed'] / 1e6; src['units'] = 'live'
            elif units_metric == 'tvl' and src.get('tvl') == 'live':
                m['units'] = m['tvl']; src['units'] = 'live'
        except Exception:
            pass
    try:
        pr = _get(llama_url(f'/prices/current/coingecko:{gecko}', section='coins'))['coins'][f'coingecko:{gecko}']['price']
        m['price'] = pr; src['price'] = 'live'
    except Exception as e:
        print(f'[marks] price pull failed -> fallback ({type(e).__name__})')
    mc = (p or {}).get('mcap') or _cg().get('mcap')
    if mc and m.get('price'):
        m['mcap'] = mc / 1e6; src['mcap'] = 'live'
        m['circ_supply'] = m['mcap'] / m['price']; src['circ_supply'] = 'live'

    # --- revenue & holders revenue (free-mirrored endpoint) ---
    def annu(dt):
        d = _get(llama_url(f'/summary/fees/{slug}?dataType={dt}'))
        v = d.get(wkey)
        return v * 365.0 / window_days / 1e6 if v is not None else None
    for field, dt in (('fees', 'dailyFees'), ('revenue', 'dailyRevenue'), ('holders_rev', 'dailyHoldersRevenue')):
        try:
            v = annu(dt)
            if v is not None:
                m[field] = v; src[field] = 'live'
        except Exception:
            pass

    # --- FDV (max) supply + forward emissions over the hold (Pro emission endpoint;
    # FDV falls back to CoinGecko fully_diluted_valuation / price without a key) ---
    body = _emission_body(slug)
    ser, max_supply = _curve_from_body(body)
    if max_supply:
        m['fdv_supply'] = max_supply / 1e6; src['fdv_supply'] = 'live'
    elif m.get('price'):
        fdv_usd = _cg().get('fdv')
        if fdv_usd:
            m['fdv_supply'] = fdv_usd / 1e6 / m['price']; src['fdv_supply'] = 'live'
    if ser:
        now = time.time()
        new_tok = _supply_at(ser, now + hold_yrs * 365 * 86400) - _supply_at(ser, now)
        m['emissions_hold_mm'] = max(0.0, new_tok) / 1e6; src['emissions_hold_mm'] = 'live'
        nu = _next_unlock(body, ser, now)
        if nu:
            ts, tok, kind = nu
            m['next_unlock_ts'] = ts; m['next_unlock_mm'] = tok / 1e6
            m['next_unlock_kind'] = kind
            src['next_unlock_ts'] = src['next_unlock_mm'] = src['next_unlock_kind'] = 'live'

    # --- treasury split (Pro) ---
    treasuries_url = llama_url('/treasuries', pro_only=True)
    if treasuries_url:
        try:
            for e in _get(treasuries_url):
                if isinstance(e, dict) and e.get('gecko_id') == gecko:
                    tb = e.get('tokenBreakdowns') or {}
                    m['treasury_stables'] = (tb.get('stablecoins') or 0) / 1e6
                    m['treasury_own'] = (tb.get('ownTokens') or 0) / 1e6
                    src['treasury_stables'] = src['treasury_own'] = 'live'
                    break
        except Exception:
            pass

    return m, src


def save_snapshot(marks, prov, window_days, path=DEFAULT_SNAPSHOT):
    """Write marks + provenance + timestamp to the snapshot cache. Saved as-is,
    even on a partial pull (mixed live/fallback provenance is recorded)."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    payload = {
        'asof': datetime.now().isoformat(timespec='seconds'),
        'window_days': window_days,
        'provenance': prov,
        'marks': marks,
    }
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    return path


def load_snapshot(path=DEFAULT_SNAPSHOT):
    """Return the saved snapshot payload, or None if absent/unreadable."""
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


@report_errors(reraise=True)
def get_marks(use_live, slug, gecko, window_days, hold_yrs,
              snapshot_path=DEFAULT_SNAPSHOT, units_metric='borrowed', fallback=None):
    """Acquire marks per USE_LIVE_DATA.

    Returns (marks, provenance, meta) where meta = {source, asof, window_days}.
      use_live=True  -> live pull, snapshot auto-saved.
      use_live=False -> most recent snapshot; the caller-supplied `fallback`
                        dict only if no snapshot exists; raises when neither is
                        available (run once live, or pass FALLBACK_MARKS).
    """
    if use_live:
        marks, prov = pull_marks(slug, gecko, window_days, hold_yrs,
                                 units_metric=units_metric, fallback=fallback)
        save_snapshot(marks, prov, window_days, snapshot_path)
        meta = {'source': 'live', 'asof': datetime.now().isoformat(timespec='seconds'),
                'window_days': window_days}
        return marks, prov, meta

    snap = load_snapshot(snapshot_path)
    if snap is not None:
        meta = {'source': f'snapshot ({snapshot_path})', 'asof': snap.get('asof'),
                'window_days': snap.get('window_days', window_days)}
        return dict(snap['marks']), dict(snap['provenance']), meta

    if fallback:
        meta = {'source': 'FALLBACK_MARKS (no snapshot found)', 'asof': None,
                'window_days': window_days}
        return dict(fallback), {k: 'fallback' for k in fallback}, meta

    raise FileNotFoundError(
        f'no marks snapshot at {snapshot_path!r} and no fallback supplied; '
        f'run once with USE_LIVE_DATA=True or set FALLBACK_MARKS in the notebook config')


def fdv_mm(marks, price=None):
    """Fully-diluted value ($mm) = FDV (max) supply x price (spot by default).
    The supply metric behind every displayed multiple: the model's exit price is
    exit value / FDV supply, so levels and multiples are stated on FDV (with mcap
    as the circulating companion), not on an enterprise-value bridge."""
    return marks['fdv_supply'] * (marks['price'] if price is None else price)


# --- DEPRECATED: house-EV convention (mcap - treasury net cash + forward
# emissions). No caller displays EV any more (the model and all comps run on
# FDV/mcap); kept only so the convention is easy to revive. ---

def net_cash(marks):
    """DEPRECATED (see fdv_mm). Treasury net cash of the old EV bridge:
    stablecoins + own-tokens, no debt."""
    return marks['treasury_stables'] + marks['treasury_own']


def enterprise_value(marks):
    """DEPRECATED (see fdv_mm). RV house EV: mcap - treasury net cash + forward
    emissions over the hold valued at the current price."""
    return marks['mcap'] - net_cash(marks) + marks['emissions_hold_mm'] * marks['price']


def enterprise_value_at(marks, price):
    """DEPRECATED (see fdv_mm). House EV recomputed at an arbitrary price/token."""
    circ = marks['mcap'] / marks['price']        # circulating supply implied by spot
    return (circ + marks['emissions_hold_mm']) * price - net_cash(marks)


@report_errors(default=[])
def tvl_history_mm(slug):
    """DeFiLlama historical total TVL as a sorted [(ts, $mm)] series, or [].

    Same line used for the current `tvl` mark (top-level `tvl`, all chains/categories
    aggregated); the full series lets the one-pager state TVL's 30d change and
    drawdown from its ATH.
    """
    try:
        p = _get(llama_url(f'/protocol/{slug}'))
        arr = p.get('tvl') or []
        ser = [(x['date'], x['totalLiquidityUSD'] / 1e6) for x in arr
               if isinstance(x.get('totalLiquidityUSD'), (int, float))]
        return sorted(ser)
    except Exception:
        return []


@report_errors(default=[])
def units_history_mm(slug, metric='borrowed'):
    """DeFiLlama historical `units` series as a sorted [(ts, $mm)] list, or [].

    metric='borrowed' reads the protocol's borrowed line (the source of the
    current `units` mark for lending-style deals); metric='tvl' reads the total
    TVL line (identical to tvl_history_mm). The full series lets the one-pager
    state the metric's 30d change and drawdown from its ATH.
    """
    if metric == 'tvl':
        return tvl_history_mm(slug)
    try:
        p = _get(llama_url(f'/protocol/{slug}'))
        arr = ((p.get('chainTvls') or {}).get('borrowed') or {}).get('tvl') or []
        ser = [(x['date'], x['totalLiquidityUSD'] / 1e6) for x in arr
               if isinstance(x.get('totalLiquidityUSD'), (int, float))]
        return sorted(ser)
    except Exception:
        return []


@report_errors()
def category_tam_now_mm(category, metric='borrowed'):
    """Current TAM = the `metric` total across a DeFiLlama category ($mm), summed
    live from /protocols (e.g. category='Lending', metric='borrowed' gives total
    active loans across DeFi lending). The category's historical series is
    category_history_mm (Pro). Returns None on failure."""
    try:
        data = _get(llama_url('/protocols'))
        if metric == 'borrowed':
            tot = sum((p.get('chainTvls') or {}).get('borrowed') or 0
                      for p in data if p.get('category') == category)
        else:
            tot = sum(p.get('tvl') or 0
                      for p in data if p.get('category') == category)
        return tot / 1e6 if tot else None
    except Exception:
        return None


@report_errors(default=[])
def category_history_mm(category, metric='borrowed'):
    """A DeFiLlama category's historical `metric` total as a sorted [(ts, $mm)]
    series, or [] without a Pro key or on failure.

    Pro /api/categories returns a daily chart keyed by timestamp, each point
    carrying every category's tvl / borrowed / staking / pool2 totals, so
    category='Lending', metric='borrowed' is the history behind
    category_tam_now_mm. Not on the free API (HTTP 402).
    """
    url = llama_url('/categories', pro_only=True)
    if url is None:
        return []
    try:
        chart = _memo(('categories',), lambda: _get(url)).get('chart') or {}
        ser = []
        for ts, cats in chart.items():
            v = ((cats or {}).get(category) or {}).get(metric)
            if isinstance(v, (int, float)):
                ser.append((float(ts), v / 1e6))
        return sorted(ser)
    except Exception:
        return []


def _year_mean(ser, year):
    """Average of a [(ts, value)] series over calendar `year`, or None if no points."""
    vals = [v for ts, v in ser if datetime.fromtimestamp(ts, timezone.utc).year == year]
    return sum(vals) / len(vals) if vals else None


def _trailing_mean(ser, days=365):
    """Average of a [(ts, value)] series over the trailing `days`, or None if no points."""
    cutoff = time.time() - days * 86400
    vals = [v for ts, v in ser if ts >= cutoff]
    return sum(vals) / len(vals) if vals else None


def units_history_cache(cache_path=DEFAULT_UNITS_CACHE):
    """Read the cached per-period TAM / units averages ($mm): {period: {tam, units}}.
    Returns {} if the cache is missing. See DEFAULT_UNITS_CACHE for provenance."""
    try:
        with open(cache_path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


@report_errors(default={})
def ensure_units_history(slug, category, metric, years, use_live=True,
                         cache_path=DEFAULT_UNITS_CACHE):
    """Fill the units-history cache from DeFiLlama and return it: {period: {tam, units}},
    per-calendar-year averages in $mm plus a 'TTM' row averaged over the trailing 365 days.

    TAM is the category's `metric` history (category_history_mm, Pro only); units is
    the protocol's own `metric` series (units_history_mm). Only fields that are
    missing from the cache are written, so hand-entered values are never
    overwritten; delete a value to have it regenerated. Offline (use_live=False), or
    with metric=None, the cache is returned as-is. Without a Pro key only the units
    side can be filled.
    """
    cache = units_history_cache(cache_path)
    if not use_live or metric is None:
        return cache
    periods = [str(y) for y in years] + ['TTM']
    missing = lambda field: [p for p in periods if (cache.get(p) or {}).get(field) is None]
    need_tam, need_units = (missing('tam') if category else []), missing('units')
    if not need_tam and not need_units:
        return cache

    tam_ser = category_history_mm(category, metric) if need_tam else []
    units_ser = units_history_mm(slug, metric) if need_units else []
    changed = False
    for field, ser, need in (('tam', tam_ser, need_tam), ('units', units_ser, need_units)):
        for p in need:
            if not ser:
                continue
            v = _trailing_mean(ser) if p == 'TTM' else _year_mean(ser, int(p))
            if v is not None:
                cache.setdefault(p, {})[field] = v
                changed = True
    if changed:
        os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(cache, f, indent=2)
        print(f'[units] filled {cache_path} from DeFiLlama '
              f'(tam: {len(need_tam)} period(s), units: {len(need_units)} period(s))')
    return cache


@report_errors(default=[])
def revenue_ann_history_mm(slug, window_days):
    """Trailing-window annualized revenue as a sorted [(ts, $mm)] series, or [].

    Recreates, at every historical day, the same trailing-window-annualized basis
    used for the current `revenue` mark (rolling `window_days` sum x 365/window),
    so the one-pager's 30d change and ATH drawdown are stated on a like basis.
    """
    try:
        ch = _fee_series(slug, 'dailyRevenue')
        vals = [(ts, v) for ts, v in ch if isinstance(v, (int, float))]
        w = window_days
        ser = []
        for i in range(w - 1, len(vals)):
            s = sum(v for _, v in vals[i - w + 1:i + 1])
            ser.append((vals[i][0], s * 365.0 / w / 1e6))
        return ser
    except Exception:
        return []


_PRICE_SPAN_CAP = 500   # DeFiLlama rejects larger spans with HTTP 400


@report_errors(default=[])
def price_history_usd(gecko, days=1500, period='3d'):
    """Token USD price as a sorted [(ts, price)] series from DeFiLlama coins, or [].

    Pulls the historical price chart for `coingecko:<gecko>` at `period` spacing over
    roughly the trailing `days` (DeFiLlama proxies CoinGecko, no key needed). The API
    caps a chart near 500 points and builds the window forward from `start` for `span`
    steps, so we clamp span to that cap and derive `start` from it; that keeps the
    window ending ~now (a far-back start with a clamped span would stop before today
    and return nothing) and means a max-timeframe request just maxes out at the cap
    (~4.1y at 3d spacing) rather than 400-ing. Younger tokens return their full life.
    Same coins.llama.fi source as the current price mark, so the one-pager's price
    chart shares that basis. Any failure -> [] so the chart degrades to nothing.
    """
    try:
        period_days = max(1, int(''.join(filter(str.isdigit, period)) or 3))
        span = max(2, min(_PRICE_SPAN_CAP, days // period_days))
        start = int(time.time()) - span * period_days * 86400
        url = llama_url(f'/chart/coingecko:{gecko}?start={start}&span={span}&period={period}',
                        section='coins')
        coin = (_get(url).get('coins') or {}).get(f'coingecko:{gecko}') or {}
        ser = [(p['timestamp'], p['price']) for p in (coin.get('prices') or [])
               if isinstance(p.get('price'), (int, float))]
        return sorted(ser)
    except Exception:
        return []


# Undocumented dataset: not known to be mirrored on the Pro host, so it stays on
# the free host regardless of key.
_TOKEN_RIGHTS_URL = 'https://api.llama.fi/token-rights'


@report_errors(default={})
def token_rights(slug):
    """Token-rights record for `slug` from DeFiLlama's (undocumented) token-rights
    dataset, or {} on any failure.

    The endpoint returns ONE array covering every listed protocol; rows are keyed by
    a 'DefiLlama ID' like 'parent#maple-finance', so we match on the slug appearing
    there. Not in the documented API, hence its own helper with a quiet {} fallback,
    mirroring the rest of this module. Surfaces the fee-switch status, token type, and
    value-accrual mechanism the one-pager carries (fee_switch / token_rights); see
    pipeline.build_metrics for the field mapping.
    """
    try:
        for rec in _get(_TOKEN_RIGHTS_URL):
            if isinstance(rec, dict) and slug in str(rec.get('DefiLlama ID', '')):
                return rec
    except Exception:
        pass
    return {}


def _fee_series(slug, datatype):
    """The daily `datatype` series for `slug` ([[ts, value], ...]) from /summary/fees,
    downloaded once per process (see _memo) so every calendar-year, TTM and rolling
    slice reuses it instead of re-pulling the same payload."""
    url = llama_url(f'/summary/fees/{slug}?dataType={datatype}')
    return _memo(('series', slug, datatype), lambda: _get(url).get('totalDataChart') or [])


def _year_sum(slug, datatype, year):
    """Sum a DeFiLlama daily fees/revenue series over a calendar year ($mm)."""
    ch = _fee_series(slug, datatype)
    s = 0.0
    for ts, v in ch:
        if isinstance(v, (int, float)) and datetime.fromtimestamp(ts, timezone.utc).year == year:
            s += v
    return s / 1e6


def _trailing_sum(slug, datatype, days=365):
    """Sum a DeFiLlama daily fees/revenue series over the trailing `days` ($mm)."""
    ch = _fee_series(slug, datatype)
    cutoff = time.time() - days * 86400
    s = sum(v for ts, v in ch if isinstance(v, (int, float)) and ts >= cutoff)
    return s / 1e6


def _statement_extras(slug, period, revenue_mm):
    """(incentives $mm, earnings $mm, basis) for `period`, or (None, None, None) without a
    key / data. period is an int calendar year or 'TTM'.

    Incentives come from the daily emission chart (_incentives_chart) summed over the
    same window as the core lines (calendar year, or trailing 365 days for TTM), so
    earnings = revenue - incentives sits on exactly the revenue basis. A protocol with
    an emission entry but no tracked incentives gets 0 / earnings = revenue, which is
    DefiLlama's convention. Only when the protocol has no emission entry at all does
    this fall back to the Pro financial statement's Incentives / Earnings lines
    (yearly aggregate, or the 12 most recent complete months for TTM)."""
    if not has_pro_key():
        return None, None, None
    chart = _incentives_chart(slug)
    if chart is not None:
        if period == 'TTM':
            cutoff = time.time() - 365 * 86400
            inc = sum(v for ts, v in chart if ts >= cutoff) / 1e6
            basis = 'trailing 365 days'
        else:
            inc = sum(v for ts, v in chart
                      if datetime.fromtimestamp(ts, timezone.utc).year == period) / 1e6
            basis = 'calendar year'
        if not chart:
            basis += ' (no incentives tracked)'
        return inc, revenue_mm - inc, basis
    ag = (financial_statement(slug) or {}).get('aggregates') or {}
    line = lambda row, name: (row.get(name) or {}).get('value') if isinstance(row, dict) else None
    num = lambda v: v if isinstance(v, (int, float)) else None
    if period == 'TTM':
        months = ag.get('monthly') or {}
        current = datetime.now(timezone.utc).strftime('%Y-%m')
        rows = [months[k] for k in sorted(k for k in months if k < current)[-12:]]
        vals = [num(line(r, _FINSTMT_INCENTIVES)) for r in rows]
        vals = [v for v in vals if v is not None]
        if not vals:
            return None, None, None
        inc = sum(vals) / 1e6
        return inc, revenue_mm - inc, 'trailing 365 days; incentives from the statement over the 12 most recent complete months'
    row = (ag.get('yearly') or {}).get(str(period))
    inc, earn = num(line(row, _FINSTMT_INCENTIVES)), num(line(row, _FINSTMT_EARNINGS))
    if inc is None and earn is None:
        return None, None, None
    inc = inc / 1e6 if inc is not None else revenue_mm - earn / 1e6
    earn = earn / 1e6 if earn is not None else revenue_mm - inc
    return inc, earn, 'calendar year'


def _finish_income(inc, slug, period, default_basis):
    """Attach year / source / basis and the Pro incentives + earnings lines to a
    core income dict, in place; returns it."""
    incentives, earnings, basis = _statement_extras(slug, period, inc['revenue'])
    inc.update(year=period, incentives=incentives, earnings=earnings,
               source=('daily-series + emission chart' if basis and 'statement' not in basis
                       else 'daily-series + financial-statement' if basis else 'daily-series'),
               basis=basis or default_basis)
    return inc


def _write_income_cache(inc, key, cache_path):
    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
    cache = {}
    try:
        with open(cache_path) as f:
            cache = json.load(f)
    except (OSError, json.JSONDecodeError):
        pass
    cache[key] = inc
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2)


@report_errors()
def defillama_income(slug, year, use_live=True, cache_path=DEFAULT_INCOME_CACHE):
    """Calendar-year income statement from DeFiLlama daily series ($mm):
    {year, fees, supply_side, revenue, protocol_rev, holders_rev, incentives,
    earnings, source, basis}. Ties out as fees - supply_side = revenue =
    protocol_rev + holders_rev (as the adapters report the lines).

    The core lines are sums of the daily series (each series downloaded once per
    process, on the Pro host when a key is set). `incentives` and `earnings` come
    from the Pro financial statement (one request per protocol, shared with
    financial_statement) and are None without a key. Mirrors get_marks: when
    use_live, pulls and caches (keyed by year); offline (or on a live failure)
    reads the cache. Returns the dict, or None if neither a live pull nor a
    cached value is available.
    """
    if use_live:
        try:
            inc = {k: _year_sum(slug, dt, year) for k, dt in _INCOME_TYPES}
            _finish_income(inc, slug, year, 'calendar year')
            _write_income_cache(inc, str(year), cache_path)
            return inc
        except Exception as e:
            print(f'[income] live FY{year} pull failed -> cache ({type(e).__name__})')
    try:
        with open(cache_path) as f:
            return json.load(f).get(str(year))
    except (OSError, json.JSONDecodeError):
        return None


@report_errors()
def defillama_income_ttm(slug, use_live=True, cache_path=DEFAULT_INCOME_CACHE):
    """Trailing-twelve-month (TTM) income statement from DeFiLlama daily series ($mm):
    same keys as defillama_income ({fees, supply_side, revenue, protocol_rev, holders_rev})
    but summed over the trailing 365 days rather than a calendar year, so it lines up with
    the equity comps' TTM (ttm_income_stmt) basis. Ties out the same way:
    fees - supply_side = revenue = protocol_rev + holders_rev.

    `incentives` (Pro financial statement, summed over the 12 most recent complete
    months since the monthly aggregates only close at month-end) and `earnings`
    (= this TTM revenue - incentives) are None without a key; `basis` records the
    mixed window. Mirrors defillama_income's live/offline behavior, cached under
    the 'TTM' key in the same income cache. Returns the dict, or None if neither a
    live pull nor a cached value exists.
    """
    if use_live:
        try:
            inc = {k: _trailing_sum(slug, dt) for k, dt in _INCOME_TYPES}
            _finish_income(inc, slug, 'TTM', 'trailing 365 days')
            _write_income_cache(inc, 'TTM', cache_path)
            return inc
        except Exception as e:
            print(f'[income] live TTM pull failed -> cache ({type(e).__name__})')
    try:
        with open(cache_path) as f:
            return json.load(f).get('TTM')
    except (OSError, json.JSONDecodeError):
        return None


@report_errors()
def financial_statement(slug, use_live=True, cache_dir=DEFAULT_FINSTMT_DIR):
    """Full DeFiLlama income-statement report for `slug` (Pro endpoint), or None.

    Returns the raw payload: protocol metadata, methodology, and `aggregates`
    (yearly/quarterly/monthly), each line item carrying `value` plus an optional
    `by-label` breakdown (e.g. Gross Protocol Revenue -> Borrow Interest /
    Management Fees / Service Fees / Strategy Fees).

    Mirrors defillama_income: when use_live, pulls and caches one file per slug at
    `{cache_dir}/{slug}.json`; offline (or on a live failure) reads that file.
    The cache dir is created on demand (exist_ok) so a fresh fork/clone without
    the folder still works.
    """
    cache_path = os.path.join(cache_dir, f'{slug}.json')
    if use_live:
        url = llama_url(f'/v2/metrics/financial-statement/protocol/{slug}', pro_only=True)
        if url:
            try:
                data = _memo(('finstmt', slug), lambda: _get(url))
                os.makedirs(cache_dir, exist_ok=True)
                with open(cache_path, 'w') as f:
                    json.dump(data, f, indent=2)
                return data
            except Exception as e:
                print(f'[finstmt] live pull for {slug} failed -> cache ({type(e).__name__})')
        else:
            print('[finstmt] no DEFILLAMA_API_KEY set -> cache')
    try:
        with open(cache_path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
