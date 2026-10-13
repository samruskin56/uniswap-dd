"""Hyperliquid AQAv2 reserve yield: what the protocol earns on USDC reserves, per 30-day period, and
an estimate of future periods under a Fed funds path.

How AQAv2 charges reserve yield (docs: hypercore/aligned-quote-assets)
---------------------------------------------------------------------
Every UTC date the node records two numbers for USDC and books one day of interest::

    interest = usdcSupply * effectiveRate / 365.25

``usdcSupply`` is the native-USDC balance on HyperEVM of the treasury address the treasury deployer
(Coinbase) designated, read at the first block at or after 00:00 UTC. That address holds exactly
90% of the USDC backing HyperCore; the technical deployer's (Circle's) linked contract holds the
other 10% and the two rebalance every block to keep the docs' 9:1 split. So the "approximately
90% of cost-adjusted reserve yield" the docs promise is delivered through the balance: the
treasury is charged 100% of the AQA rate on 90% of the money. ``effectiveRate`` is the AQA rate
itself, an onchain stake-weighted median of validator-reported values, published once a day; it
is the reserve yield after issuer costs, not a 90% share of anything. Both facts were verified
on 2026-09-30: the identity holds to 1e-8 on every published day, and the treasury balance at
midnight reproduces ``usdcSupply`` to the cent on 42 of 46 days (within $1,703 on the rest).

Days are summed in 30-day intervals that start on the activation date (USDC: 2026-08-27). Each
interval's total is deducted from the system interest address ``0x50...00 + token_index`` and
sent to the Assistance Fund at 00:00 UTC 8 days after the interval's last date. Period 1 is
2026-08-27..2026-09-25, paid 2026-10-03; period 2 is 2026-09-26..2026-10-25, paid 2026-11-02.

The rate and Fed funds
----------------------
From 2026-08-16 to 2026-09-16 the AQA rate sat at 3.135% against an effective Fed funds rate of
3.63%, a gap of 49.5bp. When EFFR stepped to 3.88% on 2026-09-17 the AQA rate did not step: it
drifted up as reserves rolled into new bills, 7.8bp of the 25bp in the first 14 days. The forward
model therefore treats the rate as a trailing L-day mean of the reference minus a fixed cost
(default 50bp), with L fitted to the observed ramp (about 45 days; reserve-fund weighted average
maturity says 40-60). Under a fixed cost every Fed move passes through one for one once L days
have elapsed: +25bp is +$1.26M per 30-day period on a 6.1B base.

RPC caveat
----------
The public HyperEVM RPC (rpc.hyperliquid.xyz/evm) is not an archive node: asked for an old block
it returns today's state with no error. Historical balance reads go through an archive RPC and
``pull_treasury_balances`` refuses to start until the node it uses has proven it serves history.

Files
-----
raw  data/raw/hyperliquid/aqav2/interest-history-usdc.json          aqav2InterestHistory, as returned + _pulled_ts
     data/raw/hyperliquid/aqav2/interest-address-ledger-usdc.json   userNonFundingLedgerUpdates for 0x50..00, as returned
     data/raw/hyperliquid/aqav2/treasury-balances-usdc.json         treasury / linked / total USDC at each 00:00 UTC block
     data/raw/fred/EFFR.json                                        reference rate, fred.pull_series
No processed file: the series is small and every frame is rebuilt from raw at call time.
"""
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.helper_functions import save_json, is_fresh, report_errors, error_code, run_parallel

INFO_URL = 'https://api.hyperliquid.xyz/info'
INFO_TIMEOUT = 60
EVM_TIMEOUT = 45

DAYS_PER_YEAR = 365.25       # the official series' day count (hl.eco's estimator uses 365)
PERIOD_DAYS = 30             # UTC dates per accrual interval
PAYMENT_LAG_DAYS = 8         # paid at 00:00 UTC on the 8th day after the interval's last date

TOKENS = {0: 'USDC'}         # AQAv2 token index -> symbol (USDC is the only aligned asset today)
AQAV2_ACTIVATION = {'USDC': '2026-08-27'}   # first accrual date; asserted against the first active row
ASSISTANCE_FUND = '0xfefefefefefefefefefefefefefefefefefefefe'
USDC_EVM = '0xb88339CB7199b77E23DB6E890353E22632Ba630f'          # native USDC on HyperEVM (6 decimals)
USDT0_EVM = '0xb8ce59fc3717ada4c02eadf9682a9e934f625ebb'         # only used to prove an RPC serves history
TREASURY = {'USDC': '0xc20699185c15D0a2fD65779BB5d69f5b0B113c00'}   # treasury deployer (Coinbase)
LINKED = {'USDC': '0x6b9e773128f453f5c2c60935ee2de2cbc5390a24'}     # technical deployer's linked contract (Circle)

ARCHIVE_RPCS = ['https://rpc.hyperlend.finance/archive', 'https://hyperliquid.drpc.org', 'https://rpc.purroofgroup.com']
NON_ARCHIVE_RPC = 'https://rpc.hyperliquid.xyz/evm'              # refused for historical state; fine (and fast) for block headers
TREASURY_PULL_WORKERS = 6
ARCHIVE_CHECK_BLOCK = 43292002                                    # 2026-08-16 00:00 UTC
REF_BLOCK, REF_BLOCK_DAY = 47244375, '2026-09-30'                 # first block at/after 00:00 UTC that day

PAYMENT_TOLERANCE_USDC = 1.0
BASE_TOLERANCE_USDC = 2000.0   # the node's sampled block can differ from ours by a second
DEFAULT_MAX_STALE_DAYS = 0     # repull when the newest row is older than today UTC

DEFAULT_REFERENCE_SERIES = 'EFFR'
FIXED_COST_BP = 50             # user decision: fixed haircut below the reference, not fitted
DEFAULT_PASSTHROUGH_DAYS = 45
PASSTHROUGH_GRID = range(10, 91)
# FOMC decision days; a new EFFR prints the next day
FOMC_DECISION_DATES = ['2026-10-28', '2026-12-09', '2027-01-27', '2027-03-17', '2027-04-28',
                       '2027-06-09', '2027-07-28', '2027-09-15', '2027-10-27', '2027-12-08']

DEFAULT_SCENARIOS = {
    'Flat': {'fed_path': None},
    'Cuts': {'fed_path': {'bp_per_meeting': -25, 'n_meetings': 4}},
    'Hikes': {'fed_path': {'bp_per_meeting': 25, 'n_meetings': 2}},
}


def system_interest_address(token=0):
    """``0x50`` followed by the token index, zero-padded to 40 hex chars: token 0 -> ``0x5000...0000``."""
    return '0x50' + f'{token:038x}'


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _today():
    return datetime.now(timezone.utc).date()


def _day_ts(day):
    """Unix seconds at 00:00 UTC of a YYYY-MM-DD string or date."""
    d = day if isinstance(day, datetime) else datetime.fromisoformat(str(day)[:10])
    return int(d.replace(tzinfo=timezone.utc).timestamp())


def _age(ts):
    if ts is None:
        return 'never'
    seconds = max(0, time.time() - ts)
    if seconds < 3600:
        return f'{int(seconds // 60)}m ago'
    if seconds < 86400:
        return f'{seconds / 3600:.0f}h ago'
    return f'{seconds / 86400:.1f}d ago'


def _fmt_usd(v):
    if v is None or pd.isna(v):
        return '-'
    if abs(v) >= 1e9:
        return f'${v / 1e9:,.2f}b'
    if abs(v) >= 1e6:
        return f'${v / 1e6:,.2f}m'
    if abs(v) >= 1e3:
        return f'${v / 1e3:,.0f}k'
    return f'${v:,.0f}'


def _info(body, retries=3, timeout=INFO_TIMEOUT):
    """POST ``body`` to the /info endpoint and return the parsed JSON.

    Retries on connection errors, 429 and 5xx with 2s/4s/8s backoff (the shared
    helper_functions session retries GET only). Other 4xx raise at once.
    """
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(INFO_URL, json=body, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
            if 400 <= r.status_code < 500:
                r.raise_for_status()
            return r.json()
        except requests.HTTPError as e:
            code = getattr(e.response, 'status_code', None)
            if code is not None and 400 <= code < 500 and code != 429:
                raise
            last = e
        except (requests.RequestException, ValueError) as e:
            last = e
        if attempt < retries - 1:
            time.sleep(2 ** (attempt + 1))
    raise last


def _interest_path(output_dir, token=0):
    return f'{output_dir}/aqav2/interest-history-{TOKENS[token].lower()}.json'


def _ledger_path(output_dir, token=0):
    return f'{output_dir}/aqav2/interest-address-ledger-{TOKENS[token].lower()}.json'


def _treasury_path(output_dir, token=0):
    return f'{output_dir}/aqav2/treasury-balances-{TOKENS[token].lower()}.json'


def period_bounds(k, activation):
    """(start, end, payment_date) of period ``k`` (1-based) as ``pd.Timestamp``: 30 UTC dates from
    ``activation + 30(k-1)``, paid 8 days after the last date (the docs' 'day 38')."""
    start = pd.Timestamp(activation) + pd.Timedelta(days=PERIOD_DAYS * (k - 1))
    end = start + pd.Timedelta(days=PERIOD_DAYS - 1)
    return start, end, end + pd.Timedelta(days=PAYMENT_LAG_DAYS)


# ---------------------------------------------------------------------------
# 1. Pulls: the official daily series, the payment ledger, the treasury balances
# ---------------------------------------------------------------------------

@report_errors()
def pull_aqav2_interest_history(output_dir, token=0, max_stale_days=DEFAULT_MAX_STALE_DAYS, force=False):
    """Save the node's daily AQAv2 series (``aqav2InterestHistory``) exactly as returned.

    The endpoint returns the full history on every call, so the file is overwritten whole; it is
    kept when its newest date is within ``max_stale_days`` of today UTC (0 = newest date is today).
    ``None`` from the API means the token is not an AQAv2 asset. Returns the saved dict.
    """
    symbol = TOKENS[token]
    path = _interest_path(output_dir, token)
    if not force and is_fresh(path, lambda d: _day_ts(d['history'][-1][0]), max_stale_days):
        data = _load_json(path)
        print(f"aqav2 interest history ({symbol}): fresh through {data['history'][-1][0]}, kept")
        return data
    body = {'type': 'aqav2InterestHistory', 'token': token}
    rows = _info(body)
    if rows is None:
        print(f"aqav2 interest history ({symbol}): token {token} is not an AQAv2 asset (API returned null)")
        return None
    ok = isinstance(rows, list) and all(isinstance(r, list) and len(r) == 2 and isinstance(r[0], str) and isinstance(r[1], dict) for r in rows)
    if not ok or not rows:
        raise ValueError(f"aqav2InterestHistory changed shape (paginated?): {str(rows)[:120]}")
    rows.sort(key=lambda r: r[0])
    data = {'_pulled_ts': int(time.time()), 'request': body, 'token': token, 'symbol': symbol, 'history': rows}
    save_json(data, path)
    n_active = sum(1 for _, r in rows if r.get('active'))
    print(f"aqav2 interest history ({symbol}): pulled {len(rows)} rows, {rows[0][0]}..{rows[-1][0]} ({n_active} active)")
    return data


@report_errors()
def pull_aqav2_payments(output_dir, token=0, start_date=None, max_stale_days=DEFAULT_MAX_STALE_DAYS, force=False):
    """Save every ledger update of the system interest address (``userNonFundingLedgerUpdates``), as returned.

    Interval payments appear here as transfers to the Assistance Fund. ``start_date`` defaults to
    the day before activation. The file is kept when it was pulled within ``max_stale_days`` of
    today UTC. Returns the saved dict.
    """
    symbol = TOKENS[token]
    path = _ledger_path(output_dir, token)
    if not force and is_fresh(path, lambda d: d.get('_pulled_ts'), max_stale_days):
        data = _load_json(path)
        print(f"aqav2 payments ({symbol}): ledger fresh (pulled {_age(data.get('_pulled_ts'))}), kept")
        return data
    activation = datetime.fromisoformat(AQAV2_ACTIVATION[symbol]).date()
    start = datetime.fromisoformat(start_date).date() if start_date else activation - timedelta(days=1)
    user = system_interest_address(token)
    body = {'type': 'userNonFundingLedgerUpdates', 'user': user, 'startTime': _day_ts(start.isoformat()) * 1000}
    updates = _info(body)
    if not isinstance(updates, list):
        raise ValueError(f"userNonFundingLedgerUpdates changed shape: {str(updates)[:120]}")
    data = {'_pulled_ts': int(time.time()), 'request': body, 'user': user, 'token': token, 'symbol': symbol,
            'assistance_fund': ASSISTANCE_FUND, 'updates': updates}
    save_json(data, path)
    n_pay = len(aqav2_payments(data, token))
    today = _today()
    k = 1
    while period_bounds(k, activation)[2].date() < today:
        k += 1
    due = period_bounds(k, activation)[2].date()
    print(f"aqav2 payments ({symbol}): {len(updates)} ledger update(s), {n_pay} payment(s) to the Assistance Fund; "
          f"period {k} due {due}")
    return data


def _archive_rpc_urls(rpc_url=None):
    urls = [rpc_url or os.getenv('HYPEREVM_ARCHIVE_RPC_URL')] + list(ARCHIVE_RPCS)
    urls = list(dict.fromkeys(u.rstrip('/') for u in urls if u))
    if NON_ARCHIVE_RPC in urls:
        raise ValueError(f"{NON_ARCHIVE_RPC} is not an archive node (it returns latest state for old blocks); "
                         f"use one of {ARCHIVE_RPCS} or set HYPEREVM_ARCHIVE_RPC_URL")
    return urls


def _evm_rpc(url, method, params, retries=3):
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(url, json={'jsonrpc': '2.0', 'id': 1, 'method': method, 'params': params}, timeout=EVM_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            if 'result' in data and data['result'] not in (None, '0x'):
                return data['result']
            last = RuntimeError(f"{url}: {data.get('error') or 'empty result'}")
        except (requests.RequestException, ValueError) as e:
            last = e
        time.sleep(1.5 * (attempt + 1))
    raise last


def _evm_balance(url, token_addr, holder, block):
    data = '0x70a08231' + holder.lower().replace('0x', '').rjust(64, '0')
    return int(_evm_rpc(url, 'eth_call', [{'to': token_addr, 'data': data}, hex(block) if isinstance(block, int) else block]), 16) / 1e6


def _evm_total_supply(url, token_addr, block):
    return int(_evm_rpc(url, 'eth_call', [{'to': token_addr, 'data': '0x18160ddd'}, hex(block) if isinstance(block, int) else block]), 16) / 1e6


def _evm_block_ts(url, n):
    return int(_evm_rpc(url, 'eth_getBlockByNumber', [hex(n), False])['timestamp'], 16)


def _pick_archive_rpc(urls):
    """First URL that provably serves history: native USDC total supply at the check block must differ from latest."""
    failures = []
    for url in urls:
        try:
            old = _evm_total_supply(url, USDC_EVM, ARCHIVE_CHECK_BLOCK)
            new = _evm_total_supply(url, USDC_EVM, 'latest')
            if abs(old - new) > 1:
                return url
            failures.append(f"{url}: returns latest state for block {ARCHIVE_CHECK_BLOCK} (not archive)")
        except Exception as e:
            failures.append(f"{url}: {error_code(e)}{type(e).__name__}: {e}")
    raise RuntimeError("no archive RPC available: " + '; '.join(failures))


def _midnight_block(url, day, latest, latest_ts, blocks_per_sec):
    """First block whose timestamp is >= 00:00 UTC on ``day`` (block headers are reliable on any node)."""
    target = _day_ts(day)
    est = int(latest - (latest_ts - target) * blocks_per_sec)
    lo, hi = max(1, est - 30000), min(latest, est + 30000)
    while lo > 1 and _evm_block_ts(url, lo) >= target:
        lo = max(1, lo - 60000)
    while hi < latest and _evm_block_ts(url, hi) < target:
        hi = min(latest, hi + 60000)
    while lo < hi:
        mid = (lo + hi) // 2
        if _evm_block_ts(url, mid) < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


@report_errors()
def pull_treasury_balances(output_dir, token=0, rpc_url=None, dates=None):
    """Read the treasury address's native-USDC balance (plus the linked contract's and the total
    supply) at the first HyperEVM block at/after 00:00 UTC for every date in the saved interest
    history, through an archive RPC. Append-only: dates already on file are never re-read.

    This is the independent recreation of ``usdcSupply``: the two agree to the cent on almost
    every day. ``dates`` overrides the date list (ISO strings). Returns the saved dict.
    """
    symbol = TOKENS[token]
    path = _treasury_path(output_dir, token)
    if dates is None:
        hist_path = _interest_path(output_dir, token)
        if not os.path.exists(hist_path):
            print(f"treasury balances ({symbol}): no interest history on file yet, nothing to tie out")
            return None
        dates = [d for d, _ in _load_json(hist_path)['history']]
    data = _load_json(path) if os.path.exists(path) else {'rows': []}
    have = {r['date'] for r in data['rows']}
    todo = sorted(d for d in dates if d not in have)
    if not todo:
        print(f"treasury balances ({symbol}): {len(have)} dates on file, none missing, kept")
        return data
    url = _pick_archive_rpc(_archive_rpc_urls(rpc_url))
    # Block headers are identical on every node and the public one answers in a fraction of the
    # archive node's time, so the midnight-block search runs there; only the state reads need the archive.
    header_url = NON_ARCHIVE_RPC
    try:
        latest = int(_evm_rpc(header_url, 'eth_blockNumber', []), 16)
        latest_ts = _evm_block_ts(header_url, latest)
    except Exception:
        header_url = url
        latest = int(_evm_rpc(url, 'eth_blockNumber', []), 16)
        latest_ts = _evm_block_ts(url, latest)
    ref_ts = _day_ts(REF_BLOCK_DAY)
    blocks_per_sec = (latest - REF_BLOCK) / (latest_ts - ref_ts) if latest_ts > ref_ts + 3600 else 1.0166

    def read(day):
        n = _midnight_block(header_url, day, latest, latest_ts, blocks_per_sec)
        return {'date': day, 'block': n,
                'treasury': _evm_balance(url, USDC_EVM, TREASURY[symbol], n),
                'linked': _evm_balance(url, USDC_EVM, LINKED[symbol], n),
                'evm_total': _evm_total_supply(url, USDC_EVM, n)}

    results = run_parallel(todo, read, max_workers=TREASURY_PULL_WORKERS, desc='treasury balances')
    failed = [d for d, r in zip(todo, results) if r is None]
    data['rows'].extend(r for r in results if r is not None)
    data['rows'].sort(key=lambda r: r['date'])
    data.update({'_pulled_ts': int(time.time()), 'rpc': url, 'token': token, 'symbol': symbol,
                 'usdc_evm': USDC_EVM, 'treasury': TREASURY[symbol], 'linked': LINKED[symbol]})
    save_json(data, path)
    print(f"treasury balances ({symbol}): read {len(todo) - len(failed)} new date(s) from {url} "
          f"({len(data['rows'])} on file, {data['rows'][0]['date']}..{data['rows'][-1]['date']})"
          + (f"; FAILED {len(failed)}: {', '.join(failed)} (rerun to retry)" if failed else ''))
    return data


def load_interest_history(output_dir, token=0):
    return _load_json(_interest_path(output_dir, token))


def load_ledger(output_dir, token=0):
    path = _ledger_path(output_dir, token)
    return _load_json(path) if os.path.exists(path) else {}


def load_treasury_balances(output_dir, token=0):
    path = _treasury_path(output_dir, token)
    return _load_json(path) if os.path.exists(path) else {}


# ---------------------------------------------------------------------------
# 2. Frames: daily rows, payments, periods
# ---------------------------------------------------------------------------

def aqav2_daily(raw, treasury=None):
    """One row per published UTC date: the official numbers as floats, the formula check, the period
    each active day belongs to and, when ``treasury`` (load_treasury_balances) is given, the
    on-chain balances read at midnight and their difference to the charged base."""
    symbol = raw.get('symbol', 'USDC')
    df = pd.DataFrame([{'date': d, 'usdc_supply': float(r['usdcSupply']), 'effective_rate': float(r['effectiveRate']),
                        'interest': float(r['interest']), 'active': bool(r.get('active'))} for d, r in raw['history']])
    df['date'] = pd.to_datetime(df['date'])
    if df['date'].duplicated().any():
        raise ValueError('duplicate dates in aqav2InterestHistory')
    df = df.sort_values('date').reset_index(drop=True)
    df['interest_check'] = df['usdc_supply'] * df['effective_rate'] / DAYS_PER_YEAR
    df['formula_diff'] = df['interest'] - df['interest_check']
    active = df[df['active']]
    if active.empty:
        raise ValueError('no active rows in aqav2InterestHistory')
    activation = active['date'].iloc[0]
    expected = pd.Timestamp(AQAV2_ACTIVATION[symbol])
    if activation != expected:
        raise ValueError(f"first active date {activation.date()} != documented activation {expected.date()}")
    days = (df['date'] - activation).dt.days
    df['period'] = pd.array([int(v // PERIOD_DAYS) + 1 if a and v >= 0 else None for v, a in zip(days, df['active'])], dtype='Int64')
    df['day_in_period'] = pd.array([int(v % PERIOD_DAYS) + 1 if a and v >= 0 else None for v, a in zip(days, df['active'])], dtype='Int64')
    df['payment_date'] = [period_bounds(int(k), activation)[2] if pd.notna(k) else pd.NaT for k in df['period']]
    df.attrs['activation'] = activation
    df.attrs['symbol'] = symbol
    if treasury and treasury.get('rows'):
        t = pd.DataFrame(treasury['rows'])
        t['date'] = pd.to_datetime(t['date'])
        t = t.rename(columns={'treasury': 'treasury_balance', 'linked': 'linked_balance', 'block': 'evm_block'})
        df = df.merge(t[['date', 'evm_block', 'treasury_balance', 'linked_balance', 'evm_total']], on='date', how='left')
        df['base_diff'] = df['treasury_balance'] - df['usdc_supply']
        df['treasury_share'] = df['treasury_balance'] / (df['treasury_balance'] + df['linked_balance'])
        df.attrs['activation'] = activation
        df.attrs['symbol'] = symbol
    return df


def aqav2_payments(raw_ledger, token=0):
    """Transfers from the system interest address to the Assistance Fund, one row each
    (``time`` UTC, ``date``, ``hash``, ``type``, ``amount``). Matched on destination and token, not
    on the update type, which is unknown until the first payment lands."""
    symbol = TOKENS[token]
    rows = []
    for u in (raw_ledger or {}).get('updates', []):
        delta = u.get('delta') or {}
        if str(delta.get('destination', '')).lower() != ASSISTANCE_FUND or delta.get('token') != symbol:
            continue
        ts = pd.Timestamp(u['time'], unit='ms', tz='UTC')
        rows.append({'time': ts, 'date': ts.tz_localize(None).normalize(), 'hash': u.get('hash'),
                     'type': delta.get('type'), 'amount': float(delta.get('amount', 0))})
    return pd.DataFrame(rows, columns=['time', 'date', 'hash', 'type', 'amount'])


def aqav2_periods(daily, payments=None, today=None):
    """One row per 30-day period from activation to the period containing the latest active day:
    dates, days sampled, accrued interest (sum of the official daily interest), average base and
    rate, and the actual payment once the ledger shows it. Index = period number."""
    activation = daily.attrs['activation']
    today = pd.Timestamp(today or _today())
    active = daily[daily['active']]
    n_periods = int(active['period'].max())
    payments = payments if payments is not None else pd.DataFrame(columns=['time', 'date', 'hash', 'type', 'amount'])
    used = set()
    rows = []
    for k in range(1, n_periods + 1):
        start, end, pay = period_bounds(k, activation)
        rows_k = active[active['period'] == k]
        window = payments[(payments['date'] >= pay - pd.Timedelta(days=3)) & (payments['date'] <= pay + pd.Timedelta(days=3))]
        used |= set(window.index)
        paid = float(window['amount'].sum()) if not window.empty else None
        accrued = float(rows_k['interest'].sum())
        if paid is not None:
            status = 'paid'
        elif len(rows_k) < PERIOD_DAYS:
            status = 'in-progress'
        elif today >= pay:
            status = 'overdue'
        else:
            status = 'accrued-awaiting-payment'
        rows.append({'period': k, 'start': start, 'end': end, 'payment_date': pay, 'days_sampled': len(rows_k),
                     'status': status, 'interest_accrued': accrued,
                     'avg_supply': float(rows_k['usdc_supply'].mean()) if len(rows_k) else None,
                     'avg_rate': float(rows_k['effective_rate'].mean()) if len(rows_k) else None,
                     'actual_paid': paid, 'paid_date': window['date'].min() if paid is not None else pd.NaT,
                     'difference': (paid - accrued) if paid is not None else None,
                     'payment_tx': ', '.join(window['hash'].astype(str)) if paid is not None else None})
    out = pd.DataFrame(rows).set_index('period')
    out.attrs['unmatched_payments'] = payments[~payments.index.isin(used)]
    out.attrs['activation'] = activation
    return out


# ---------------------------------------------------------------------------
# 3. Reference rate, pass-through, projection
# ---------------------------------------------------------------------------

def load_reference_rate(fred_path):
    """A FRED observations file (fred.pull_series) as a daily Series of fractions indexed by calendar
    date, forward-filled over weekends, holidays and the publication lag through today UTC."""
    obs = _load_json(fred_path)['observations']
    s = pd.Series({pd.Timestamp(o['date']): float(o['value']) / 100 for o in obs if o.get('value') not in (None, '.', '')})
    s = s.sort_index()
    idx = pd.date_range(s.index.min(), max(s.index.max(), pd.Timestamp(_today())), freq='D')
    return s.reindex(idx).ffill()


def _trailing_mean(ref, end_date, window):
    """Mean of ``ref`` over the ``window`` days ending the day before ``end_date`` (the rate for a date
    is published the evening before, so it can only reflect reference prints up to then)."""
    stop = end_date - pd.Timedelta(days=1)
    start = stop - pd.Timedelta(days=window - 1)
    w = ref.loc[start:stop]
    return float(w.mean()) if len(w) == window else None


def calibrate_passthrough(daily, ref, cost_bp=FIXED_COST_BP, passthrough_days=None, quiet=False):
    """Fit the lag of the rate model ``aqa_t = mean(ref over the L days before t) - cost``.

    ``cost_bp`` is the user's fixed haircut (never fitted). ``L`` is grid-searched over
    PASSTHROUGH_GRID by least squares on every published day unless ``passthrough_days`` is
    given. Also reports ``implied_cost_bp``, the reference-minus-AQA gap over the days before the
    reference first moved, so drift from the assumed cost is visible.
    """
    cost = cost_bp / 1e4
    rows = daily[['date', 'effective_rate']]
    def sse(L):
        errs = [(float(r) - (m - cost)) for d, r in zip(rows['date'], rows['effective_rate'])
                for m in [_trailing_mean(ref, d, L)] if m is not None]
        return (sum(e * e for e in errs) / len(errs)) if errs else float('inf'), len(errs)
    if passthrough_days is None:
        fits = {L: sse(L) for L in PASSTHROUGH_GRID}
        L = min(fits, key=lambda k: fits[k][0])
    else:
        L = int(passthrough_days)
        fits = {L: sse(L)}
    mse, n = fits[L]
    rmse_bp = mse ** 0.5 * 1e4 if n else None
    ref_prev = [float(ref.loc[d - pd.Timedelta(days=1)]) if (d - pd.Timedelta(days=1)) in ref.index else None for d in rows['date']]
    first_ref = next((v for v in ref_prev if v is not None), None)
    steady = [(v - r) for v, r in zip(ref_prev, rows['effective_rate']) if v is not None and v == first_ref]
    implied_cost_bp = (sum(steady) / len(steady) * 1e4) if steady else None
    latest_date = rows['date'].iloc[-1]
    latest_actual = float(rows['effective_rate'].iloc[-1])
    m = _trailing_mean(ref, latest_date, L)
    latest_model = (m - cost) if m is not None else None
    latest_ref = float(ref.iloc[-1])
    out = {'cost_bp': cost_bp, 'implied_cost_bp': implied_cost_bp, 'passthrough_days': L, 'rmse_bp': rmse_bp,
           'n_days': n, 'latest_ref': latest_ref, 'latest_model_rate': latest_model, 'latest_actual_rate': latest_actual,
           'steady_state_rate': latest_ref - cost, 'fitted': passthrough_days is None}
    if not quiet:
        imp = f"{implied_cost_bp:.1f}bp" if implied_cost_bp is not None else 'n/a'
        print(f"AQA rate model: EFFR trailing {L}d mean - {cost_bp}bp (implied pre-move cost {imp}); "
              f"RMSE {rmse_bp:.1f}bp over {n} days; latest model {latest_model:.3%} vs actual {latest_actual:.3%}; "
              f"steady state at EFFR {latest_ref:.2%} = {latest_ref - cost:.3%}")
    return out


def sensitivity_per_25bp(daily, quiet=False):
    """What a 25bp move in the reference is worth at the latest charged balance under a fixed cost
    (full pass-through once the lag has elapsed)."""
    latest = daily[daily['active']].iloc[-1]
    base, rate = float(latest['usdc_supply']), float(latest['effective_rate'])
    per_year = base * 0.0025
    out = {'rate_bp': 25.0, 'base': base, 'per_day': per_year / DAYS_PER_YEAR, 'per_period': per_year / DAYS_PER_YEAR * PERIOD_DAYS,
           'per_year': per_year, 'pct_of_current': 0.0025 / rate}
    if not quiet:
        print(f"per +25bp in EFFR at the {latest['date'].date()} base of {_fmt_usd(base)}: {_fmt_usd(out['per_day'])}/day, "
              f"{_fmt_usd(out['per_period'])}/period, {_fmt_usd(out['per_year'])}/yr ({out['pct_of_current']:+.1%} of the current rate)")
    return out


def fed_path(bp_per_meeting, n_meetings, start=None):
    """``{decision_date: change_in_bp}`` for the next ``n_meetings`` FOMC decisions after ``start`` (today UTC)."""
    start = pd.Timestamp(start or _today())
    dates = [d for d in FOMC_DECISION_DATES if pd.Timestamp(d) > start][:n_meetings]
    return {d: bp_per_meeting for d in dates}


def _resolve_fed_path(spec, today):
    if spec is None:
        return {}
    if isinstance(spec, dict) and 'bp_per_meeting' in spec:
        return fed_path(spec['bp_per_meeting'], spec.get('n_meetings', 1), start=today)
    return {str(pd.Timestamp(k).date()): v for k, v in dict(spec).items()}


def project_reference(ref, fed_path_dict, end_date):
    """Actual reference history extended day by day to ``end_date`` at its last print, each Fed move
    (``{decision_date: bp}``) applied the day after its decision date."""
    last_date, level = ref.index[-1], float(ref.iloc[-1])
    idx = pd.date_range(last_date + pd.Timedelta(days=1), pd.Timestamp(end_date), freq='D')
    moves = {pd.Timestamp(d) + pd.Timedelta(days=1): bp / 1e4 for d, bp in (fed_path_dict or {}).items()}
    vals = []
    for d in idx:
        level += moves.get(d, 0.0)
        vals.append(level)
    return pd.concat([ref, pd.Series(vals, index=idx)]) if len(idx) else ref.copy()


def project_rate(ref_full, cost_bp, passthrough_days):
    """The modelled AQA rate for every date in ``ref_full``: trailing mean of the reference over the
    ``passthrough_days`` before the date, minus the fixed cost."""
    cost = cost_bp / 1e4
    vals = {}
    for d in ref_full.index:
        m = _trailing_mean(ref_full, d, passthrough_days)
        if m is not None:
            vals[d] = m - cost
    return pd.Series(vals)


# ---------------------------------------------------------------------------
# 4. Estimate future periods
# ---------------------------------------------------------------------------

def estimate_aqav2_payments(daily, ref, n_periods=12, fed_path=None, cost_bp=FIXED_COST_BP, passthrough_days=None,
                            supply='latest', treasury_cagr=0.0, fixed_rate=None, scenarios=None, today=None):
    """Estimated interest per 30-day period, from the in-progress period through ``n_periods`` periods.

    Days the node has already published are taken as actual. Every remaining UTC date ``t`` is
    ``S_t * r_t / 365.25`` where ``S_t`` is the latest treasury balance grown at ``treasury_cagr``
    (annual, compounded daily; default flat) or a float / callable ``supply``, and ``r_t`` is the
    rate model: trailing ``passthrough_days`` mean of the reference (extended along ``fed_path``,
    ``{decision_date: bp}`` or ``{'bp_per_meeting': -25, 'n_meetings': 4}``) minus ``cost_bp``.
    ``passthrough_days=None`` fits the lag with calibrate_passthrough(). ``fixed_rate`` (a fraction)
    bypasses the reference entirely. ``scenarios`` maps a name to overrides of any of these
    knobs and yields one block of rows per scenario.

    Returns a long DataFrame, one row per (scenario, period), with ``attrs['assumptions']``.
    """
    today = pd.Timestamp(today or _today())
    activation = daily.attrs['activation']
    active = daily[daily['active']]
    latest = active.iloc[-1]
    latest_date, latest_base, latest_rate = latest['date'], float(latest['usdc_supply']), float(latest['effective_rate'])
    k0 = int(latest['period'])
    if passthrough_days is None and fixed_rate is None:
        passthrough_days = calibrate_passthrough(daily, ref, cost_bp=cost_bp, quiet=True)['passthrough_days']
    end_date = period_bounds(k0 + n_periods - 1, activation)[2]
    base_defaults = {'fed_path': fed_path, 'cost_bp': cost_bp, 'passthrough_days': passthrough_days, 'supply': supply,
                     'treasury_cagr': treasury_cagr, 'fixed_rate': fixed_rate}
    scenario_specs = scenarios or {'Estimate': {}}
    rows, assumptions = [], {'latest_date': str(latest_date.date()), 'latest_base': latest_base, 'latest_rate': latest_rate,
                             'latest_ref': float(ref.iloc[-1]) if ref is not None and len(ref) else None,
                             'cost_bp': cost_bp, 'passthrough_days': passthrough_days, 'scenarios': {}}
    for name, spec in scenario_specs.items():
        p = {**base_defaults, **(spec or {})}
        path = _resolve_fed_path(p['fed_path'], today)
        if p['fixed_rate'] is not None:
            rate_at = lambda d, r=float(p['fixed_rate']): r
        else:
            ref_full = project_reference(ref, path, end_date)
            rate_series = project_rate(ref_full, p['cost_bp'], p['passthrough_days'])
            rate_at = lambda d, s=rate_series: float(s.get(d, latest_rate))
        if callable(p['supply']):
            base_at = p['supply']
        else:
            s0 = latest_base if p['supply'] == 'latest' else float(p['supply'])
            g = float(p['treasury_cagr'] or 0.0)
            base_at = lambda d, s0=s0, g=g: s0 * (1 + g) ** ((d - latest_date).days / DAYS_PER_YEAR)
        assumptions['scenarios'][name] = {'fed_path': path, 'cost_bp': p['cost_bp'], 'passthrough_days': p['passthrough_days'],
                                          'supply': p['supply'] if not callable(p['supply']) else 'callable',
                                          'treasury_cagr': p['treasury_cagr'], 'fixed_rate': p['fixed_rate']}
        for k in range(k0, k0 + n_periods):
            start, end, pay = period_bounds(k, activation)
            actual_rows = active[active['period'] == k]
            actual = float(actual_rows['interest'].sum())
            days_actual = len(actual_rows)
            projected, bases, rates = 0.0, list(actual_rows['usdc_supply']), list(actual_rows['effective_rate'])
            refs = []
            for d in pd.date_range(start, end, freq='D'):
                if d <= latest_date:
                    continue
                S, r = base_at(d), rate_at(d)
                projected += S * r / DAYS_PER_YEAR
                bases.append(S)
                rates.append(r)
                if p['fixed_rate'] is None:
                    refs.append(float(ref_full.get(d, float('nan'))))
            rows.append({'scenario': name, 'period': k, 'start': start, 'end': end, 'payment_date': pay,
                         'days_actual': days_actual, 'days_projected': PERIOD_DAYS - days_actual,
                         'interest_actual': actual, 'interest_projected': projected, 'interest_total': actual + projected,
                         'avg_supply_assumed': sum(bases) / len(bases), 'avg_rate_assumed': sum(rates) / len(rates),
                         'avg_reference_assumed': (sum(refs) / len(refs)) if refs else None,
                         'status': 'in-progress' if days_actual else 'projected'})
    out = pd.DataFrame(rows)
    out.attrs['assumptions'] = assumptions
    return out


# ---------------------------------------------------------------------------
# 5. Checks and the orchestrator
# ---------------------------------------------------------------------------

def check_aqav2(daily, periods, payments=None, tolerance=1e-6, quiet=False):
    """Tie-outs. Returns the list of discrepancies (empty when clean) and prints one summary line.

    Formula identity on every active day; activation date and gap-free calendar; the documented
    period boundaries; the on-chain treasury balance against the charged base where read; the 9:1
    split; each paid period against its accrued total; unmatched or overdue payments.
    """
    issues = []
    activation = daily.attrs['activation']
    active = daily[daily['active']]
    worst = float(active['formula_diff'].abs().max())
    if worst >= tolerance:
        issues.append(f"interest != supply x rate / {DAYS_PER_YEAR} (max diff {worst:.2e})")
    gaps = active['date'].diff().dt.days.dropna()
    if (gaps != 1).any():
        issues.append(f"gaps in active dates: {int((gaps != 1).sum())}")
    b1, b2 = period_bounds(1, activation), period_bounds(2, activation)
    exp1 = tuple(pd.Timestamp(d) for d in ('2026-08-27', '2026-09-25', '2026-10-03'))
    exp2 = tuple(pd.Timestamp(d) for d in ('2026-09-26', '2026-10-25', '2026-11-02'))
    if daily.attrs.get('symbol') == 'USDC' and (b1 != exp1 or b2 != exp2):
        issues.append(f"period boundaries {b1} / {b2} differ from the documented USDC schedule")
    base_note = ''
    if 'base_diff' in daily.columns and daily['base_diff'].notna().any():
        read = daily[daily['base_diff'].notna()]
        off = read[read['base_diff'].abs() > BASE_TOLERANCE_USDC]
        exact = int((read['base_diff'].abs() < 0.005).sum())
        for _, r in off.iterrows():
            issues.append(f"{r['date'].date()}: treasury balance {r['treasury_balance']:,.2f} vs usdcSupply {r['usdc_supply']:,.2f} ({r['base_diff']:+,.2f})")
        bad_share = read[(read['treasury_share'] - 0.9).abs() > 1e-6]
        if not bad_share.empty:
            issues.append(f"treasury share of treasury+linked not 0.9 on {len(bad_share)} day(s)")
        base_note = f"; base = treasury balance at midnight on all {len(read)} days read ({exact} exact to the cent)"
    for k, r in periods.iterrows():
        if r['status'] == 'paid' and abs(r['difference']) > PAYMENT_TOLERANCE_USDC:
            issues.append(f"period {k}: paid {r['actual_paid']:,.2f} vs accrued {r['interest_accrued']:,.2f} ({r['difference']:+,.2f})")
        if r['status'] == 'overdue':
            issues.append(f"period {k}: due {r['payment_date'].date()}, no payment to the Assistance Fund seen")
    unmatched = periods.attrs.get('unmatched_payments')
    if unmatched is not None and len(unmatched):
        issues.append(f"{len(unmatched)} payment(s) to the Assistance Fund match no period: "
                      + ', '.join(f"{d.date()} {a:,.2f}" for d, a in zip(unmatched['date'], unmatched['amount'])))
    if not quiet:
        paid = int((periods['status'] == 'paid').sum())
        summary = (f"aqav2 ({daily.attrs.get('symbol', 'USDC')}): {len(active)} active days in {len(periods)} period(s), {paid} paid; "
                   f"identity holds to {worst:.1e} on every day{base_note}")
        print(summary + (f"; TIE-OUT: {len(issues)} issue(s)" if issues else "; tie-out clean"))
        for i in issues:
            print(f"    {i}")
    return issues


@report_errors()
def refresh(output_dir, token=0, max_stale_days=DEFAULT_MAX_STALE_DAYS, rpc_url=None):
    """Pull the daily series, the payment ledger and the treasury balances (each skipped when fresh),
    then build the period table and print the tie-outs. Returns the periods DataFrame."""
    pull_aqav2_interest_history(output_dir, token=token, max_stale_days=max_stale_days)
    pull_aqav2_payments(output_dir, token=token, max_stale_days=max_stale_days)
    pull_treasury_balances(output_dir, token=token, rpc_url=rpc_url)
    daily = aqav2_daily(load_interest_history(output_dir, token), load_treasury_balances(output_dir, token))
    payments = aqav2_payments(load_ledger(output_dir, token), token)
    periods = aqav2_periods(daily, payments)
    check_aqav2(daily, periods, payments)
    return periods
