"""MetaDAO launchpad: ICO prices from on-chain launch accounts, launch-window prices from
MetaDAO's market API (https://market-api.metadao.fi).

How a MetaDAO sale is priced
----------------------------
Every participant pays the same price. The launchpad program mints a fixed
``TOKENS_TO_PARTICIPANTS = 10,000,000`` tokens to funders as a group (a constant in the
program source for v0.6, v0.7 and v0.8), split pro rata to the USDC each wallet had
accepted. So ``ico_price_usd = usdc_accepted / 10,000,000``. The accepted amount can be
below the amount committed: founders set a discretionary cap and the rest is refunded
(Umbra: $154.9M committed, $3.0M accepted, $0.30 per token). It is read from each
launch's on-chain ``Launch`` account: ``total_approved_amount`` on v0.7/v0.8,
``final_raise_amount`` on v0.6, and ``total_committed_amount`` on v0.4/v0.5 (no cap).

Price history
-------------
``/dexscreener/events?fromBlock=&toBlock=`` returns every swap on the MetaDAO spot AMM
(``pairId``, token amounts ``asset0In/Out``, USDC amounts ``asset1In/Out``, block
timestamp), at most 500,000 slots (~2.3 days) per request, with history back to the
first launches in October 2025. Hourly VWAP = USDC volume / token volume per hour.
Daily VWAP rebuilt from these swaps matches MetaDAO's own daily endpoint
(``/api/market-data``) to four decimals, which is the tie-out ``build_launch_prices``
prints. Both endpoints are undocumented: the raw responses are saved unchanged so a
change upstream never silently changes an existing chart.

Files
-----
raw   data/raw/metadao/launches/{version}.json   getProgramAccounts response, as returned
      data/raw/metadao/tickers.json               /api/tickers
      data/raw/metadao/assets/{mint}.json         /dexscreener/asset?id=
      data/raw/metadao/pairs/{pair_id}.json       /dexscreener/pair?id=
      data/raw/metadao/swaps/{mint}.json          swap events for the launch pair, first n days
      data/raw/metadao/market-data.json           /api/market-data (daily, tie-out only)
config config/metadao_launches.json               hand-editable; filled only where missing
processed data/processed/metadao/launch-prices-{n}d.json  what the chart reads
"""
import base64
import hashlib
import json
import os
import struct
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.helper_functions import (fetch_and_save_json, save_json, is_fresh, pull_progress,
                                    report_pull, report_errors, error_code)

BASE_URL = 'https://market-api.metadao.fi'
# The API answers 403 to requests without a User-Agent.
HEADERS = {'User-Agent': 'dd-tools research notebook', 'Accept': 'application/json'}
# Documented limit: 60 requests per minute per IP.
SECONDS_BETWEEN_CALLS = 1.05
MAX_SLOTS_PER_REQUEST = 500_000
SLOT_SECONDS = 0.4  # first estimate only; every slot lookup is corrected from event timestamps

DEFAULT_RPC_URL = 'https://api.mainnet-beta.solana.com'
RPC_TIMEOUT = 120
# Earliest day the swap-events feed covers; launches that closed before it have no price history.
EVENTS_HISTORY_START = '2025-10-09'
DEFAULT_MAX_STALE_DAYS = 1

TOKENS_TO_PARTICIPANTS = 10_000_000
USDC_DECIMALS = 6
DEFAULT_MIN_ACCEPTED_USD = 100_000

# Launchpad program IDs, from the SDK's sdk/src/constants.ts.
PROGRAMS = {
    'v0.4': 'AfJJJ5UqxhBKoE3grkKAZZsoXDE9kncbMKvqSHGsCNrE',
    'v0.5': 'mooNhciQJi1LqHDmse2JPic2NqG2PXCanbE3ZYzP3qA',
    'v0.6': 'MooNyh4CBUYEKyXVnjGYQ8mEiJDpGvJMdvrZx1iGeHV',
    'v0.7': 'moontUzsdepotRGe5xsfip7vLPTJnVuafqdUWexVnPM',
    'v0.8': 'moonDJUoHteKkGATejA5bdJVwJ6V6Dg74gyqyJTx73n',
}
# Anchor account discriminator: first 8 bytes of sha256("account:Launch").
LAUNCH_DISCRIMINATOR = hashlib.sha256(b'account:Launch').digest()[:8]

# Borsh layouts of the Launch account per program version, in declaration order, taken
# from the SDK IDLs (sdk/src/launchpad/v0.X/types/*.ts). Types: u8, u32, u64, i64, bool,
# pubkey, opt:<type>, vec:pubkey, state (u8 enum index into LAUNCH_STATES).
_V06_HEAD = [
    ('pda_bump', 'u8'), ('minimum_raise_amount', 'u64'), ('monthly_spending_limit_amount', 'u64'),
    ('monthly_spending_limit_members', 'vec:pubkey'), ('launch_authority', 'pubkey'),
    ('launch_signer', 'pubkey'), ('launch_signer_pda_bump', 'u8'), ('launch_quote_vault', 'pubkey'),
    ('launch_base_vault', 'pubkey'), ('base_mint', 'pubkey'), ('quote_mint', 'pubkey'),
    ('unix_timestamp_started', 'opt:i64'), ('unix_timestamp_closed', 'opt:i64'),
    ('total_committed_amount', 'u64'),
]
_V06_TAIL = [
    ('state', 'state'), ('seq_num', 'u64'), ('seconds_for_launch', 'u32'), ('dao', 'opt:pubkey'),
    ('dao_vault', 'opt:pubkey'), ('performance_package_grantee', 'pubkey'),
    ('performance_package_token_amount', 'u64'), ('months_until_insiders_can_unlock', 'u8'),
    ('team_address', 'pubkey'),
]
_V07_EXTRA = [
    ('total_approved_amount', 'u64'), ('additional_tokens_amount', 'u64'),
    ('additional_tokens_recipient', 'opt:pubkey'), ('additional_tokens_claimed', 'bool'),
    ('unix_timestamp_completed', 'opt:i64'),
]
LAYOUTS = {
    'v0.4': [
        ('pda_bump', 'u8'), ('minimum_raise_amount', 'u64'), ('launch_authority', 'pubkey'),
        ('launch_signer', 'pubkey'), ('launch_signer_pda_bump', 'u8'), ('launch_usdc_vault', 'pubkey'),
        ('launch_token_vault', 'pubkey'), ('base_mint', 'pubkey'), ('quote_mint', 'pubkey'),
        ('unix_timestamp_started', 'i64'), ('total_committed_amount', 'u64'), ('state', 'state'),
        ('seq_num', 'u64'), ('seconds_for_launch', 'u32'), ('dao', 'opt:pubkey'), ('dao_vault', 'opt:pubkey'),
    ],
    'v0.5': [
        ('pda_bump', 'u8'), ('minimum_raise_amount', 'u64'), ('monthly_spending_limit_amount', 'u64'),
        ('monthly_spending_limit_members', 'vec:pubkey'), ('launch_authority', 'pubkey'),
        ('launch_signer', 'pubkey'), ('launch_signer_pda_bump', 'u8'), ('launch_quote_vault', 'pubkey'),
        ('launch_base_vault', 'pubkey'), ('base_mint', 'pubkey'), ('quote_mint', 'pubkey'),
        ('unix_timestamp_started', 'i64'), ('total_committed_amount', 'u64'), ('state', 'state'),
        ('seq_num', 'u64'), ('seconds_for_launch', 'u32'), ('dao', 'opt:pubkey'), ('dao_vault', 'opt:pubkey'),
    ],
    'v0.6': _V06_HEAD + [('final_raise_amount', 'opt:u64')] + _V06_TAIL,
    'v0.7': _V06_HEAD + _V06_TAIL + _V07_EXTRA + [
        ('is_performance_package_initialized', 'bool'), ('accumulator_activation_delay_seconds', 'u32'),
        ('has_bid_wall', 'bool'),
    ],
    'v0.8': _V06_HEAD + _V06_TAIL + _V07_EXTRA + [
        ('is_finalized', 'bool'), ('accumulator_activation_delay_seconds', 'u32'),
        ('has_bid_wall', 'bool'), ('mint_governor', 'pubkey'),
    ],
}
LAUNCH_STATES = {
    'v0.4': ['Initialized', 'Live', 'Complete', 'Refunding'],
    'v0.5': ['Initialized', 'Live', 'Complete', 'Refunding'],
    'v0.6': ['Initialized', 'Live', 'Closed', 'Complete', 'Refunding'],
    'v0.7': ['Initialized', 'Live', 'Closed', 'Complete', 'Refunding'],
    'v0.8': ['Initialized', 'Live', 'Closed', 'Complete', 'Refunding'],
}

_B58 = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'


def _b58encode(raw):
    n = int.from_bytes(raw, 'big')
    out = ''
    while n:
        n, r = divmod(n, 58)
        out = _B58[r] + out
    return '1' * (len(raw) - len(raw.lstrip(b'\0'))) + out


def _get(path, params=None, timeout=60):
    """GET a market-api path and return the parsed JSON (raises on HTTP errors)."""
    r = requests.get(f'{BASE_URL}{path}', params=params, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _utc(ts):
    return datetime.fromtimestamp(ts, timezone.utc)


def _iso(ts):
    return _utc(ts).strftime('%Y-%m-%dT%H:%M:%SZ') if ts is not None else None


def _date(ts):
    return _utc(ts).date().isoformat() if ts is not None else None


def _day_ts(day):
    return int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp())


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _now_iso():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _age(iso_or_ts):
    """Human age of an ISO timestamp or unix seconds, e.g. '3h ago'."""
    if iso_or_ts is None:
        return 'never'
    ts = iso_or_ts if isinstance(iso_or_ts, (int, float)) else datetime.strptime(iso_or_ts, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc).timestamp()
    seconds = max(0, time.time() - ts)
    if seconds < 3600:
        return f'{int(seconds // 60)}m ago'
    if seconds < 86400:
        return f'{seconds / 3600:.0f}h ago'
    return f'{seconds / 86400:.1f}d ago'


def _file_is_fresh(path, extract, max_stale_days):
    """True when ``path`` exists and ``extract(json)`` (ISO string or unix seconds) is within max_stale_days.

    ``max_stale_days=None`` means an existing file is always fresh (never repulled).
    """
    if not os.path.exists(path):
        return False
    if max_stale_days is None:
        return True
    try:
        v = extract(_load_json(path))
    except Exception:
        return False
    if v is None:
        return False
    ts = v if isinstance(v, (int, float)) else datetime.strptime(v, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc).timestamp()
    return time.time() - ts <= max_stale_days * 86400


def _fmt_usd(v):
    if v is None:
        return '-'
    if v >= 1e6:
        return f'${v / 1e6:.1f}m'
    if v >= 1e3:
        return f'${v / 1e3:.0f}k'
    return f'${v:,.0f}'


def _rpc_url(rpc_url):
    return rpc_url or os.getenv('SOLANA_RPC_URL') or DEFAULT_RPC_URL


# ---------------------------------------------------------------------------
# 1. On-chain launch accounts -> ICO price
# ---------------------------------------------------------------------------

def _rpc(url, method, params):
    body = {'jsonrpc': '2.0', 'id': 1, 'method': method, 'params': params}
    last = None
    for attempt in range(2):
        try:
            r = requests.post(url, json=body, timeout=RPC_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            if 'error' in data:
                raise RuntimeError(f"RPC error: {data['error']}")
            return data['result']
        except Exception as e:
            last = e
            if attempt == 0:
                time.sleep(2)
    raise last


@report_errors()
def pull_launches(metadao_output_dir, rpc_url=None, force=False, versions=None, max_stale_days=None):
    """Save every launchpad Launch account, per program version, exactly as the RPC returns it.

    One ``getProgramAccounts`` per program in PROGRAMS, filtered by the Launch account
    discriminator, to ``{output_dir}/launches/{version}.json``. Completed launches never
    change, so an existing file is kept (``max_stale_days=None``, the default); pass a
    number of days to repull older files, or ``force=True`` to repull regardless. The whole set takes a few seconds on the public RPC. ``rpc_url``
    defaults to ``SOLANA_RPC_URL`` in API_KEYS.env, then Solana's public mainnet RPC; if
    the public RPC times out, a free Helius key (https://www.helius.dev) in
    SOLANA_RPC_URL is the fix.
    """
    url = _rpc_url(rpc_url)
    pulled, kept, failed = [], [], []
    oldest = None
    started = time.time()
    for version in (versions or list(PROGRAMS)):
        program_id = PROGRAMS[version]
        path = f'{metadao_output_dir}/launches/{version}.json'
        if not force and _file_is_fresh(path, lambda d: d.get('pulled'), max_stale_days):
            kept.append(version)
            pulled_at = _load_json(path).get('pulled')
            oldest = pulled_at if oldest is None or (pulled_at and pulled_at < oldest) else oldest
            continue
        try:
            accounts = _rpc(url, 'getProgramAccounts', [program_id, {
                'encoding': 'base64',
                'filters': [{'memcmp': {'offset': 0, 'bytes': _b58encode(LAUNCH_DISCRIMINATOR)}}],
            }])
            save_json({'program': program_id, 'version': version, 'rpc': url,
                       'pulled': _now_iso(), 'accounts': accounts}, path)
            pulled.append(f'{version} ({len(accounts)})')
        except Exception as e:
            print(f"  {version} ({program_id}) failed: {error_code(e)}{type(e).__name__}: {e}")
            if url == DEFAULT_RPC_URL:
                print("  The public RPC can be slow on getProgramAccounts; set SOLANA_RPC_URL in API_KEYS.env to a free Helius endpoint.")
            failed.append(version)
    parts = []
    if pulled:
        parts.append(f"repulled {', '.join(pulled)} in {time.time() - started:.0f}s")
    if kept:
        parts.append(f"{len(kept)} program file(s) fresh (oldest {_age(oldest)}), kept")
    if failed:
        parts.append(f"FAILED {', '.join(failed)}")
    print(f"launch accounts: {'; '.join(parts)}")


class _Reader:
    def __init__(self, raw):
        self.raw = raw
        self.i = 8  # skip the discriminator

    def take(self, kind):
        if self.i >= len(self.raw):
            raise ValueError(f'field {kind} at offset {self.i} runs past the end of the account')
        if kind == 'u8':
            v = self.raw[self.i]; self.i += 1; return v
        if kind == 'bool':
            v = self.raw[self.i] != 0; self.i += 1; return v
        if kind == 'u32':
            v = struct.unpack_from('<I', self.raw, self.i)[0]; self.i += 4; return v
        if kind == 'u64':
            v = struct.unpack_from('<Q', self.raw, self.i)[0]; self.i += 8; return v
        if kind == 'i64':
            v = struct.unpack_from('<q', self.raw, self.i)[0]; self.i += 8; return v
        if kind == 'pubkey':
            if self.i + 32 > len(self.raw):
                raise ValueError(f'pubkey at offset {self.i} runs past the end of the account')
            v = _b58encode(self.raw[self.i:self.i + 32]); self.i += 32; return v
        if kind.startswith('opt:'):
            tag = self.take('u8')
            if tag > 1:
                raise ValueError(f'option tag {tag} at offset {self.i - 1}')
            return self.take(kind[4:]) if tag else None
        if kind == 'vec:pubkey':
            n = self.take('u32')
            if n * 32 > len(self.raw) - self.i:
                raise ValueError(f'vec length {n} at offset {self.i - 4} exceeds the account')
            return [self.take('pubkey') for _ in range(n)]
        if kind == 'state':
            return self.take('u8')
        raise ValueError(f'unknown field type {kind}')


def decode_launch(raw, version):
    """Decode one Launch account (bytes, discriminator included) with the version's layout.

    Raises ValueError if the layout does not fit: the account must be at least as long as
    the fields consumed, and anything after them must be zero padding (Anchor allocates
    the account with room to spare).
    """
    reader = _Reader(raw)
    fields = {}
    for name, kind in LAYOUTS[version]:
        fields[name] = reader.take(kind)
    if reader.i > len(raw):
        raise ValueError(f'{version} layout needs {reader.i} bytes, account has {len(raw)}')
    if any(raw[reader.i:]):
        raise ValueError(f'{version} layout leaves {len(raw) - reader.i} non-zero trailing bytes')
    states = LAUNCH_STATES[version]
    fields['state'] = states[fields['state']] if fields['state'] < len(states) else f"unknown({fields['state']})"
    return fields


def decode_launches(metadao_output_dir, quiet=False):
    """Return one dict per decoded Launch account across the saved program files.

    USDC amounts are converted to dollars. ``usdc_accepted`` is ``total_approved_amount``
    (v0.7+), ``final_raise_amount`` (v0.6) or, on the capless v0.4/v0.5 programs, the
    committed amount. ``ico_price_usd`` is only set for Complete launches; when a
    completed launch shows an accepted amount of zero the committed amount is used and
    the row is noted. ``sale_closed`` falls back to start + duration on programs without a
    close timestamp.
    """
    rows = []
    bad = []
    for version in PROGRAMS:
        path = f'{metadao_output_dir}/launches/{version}.json'
        if not os.path.exists(path):
            continue
        for account in _load_json(path)['accounts']:
            raw = base64.b64decode(account['account']['data'][0])
            try:
                f = decode_launch(raw, version)
            except Exception as e:
                bad.append(f"{version} {account['pubkey'][:8]}... ({e})")
                continue
            scale = 10 ** USDC_DECIMALS
            committed = f['total_committed_amount'] / scale
            if version in ('v0.4', 'v0.5'):
                accepted = committed
            elif version == 'v0.6':
                accepted = None if f.get('final_raise_amount') is None else f['final_raise_amount'] / scale
            else:
                accepted = f['total_approved_amount'] / scale
            started = f.get('unix_timestamp_started')
            closed = f.get('unix_timestamp_closed')
            if closed is None and started is not None and f['state'] not in ('Initialized', 'Live'):
                closed = started + f['seconds_for_launch']
            notes = None
            price = None
            if f['state'] == 'Complete':
                if not accepted:
                    accepted = committed
                    notes = 'accepted amount not recorded on-chain; committed amount used'
                price = accepted / TOKENS_TO_PARTICIPANTS
            rows.append({
                'mint': f['base_mint'], 'launchpad_version': version, 'launch_account': account['pubkey'],
                'state': f['state'], 'sale_started': _date(started), 'sale_closed': _date(closed),
                'sale_completed': _date(f.get('unix_timestamp_completed')),
                'usdc_committed': round(committed, 2),
                'usdc_accepted': None if accepted is None else round(accepted, 2),
                'tokens_to_participants': TOKENS_TO_PARTICIPANTS,
                'ico_price_usd': None if price is None else round(price, 8),
                'notes': notes,
            })
    if bad and not quiet:
        print(f"decoded {len(rows)} launch accounts; {len(bad)} skipped as undecodable: {'; '.join(bad)}")
    return rows


# ---------------------------------------------------------------------------
# 2. Config (hand-maintained, filled only where missing)
# ---------------------------------------------------------------------------

def _empty_config():
    return {
        '_metadata': {
            'description': 'MetaDAO launchpad raises. Hand-editable: metadao.fill_launch_config only adds missing '
                           'launches and missing keys, it never overwrites a value. Set include=false to drop a '
                           'launch from the charts without deleting it.',
            'price_formula': 'ico_price_usd = usdc_accepted / tokens_to_participants (the program mints a fixed '
                             '10,000,000 tokens to funders as a group, pro rata to accepted USDC)',
            'usdc_accepted': 'total_approved_amount (v0.7+), final_raise_amount (v0.6) or committed (v0.4/v0.5) '
                             'from the on-chain Launch account, in USDC',
            'programs': dict(PROGRAMS),
            'launch_discriminator_base64': base64.b64encode(LAUNCH_DISCRIMINATOR).decode(),
        },
        'launches': {},
    }


def load_launch_config(config_path):
    return _load_json(config_path) if os.path.exists(config_path) else _empty_config()


def save_launch_config(config, config_path):
    os.makedirs(os.path.dirname(config_path) or '.', exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
        f.write('\n')


def _pull_tickers(metadao_output_dir, max_stale_days=DEFAULT_MAX_STALE_DAYS):
    path = f'{metadao_output_dir}/tickers.json'
    if not _file_is_fresh(path, lambda d: d.get('_pulled_ts'), max_stale_days):
        data = _get('/api/tickers')
        save_json({'_pulled_ts': int(time.time()), 'tickers': data}, path)
    return {t['base_currency']: t for t in _load_json(path)['tickers']}


def _asset_info(mint, metadao_output_dir):
    """Symbol/name for any mint via /dexscreener/asset, cached raw under assets/."""
    path = f'{metadao_output_dir}/assets/{mint}.json'
    if not os.path.exists(path):
        data = fetch_and_save_json(f'{BASE_URL}/dexscreener/asset', path, headers=HEADERS, params={'id': mint})
        time.sleep(SECONDS_BETWEEN_CALLS)
        if data is None:
            return {}
    return _load_json(path).get('asset') or {}


@report_errors()
def fill_launch_config(config_path, decoded, metadao_output_dir, min_accepted_usd=DEFAULT_MIN_ACCEPTED_USD):
    """Add completed launches that are missing from the config; never change what is there.

    New rows get ``include = usdc_accepted >= min_accepted_usd`` so dust test raises are
    kept for reference but left out of the charts. Ticker and name come from
    ``/api/tickers`` when the DAO is listed, else ``/dexscreener/asset``; unresolved ones
    are printed for hand entry. Existing rows only gain keys they lack.
    """
    config = load_launch_config(config_path)
    launches = config.setdefault('launches', {})
    tickers = _pull_tickers(metadao_output_dir)
    added, filled, unnamed = [], [], []
    for row in sorted(decoded, key=lambda r: (r['sale_closed'] or '', r['mint'])):
        if row['state'] != 'Complete':
            continue
        mint = row['mint']
        existing = launches.get(mint)
        if existing is None:
            info = tickers.get(mint) or {}
            ticker, name = info.get('base_symbol'), info.get('base_name')
            if not ticker:
                asset = _asset_info(mint, metadao_output_dir)
                ticker, name = asset.get('symbol'), asset.get('name')
            new = {
                'ticker': ticker, 'name': name,
                'launchpad_version': row['launchpad_version'], 'launch_account': row['launch_account'],
                'sale_started': row['sale_started'], 'sale_closed': row['sale_closed'],
                'sale_completed': row['sale_completed'],
                'usdc_committed': row['usdc_committed'], 'usdc_accepted': row['usdc_accepted'],
                'tokens_to_participants': row['tokens_to_participants'], 'ico_price_usd': row['ico_price_usd'],
                'pair_id': None,
                'include': bool(row['usdc_accepted'] and row['usdc_accepted'] >= min_accepted_usd),
                'source': 'on-chain Launch account (metadao.pull_launches)',
                'notes': row['notes'],
            }
            launches[mint] = new
            added.append(f"{ticker or mint[:8]} ({_fmt_usd(row['usdc_accepted'])}{'' if new['include'] else ', excluded'})")
            if not ticker:
                unnamed.append(mint)
        else:
            for key, value in row.items():
                if key in ('mint', 'state'):
                    continue
                if key not in existing:
                    existing[key] = value
                    filled.append(f"{existing.get('ticker') or mint[:8]}.{key}")
            for key, value in (('pair_id', None), ('include', False), ('source', None), ('notes', None)):
                if key not in existing:
                    existing[key] = value
                    filled.append(f"{existing.get('ticker') or mint[:8]}.{key}")
    if added or filled:
        save_launch_config(config, config_path)
    n_included = sum(1 for l in launches.values() if l.get('include'))
    summary = f"config: {len(launches)} launches, {n_included} included"
    if added:
        shown = added[:8] + ([f'... {len(added) - 8} more'] if len(added) > 8 else [])
        summary += f"; added {len(added)}: {', '.join(shown)}"
    else:
        summary += "; no new launches"
    if filled:
        summary += f"; filled {len(filled)} missing key(s)"
    print(summary)
    if unnamed:
        print(f"  no ticker found, fill by hand: {', '.join(unnamed)}")
    return config


def included_launches(config_path):
    """{mint: row} for launches with include=true and an ICO price."""
    config = load_launch_config(config_path)
    return {m: r for m, r in config['launches'].items() if r.get('include') and r.get('ico_price_usd')}


# ---------------------------------------------------------------------------
# 3. Pairs and swap events (market-api.metadao.fi)
# ---------------------------------------------------------------------------

def _latest_block():
    b = _get('/dexscreener/latest-block')['block']
    return b['blockNumber'], b['blockTimestamp']


def _events(from_slot, to_slot):
    data = _get('/dexscreener/events', params={'fromBlock': from_slot, 'toBlock': to_slot}, timeout=120)
    time.sleep(SECONDS_BETWEEN_CALLS)
    return data.get('events') or []


def _slot_at(target_ts, latest=None, tolerance=600):
    """Approximate the slot at a unix timestamp from event timestamps (no RPC needed)."""
    latest_slot, latest_ts = latest or _latest_block()
    slot = latest_slot - int((latest_ts - target_ts) / SLOT_SECONDS)
    for _ in range(10):
        slot = max(slot, 1)
        events = _events(slot, slot + 20_000)
        if not events:
            slot -= 20_000
            continue
        seen = events[0]['block']['blockTimestamp']
        if abs(seen - target_ts) <= tolerance:
            return slot
        slot -= int((seen - target_ts) / SLOT_SECONDS)
    return slot


def _pair_info(pair_id, metadao_output_dir):
    path = f'{metadao_output_dir}/pairs/{pair_id}.json'
    if not os.path.exists(path):
        data = fetch_and_save_json(f'{BASE_URL}/dexscreener/pair', path, headers=HEADERS, params={'id': pair_id})
        time.sleep(SECONDS_BETWEEN_CALLS)
        if data is None:
            return None
    return _load_json(path).get('pair')


def _find_launch_pair(mint, sale_closed, metadao_output_dir, window_days=2):
    """Scan the swaps in the days after the sale close for a pair whose base asset is the mint."""
    slot = _slot_at(_day_ts(sale_closed))
    end_slot = slot + int(window_days * 86400 / SLOT_SECONDS)
    seen = set()
    while slot < end_slot:
        to_slot = min(slot + MAX_SLOTS_PER_REQUEST, end_slot)
        for event in _events(slot, to_slot):
            pair_id = event['pairId']
            if pair_id in seen:
                continue
            seen.add(pair_id)
            info = _pair_info(pair_id, metadao_output_dir)
            if info and info.get('asset0Id') == mint:
                return info
        slot = to_slot + 1
    return None


@report_errors()
def pull_pairs(config_path, metadao_output_dir, max_days_after_close=3):
    """Find each included launch's AMM pair and fill ``pair_id`` into the config (fill-only).

    The listed DAO's current ``pool_id`` from ``/api/tickers`` is used when that pool was
    created within ``max_days_after_close`` of the sale close; otherwise (delisted token,
    or a pool migrated long after launch, e.g. Omnipair) the swaps in the days after the
    close are scanned for a pair whose base asset is the mint. Pair details are saved
    raw under ``pairs/`` and carry the exact trading start (``createdAtBlockTimestamp``).
    """
    config = load_launch_config(config_path)
    rows = {m: r for m, r in config['launches'].items() if r.get('include') and r.get('ico_price_usd')}
    todo = {m: r for m, r in rows.items()
            if not (r.get('pair_id') and os.path.exists(f"{metadao_output_dir}/pairs/{r['pair_id']}.json"))}
    prehistory = [r.get('ticker') or m[:8] for m, r in todo.items() if (r.get('sale_closed') or '') < EVENTS_HISTORY_START]
    todo = {m: r for m, r in todo.items() if (r.get('sale_closed') or '') >= EVENTS_HISTORY_START}
    tickers = _pull_tickers(metadao_output_dir) if todo else {}
    unresolved, newly = [], []
    for mint, row in todo.items():
        label = row.get('ticker') or mint[:8]
        close_ts = _day_ts(row['sale_closed'])
        info = None
        listed = tickers.get(mint)
        if listed and listed.get('pool_id'):
            candidate = _pair_info(listed['pool_id'], metadao_output_dir)
            if candidate and candidate.get('asset0Id') == mint:
                lag_days = (candidate['createdAtBlockTimestamp'] - close_ts) / 86400
                if -1 <= lag_days <= max_days_after_close:
                    info = candidate
                else:
                    print(f"  {label}: listed pool created {lag_days:.0f} days after the sale close (a migration?); searching the launch window")
        if info is None:
            try:
                info = _find_launch_pair(mint, row['sale_closed'], metadao_output_dir)
            except Exception as e:
                print(f"  {label}: pair search failed: {error_code(e)}{type(e).__name__}: {e}")
        if info is None:
            unresolved.append(label)
            continue
        row['pair_id'] = info['id']
        newly.append(label)
    if newly:
        save_launch_config(config, config_path)
    resolved = sum(1 for r in rows.values() if r.get('pair_id'))
    summary = f"pairs: {resolved}/{len(rows)} included launches resolved"
    if newly:
        summary += f"; new: {', '.join(newly)}"
    if prehistory:
        summary += f"; closed before the swap feed starts ({EVENTS_HISTORY_START}), no price history: {', '.join(prehistory)}"
    if unresolved:
        summary += f"; no launch-window pair found: {', '.join(unresolved)}"
    print(summary)


def _swaps_path(metadao_output_dir, mint):
    return f'{metadao_output_dir}/swaps/{mint}.json'


@report_errors()
def pull_launch_swaps(config_path, metadao_output_dir, n_days=7, force=False):
    """Save every swap on each included launch's pair for its first ``n_days`` of trading.

    Reads forward from the pair's creation slot in 500k-slot requests until the events
    pass ``trading_start + n_days``; only that pair's events are kept, unchanged, in
    ``swaps/{mint}.json`` with the pair id, slot range and window. A file that already
    covers a completed ``n_days`` window is skipped unless ``force=True``; a launch whose
    window is still running is saved as partial and repulled by the next run.
    """
    rows = included_launches(config_path)
    failures, partial, pulled = [], [], []
    skipped = [r.get('ticker') or m[:8] for m, r in rows.items() if not r.get('pair_id')]
    todo = {}
    complete = 0
    for mint, row in rows.items():
        if not row.get('pair_id'):
            continue
        path = _swaps_path(metadao_output_dir, mint)
        if os.path.exists(path) and not force:
            existing = _load_json(path)
            if existing.get('n_days', 0) >= n_days and existing.get('window_complete'):
                complete += 1
                continue
        todo[mint] = row
    for mint, row in (pull_progress(list(todo.items()), 'MetaDAO launch swaps') if todo else []):
        label = row.get('ticker') or mint[:8]
        path = _swaps_path(metadao_output_dir, mint)
        pair = _pair_info(row['pair_id'], metadao_output_dir)
        if not pair:
            failures.append(('swaps', label))
            continue
        start_slot, start_ts = pair['createdAtBlockNumber'], pair['createdAtBlockTimestamp']
        end_ts = start_ts + n_days * 86400
        # Upper bound on slots, in case a chunk returns no events at all.
        max_slot = start_slot + int(n_days * 86400 / 0.35)
        events, slot, done = [], start_slot, False
        last_seen_ts = start_ts
        try:
            latest_slot, latest_ts = _latest_block()
            while not done and slot <= min(max_slot, latest_slot):
                to_slot = slot + MAX_SLOTS_PER_REQUEST
                for event in _events(slot, to_slot):
                    ts = event['block']['blockTimestamp']
                    if ts > end_ts:
                        done = True
                        break
                    last_seen_ts = max(last_seen_ts, ts)
                    if event['pairId'] == row['pair_id']:
                        events.append(event)
                slot = to_slot + 1
        except Exception as e:
            print(f"  {label}: {error_code(e)}{type(e).__name__}: {e}")
            failures.append(('swaps', label))
            continue
        # The window is complete once the chain has moved past its end; a launch younger
        # than n_days is saved as partial and repulled by the next run.
        window_complete = done or latest_ts >= end_ts
        if not window_complete:
            partial.append(f"{label} ({(last_seen_ts - start_ts) / 86400:.1f} of {n_days} days so far)")
        else:
            pulled.append(f"{label} ({len(events):,} swaps)")
        save_json({'mint': mint, 'ticker': row.get('ticker'), 'pair_id': row['pair_id'],
                   'trading_start_ts': start_ts, 'trading_start': _iso(start_ts), 'n_days': n_days,
                   'window_complete': window_complete,
                   'from_slot': start_slot, 'to_slot': slot - 1,
                   'pulled': _now_iso(), 'events': events}, path)
    summary = f"swaps: {complete + len(pulled)} launches complete on file"
    if pulled:
        summary += f"; pulled {', '.join(pulled)}"
    if partial:
        summary += f"; window still running, repulled next time: {', '.join(partial)}"
    if failures:
        summary += f"; FAILED {', '.join(l for _, l in failures)}"
    if skipped:
        summary += f"; no pair: {', '.join(skipped)}"
    print(summary)
    return len(pulled) + len(partial)


@report_errors()
def pull_market_data(metadao_output_dir, start_date='2025-10-01', max_stale_days=None, force=False):
    """Save MetaDAO's daily per-token market data (volumes, fees) for the tie-out, as returned.

    An existing file is kept unless ``force=True`` or it is older than ``max_stale_days``.
    """
    path = f'{metadao_output_dir}/market-data.json'
    if not force and _file_is_fresh(path, lambda d: d.get('_pulled_ts'), max_stale_days):
        print(f"market data: fresh (pulled {_age(_load_json(path).get('_pulled_ts'))}), kept")
        return
    end_date = datetime.now(timezone.utc).date().isoformat()
    data = _get('/api/market-data', params={'startDate': start_date, 'endDate': end_date}, timeout=120)
    data['_pulled_ts'] = int(time.time())
    save_json(data, path)
    n = (data.get('futarchyAMM') or {}).get('count', 0)
    print(f"market data: pulled {start_date}..{end_date} ({n:,} token-days)")


# ---------------------------------------------------------------------------
# 4. Processed prices (what the chart reads) and tables
# ---------------------------------------------------------------------------

def _usdc(event):
    return (event.get('asset1In') or 0) + (event.get('asset1Out') or 0)


def _tokens(event):
    return (event.get('asset0In') or 0) + (event.get('asset0Out') or 0)


def _daily_vwap_from_market_data(metadao_output_dir):
    path = f'{metadao_output_dir}/market-data.json'
    if not os.path.exists(path):
        return {}
    out = {}
    for row in (_load_json(path).get('futarchyAMM') or {}).get('data') or []:
        base = float(row.get('spot_base_volume') or 0)
        quote = float(row.get('spot_target_volume') or 0)
        if base > 0:
            out[(row['token'], row['date'])] = quote / base
    return out


@report_errors()
def build_launch_prices(config_path, metadao_output_dir, processed_dir, n_days=7, tolerance=0.005, force=False):
    """Build ``{processed_dir}/launch-prices-{n_days}d.json``: hourly VWAP per launch from its swaps.

    Hour ``k`` covers ``trading_start + k`` hours; ``vwap`` is USDC volume / token volume
    over the hour (null when nothing traded) and ``multiple`` is vwap / ico_price_usd.
    Prints two checks: the ICO price identity (accepted == price x 10M) and, for every
    full calendar day in the window, the swap-derived daily VWAP against MetaDAO's daily
    endpoint, flagging days that differ by more than ``tolerance``. The file is only
    rebuilt when the config, market data or a swaps file is newer than it (``force=True``
    rebuilds regardless).
    """
    rows = included_launches(config_path)
    out_path = f'{processed_dir}/launch-prices-{n_days}d.json'
    inputs = [config_path, f'{metadao_output_dir}/market-data.json'] + [_swaps_path(metadao_output_dir, m) for m in rows]
    if os.path.exists(out_path) and not force:
        newest_input = max((os.path.getmtime(p) for p in inputs if os.path.exists(p)), default=0)
        if os.path.getmtime(out_path) >= newest_input:
            existing = _load_json(out_path)
            print(f"{out_path}: up to date ({len(existing['launches'])} launches, built {_age(existing['_metadata'].get('built'))}), kept")
            return existing
    daily_ref = _daily_vwap_from_market_data(metadao_output_dir)
    out = {'_metadata': {
        'n_days': n_days, 'freq': '1h',
        'price': 'hourly VWAP = USDC volume / token volume of every swap on the launch pair (MetaDAO spot AMM)',
        'multiple': 'vwap / ico_price_usd; ico_price_usd = usdc_accepted / 10,000,000 tokens',
        'built': _now_iso(),
    }, 'launches': {}}
    checks, skipped = [], []
    compared = 0
    for mint, row in rows.items():
        label = row.get('ticker') or mint[:8]
        price = row['ico_price_usd']
        implied = price * row['tokens_to_participants']
        if abs(implied - row['usdc_accepted']) > 0.01 * max(1, row['usdc_accepted']):
            checks.append(f"{label}: ico_price x tokens = ${implied:,.0f} but usdc_accepted = ${row['usdc_accepted']:,.0f}")
        path = _swaps_path(metadao_output_dir, mint)
        if not row.get('pair_id') or not os.path.exists(path):
            skipped.append(label)
            continue
        swaps = _load_json(path)
        start_ts = swaps['trading_start_ts']
        hours = n_days * 24
        usdc = [0.0] * hours
        tokens = [0.0] * hours
        trades = [0] * hours
        by_day = {}
        for event in swaps['events']:
            ts = event['block']['blockTimestamp']
            k = (ts - start_ts) // 3600
            if 0 <= k < hours:
                usdc[k] += _usdc(event)
                tokens[k] += _tokens(event)
                trades[k] += 1
                d = by_day.setdefault(_date(ts), [0.0, 0.0])
                d[0] += _usdc(event)
                d[1] += _tokens(event)
        series = []
        for k in range(hours):
            vwap = usdc[k] / tokens[k] if tokens[k] > 0 else None
            series.append({'hour': k, 'ts': _iso(start_ts + k * 3600), 'vwap': vwap,
                           'usdc_volume': usdc[k], 'token_volume': tokens[k], 'trades': trades[k],
                           'multiple': None if vwap is None else vwap / price})
        out['launches'][mint] = {
            'ticker': row.get('ticker'), 'name': row.get('name'), 'ico_price_usd': price,
            'usdc_accepted': row['usdc_accepted'], 'usdc_committed': row['usdc_committed'],
            'sale_closed': row['sale_closed'], 'trading_start': swaps['trading_start'],
            'pair_id': row['pair_id'], 'hours': series,
        }
        # Tie-out on full calendar days inside the window (the first day starts at the first
        # trade, as MetaDAO's daily row does; the last day is only partly covered).
        window_end = start_ts + hours * 3600
        for day, (q, b) in sorted(by_day.items()):
            ref = daily_ref.get((mint, day))
            if _day_ts(day) + 86400 > window_end or ref is None or b <= 0:
                continue
            compared += 1
            diff = (q / b) / ref - 1
            if abs(diff) > tolerance:
                checks.append(f"{label} {day}: swaps VWAP {q / b:.4f} vs MetaDAO daily {ref:.4f} ({diff:+.2%})")
    os.makedirs(processed_dir, exist_ok=True)
    save_json(out, out_path)
    summary = f"{out_path}: {len(out['launches'])} launches x {n_days * 24} hourly points"
    if skipped:
        summary += f"; no swap history: {', '.join(skipped)}"
    if checks:
        summary += f"; TIE-OUT: {len(checks)} discrepancies"
    else:
        summary += f"; tie-out clean (ICO price identity on every launch; daily VWAP within {tolerance:.1%} of MetaDAO on all {compared} launch-days)"
    print(summary)
    for c in checks:
        print(f"    {c}")
    return out


@report_errors()
def refresh(config_path, metadao_output_dir, processed_dir, n_days=7,
            min_accepted_usd=DEFAULT_MIN_ACCEPTED_USD, rpc_url=None):
    """Check MetaDAO for new launches and fetch only those; launches already on file are never repulled.

    One fresh ``/api/tickers`` call decides: a listed DAO whose mint is in neither the config
    nor the saved launch accounts, and whose pool started on or after the day the launch
    files were last pulled, is a new raise. Only then are the launch accounts repulled (a
    few seconds) and the new rows added to the config (existing rows untouched). Pairs and
    swaps are fetched only for launches that lack them or whose first ``n_days`` had not
    finished at the previous run; the daily market data (tie-out only) is repulled only when
    swaps were fetched in this run; the processed file is rebuilt only when an input changed.
    The first run, with nothing on file, pulls everything. ``metadao.pull_launches(...,
    force=True)`` is the manual override to repull every launch account.
    """
    launch_files = [f'{metadao_output_dir}/launches/{v}.json' for v in PROGRAMS]
    pulled_stamps = [_load_json(p).get('pulled') or '' for p in launch_files if os.path.exists(p)]
    pulled_dates = [stamp[:10] for stamp in pulled_stamps]
    tickers = _pull_tickers(metadao_output_dir, max_stale_days=0)
    if pulled_dates:
        known = set(load_launch_config(config_path).get('launches', {}))
        known |= {r['mint'] for r in decode_launches(metadao_output_dir, quiet=True)}
        since = (datetime.fromisoformat(min(pulled_dates)) - timedelta(days=1)).date().isoformat()
        new_daos = [t['base_symbol'] for m, t in tickers.items()
                    if m not in known and (t.get('startDate') or '') >= since]
        if new_daos:
            print(f"new launch(es) on MetaDAO: {', '.join(new_daos)}; pulling their launch accounts")
    else:
        new_daos = ['first run']
    if new_daos:
        pull_launches(metadao_output_dir, rpc_url=rpc_url, force=True)
    else:
        n_on_file = len(load_launch_config(config_path).get('launches', {}))
        print(f"no new launches on MetaDAO ({len(tickers)} DAOs listed, {n_on_file} launches on file, "
              f"accounts last pulled {_age(min(pulled_stamps))})")
    # Local and cheap: re-adds any config row that was deleted by hand and fills missing keys.
    fill_launch_config(config_path, decode_launches(metadao_output_dir, quiet=True), metadao_output_dir,
                       min_accepted_usd=min_accepted_usd)
    pull_pairs(config_path, metadao_output_dir)
    fetched = pull_launch_swaps(config_path, metadao_output_dir, n_days=n_days) or 0
    pull_market_data(metadao_output_dir, force=fetched > 0)
    return build_launch_prices(config_path, metadao_output_dir, processed_dir, n_days=n_days)


def load_launch_prices(processed_path):
    return _load_json(processed_path)


def first_days_table(processed_path, n_days=None, min_fundraise_usd=0):
    """One row per launch: day_k = VWAP over trading hours 24(k-1)..24k-1 divided by the ICO price.

    Also ``max_hourly`` / ``min_hourly`` (hourly multiples over the window), ``hours_with_trades``,
    and the launch facts (sale close, USDC accepted, ICO price). Indexed by ticker. Launches
    that raised less than ``min_fundraise_usd`` are left out (same filter as the chart).
    """
    data = load_launch_prices(processed_path)
    n_days = n_days or data['_metadata']['n_days']
    rows = []
    for mint, launch in data['launches'].items():
        if (launch.get('usdc_accepted') or 0) < min_fundraise_usd:
            continue
        hours = launch['hours'][:n_days * 24]
        r = {'ticker': launch['ticker'], 'sale_closed': launch['sale_closed'],
             'usdc_accepted': launch['usdc_accepted'], 'ico_price_usd': launch['ico_price_usd']}
        for k in range(1, n_days + 1):
            chunk = hours[(k - 1) * 24:k * 24]
            q = sum(h['usdc_volume'] for h in chunk)
            b = sum(h['token_volume'] for h in chunk)
            r[f'day_{k}'] = (q / b) / launch['ico_price_usd'] if b > 0 else None
        mults = [h['multiple'] for h in hours if h['multiple'] is not None]
        r['max_hourly'] = max(mults) if mults else None
        r['min_hourly'] = min(mults) if mults else None
        r['hours_with_trades'] = len(mults)
        rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values('sale_closed').set_index('ticker')


# ---------------------------------------------------------------------------
# 5. Is participating profitable? Horizon multiples, fill-adjusted returns, cohort curve
# ---------------------------------------------------------------------------

HORIZONS = (1, 7, 30, 90)


def daily_vwap_series(metadao_output_dir):
    """{mint: {date: vwap}} from the saved daily market data (MetaDAO spot AMM rows)."""
    out = {}
    path = f'{metadao_output_dir}/market-data.json'
    if not os.path.exists(path):
        return out
    for row in (_load_json(path).get('futarchyAMM') or {}).get('data') or []:
        base = float(row.get('spot_base_volume') or 0)
        quote = float(row.get('spot_target_volume') or 0)
        if base > 0:
            out.setdefault(row['token'], {})[row['date']] = quote / base
    return out


def daily_multiples(processed_path, metadao_output_dir, max_days=90, min_fundraise_usd=0):
    """Daily multiple of the ICO price per launch: rows = days since first trade (0..max_days), columns = tickers.

    Day k is the UTC calendar day ``trading_start + k``; the value is that day's VWAP on the
    MetaDAO spot AMM divided by the ICO price, NaN when nothing traded or the day is in the
    future. Returns ``(DataFrame, launches)`` where ``launches`` is the processed file's
    per-launch info (filtered by ``min_fundraise_usd``) with ``latest_date`` added.
    """
    data = load_launch_prices(processed_path)
    series = daily_vwap_series(metadao_output_dir)
    today = datetime.now(timezone.utc).date()
    cols, launches = {}, {}
    for mint, launch in data['launches'].items():
        if (launch.get('usdc_accepted') or 0) < min_fundraise_usd:
            continue
        s = series.get(mint, {})
        start = datetime.fromisoformat(launch['trading_start'][:10]).date()
        vals = []
        for k in range(max_days + 1):
            day = start + timedelta(days=k)
            v = s.get(day.isoformat())
            vals.append(v / launch['ico_price_usd'] if v and day <= today else float('nan'))
        ticker = launch['ticker'] or mint[:8]
        cols[ticker] = vals
        info = dict(launch)
        info.pop('hours', None)
        info['latest_date'] = max(s) if s else None
        info['latest_multiple'] = (s[max(s)] / launch['ico_price_usd']) if s else None
        info['days_traded'] = (today - start).days
        launches[ticker] = info
    df = pd.DataFrame(cols, index=pd.RangeIndex(0, max_days + 1, name='day'))
    return df, launches


def horizon_returns(processed_path, metadao_output_dir, horizons=HORIZONS, min_fundraise_usd=0):
    """One row per launch: fill rate, multiple of ICO price at each horizon and today, and the return per $1 committed.

    ``mult_{h}d`` is the daily VWAP on day ``h`` after the first trade (falling back to the
    last traded day within the previous 3 days), NaN when the launch is younger than ``h``
    days. ``fill_rate = usdc_accepted / usdc_committed`` is the pro-rata share of a
    participant's commitment that bought tokens; the rest was refunded. So
    ``ret_{h}d = fill_rate * (mult_{h}d - 1)`` is what $1 committed to the raise earned by
    day ``h`` (the refunded part earns nothing). ``mult_latest`` uses the last day the
    token traded on the MetaDAO AMM (``latest_date``), which for delisted tokens can be
    months old. Sorted by sale close.
    """
    df, launches = daily_multiples(processed_path, metadao_output_dir, max_days=max(horizons), min_fundraise_usd=min_fundraise_usd)
    rows = []
    for ticker, info in launches.items():
        col = df[ticker]
        fill = info['usdc_accepted'] / info['usdc_committed'] if info.get('usdc_committed') else float('nan')
        r = {'ticker': ticker, 'sale_closed': info['sale_closed'], 'usdc_committed': info['usdc_committed'],
             'usdc_accepted': info['usdc_accepted'], 'fill_rate': fill, 'ico_price_usd': info['ico_price_usd']}
        for h in horizons:
            if info['days_traded'] < h:
                m = float('nan')
            else:
                window = col.loc[max(0, h - 3):h].dropna()
                m = window.iloc[-1] if len(window) else float('nan')
            r[f'mult_{h}d'] = m
            r[f'ret_{h}d'] = fill * (m - 1) if pd.notna(m) else float('nan')
        r['mult_latest'] = info['latest_multiple']
        r['latest_date'] = info['latest_date']
        r['ret_latest'] = fill * (info['latest_multiple'] - 1) if info['latest_multiple'] else float('nan')
        rows.append(r)
    out = pd.DataFrame(rows)
    return out.sort_values('sale_closed').set_index('ticker') if not out.empty else out


def cohort_summary(returns_df, horizons=HORIZONS):
    """Summary rows for horizon_returns(): launches with data, mean and median multiple, hit rate, and the
    equal-weight strategy return (commit the same amount to every raise; the mean of ret_{h})."""
    cols = [f'{h}d' for h in horizons] + ['latest']
    rows = {}
    rows['launches with data'] = {c: returns_df[f'mult_{c}'].notna().sum() for c in cols}
    rows['mean multiple'] = {c: returns_df[f'mult_{c}'].mean() for c in cols}
    rows['median multiple'] = {c: returns_df[f'mult_{c}'].median() for c in cols}
    rows['hit rate (above ICO price)'] = {c: (returns_df[f'mult_{c}'].dropna() > 1).mean() if returns_df[f'mult_{c}'].notna().any() else float('nan') for c in cols}
    rows['return per $1 committed to every raise'] = {c: returns_df[f'ret_{c}'].mean() for c in cols}
    return pd.DataFrame(rows).T
