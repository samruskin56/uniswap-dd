"""Comparable-set pull for the liquid-token valuation.

All multiples are stated on FDV (price x max supply), the model's own supply
metric, with mcap (circulating) alongside; the house-EV convention is
deprecated. Equities: yfinance, spot mcap (the fully-diluted proxy), TTM
revenue and EBITDA. Tokens: CoinGecko mcap/FDV + DeFiLlama units/TVL and TTM
net protocol revenue (the protocol's own take, same altitude as equity
revenue). FDV/EBITDA is the tie-out to the model's exit multiple; tokens do
not report opex, so their EBITDA is blank.

Peer selection is a deal input: the notebook config supplies `equity_groups`
({group label: [tickers]}) and `token_group` ([(name, slug, gecko)]) -- there
are no module-level comp sets.
"""

import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

from .marks import _get, WINDOW_KEY, financial_statement
from utils.helper_functions import report_errors, llama_url

_GECKO = 'https://api.coingecko.com/api/v3'
# Last-good CoinGecko market data, merged on every successful pull (the free
# tier 429s for minutes at a time, so a retry alone cannot ride it out).
_TECHNICALS_CACHE = os.path.join('data', 'technicals_snapshot.json')


@report_errors(reraise=True)
def pull_equity(t, group):
    """One equity comp: spot mcap/price (yfinance .info; mcap doubles as the
    fully-diluted proxy), TTM revenue and EBITDA (ttm_income_stmt). Blank where
    yfinance has no figure."""
    try:
        import yfinance as yf
        tk = yf.Ticker(t)
        i = tk.info
        mcap, price = i.get('marketCap'), i.get('currentPrice')

        rev = ebitda = None
        try:
            ttm = tk.ttm_income_stmt
            if ttm is not None and not ttm.empty:
                c = ttm.columns[0]
                _row = lambda n: (float(ttm.loc[n, c])
                                  if (n in ttm.index and pd.notna(ttm.loc[n, c])) else None)
                rev, ebitda = _row('Total Revenue'), _row('EBITDA')
        except Exception:
            pass
        if rev is None:
            rev = i.get('totalRevenue')

        return dict(Company=t, Group=group, Price=price,
                    Mcap=mcap / 1e6 if mcap else np.nan,
                    Fdv=mcap / 1e6 if mcap else np.nan,   # FD proxy for equities
                    Rev=rev / 1e6 if rev else np.nan,
                    EBITDA=ebitda / 1e6 if ebitda is not None else np.nan)
    except Exception as e:
        return dict(Company=t, Group=group, err=type(e).__name__)


@report_errors(reraise=True)
def pull_token(name, slug, gecko, market=None, group='Token peers'):
    """One token comp: DeFiLlama units (borrowed) / TVL + TTM net protocol
    revenue (dailyRevenue, the protocol's own take); price/mcap/FDV from
    `market` (a pull_market_technicals record), falling back to DeFiLlama price
    + mcap (FDV falls back to mcap). EBITDA blank (protocol opex not reported),
    so the token read is FDV/Rev."""
    try:
        p = _get(llama_url(f'/protocol/{slug}'))
        tvl_arr = p.get('tvl')
        tvl = tvl_arr[-1].get('totalLiquidityUSD') if isinstance(tvl_arr, list) and tvl_arr else None
        units = (p.get('currentChainTvls') or {}).get('borrowed')
        rev = _get(llama_url(f'/summary/fees/{slug}?dataType=dailyRevenue')).get(WINDOW_KEY[365])
        mk = market or {}
        pr = mk.get('price')
        if pr is None:
            pr = _get(llama_url(f'/prices/current/coingecko:{gecko}', section='coins'))['coins'].get(f'coingecko:{gecko}', {}).get('price')
        mcap = mk.get('mcap') or p.get('mcap')
        fdv = mk.get('fdv') or mcap
        return dict(Company=name, Group=group, Price=pr,
                    Mcap=mcap / 1e6 if mcap else np.nan, Fdv=fdv / 1e6 if fdv else np.nan,
                    Rev=rev / 1e6 if rev else np.nan,
                    Tvl=tvl / 1e6 if tvl else np.nan, Units=units / 1e6 if units else np.nan,
                    EBITDA=np.nan)
    except Exception as e:
        return dict(Company=name, Group=group, err=type(e).__name__)


def _technicals_cache_merge(rows):
    """Merge fresh per-id market rows into the last-good cache file."""
    cache = {'asof': None, 'data': {}}
    try:
        with open(_TECHNICALS_CACHE) as f:
            cache = json.load(f)
    except (OSError, json.JSONDecodeError):
        pass
    cache['asof'] = datetime.now().isoformat(timespec='seconds')
    cache.setdefault('data', {}).update(rows)
    try:
        os.makedirs(os.path.dirname(_TECHNICALS_CACHE) or '.', exist_ok=True)
        with open(_TECHNICALS_CACHE, 'w') as f:
            json.dump(cache, f, indent=2)
    except OSError:
        pass


@report_errors(default={})
def pull_market_technicals(gecko_ids, retries=2):
    """CoinGecko market data per gecko id: dict(price, mcap, fdv, chg_30d,
    ath_dd); $ figures raw USD, changes fractions. The free tier hard-429s for
    minutes at a time, so after a short retry the last-good on-disk cache
    serves (merged on every successful pull); {} only when both fail."""
    if not gecko_ids:
        return {}
    ids = ','.join(gecko_ids)
    url = f'{_GECKO}/coins/markets?vs_currency=usd&ids={ids}&price_change_percentage=30d'
    data = None
    for attempt in range(retries):
        try:
            data = _get(url)
            break
        except Exception:
            if attempt + 1 < retries:
                time.sleep(5)
    out = {}
    for d in data if isinstance(data, list) else []:
        chg = d.get('price_change_percentage_30d_in_currency')
        ath = d.get('ath_change_percentage')
        out[d.get('id')] = dict(
            price=d.get('current_price'),
            mcap=d.get('market_cap'),
            fdv=d.get('fully_diluted_valuation'),
            chg_30d=chg / 100.0 if chg is not None else None,
            ath_dd=ath / 100.0 if ath is not None else None,
        )
    if out:
        _technicals_cache_merge(out)
        return out
    try:
        with open(_TECHNICALS_CACHE) as f:
            cache = json.load(f)
        cached = {g: cache['data'][g] for g in gecko_ids if g in cache.get('data', {})}
        if cached:
            print(f"[technicals] live CoinGecko failed -> cache (asof {cache.get('asof')})")
        return cached
    except (OSError, json.JSONDecodeError, KeyError):
        return {}


@report_errors(reraise=True)
def build_comps(subject, equity_groups, token_group):
    """Pull the comp set and return a DataFrame indexed by Company. `subject` is
    the subject-protocol row dict (Group='SUBJECT'); `equity_groups` maps group
    label -> ticker list; `token_group` is [(name, slug, gecko)]. Token mcap/FDV
    come from one batched CoinGecko call."""
    records = [subject]
    for grp, tickers in equity_groups.items():
        for t in tickers:
            records.append(pull_equity(t, grp))
    mkts = pull_market_technicals([g for _n, _s, g in token_group])
    for name, slug, gecko in token_group:
        records.append(pull_token(name, slug, gecko, market=mkts.get(gecko)))

    return pd.DataFrame(records).set_index('Company')


@report_errors(default={})
def fetch_comp_financial_statements(token_group, use_live=True):
    """Fetch + cache each token comp's DeFiLlama income statement to
    data/financial_statements/{slug}.json. Returns {slug: payload}."""
    out = {}
    for name, slug, _gecko in token_group:
        data = financial_statement(slug, use_live=use_live)
        if data is None:
            print(f'[finstmt] {name} ({slug}): no data (live failed and no cache)')
            continue
        out[slug] = data
    return out
