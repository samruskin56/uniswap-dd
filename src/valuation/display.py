"""Formatting helpers for the valuation notebook output tables.

Each function returns a display-ready (string-formatted) DataFrame so notebook
cells stay one-liners. Numeric kinds:
  usd_mm / usd_mm0  -> $x.xm / $xm        price -> $x.xxxx
  pct0 / pct1       -> x% / x.x%          bps   -> x bps
  mult / x0         -> x.xx / x x         years -> x.xx yr      int -> x

Row labels: DEFAULT_LABELS carries sector-neutral names for the model's stable
row keys; every table takes a `labels` dict merged over it so the notebook can
name the deal's own semantics once, e.g.
    LABELS = {'tam': 'Active Loans TAM', 'units': 'Maple Active Loans',
              'fee_rate': 'Gross Loan Yield', 'revenue': 'Maple Revenue'}
Scenario-driver rows use the same mechanism keyed by the scenario column name
(share_target, fee_rate_exit, ...).
"""

import pandas as pd

from .model import ROW_EBITDA, ROW_REVENUE
from utils.helper_functions import report_errors


def _fmt(value, kind):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return '—'
    if kind == 'usd_mm':   return f'${value:,.1f}m'
    if kind == 'usd_mm0':  return f'${value:,.0f}m'
    if kind == 'price':    return f'${value:,.4f}'
    if kind == 'pct0':     return f'{value * 100:,.0f}%'
    if kind == 'pct1':     return f'{value * 100:,.1f}%'
    if kind == 'bps':      return f'{value:,.0f} bps'
    if kind == 'mult':     return f'{value:,.1f}x'
    if kind == 'x0':       return f'{value:,.0f}x'
    if kind == 'years':    return f'{value:,.2f} yr'
    if kind == 'int':      return f'{value:,.0f}'
    return str(value)


# Sector-neutral default labels for the model's row keys (projection /
# historical statements) and scenario driver columns. Override per deal via the
# `labels` argument on each table.
DEFAULT_LABELS = {
    # projection rows
    'tam':           'Addressable Market (TAM)',
    'tam_growth':    'TAM Growth (p.a.)',
    'share':         'Market Share',
    'units':         'Units',
    'fee_rate':      'Monetization Rate',
    'gross_revenue': 'Gross Revenue',
    'take_rate':     'Take Rate',
    ROW_REVENUE:     'Net Revenue',
    'ebitda_margin': 'EBITDA Margin',
    ROW_EBITDA:      'EBITDA',
    # scenario driver columns
    'prob':               'Probability',
    'mkt_growth':         'TAM Growth (p.a.)',
    'share_target':       'Market Share at Exit',
    'fee_rate_exit':      'Monetization Rate at Exit',
    'take_rate_exit':     'Take Rate at Exit',
    'ebitda_margin_exit': 'EBITDA Margin at Exit',
    'exit_pe':            'Exit FDV/EBITDA',
}


def _labels(labels):
    return {**DEFAULT_LABELS, **(labels or {})}


# ----- (1) assumptions -------------------------------------------------------

# (driver col, kind) for the by-scenario assumptions table.
SCENARIO_ROWS = [
    ('prob',               'pct0'),
    ('mkt_growth',         'pct0'),
    ('share_target',       'pct1'),
    ('fee_rate_exit',      'pct1'),
    ('take_rate_exit',     'pct0'),
    ('ebitda_margin_exit', 'pct0'),
    ('exit_pe',            'x0'),
]


def global_assumptions_table(items):
    """items: list of (label, value, kind). One-column formatted table."""
    return pd.DataFrame(
        [{'Assumption': lbl, 'Value': _fmt(val, kind)} for lbl, val, kind in items]
    ).set_index('Assumption')


def scenario_assumptions_table(scenario, labels=None):
    """By-scenario drivers, rows = drivers, cols = cases."""
    lab = _labels(labels)
    data = {lab[col]: {c: _fmt(scenario.loc[c, col], kind) for c in scenario.index}
            for col, kind in SCENARIO_ROWS}
    return pd.DataFrame(data).T[list(scenario.index)]


# ----- (2-3) statements ------------------------------------------------------

# (row key, kind, indented sub-line) in statement order.
REVENUE_ROWS = [
    ('tam',           'usd_mm0', False),
    ('tam_growth',    'pct0',    True),
    ('share',         'pct1',    True),
    ('units',         'usd_mm0', False),
    ('fee_rate',      'pct1',    True),
    ('gross_revenue', 'usd_mm',  False),
    ('take_rate',     'pct1',    True),
    (ROW_REVENUE,     'usd_mm',  False),
]

INCOME_ROWS = [
    ('tam',           'usd_mm0', False),
    ('share',         'pct1',    True),
    ('units',         'usd_mm0', False),
    ('fee_rate',      'pct1',    True),
    ('gross_revenue', 'usd_mm',  False),
    ('take_rate',     'pct1',    True),
    (ROW_REVENUE,     'usd_mm',  False),
    ('ebitda_margin', 'pct0',    True),
    (ROW_EBITDA,      'usd_mm',  False),
]


def _rowspec(spec, labels):
    """Resolve a (key, kind, indent) spec to (key, display label, kind)."""
    lab = _labels(labels)
    return [(key, ('  ' if indent else '') + lab[key], kind) for key, kind, indent in spec]


def _statement(proj, rowspec):
    cols = list(proj.columns)
    data = {lbl: {c: _fmt(proj.loc[src, c], kind) for c in cols} for src, lbl, kind in rowspec}
    return pd.DataFrame(data).T.loc[[lbl for _, lbl, _ in rowspec]][cols]


def revenue_build_table(proj, labels=None):
    return _statement(proj, _rowspec(REVENUE_ROWS, labels))


def income_statement(proj, hist=None, labels=None):
    """Income statement; `hist` (from model.historical_income) prepends actual
    columns under a Historical | Projected header."""
    rowspec = _rowspec(INCOME_ROWS, labels)
    if hist is None:
        return _statement(proj, rowspec)
    hist_cols, proj_cols = list(hist.columns), list(proj.columns)
    col_tuples = [('Historical', c) for c in hist_cols] + [('Projected', c) for c in proj_cols]
    data, index = [], []
    for src, lbl, kind in rowspec:
        index.append(lbl)
        data.append([_fmt(hist.loc[src, c], kind) for c in hist_cols]
                    + [_fmt(proj.loc[src, c], kind) for c in proj_cols])
    return pd.DataFrame(data, index=index, columns=pd.MultiIndex.from_tuples(col_tuples))


# ----- entry / exit ----------------------------------------------------------

@report_errors()
def entry_exit_report(val, summary, scenario):
    """Print the headline entry multiple + fair value; return the per-case table."""
    s = summary
    print('ENTRY MULTIPLE (at entry): '
          f"{s['entry_mult']:,.0f}x   "
          f"= FDV ${s['fdv_entry']:,.0f}m @ entry ${s['returns_entry']:.4f} "
          f"/ annualized EBITDA ${s['ebitda_now']:,.1f}m")
    print('  fair-value-implied x-check: '
          f"{s['entry_mult_implied']:,.0f}x   "
          f"= implied FDV ${s['implied_fdv_fair']:,.0f}m / same EBITDA")
    print()
    tag = 'UNDERVALUED' if s['prem'] > 0 else 'OVERVALUED'
    print(f"Market price ${s['market_price']:.4f}   |   hurdle {s['hurdle'] * 100:.0f}%   |   "
          f"FAIR VALUE TODAY (prob-wtd PV) ${s['fair_value_today']:.4f}  ->  {s['prem'] * 100:+.0f}% ({tag})")
    print(f"MOIC / IRR entry price: ${s['returns_entry']:.4f}  [= {s['entry_basis']}]")
    print(f"Prob-weighted: {s['pw_moic']:.1f}x MOIC, {s['pw_irr'] * 100:.0f}% IRR   "
          f"(prob-weighted exit price ${s['pw_undisc']:.4f})")
    print()

    rows = []
    for case in scenario.index:
        r = val.loc[case]
        rows.append({
            'Prob':            _fmt(r['prob'], 'pct0'),
            'Exit-Year EBITDA': _fmt(r['exit_ebitda'], 'usd_mm'),
            'Exit FDV/EBITDA': _fmt(r['exit_pe'], 'x0'),
            'Exit Value':      _fmt(r['terminal_ev'], 'usd_mm0'),
            'Exit Price':      _fmt(r['undisc_price'], 'price'),
            'PV Price':        _fmt(r['pv_price'], 'price'),
            'MOIC':            _fmt(r['moic'], 'mult'),
            'IRR':             _fmt(r['irr'], 'pct0'),
        })
    disp = pd.DataFrame(rows, index=list(scenario.index))
    disp.loc['Prob-weighted'] = ['100%', '', '', '',
                                 _fmt(s['pw_undisc'], 'price'),
                                 _fmt(s['fair_value_today'], 'price'),
                                 _fmt(s['pw_moic'], 'mult'),
                                 _fmt(s['pw_irr'], 'pct0')]
    return disp


# ----- MOIC waterfall --------------------------------------------------------

@report_errors()
def moic_waterfall_chart(waterfalls, entry, meta=None):
    """Per-case multiplicative MOIC bridges from model.moic_waterfall, on a
    shared log y-axis. Bar k spans [cum_{k-1}, cum_k]; x1.0 no-ops draw as a
    dash; the grey MOIC total is an absolute column from the axis floor.
    Returns the Figure."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    cases = list(waterfalls)
    fig, axes = plt.subplots(1, len(cases), figsize=(5.2 * len(cases), 4.8), sharey=True)
    axes = [axes] if len(cases) == 1 else list(axes)

    all_vals = [1.0] + [v for df in waterfalls.values() for v in df['cum_moic']]
    ymin, ymax = min(all_vals) / 1.8, max(all_vals) * 1.8

    for ax, case in zip(axes, cases):
        df = waterfalls[case]
        steps = list(df['step'])
        cums = [1.0] + list(df['cum_moic'])
        moic = cums[-1]
        xs = range(len(steps) + 1)

        for i, step in enumerate(steps):
            lo, hi = sorted((cums[i], cums[i + 1]))
            up = df['factor'].iloc[i] >= 1
            if hi > lo:
                ax.bar(i, hi - lo, bottom=lo, color='#2e7d32' if up else '#c62828',
                       alpha=0.85, width=0.62)
            else:
                ax.plot([i - 0.31, i + 0.31], [lo, lo], color='#9e9e9e', lw=2)
            ax.annotate(f"×{df['factor'].iloc[i]:.1f}", (i, hi), textcoords='offset points',
                        xytext=(0, 4), ha='center', fontsize=8)
        ax.bar(len(steps), moic - ymin, bottom=ymin, color='#546e7a', alpha=0.9, width=0.62)
        ax.annotate(f'{moic:.1f}x', (len(steps), moic), textcoords='offset points',
                    xytext=(0, 4), ha='center', fontsize=9, fontweight='bold')

        ax.axhline(1.0, color='black', lw=0.8, ls='--', alpha=0.6)
        ax.set_yscale('log')
        ax.set_ylim(ymin, ymax)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}x'))
        ax.set_xticks(list(xs))
        ax.set_xticklabels(steps + ['MOIC'], rotation=40, ha='right', fontsize=8)
        ax.set_title(f'{case}: {moic:.1f}x', fontsize=10)
        ax.grid(axis='y', alpha=0.25)

    axes[0].set_ylabel('MOIC (multiplicative bridge, log scale)')
    if meta:
        p0s = {f"{m['p0']:.0f}x" for m in meta.values()}
        note = ' / '.join(sorted(p0s))
        fig.text(0.01, -0.02,
                 f'Baseline 1.0x = flat fundamentals sold at the entry-implied multiple P0 = {note}.',
                 fontsize=8, color='#444444')
    fig.suptitle(f'MOIC bridge by assumption (entry ${entry:.4f})', fontsize=11, y=1.02)
    fig.tight_layout()
    return fig


# ----- comps -----------------------------------------------------------------

@report_errors()
def comps_table(comps, subject_name='subject'):
    """Comps table: Revenue (TTM, net take for tokens), EBITDA (TTM), EBITDA
    margin, FDV/Rev, FDV/EBITDA. FDV/EBITDA is the tie-out to the model's exit
    multiple (equities' FDV is proxied by market cap); margins and multiples
    need a positive denominator. Appends a Median (ex-subject) row."""
    work = comps.copy()
    rev = pd.to_numeric(work['Rev'], errors='coerce')
    ebitda = pd.to_numeric(work['EBITDA'], errors='coerce')
    fdv = pd.to_numeric(work['Fdv'], errors='coerce')

    work['_EBMGN'] = ebitda / rev.where(rev > 0)
    work['_FDVREV'] = fdv / rev.where(rev > 0)
    work['_FDVEB'] = fdv / ebitda.where(ebitda > 0)

    disp = ['Rev', 'EBITDA', '_EBMGN', '_FDVREV', '_FDVEB']
    peers = work[work['Group'] != 'SUBJECT']
    med = peers[disp].median(numeric_only=True)
    work.loc[f'Median (ex-{subject_name})'] = {'Group': '', 'Price': float('nan'), 'Mcap': float('nan'),
                                               'Fdv': float('nan'), **med.to_dict()}

    fmt = work.copy()
    fmt['Price'] = fmt['Price'].map(lambda x: f'${x:,.2f}' if pd.notna(x) else '—')
    for c in ['Mcap', 'Fdv', 'Rev', 'EBITDA']:
        fmt[c] = fmt[c].map(lambda x: f'${x:,.0f}m' if pd.notna(x) else '—')
    fmt['_EBMGN'] = fmt['_EBMGN'].map(lambda x: f'{x * 100:,.0f}%' if pd.notna(x) else '—')
    for c in ['_FDVREV', '_FDVEB']:
        fmt[c] = fmt[c].map(lambda x: f'{x:,.1f}x' if pd.notna(x) else '—')
    cols = ['Group', 'Price', 'Mcap', 'Fdv', 'Rev', 'EBITDA', '_EBMGN', '_FDVREV', '_FDVEB']
    out = fmt[cols].rename(columns={'Fdv': 'FDV', 'Rev': 'Revenue (TTM)',
                                    'EBITDA': 'EBITDA (TTM)',
                                    '_EBMGN': 'EBITDA Mgn', '_FDVREV': 'FDV/Rev',
                                    '_FDVEB': 'FDV/EBITDA'})
    out.index.name = None
    return out


@report_errors()
def group_median_table(comps):
    """By-group medians (ex-subject): FDV/Rev, FDV/EBITDA, EBITDA margin."""
    peers = comps[comps['Group'] != 'SUBJECT'].copy()
    rev = pd.to_numeric(peers['Rev'], errors='coerce')
    ebitda = pd.to_numeric(peers['EBITDA'], errors='coerce')
    fdv = pd.to_numeric(peers['Fdv'], errors='coerce')

    order = ['FDV/Rev', 'FDV/EBITDA', 'EBITDA Mgn']
    metrics = pd.DataFrame({
        'Group':       peers['Group'],
        'FDV/Rev':     fdv / rev.where(rev > 0),
        'FDV/EBITDA':  fdv / ebitda.where(ebitda > 0),
        'EBITDA Mgn':  ebitda / rev.where(rev > 0),
    })
    med = metrics.groupby('Group')[order].median(numeric_only=True)

    out = med[order].copy().astype(object)
    out['FDV/Rev'] = med['FDV/Rev'].map(lambda x: f'{x:,.1f}x' if pd.notna(x) else '—')
    out['FDV/EBITDA'] = med['FDV/EBITDA'].map(lambda x: f'{x:,.1f}x' if pd.notna(x) else '—')
    out['EBITDA Mgn'] = med['EBITDA Mgn'].map(lambda x: f'{x * 100:,.0f}%' if pd.notna(x) else '—')
    out.index.name = None
    return out
