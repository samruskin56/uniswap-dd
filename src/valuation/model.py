"""Scenario projection and target-price valuation for a liquid token.

build_projection rolls the driver tree per case: TAM -> market share -> units
-> monetization rate -> gross revenue -> take rate -> net revenue -> EBITDA
margin -> EBITDA. `units` is the protocol's traction quantity (a stock like
active loans / AUM, or an annual flow like trading volume); `fee_rate` is the
monetization rate on it (a p.a. yield on a stock, a per-unit take on a flow) --
gross_revenue = units x fee_rate either way. EBITDA is the bottom line and a
loose UFCF proxy (no tax / D&A / capex / NWC). Drivers ramp linearly from a
common first-forecast-year start to per-case exits.

Valuation is entry/exit only: exit value = exit-year EBITDA x exit_pe; exit
price = exit value / FDV supply. No interim flows, no net-cash bridge. Fair
value today = prob-weighted PV of the exit price at the hurdle. Entry price per
ENTRY_BASES; MOIC = exit price / entry; IRR = MOIC^(1/hold) - 1.

Row keys are stable machine identifiers; human labels (which name the deal's
own semantics, e.g. "Maple Active Loans") live in the display layer's LABELS
override (see display.DEFAULT_LABELS).
"""

from datetime import date
import pandas as pd

ROW_EBITDA = 'ebitda'     # row key for the model's bottom line (= operating income here; no D&A modeled)
ROW_REVENUE = 'revenue'   # row key for net protocol revenue

# Projection rows in statement order.
ROW_ORDER = ['tam', 'tam_growth', 'share', 'units', 'fee_rate',
             'gross_revenue', 'take_rate', ROW_REVENUE, 'ebitda_margin', ROW_EBITDA]


def make_context(marks, valuation_date, exit_year, tam_start,
                 ebitda_margin_start, fee_rate_start, take_rate_start, share_start):
    """Time / anchor context shared by projection and valuation. Each *_start is
    the FIRST forecast year's level and ramps to the per-case exit target.
    share_now and fee_rate_now are live diagnostics only (None when the marks
    carry no `units`)."""
    forecast_years = list(range(valuation_date.year, exit_year + 1))
    year_labels = [f'FY{y}' for y in forecast_years]
    tau = [(date(y, 12, 31) - valuation_date).days / 365.25 for y in forecast_years]
    units_now = marks.get('units')
    fees_now = marks.get('fees')
    return dict(
        forecast_years=forecast_years,
        year_labels=year_labels,
        tau=tau,
        tau_exit=tau[-1],
        tam_start=tam_start,
        share_now=(units_now / tam_start) if units_now else None,
        share_start=share_start,
        fee_rate_now=(fees_now / units_now) if (fees_now and units_now) else None,
        fee_rate_start=fee_rate_start,
        take_rate_start=take_rate_start,
        ebitda_margin_start=ebitda_margin_start,
    )


def build_projection(case, scenario, ctx):
    """Line-item engine for one case. Rows = ROW_ORDER keys, cols = year-ends
    ($mm). TAM holds the input level in year 1, then compounds the case CAGR.
    The four ramped drivers step linearly from start to the case exit target."""
    s = scenario.loc[case]
    n = len(ctx['tau'])
    steps = max(n - 1, 1)
    years, labels = ctx['forecast_years'], ctx['year_labels']

    base_year = years[0]
    tam = [ctx['tam_start'] * (1 + s.mkt_growth) ** (y - base_year) for y in years]
    tam_growth = [None] + [tam[i] / tam[i - 1] - 1 for i in range(1, n)]
    _ss, _se = ctx['share_start'], s.share_target
    share = [_ss + (_se - _ss) * (i / steps) for i in range(n)]
    units = [tam[i] * share[i] for i in range(n)]

    _fs, _fe = ctx['fee_rate_start'], s.fee_rate_exit
    fee_rate = [_fs + (_fe - _fs) * (i / steps) for i in range(n)]
    gross_revenue = [units[i] * fee_rate[i] for i in range(n)]

    _ts, _te = ctx['take_rate_start'], s.take_rate_exit
    take_rate = [_ts + (_te - _ts) * (i / steps) for i in range(n)]
    revenue = [gross_revenue[i] * take_rate[i] for i in range(n)]

    _ms, _me = ctx['ebitda_margin_start'], s.ebitda_margin_exit
    ebitda_margin = [_ms + (_me - _ms) * (i / steps) for i in range(n)]
    ebitda = [revenue[i] * ebitda_margin[i] for i in range(n)]

    rows = {
        'tam':           tam,
        'tam_growth':    tam_growth,
        'share':         share,
        'units':         units,
        'fee_rate':      fee_rate,
        'gross_revenue': gross_revenue,
        'take_rate':     take_rate,
        ROW_REVENUE:     revenue,
        'ebitda_margin': ebitda_margin,
        ROW_EBITDA:      ebitda,
    }
    return pd.DataFrame(rows, index=labels).T


def historical_income(periods):
    """Historical income-statement columns from DeFiLlama actuals, no model
    assumptions. `periods` = ordered (label, dict) with revenue, fees, tam,
    units, ann_factor (annualizes gross revenue for the monetization-rate row).
    EBITDA left blank: protocol opex is not observed."""
    cols = {}
    for label, inc in periods:
        fees = inc.get('fees')
        rev = inc['revenue']
        tam = inc.get('tam')
        units = inc.get('units')
        af = inc.get('ann_factor', 1.0)
        cols[label] = {
            'tam':           tam,
            'share':         (units / tam) if (units and tam) else None,
            'units':         units,
            'fee_rate':      (fees * af / units) if (fees and units) else None,
            'gross_revenue': fees,
            'take_rate':     (rev / fees) if fees else None,
            ROW_REVENUE:     rev,
            'ebitda_margin': None,
            ROW_EBITDA:      None,
        }
    return pd.DataFrame(cols)


def value_case(case, scenario, ctx, marks, hurdle):
    """Value one case, entry-independent. Exit value = exit-year EBITDA x
    exit_pe; exit price (undisc_price) = exit value / FDV supply; pv_price =
    exit price discounted at the hurdle. No interim flows, no net-cash bridge."""
    proj = build_projection(case, scenario, ctx)
    fdv = marks['fdv_supply']

    pe = scenario.loc[case, 'exit_pe']
    prob = scenario.loc[case, 'prob']
    exit_ebitda = list(proj.loc[ROW_EBITDA])[-1]
    terminal_ev = exit_ebitda * pe
    undisc_price = terminal_ev / fdv
    pv_price = undisc_price / (1 + hurdle) ** ctx['tau_exit']

    return dict(case=case, prob=prob, exit_pe=pe,
                exit_ebitda=exit_ebitda, terminal_ev=terminal_ev,
                undisc_price=undisc_price, pv_price=pv_price)


# Drivers that live in ctx rather than the scenario table (usable in moic_grid).
CTX_DRIVERS = ('tam_start', 'share_start', 'fee_rate_start',
               'take_rate_start', 'ebitda_margin_start')


def moic_grid(case, scenario, ctx, marks, hurdle, entry, row_axis, col_axis):
    """MOIC sensitivity for one case over any two drivers, entry held fixed.
    Each axis = (driver, values); a driver is a scenario column or a CTX_DRIVERS
    key. The cell at the run values reproduces the reported case MOIC. Returns a
    float DataFrame (index = row values, columns = col values)."""
    (row_drv, row_vals), (col_drv, col_vals) = row_axis, col_axis

    def _perturbed(pairs):
        scen = scenario.copy().astype(float)   # exit_pe is int64; keep float grid values exact
        c = dict(ctx)
        for name, v in pairs:
            if name in scen.columns:
                scen.loc[case, name] = v
            elif name in CTX_DRIVERS:
                c[name] = v
            else:
                raise KeyError(f'unknown driver {name!r}')
        return scen, c

    grid = []
    for rv in row_vals:
        row = []
        for cv in col_vals:
            scen, c = _perturbed([(row_drv, rv), (col_drv, cv)])
            row.append(value_case(case, scen, c, marks, hurdle)['undisc_price'] / entry)
        grid.append(row)
    return pd.DataFrame(grid, index=row_vals, columns=col_vals)


# (scenario driver, waterfall bar label). MOIC is a pure product of these, so the
# decomposition is exact and order-independent; order below is display only.
WATERFALL_STEPS = [
    ('mkt_growth',         'TAM growth'),
    ('share_target',       'Share gains'),
    ('fee_rate_exit',      'Monetization'),
    ('take_rate_exit',     'Take rate'),
    ('ebitda_margin_exit', 'Margin expansion'),
    ('exit_pe',            'Exit multiple'),
]


def moic_waterfall(case, scenario, ctx, marks, entry):
    """Multiplicative MOIC bridge from 1.0x at entry to the case MOIC, one exact
    factor per driver. Baseline: flat fundamentals sold at the entry-implied
    multiple P0 = entry x FDV supply / flat FY1 EBITDA (same FDV convention as
    the headline entry multiple, which divides by current annualized EBITDA
    instead of the flat model FY1). Returns (df, meta): df columns
    ['step', 'factor', 'cum_moic']; meta = dict(e0, p0, case_moic). Asserts the
    factors multiply to the value_case MOIC."""
    s = scenario.loc[case]
    n = len(ctx['forecast_years'])
    fdv = marks['fdv_supply']

    e0 = (ctx['tam_start'] * ctx['share_start'] * ctx['fee_rate_start']
          * ctx['take_rate_start'] * ctx['ebitda_margin_start'])
    p0 = entry * fdv / e0

    factors = {
        'mkt_growth':         (1 + s.mkt_growth) ** (n - 1),
        'share_target':       s.share_target / ctx['share_start'],
        'fee_rate_exit':      s.fee_rate_exit / ctx['fee_rate_start'],
        'take_rate_exit':     s.take_rate_exit / ctx['take_rate_start'],
        'ebitda_margin_exit': s.ebitda_margin_exit / ctx['ebitda_margin_start'],
        'exit_pe':            s.exit_pe / p0,
    }

    rows, cum = [], 1.0
    for drv, label in WATERFALL_STEPS:
        f = float(factors[drv])
        cum *= f
        rows.append(dict(step=label, factor=f, cum_moic=cum))
    df = pd.DataFrame(rows)

    case_moic = value_case(case, scenario, ctx, marks, 0.0)['undisc_price'] / entry
    assert abs(cum - case_moic) < 1e-9 * max(1.0, case_moic), \
        f'waterfall does not close: {cum} vs {case_moic}'
    return df, dict(e0=e0, p0=p0, case_moic=case_moic)


def entry_ebitda_now(marks, ebitda_margin_start):
    """Current annualized EBITDA = annualized revenue x the start margin (the
    projection's t=0 basis, behind the display-only entry multiple)."""
    return marks['revenue'] * ebitda_margin_start


# Entry-price bases for run_scenarios.
ENTRY_BASES = ('blended_pv', 'base_pv', 'market', 'manual')


def entry_candidates(scenario, ctx, marks, hurdle, base_case):
    """The data-driven entry candidates behind the ENTRY_BASIS toggle, computable
    before a basis is chosen (value_case is entry-independent). run_scenarios
    resolves the same figures."""
    results = [value_case(c, scenario, ctx, marks, hurdle) for c in scenario.index]
    return dict(
        blended_pv=sum(r['pv_price'] * r['prob'] for r in results),
        base_pv=next(r['pv_price'] for r in results if r['case'] == base_case),
        market=marks['price'],
    )


def run_scenarios(scenario, ctx, marks, hurdle, base_case,
                  entry_basis='blended_pv', entry_price=None):
    """Value every case, probability-weight, set the entry, compute MOIC/IRR.

    entry_basis: 'blended_pv' (prob-weighted fair value; pw IRR = hurdle by
    construction), 'base_pv' (base-case PV; base IRR = hurdle), 'market', or
    'manual' (requires entry_price). MOIC = exit price / entry; IRR =
    MOIC^(1/hold) - 1. The entry multiples in summary are display-only
    diagnostics on the FDV convention (price x FDV supply / current annualized
    EBITDA, same convention as the waterfall's P0 and the exit multiple);
    nothing feeds back into exit_pe. Returns (val_df, summary)."""
    cases = list(scenario.index)
    results = [value_case(c, scenario, ctx, marks, hurdle) for c in cases]

    price = marks['price']
    tau_exit = ctx['tau_exit']

    fair_value_today = sum(r['pv_price'] * r['prob'] for r in results)
    pw_undisc = sum(r['undisc_price'] * r['prob'] for r in results)
    base_pv_price = next(r['pv_price'] for r in results if r['case'] == base_case)

    if entry_basis == 'blended_pv':
        returns_entry, entry_note = fair_value_today, 'blended (prob-weighted) PV'
    elif entry_basis == 'base_pv':
        returns_entry, entry_note = base_pv_price, f'base-case ({base_case}) PV'
    elif entry_basis == 'market':
        returns_entry, entry_note = price, 'current market price'
    elif entry_basis == 'manual':
        if entry_price is None:
            raise ValueError("entry_basis='manual' requires an explicit entry_price")
        returns_entry, entry_note = entry_price, f'manual ENTRY_PRICE (${entry_price:.4f})'
    else:
        raise ValueError(f'unknown entry_basis {entry_basis!r}; expected one of {ENTRY_BASES}')

    for r in results:
        r['moic'] = r['undisc_price'] / returns_entry
        r['irr'] = r['moic'] ** (1 / tau_exit) - 1
    val = pd.DataFrame(results).set_index('case')

    pw_moic = pw_undisc / returns_entry
    pw_irr = pw_moic ** (1 / tau_exit) - 1
    prem = fair_value_today / price - 1

    ebitda_now = entry_ebitda_now(marks, ctx['ebitda_margin_start'])
    fdv_entry = returns_entry * marks['fdv_supply']
    implied_fdv_fair = fair_value_today * marks['fdv_supply']

    summary = dict(returns_entry=returns_entry, entry_basis=entry_note,
                   base_pv_price=base_pv_price, market_price=price, hurdle=hurdle,
                   fair_value_today=fair_value_today, pw_undisc=pw_undisc,
                   pw_moic=pw_moic, pw_irr=pw_irr, prem=prem,
                   ebitda_now=ebitda_now, fdv_entry=fdv_entry,
                   entry_mult=fdv_entry / ebitda_now,
                   implied_fdv_fair=implied_fdv_fair,
                   entry_mult_implied=implied_fdv_fair / ebitda_now)
    return val, summary
