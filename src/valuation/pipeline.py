"""Formatting bridge from the valuation notebook to the one-pager.

The notebook is the single author of every assumption; build_metrics only
formats the notebook's objects into the display-string NUMBERS dict the
one-pager template expects. It never re-derives or defaults a valuation input,
so the one-pager cannot diverge from the notebook. `manual` carries the few
non-modeled fields (chiefly position sizing).

    from valuation.pipeline import build_metrics
    NUMBERS = build_metrics(valuation, comps_df=comps_df, notional_m=2.5)
"""

from datetime import date, datetime, timezone

import pandas as pd

from . import marks as marks_mod, comps as comps_mod
from utils.helper_functions import report_errors

# Everything build_metrics needs beyond the valuation objects themselves; the notebook
# passes these from its own knobs (there are NO module-level defaults to fall back to).
# protocol_label / unit_label / units_metric name the deal's semantics (e.g. "Maple",
# "Active loans", 'borrowed').
_REQUIRED_CFG = ("base_case", "defillama_slug", "gecko_id",
                 "annualize_window_days", "peer_comps",
                 "protocol_label", "unit_label", "units_metric")


# ---- display-string formatters (the template expects pre-formatted strings) ----
def _price(x):
    return f"${x:,.2f}" if abs(x) >= 0.10 else f"${x:,.3f}"


def _mm(x):
    """$ value in millions -> $X.Xb / $XXXm."""
    return f"${x / 1000:,.1f}b" if abs(x) >= 1000 else f"${x:,.0f}m"


def _cnt(x):
    """Token count in millions -> X.XXb / XXXm."""
    return f"{x / 1000:,.2f}b" if abs(x) >= 1000 else f"{x:,.0f}m"


def _x(x):
    return f"{x:,.1f}x"


def _pct(x):
    return f"{x * 100:,.0f}%"


def _signed_pct(x):
    """Signed percent for a change/drawdown fraction, e.g. +0.08 -> '+8%'."""
    return f"{x * 100:+,.0f}%"


def _delta_30d(ser):
    """Trailing-30d % change of a sorted [(ts, value)] series, signed, or 'n/a'.

    Compares the last point to the latest point at least 30 days earlier."""
    if len(ser) < 2:
        return "n/a"
    last_ts, last_v = ser[-1]
    cutoff = last_ts - 30 * 86400
    prior = next((v for ts, v in reversed(ser) if ts <= cutoff), None)
    if not prior:
        return "n/a"
    return _signed_pct(last_v / prior - 1)


def _drawdown_ath(ser):
    """Drawdown of a sorted [(ts, value)] series from its all-time high, or 'n/a'.

    Returns a signed percent (<= 0); 0% means the latest point is the ATH."""
    if not ser:
        return "n/a"
    peak = max(v for _, v in ser)
    if peak <= 0:
        return "n/a"
    return _signed_pct(ser[-1][1] / peak - 1)


# DeFiLlama "Value Accrual" -> short label for the token_rights line.
_ACCRUAL_LABEL = {"Buybacks": "Buyback", "Buyback": "Buyback",
                  "Dividends": "Rev", "Revenue Share": "Rev",
                  "Burns": "Burn", "Burn": "Burn"}


def _fee_switch_str(tr):
    """DeFiLlama 'Fee Switch Status' ('ON'/'OFF'/...) -> 'On'/'Off', or None."""
    s = (tr.get("Fee Switch Status") or "").strip()
    return s.title() if s else None


def _token_rights_str(tr):
    """"Gov / Buyback" from a token-rights record's Token Type + Value Accrual, or None.

    Governance prefix when the token confers governance; value-accrual suffix mapped
    to a short label (Buybacks -> Buyback, Dividends/Revenue Share -> Rev, ...)."""
    parts = []
    if "Governance" in (tr.get("Token Type") or []):
        parts.append("Gov")
    accrual = (tr.get("Value Accrual") or "").strip()
    if accrual:
        parts.append(_ACCRUAL_LABEL.get(accrual, accrual))
    return " / ".join(parts) if parts else None


# DeFiLlama "Equity Revenue Capture" flag -> short label for the Value Capture block.
_EQUITY_CAPTURE_LABEL = {"No": "None (single token)", "Partial": "Partial",
                         "Yes": "Yes", "Unknown": "n/a"}


def _equity_capture_str(tr):
    """DeFiLlama 'Equity Revenue Capture' (No/Partial/Yes) as a short label, or None.

    Whether a separate equity layer competes with the token for protocol value."""
    cap = (tr.get("Equity Revenue Capture") or "").strip()
    return _EQUITY_CAPTURE_LABEL.get(cap, cap) if cap else None


def _next_unlock_str(m):
    """"Sep 2026 · +0.6% circ" from the marks' next-unlock fields, or "n/a".

    % is the next event's tokens as a share of current circulating supply; a linear
    remainder is tagged so a linear vest isn't misread as a cliff."""
    ts, mm, circ = m.get("next_unlock_ts"), m.get("next_unlock_mm"), m.get("circ_supply")
    if not ts or not mm or not circ:
        return "n/a"
    when = datetime.fromtimestamp(ts, timezone.utc).strftime("%b %Y")
    p = mm / circ * 100
    pct = f"+{p:.1f}% circ" if p < 10 else f"+{p:.0f}% circ"
    if m.get("next_unlock_kind") == "linear":
        pct += " (linear)"
    return f"{when} · {pct}"


def _peer_comp_rows(cfg, comps_df):
    """Raw peer records (dicts with Mcap/Fdv/Rev/Tvl/Units), in cfg['peer_comps'] order.

    Reuses an already-computed comps_df (from build_comps) when given; otherwise
    pulls just the named token peers live. Both paths produce the same keys.
    """
    names = [p[0] for p in cfg["peer_comps"]]
    if comps_df is not None and len(comps_df):
        sub = comps_df[comps_df.index.isin(names)]
        recs = [{"Company": idx, **row} for idx, row in sub.to_dict("index").items()]
        recs.sort(key=lambda r: names.index(r["Company"]) if r["Company"] in names else 99)
        return recs
    mkts = comps_mod.pull_market_technicals([g for _n, _s, g in cfg["peer_comps"]])
    return [comps_mod.pull_token(n, s, g, market=mkts.get(g)) for n, s, g in cfg["peer_comps"]]


@report_errors(reraise=True)
def build_metrics(valuation, manual=None, comps_df=None,
                  notional_m=None, fund_size_m=50.0):
    """Return the NUMBERS dict for the one-pager from the notebook's valuation objects.

    `valuation` (required) is the dict of objects the notebook computed: cfg, marks,
    fdv_now, scenario, ctx, val, summary, meta. Auto-fills every model-derived field,
    including the comps table (subject + the `peer_comps` peers on mcap + FDV levels,
    valued as FDV / each traction metric), the technicals table (CoinGecko price / 30d
    change / ATH drawdown for the same set), and the traction table (DeFiLlama TVL,
    units and revenue with 30d change and ATH drawdown). cfg['comps_specs'] (optional)
    overrides the comps columns: a list of dicts with col (column label), subject
    (the subject's metric value, $mm), peer_key (the pull_token record key, e.g.
    'Units'/'Rev'/'Tvl'). Position sizing: pass `notional_m` ($mm deployed) and
    allocation is derived as notional / `fund_size_m` (fund size, default $50m).
    `manual` (dict) overrides/adds anything (e.g. allocation/notional, a hand-set
    table, or the traction_title/traction_caption/comps_note strings). Pass
    `comps_df` to reuse already-pulled peers.
    """
    v = valuation
    m, scen, val, summ, ctx = v["marks"], v["scenario"], v["val"], v["summary"], v["ctx"]
    missing = [k for k in _REQUIRED_CFG if k not in v.get("cfg", {})]
    if missing:
        raise ValueError(f'valuation["cfg"] is missing {missing}; '
                         f'pass them from the notebook knobs (no module defaults exist)')
    cfg = v["cfg"]
    base_case = cfg["base_case"]
    protocol_label = cfg["protocol_label"]
    unit_label = cfg["unit_label"]
    units_metric = cfg["units_metric"]

    # entry: always "target / spot" so the headline MOIC/IRR are clearly stated
    # at the underwritten entry, not at today's market price
    e, mk = summ["returns_entry"], summ["market_price"]
    entry = f"{_price(e)} / {_price(mk)}"

    horizon = f"{ctx['tau_exit']:.1f} yrs"

    scenarios = [
        dict(name=c,
             price=_price(val.loc[c, "undisc_price"]),
             moic=_x(val.loc[c, "moic"]),
             irr=_pct(val.loc[c, "irr"]),
             prob=_pct(scen.loc[c, "prob"]),
             highlight=(c == base_case))
        for c in scen.index
    ]

    maxs, circ = m["fdv_supply"], m["circ_supply"]

    # --- traction metrics (the traction table only): `subject` is the protocol's
    # current value (live marks), `hist` its DeFiLlama series for the 30d Δ / ATH Δ.
    # Revenue here is the protocol's own (net) annualized run-rate; the comps section
    # below uses its own, TTM revenue basis. The units row appears only when
    # units_metric='borrowed' (a distinct series; 'tvl' would duplicate the TVL row,
    # and a notebook-supplied units mark has no history to diff). ---
    slug, win = cfg["defillama_slug"], cfg["annualize_window_days"]
    tvl_hist = marks_mod.tvl_history_mm(slug)
    units_hist = (marks_mod.units_history_mm(slug, units_metric)
                  if units_metric == 'borrowed' else [])
    revenue_hist = marks_mod.revenue_ann_history_mm(slug, win)
    # Current TVL prefers the mark, but reused/older marks (stale kernel object or a
    # pre-`tvl` snapshot) may lack it; fall back to the pulled history's last point.
    tvl_now = m["tvl"] if "tvl" in m else (tvl_hist[-1][1] if tvl_hist else None)
    traction_metrics = [
        dict(label="TVL", subject=tvl_now, hist=tvl_hist),
    ]
    if units_hist:
        traction_metrics.append(dict(label=unit_label, subject=m.get("units"), hist=units_hist))
    traction_metrics.append(dict(label="Annualized revenue", subject=m.get("revenue"),
                                 hist=revenue_hist))

    # --- comps: subject + peers on mcap + FDV levels, valued as FDV / each metric (the
    # model's supply basis; EV deprecated). Revenue is TTM NET protocol revenue
    # (DeFiLlama trailing-12m dailyRevenue, the protocol's own take), matching the
    # notebook comps and the model's revenue line; units are point-in-time stocks. ---
    fdv_now = v["fdv_now"]
    subject_net_ttm = (marks_mod.defillama_income_ttm(slug) or {}).get("revenue") or m.get("revenue")

    def _ratio(num, denom):
        """num / metric as a display multiple, or None when either side is missing/zero."""
        if num is None or not pd.notna(num) or denom is None or not pd.notna(denom) or not denom:
            return None
        return num / denom

    comps_specs = cfg.get("comps_specs") or [
        dict(col=unit_label, subject=m.get("units"), peer_key="Units"),
        dict(col="Rev",      subject=subject_net_ttm, peer_key="Rev"),
    ]
    comps_metrics = [s["col"] for s in comps_specs]            # column labels: FDV/<col>
    comps_target = dict(
        name=protocol_label, mcap=_mm(m["mcap"]), fdv=_mm(fdv_now),
        ratios=[_x(r) if (r := _ratio(fdv_now, s["subject"])) is not None else "n/a"
                for s in comps_specs],
    )
    comps_list = []
    ratio_acc = [[] for _ in comps_specs]                      # per-metric peer ratios to average
    for rec in _peer_comp_rows(cfg, comps_df):
        fdv, mcap = rec.get("Fdv"), rec.get("Mcap")
        ratios = []
        for i, s in enumerate(comps_specs):
            r = _ratio(fdv, rec.get(s["peer_key"]))
            ratios.append(_x(r) if r is not None else "n/a")
            if r is not None:
                ratio_acc[i].append(r)
        comps_list.append(dict(name=rec.get("Company"),
                               mcap=_mm(mcap) if pd.notna(mcap) else "n/a",
                               fdv=_mm(fdv) if pd.notna(fdv) else "n/a", ratios=ratios))
    # Only the multiples are averaged (absolute levels aren't meaningful to average).
    comps_avg = dict(mcap="n/a", fdv="n/a",
                     ratios=[_x(sum(a) / len(a)) if a else "n/a" for a in ratio_acc])

    # --- technicals: price / 30d change / ATH drawdown for the same protocols ---
    # CoinGecko (one call) for the subject + the peer comps, keyed by gecko id. Price
    # is stated from this same source so every row in the table shares one basis.
    tech_specs = [(comps_target["name"], cfg["gecko_id"])] + \
                 [(n, g) for n, _s, g in cfg["peer_comps"]]
    tech = comps_mod.pull_market_technicals([g for _n, g in tech_specs])
    technicals = []
    for name, gid in tech_specs:
        t = tech.get(gid, {})
        pr, c30, dd = t.get("price"), t.get("chg_30d"), t.get("ath_dd")
        technicals.append(dict(
            name=name,
            current=_price(pr) if pr is not None else "n/a",
            d30=_signed_pct(c30) if c30 is not None else "n/a",
            ath=_signed_pct(dd) if dd is not None else "n/a",
        ))

    # --- traction: current / 30d change / ATH drawdown for each traction metric,
    # auto-derived from the DeFiLlama history on the same basis as `current`. ---
    traction = [
        dict(metric=tm["label"],
             current=_mm(tm["subject"]) if tm["subject"] is not None else "n/a",
             d30=_delta_30d(tm["hist"]), ath=_drawdown_ath(tm["hist"]))
        for tm in traction_metrics
    ]

    # --- embedded charts (base64 PNGs, house style): the price history (technicals)
    # and a traction chart of units (or TVL) vs annualized revenue, reusing the
    # histories already pulled above. Built here so the one-pager stays self-contained;
    # each degrades to "" (template omits the section) if data or matplotlib is absent.
    # The financial projections chart is disabled for now (kept as "" so it's skipped). ---
    try:
        from onepager import charts
        price_series = marks_mod.price_history_usd(cfg["gecko_id"])
        price_chart = charts.price_chart(price_series)
        area_hist, area_label = (units_hist, unit_label) if units_hist else (tvl_hist, "TVL")
        traction_chart = charts.traction_chart(area_hist, revenue_hist, units_label=area_label)
        traction_area_label = area_label
    except Exception:
        price_chart = traction_chart = ""
        traction_area_label = unit_label
    projections_chart = ""

    # --- token rights (DeFiLlama token-rights dataset): fee switch, value-accrual
    # rights, and the equity stance. Each falls back to "n/a" when DeFiLlama is empty. ---
    tr = marks_mod.token_rights(slug)

    out = dict(
        date=date.today().isoformat(),
        as_of_date=(v["meta"].get("asof") or date.today().isoformat())[:10],
        horizon=horizon,
        base_target=_price(val.loc[base_case, "undisc_price"]),
        entry=entry,
        base_moic=_x(val.loc[base_case, "moic"]),
        base_irr=_pct(val.loc[base_case, "irr"]),
        supply=f"{_cnt(maxs)} / {_cnt(circ)} / {circ / maxs * 100:.0f}%",
        unlock_years=horizon,                    # forward emissions over the hold (proxy)
        # emissions / treasury come from DeFiLlama Pro endpoints; "n/a" without a key
        unlock_pct=(f"+{m['emissions_hold_mm'] / circ * 100:.1f}% circ"
                    if m.get("emissions_hold_mm") is not None else "n/a"),
        next_unlock=_next_unlock_str(m),         # next cliff, or linear-vest completion
        treasury=(f"{_mm(m['treasury_stables'] + m['treasury_own'])} / "
                  f"{_mm(m['treasury_stables'])} / {_mm(m['treasury_own'])}"
                  if m.get("treasury_stables") is not None else "n/a"),
        fee_switch=_fee_switch_str(tr) or "n/a",
        token_rights=_token_rights_str(tr) or "n/a",
        equity_capture=_equity_capture_str(tr) or "n/a",
        scenarios=scenarios,
        prob_weighted=dict(price=_price(summ["pw_undisc"]),
                           moic=_x(summ["pw_moic"]),
                           irr=_pct(summ["pw_irr"])),
        comps_target=comps_target,
        comps=comps_list,
        comps_avg=comps_avg,
        comps_metrics=comps_metrics,
        technicals=technicals,
        traction=traction,
        price_chart=price_chart,
        traction_chart=traction_chart,
        projections_chart=projections_chart,
        # --- deal-worded template strings (overridable via `manual`) ---
        traction_title=f"{traction_area_label} & Revenue",
        traction_caption=(f"{traction_area_label} (left) and trailing-{win}d annualized "
                          f"revenue (right), historical. Source: DefiLlama."),
        comps_note=(f"FDV = token price x max supply; Mcap = circulating. Rev = TTM net "
                    f"protocol revenue (DefiLlama); {unit_label.lower()} = current level. "
                    f"Multiples on FDV, the model's supply basis."),
        # --- position sizing: allocation = notional / fund size (your decision) ---
        allocation=f"{notional_m / fund_size_m * 100:.1f}%" if notional_m is not None else "n/a",
        notional=f"${notional_m:g}m" if notional_m is not None else "n/a",
    )
    if manual:
        out.update(manual)
    return out
