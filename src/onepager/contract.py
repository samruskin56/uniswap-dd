"""Field contract for the liquid-token one-pager.

Single source of truth for what the template expects. The notebook builds a
``context`` dict; ``validate()`` checks it against this schema BEFORE rendering
so a missing or misspelled field fails loudly instead of producing a blank cell.

All values are render-ready *display strings* (e.g. "3.2x", "42%", "$1.2b").
Formatting lives in the data/metrics layer, not in the template — the template
is purely structural.
"""

# verdict -> border/text color (mirrors the original artifact's logic)
VERDICT_COLORS = {
    "BUY": "#0A7C42",
    "ACCUMULATE": "#0A7C42",
    "SELL": "#C62828",
    "REDUCE": "#C62828",
    "HOLD": "#5F5F5F",
}
DEFAULT_VERDICT_COLOR = "#1f1f60"

# Plain scalar fields (text you type, or display-formatted numbers).
SCALAR_FIELDS = [
    # header / masthead
    "date", "protocol_name", "ticker", "sector", "verdict", "horizon",
    "base_target",
    # conviction band
    "allocation", "notional", "entry", "base_moic", "base_irr",
    # narrative
    "thesis", "team", "market", "product", "overview", "management", "sentiment",
    # token economics
    "supply", "unlock_years", "unlock_pct", "next_unlock", "fee_switch", "token_rights",
    "equity_capture", "treasury",
    # footer
    "data_sources", "as_of_date",
    # embedded charts (base64 PNG data URIs; may be "" when data is unavailable)
    "price_chart", "traction_chart", "projections_chart",
    # deal-worded template strings (section header / caption / comps footnote;
    # pipeline.build_metrics derives defaults from the cfg labels)
    "traction_title", "traction_caption", "comps_note",
]

# List-of-string fields.
# Author convention: each catalyst and risk line should convey WHEN it lands and
# WHY it matters for the token (free prose, not separate fields).
# `comps_metrics` is the ordered list of traction-metric column labels; each comp
# row carries an `ev` plus a `ratios` list (EV / metric) aligned to it.
LIST_FIELDS = ["catalysts", "risks", "comps_metrics"]

# List-of-dict fields: field name -> required keys per row.
# Comps carry mcap + FDV levels and are valued as FDV / each traction metric
# (`ratios`, aligned to `comps_metrics`); see pipeline.build_metrics.
TABLE_FIELDS = {
    "scenarios": ["name", "price", "moic", "irr", "prob"],   # + optional "highlight"
    "traction": ["metric", "current", "d30", "ath"],         # ath = drawdown from the metric's peak
    "technicals": ["name", "current", "d30", "ath"],         # token price / 30d chg / drawdown from ATH
    "comps": ["name", "mcap", "fdv", "ratios"],
}

# Single-dict fields: field name -> required keys.
OBJECT_FIELDS = {
    "prob_weighted": ["price", "moic", "irr"],
    "comps_target": ["name", "mcap", "fdv", "ratios"],
    "comps_avg": ["mcap", "fdv", "ratios"],
}


def verdict_color(verdict: str) -> str:
    return VERDICT_COLORS.get(str(verdict).upper().strip(), DEFAULT_VERDICT_COLOR)


def validate(context: dict) -> None:
    """Raise ValueError listing every problem with ``context``."""
    errs = []
    expected = set(SCALAR_FIELDS) | set(LIST_FIELDS) | set(TABLE_FIELDS) | set(OBJECT_FIELDS)

    for f in SCALAR_FIELDS:
        if f not in context:
            errs.append(f"missing scalar field: {f!r}")

    for f in LIST_FIELDS:
        if f not in context:
            errs.append(f"missing list field: {f!r}")
        elif not isinstance(context[f], (list, tuple)):
            errs.append(f"{f!r} must be a list, got {type(context[f]).__name__}")

    for f, keys in TABLE_FIELDS.items():
        if f not in context:
            errs.append(f"missing table field: {f!r}")
            continue
        if not isinstance(context[f], (list, tuple)):
            errs.append(f"{f!r} must be a list of rows")
            continue
        for i, row in enumerate(context[f]):
            for k in keys:
                if k not in row:
                    errs.append(f"{f}[{i}] missing key {k!r}")

    for f, keys in OBJECT_FIELDS.items():
        if f not in context:
            errs.append(f"missing object field: {f!r}")
            continue
        for k in keys:
            if k not in context[f]:
                errs.append(f"{f} missing key {k!r}")

    extra = set(context) - expected - {"font_css", "verdict_color", "logo_src"}
    if extra:
        errs.append(f"unexpected fields (typo?): {sorted(extra)}")

    if errs:
        raise ValueError("one-pager context invalid:\n  - " + "\n  - ".join(errs))
