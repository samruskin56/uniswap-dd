"""Compact, house-style charts for the one-pager, returned as base64 PNG data URIs.

These reuse the RV chart palette and the shared ``format_large_numbers`` y-axis
formatter so they match the standalone report figures in ``src/visualization``,
but are sized for the one-pager and drop the per-chart logo / source note (the page
already carries the RV masthead and a sources footer).

Each builder returns a ``data:image/png;base64,...`` string the template embeds in an
``<img>`` (keeping the one-pager fully self-contained), or ``""`` when the input data
is missing so the template can omit the section rather than render a broken image.
"""
import base64
import datetime as dt
import io
import math

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

from utils.helper_functions import format_large_numbers, report_errors

# RV house palette (mirrors the color_list in src/visualization).
NAVY, MID, LIGHT = "#1f1f60", "#313EA5", "#637BBF"
INK, MUTE, GRID = "#333333", "#8A8A8A", "#CCCCCC"


def _nice_step(span, target=6):
    """A "nice" 1/2/2.5/5 x 10^k tick step for ~`target` divisions of `span`."""
    if span <= 0:
        return 1.0
    raw = span / target
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def _data_uri(fig, dpi=200):
    """Save a figure to a transparent base64 PNG data URI and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@report_errors(default="")
def price_chart(series, width_in=3.7, height_in=1.05):
    """``[(ts, price)]`` -> a navy price line with a dated x-axis and a USD y-axis.

    Plots the full series the caller passes (pull the max timeframe for a true
    "since launch" view). Returns "" when there are too few points to draw a line.
    """
    if not series or len(series) < 2:
        return ""
    xs = [dt.datetime.fromtimestamp(ts, dt.timezone.utc) for ts, _ in series]
    ys = [p for _, p in series]
    lo, hi = min(ys), max(ys)

    fig, ax = plt.subplots(figsize=(width_in, height_in))

    # y-axis: snap the limits to "nice" round prices so the axis edges are themselves
    # ticks -- this guarantees the lowest price band (e.g. $0.10) is labelled rather
    # than floating between gridlines.
    step = _nice_step(hi - lo)
    bottom = math.floor(lo / step) * step
    top = math.ceil(hi / step) * step
    n = int(round((top - bottom) / step))
    ax.set_yticks([bottom + i * step for i in range(n + 1)])
    ax.set_ylim(bottom, top)

    ax.plot(xs, ys, color=NAVY, linewidth=1.4, solid_capstyle="round")
    ax.fill_between(xs, ys, bottom, color=NAVY, alpha=0.08)
    ax.plot([xs[-1]], [ys[-1]], "o", color=NAVY, markersize=3.2)   # last-price dot
    ax.margins(x=0.01)

    # x-axis: dated ticks spanning the full history; y-axis: USD price (cents for
    # sub-dollar tokens). Light grid + soft spines keep it in the one-pager palette.
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))
    ax.tick_params(axis="both", labelsize=7, colors=INK, length=0)
    ax.grid(axis="y", linestyle="--", alpha=0.45, color=GRID)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    return _data_uri(fig)


@report_errors(default="")
def traction_chart(units, revenue, units_label="Units", width_in=3.9, height_in=1.05):
    """Units (active loans / AUM / TVL; navy area, left axis) + annualized revenue
    (line, right axis).

    Both inputs are ``[(ts, $mm)]`` histories on very different scales (units often
    ~100x revenue), so revenue rides a secondary axis to stay visible. `units_label`
    names the deal's traction metric in the legend. Both axes use the house
    ``format_large_numbers`` formatter. Returns "" if either series is too short.
    """
    if not units or len(units) < 2 or not revenue or len(revenue) < 2:
        return ""
    bx = [dt.datetime.fromtimestamp(ts, dt.timezone.utc) for ts, _ in units]
    by = [float(v) * 1e6 for _, v in units]         # $mm -> $ for the formatter
    rx = [dt.datetime.fromtimestamp(ts, dt.timezone.utc) for ts, _ in revenue]
    ry = [float(v) * 1e6 for _, v in revenue]

    fig, ax = plt.subplots(figsize=(width_in, height_in))
    ax.fill_between(bx, by, 0, color=NAVY, alpha=0.10)
    l1, = ax.plot(bx, by, color=NAVY, linewidth=1.4, solid_capstyle="round",
                  label=units_label)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(format_large_numbers(max(by) if by else 1))

    ax2 = ax.twinx()                                # revenue on its own scale
    l2, = ax2.plot(rx, ry, color=MID, linewidth=1.4, solid_capstyle="round",
                   label="Ann. revenue")
    ax2.set_ylim(bottom=0)
    ax2.yaxis.set_major_formatter(format_large_numbers(max(ry) if ry else 1))

    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    for a in (ax, ax2):
        a.margins(x=0.01)
        a.tick_params(axis="both", labelsize=7, colors=INK, length=0)
        a.spines["top"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.45, color=GRID)
    for side in ("left", "right", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.legend(handles=[l1, l2], loc="upper left", fontsize=7, frameon=False,
              handlelength=1.4, borderpad=0, labelspacing=0.3)
    return _data_uri(fig)


@report_errors(default="")
def projections_chart(proj, revenue_row="revenue", ebitda_row="ebitda",
                      width_in=3.9, height_in=1.0):
    """Base-case Revenue (area) + EBITDA (line) over the forecast.

    ``proj`` is a model.build_projection frame ($mm; rows = line items, cols = years);
    `revenue_row` / `ebitda_row` are its row keys (defaults match valuation.model's
    ROW_REVENUE / ROW_EBITDA). EBITDA is the model's bottom line (a loose UFCF proxy),
    so we plot it once to avoid a duplicate line. Returns "" if the rows are absent.
    """
    try:
        years = [str(c) for c in proj.columns]
        rev = [float(v) * 1e6 for v in proj.loc[revenue_row]]          # $mm -> $ for the formatter
        ebitda = [float(v) * 1e6 for v in proj.loc[ebitda_row]]
    except Exception:
        return ""
    if not years:
        return ""

    x = list(range(len(years)))
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    ax.fill_between(x, rev, color=NAVY, alpha=0.12)
    ax.plot(x, rev, color=NAVY, linewidth=1.8, marker="o", markersize=3, label="Revenue")
    ax.plot(x, ebitda, color=LIGHT, linewidth=1.8, marker="o", markersize=3,
            label="EBITDA")

    ax.set_ylim(bottom=0)
    ax.margins(x=0.03)
    ax.yaxis.set_major_formatter(format_large_numbers(max(rev) if rev else 1))
    ax.set_xticks(x)
    ax.set_xticklabels(years, fontsize=7.5, color=INK)
    ax.tick_params(axis="y", labelsize=7.5, colors=INK, length=0)
    ax.grid(axis="y", linestyle="--", alpha=0.45, color=GRID)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.legend(loc="upper left", fontsize=7.5, frameon=False, handlelength=1.4,
              borderpad=0, labelspacing=0.3)
    return _data_uri(fig)
