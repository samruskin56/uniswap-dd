# dd-tools

Due-diligence tooling for crypto VC research. Two layers:

1. **Liquid-token valuation & one-pager**: a target-price valuation engine
   (`src/valuation`), a self-contained IC one-pager generator (`src/onepager`),
   and the deal notebook template that drives them
   (`liquid-valuation-template.ipynb`).
2. **Generic data tooling**: raw market-data pulls from several APIs,
   processing into comparable metrics, charts, and a generic scenario module
   (used by the example notebooks).

## Repository structure

```
dd-tools/
├── src/
│   ├── valuation/                # Liquid-token valuation engine (deal-agnostic)
│   │   ├── marks.py              # Live DeFiLlama marks pull + on-disk snapshot cache + FDV helper
│   │   ├── model.py              # Scenario projection + target-price valuation, moic_grid, moic_waterfall
│   │   ├── comps.py              # Comparable set (yfinance equities + DeFiLlama/CoinGecko tokens, FDV basis)
│   │   ├── display.py            # Display-string formatting helpers + waterfall chart
│   │   └── pipeline.py           # Formats the notebook's valuation into the one-pager NUMBERS dict
│   ├── onepager/                 # Reusable liquid-token one-pager generator
│   │   ├── contract.py           # Field contract: single source of truth + validate()
│   │   ├── render.py             # render(context) / save() -> self-contained HTML
│   │   ├── charts.py             # Compact house-style charts as base64 PNG data URIs
│   │   └── assets/               # Embedded Montserrat font + RV lockup logo
│   ├── data/                     # Data pulls + processing (example notebooks)
│   │   ├── dune.py               # Dune Analytics
│   │   ├── fred.py               # FRED (macro series, e.g. M2)
│   │   ├── messari.py            # Messari (assets, networks, stablecoins, exchanges)
│   │   ├── blockworks.py         # Blockworks metrics
│   │   ├── metadao.py            # MetaDAO launchpad: on-chain ICO prices + market API swaps (hourly VWAP)
│   │   ├── hyperliquid.py        # Hyperliquid AQAv2 reserve yield: official daily series, 30-day periods, payments, Fed-path estimate
│   │   ├── defillama.py          # DefiLlama (fees, revenue, TVL, stablecoins, OI)
│   │   ├── yahoofinance.py       # Yahoo Finance (equity financials & market caps)
│   │   ├── data_processing.py    # Category aggregation, multiples, volatility
│   │   └── protocol_index.py     # Helpers for reading config/protocol_index.json
│   ├── visualization/            # Chart-building functions
│   └── scenario_analysis/        # Investment return scenario modeling (see its README)
├── config/
│   ├── protocol_index.json       # Protocol & equity name/slug mappings across APIs
│   └── metadao_launches.json     # MetaDAO launchpad raises (ICO price, dates, pair); filled only where missing
├── templates/
│   └── liquid-one-pager.html.j2  # One-pager design as a Jinja template
├── utils/
│   └── helper_functions.py       # Shared fetch/format/IO helpers
├── data/                         # Per-deal caches: marks snapshot, income statements (git-ignored)
├── outputs/                      # Rendered one-pager HTML (git-ignored)
├── liquid-valuation-template.ipynb  # THE deal notebook template: valuation + thesis + one-pager
├── example-pull-raw-data.ipynb   # Walkthrough: pulling raw data
├── example-analysis.ipynb        # Walkthrough: processing & charting
├── investment-memo-template.ipynb
├── API_KEYS.env.example          # Template for your git-ignored API_KEYS.env
└── requirements.txt
```

## Setup

Requires **Python 3.12+**.

```bash
# from the repository root
python -m venv .venv
.venv\Scripts\activate        # Windows (PowerShell)
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

## API keys

The data-pull modules need API keys. Keys are loaded (via `python-dotenv`) from
`API_KEYS.env` in the repository root — this file is git-ignored, so create it
locally. Copy the tracked template [`API_KEYS.env.example`](API_KEYS.env.example)
and fill in your own keys:

```bash
cp API_KEYS.env.example API_KEYS.env     # macOS / Linux
copy API_KEYS.env.example API_KEYS.env   # Windows
```

```
DEFILLAMA_API_KEY=...
DUNE_API_KEY=...
FRED_API_KEY=...
MESSARI_API_KEY=...
BLOCKWORKS_API_KEY=...
COINGECKO_API_KEY=...
TWITTER_API_KEY=...
```

(Yahoo Finance, MetaDAO's market API and Hyperliquid's `/info` endpoint do not require a key.
`SOLANA_RPC_URL` is optional: the MetaDAO launch-account pull defaults to Solana's public RPC.
`HYPEREVM_ARCHIVE_RPC_URL` is optional: the AQAv2 treasury-balance tie-out defaults to a list of
public HyperEVM archive RPCs; `FRED_API_KEY` is needed for the EFFR reference rate.)

`DEFILLAMA_API_KEY` (DeFiLlama Pro) is optional everywhere but recommended.
Every DeFiLlama URL is built by `utils.helper_functions.llama_url`: with the
key set, all pulls, including the free-mirrored endpoints, run on
`pro-api.llama.fi` (1,000 req/min, so the raw-data pulls use 16 workers
instead of 6), and the Pro-only data becomes available: emissions, treasuries,
the category TAM history behind `data/units_history.json`, the daily token
incentives behind the derived `earnings` series (revenue - incentives,
DefiLlama's Earnings; `data_processing.build_earnings_series`, tied out
monthly against the financial statement by `check_incentives_tieout`), and
the financial statement that supplies the peers' statements. Without it the free hosts serve, FDV falls
back to CoinGecko, and the Pro-only fields show as n/a. The other keys serve the example notebooks.
`BLOCKWORKS_API_KEY` is a Blockworks Research API key with a monthly usage
cap; once the cap is hit the Blockworks pull stops after one request, prints
the reset estimate and reports the skipped pairs instead of retrying.

## Liquid-token valuation & one-pager

### Starting a new deal

1. Copy this repo (or just `liquid-valuation-template.ipynb`, `src/`,
   `templates/` and `utils/`) into the deal's working directory.
2. Fill in the **DEAL CONFIG** cell at the top of the notebook: protocol name,
   DeFiLlama slug, CoinGecko id, `UNITS_METRIC` / `UNIT_LABEL`, TAM, hurdle,
   exit year, peer comps, and any `LABELS` display overrides.
3. Set the Bear / Base / Bull driver table and the `*_START` levels (live
   values print alongside for reference), then Run All.
4. Write the NARRATIVE block (thesis, catalysts, risks), set position sizing
   in MANUAL, and re-run the one-pager section. The rendered deliverable lands
   in `outputs/<slug>-one-pager.html`.

The shipped example values run Aave end-to-end, so a fresh clone works out of
the box. Keep exploratory analysis in the notebook; the one-pager is a
fixed-layout deliverable, not a scratchpad.

### The driver tree

The model is sector-neutral. Row keys are stable identifiers; each deal maps
its own semantics through the config and `LABELS`:

| Row key | Default label | Lending deal | DEX deal |
|---|---|---|---|
| `tam` | Addressable Market (TAM) | crypto loan TAM ($ borrowed) | onchain volume TAM ($/yr) |
| `share` | Market Share | share of category borrowed | share of DEX volume |
| `units` | (your `UNIT_LABEL`) | active loans (a stock) | trading volume (a flow) |
| `fee_rate` | Monetization Rate | gross loan yield (p.a.) | avg fee take on volume |
| `gross_revenue` | Gross Revenue | gross protocol fees | total swap fees |
| `take_rate` | Take Rate | protocol take of fees | fee-switch share |
| `revenue` | Net Revenue | net protocol revenue | protocol revenue |
| `ebitda_margin`, `ebitda` | EBITDA Margin / EBITDA | same | same |

For a stock-type `units` (loans, AUM, TVL) the monetization rate is a p.a.
yield; for a flow-type `units` (annual volume) it is the take per unit of flow.
The math is identical either way: `gross_revenue = units x fee_rate`.
`UNITS_METRIC` selects the DeFiLlama source for the live units mark
(`'borrowed'`, `'tvl'`, or `None` to supply it yourself in the notebook).

### How the valuation model works

The model values the token the way you would underwrite a listed equity: a
**target-price model** with Bear / Base / Bull cases, not a textbook DCF and
not an LBO. Everything is orchestrated by the notebook; the math lives in
`src/valuation/model.py`.

**1. Fundamental build (per case).** A bottom-up engine projects each year
from the valuation date to the exit year through the driver tree above. TAM is
held at its input level in the first forecast year, then compounds the case
CAGR. Market share, monetization rate, take rate and EBITDA margin each start
at a common first-year level (the `*_START` knobs, set near live DeFiLlama
marks) and ramp by a constant annual step to the per-case exit target in the
scenario table. EBITDA is the bottom line and a loose proxy for UFCF: tax,
D&A, capex and working-capital changes are not modeled.

**2. Exit value.** Per case, exit value = exit-year EBITDA x the exit
FDV/EBITDA multiple (`exit_pe`, hard-coded per case: it is an assumption you
set and sanity-check against the comps, not a derived quantity). Exit price
per token = exit value / fully-diluted supply. FDV (max supply x price) and
mcap (circulating x price) are the only supply metrics; there is no
EV-to-equity (net-cash) bridge, so treasury is not added to the token value.

**3. Returns are entry/exit only.** No interim cash flows are credited to the
holder: buybacks and revenue share accrue through the price, so they are
captured in the exit multiple rather than as separate flows.

- **Fair value today** = probability-weighted PV of the exit price at the
  hurdle. This is an underwriting number, the price at which the position
  clears the hurdle in expectation, not an intrinsic-value DCF.
- **Entry price** is a toggle (`ENTRY_BASIS`, with the candidate prices printed
  in the assumptions table): blended (prob-weighted) fair value, base-case PV,
  current market price, or a manual `ENTRY_PRICE`.
- **MOIC** = exit price / entry (undiscounted). **IRR** = MOIC^(1/hold) - 1,
  closed form. With the blended-PV entry the prob-weighted IRR equals the
  hurdle exactly, by construction (with base-case PV, the base-case IRR does);
  the notebook asserts this on every run.

**4. MOIC waterfall.** Because exit EBITDA is a pure product of the drivers,
each case's MOIC decomposes exactly (no residual, order-independent) into one
multiplicative bar per assumption: MOIC = TAM growth x share gains x
monetization x take rate x margin expansion x exit multiple. The 1.0x baseline
is a flat business (every driver at its start) sold at the multiple you paid,
P0 = entry x FDV / flat-FY1 EBITDA. When P0 is high the exit-multiple bar is a
large give-back the fundamental bars must overcome: the honest read is that
the return is earnings-growth-driven, not re-rating-driven.

**5. Sensitivity.** `model.moic_grid` runs the base-case MOIC over any two
drivers (a scenario column or a `model.CTX_DRIVERS` key) with the entry held
fixed. The notebook ships two grids: initial TAM x exit share, and exit
FDV/EBITDA x exit share. The cell at the run values reproduces the reported
MOIC, asserted on every run.

**6. Comps (cross-check only, never feeds the valuation).** Peer selection is
a deal judgement made in the notebook config: equity groups (yfinance, TTM
revenue and EBITDA, market cap as the fully-diluted proxy) and token peers
(CoinGecko mcap/FDV, DeFiLlama TTM net protocol revenue, the same altitude as
the model's revenue line). Everything at spot, levels and multiples on FDV
with mcap alongside. FDV/EBITDA is the tie-out to the model's exit multiple
(tokens are blank there; opex unreported) and the EBITDA margin anchors the
model's margin ramp.

**Single source of truth.** The notebook owns every assumption. The one-pager
formatter (`pipeline.build_metrics`) has no defaults of its own: the notebook
passes its live scenario, context and valuation objects in, and an assert ties
the rendered MOIC back to the notebook's number, so the one-pager cannot
silently diverge.

**Data.** Live marks come from DeFiLlama (`marks.py`), with every successful
pull snapshotted to `data/marks_snapshot.json`; set `USE_LIVE_DATA = False` to
rerun offline from the snapshot. Historical TAM / units context is cached in
`data/units_history.json`; `marks.ensure_units_history` fills any missing
period from the DeFiLlama Pro categories history (TAM) and the protocol's own
series (units) and never overwrites a value already there, so hand-entered or
out-of-band figures survive. The `data/` caches are per-deal workspace and
git-ignored: one deal per clone.

### One-pager generator (`src/onepager`)

- `templates/liquid-one-pager.html.j2`: the design as a plain Jinja template
  (placeholders -> vars). Section headers and footnotes that name the deal's
  traction metric come through the context (`traction_title`, `comps_note`).
- `contract.py`: single source of truth for the fields the template expects;
  `validate()` fails loudly on a missing/misspelled field. All values are
  render-ready display strings (e.g. `"2.3x"`, `"$0.92"`).
- `render.py`: `render(context)` / `save(context, path)` produce one
  self-contained HTML file (Montserrat font and RV logo embedded as base64).
- `pipeline.py` (in `src/valuation`): `build_metrics(valuation, ...)` formats
  the notebook's valuation objects into the template's NUMBERS dict (target,
  MOIC/IRR, scenarios, supply, treasury, comps, technicals, traction).
  Position sizing and other non-modeled fields pass through a `manual`
  override so nothing is silently mislabeled.

## Usage (example notebooks)

Deal work runs from `liquid-valuation-template.ipynb` (see above). The example
notebooks are walkthroughs of the generic data tooling:

- **`example-pull-raw-data.ipynb`** — fetch raw data from each API into local
  JSON files. Long-running pulls show a `tqdm` progress bar per data source.
- **`example-analysis.ipynb`** — process the raw data and build charts.
- **`investment-memo-template.ipynb`** — scenario analysis for an investment memo.

Raw data and chart outputs (`*.json`, `*.png`) are git-ignored; only
`config/*.json` is tracked.

### MetaDAO launchpad (`src/data/metadao.py`)

Charts each MetaDAO launch's first days of trading against the price its
participants paid. A MetaDAO sale is not an auction: the launchpad program mints
a fixed 10,000,000 tokens to funders as a group, pro rata to the USDC the founder
accepted, so `ico_price_usd = usdc_accepted / 10,000,000`. The accepted amount
(`total_approved_amount` on v0.7+, `final_raise_amount` on v0.6, the committed
amount on the capless v0.4/v0.5 programs) is decoded from each launch's on-chain
`Launch` account (`getProgramAccounts` on the five launchpad programs, filtered
by the Anchor discriminator) and written to `config/metadao_launches.json`. That
file is hand-editable: the pull adds missing launches and missing keys only and
never overwrites a value, so an override or `include: false` survives reruns.
New launches are included when the raise is at least $100k.

Prices come from MetaDAO's market API (`market-api.metadao.fi`), not CoinGecko:
`/dexscreener/pair` gives each launch's AMM pair and exact trading start, and
`/dexscreener/events` returns every swap by slot range (500k slots per request,
history back to October 2025). `build_launch_prices` turns the swaps into
hourly VWAP (USDC volume / token volume) for the first `n_days` and writes
`data/processed/metadao/launch-prices-{n}d.json`, which
`visualization.metadao_launch_performance` and `metadao.first_days_table` read.
It also ties the swap-derived daily VWAP out against MetaDAO's daily
`/api/market-data` endpoint and prints any full launch-day that differs by more
than 0.5%. Both endpoints are undocumented, so every response is saved unchanged
under `data/raw/metadao/`. Launches that predate the API's history (Omnipair,
mtnCapital) keep their ICO price in the config but are skipped by the charts.
The fundraise cutoff for what is charted (`min_fundraise_usd`) is set in the
analysis notebook and passed to the chart and table, so the pull's own floor only
decides which launches get price history collected.
To pick up new launches, run `metadao.refresh(...)` (the METADAO cell in the pull
notebook). Launches already on file are never repulled: one fresh tickers call
decides whether a new DAO launched since the last pull, and only then are the
launch accounts repulled and the new rows added to the config. Swaps are pulled
only for launches that lack them or whose first `n_days` had not finished at the
previous run, the daily market data is repulled only when new swaps came in, and
the processed file is rebuilt only when an input changed. Each step prints one
line; `metadao.pull_launches(..., force=True)` is the manual full repull.

### Hyperliquid AQAv2 reserve yield (`src/data/hyperliquid.py`)

Reconstructs what Hyperliquid's AQAv2 (Aligned Quote Assets v2) mechanism pays the
Assistance Fund each 30-day period, and estimates future periods under a Fed funds path.

How the charge works (docs: hypercore/aligned-quote-assets, verified against chain data):
every UTC date the node records the treasury deployer's native-USDC balance on HyperEVM
at 00:00 UTC and the validator-published AQA rate, and books
`interest = balance x rate / 365.25`. The treasury address holds exactly 90% of the
USDC backing HyperCore (the docs' 9:1 split with the technical deployer's linked
contract), so the "approximately 90% of cost-adjusted reserve yield" is delivered
through the balance; the rate is the full cost-adjusted reserve yield, about 50bp below
effective Fed funds. Days are summed in 30-day intervals from activation (2026-08-27)
and paid 8 days after each interval ends (period 1: 2026-08-27..2026-09-25, paid
2026-10-03, 14,580,776.69 USDC).

Data: the official daily series from `POST /info {"type":"aqav2InterestHistory","token":0}`
(no key), the system interest address's ledger for the actual payments, the treasury
balance re-read at each midnight block through a HyperEVM **archive** RPC as an
independent tie-out (it reproduces the charged base to the cent; the public
`rpc.hyperliquid.xyz/evm` is not an archive node and silently returns current state for
old blocks), and EFFR from FRED. Raw files live under `data/raw/hyperliquid/aqav2/`.

Estimate: days already published are actual; remaining days are the latest treasury
balance (grown at a user `treasury_cagr`, default 0) times a rate equal to the trailing
L-day mean of EFFR minus a fixed cost (default 50bp), with L fitted to the ramp after the
Sep 17 2026 hike (about 45 days) and Fed moves applied the day after each FOMC decision.
Scenarios are Fed paths. `hyperliquid.refresh(...)` in the pull notebook runs the pulls
and prints the tie-outs; the analysis notebook renders the period table, the daily and
rate charts, and the estimate table and chart.

## Scenario analysis

The investment scenario-analysis module has its own documentation:
[`src/scenario_analysis/README.md`](src/scenario_analysis/README.md).
