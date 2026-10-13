# Scenario Analysis Module

This module recreates your Excel scenario analysis worksheets in Python, matching the exact format and calculations.

## Overview

The scenario analysis calculates investment returns across multiple outcome scenarios with:
- Investment assumptions (round size, valuation, ownership)
- Multiple exit scenarios with probabilities
- Calculated MOIC, IRR, and weighted averages
- Fund-level metrics (% of fund returned)

## Quick Start

### Using the Investment Memo Template

The scenario analysis is pre-configured in `investment-memo-template.ipynb`. Just edit the parameters:

```python
# Company details
COMPANY_NAME = "Felix"
ROUND_NAME = "Seed"
ENTRY_DATE = date(2025, 5, 31)

# Round details
AMOUNT_RAISED = 4.0  # millions
PRE_MONEY_VALUATION = 36.0  # millions
RV_INVESTMENT = 1.0  # millions
FUND_SIZE = 70.0  # millions

# Scenarios: (description, exit_value, probability%, years_to_exit)
scenarios = [
    ("Disaster. Company implodes.", 0, 0.2, 1),
    ("Base case", 100, 0.35, 2),
    ("Success case", 600, 0.3, 2),
    ("Strong success", 1250, 0.1, 3),
    ("Exceptional", 4000, 0.05, 3),
]
```

Run the cell and it automatically generates the full analysis.

## Output Format

```
============================================================
RV Scenario Analysis - Felix
============================================================

ASSUMPTIONS
------------------------------------------------------------
Round:                      Seed
Entry date:                 2025-05-31
Amount being raised:        $4M
Pre-money valuation:        $36M
Post-money valuation:       $40M

Amount Reciprocal will invest:  $1M
RV ownership:                   0.025000

Fund 3 size:                $70M
% of fund returned:         0.192857

============================================================
SCENARIO ANALYSIS
============================================================

                              $ Million  Exit Value  Probability  Year  To RV  Multiple  IRR %
        Disaster. Company implodes.           0         0.20     1    0.00      0.00  -100.0
Felix continues to find PMF...               100         0.35     2    2.50      2.50    58.1
...
                        Weighted Avg.        None          NaN  1.95   13.50     13.50   126.2
```

## Function Reference

### create_scenario_analysis()

Creates a scenario analysis DataFrame matching your Excel template.

**Parameters:**
- `company_name` (str): Company name
- `round_name` (str): Round name (e.g., "Seed", "Series A")
- `entry_date` (date): Investment date
- `amount_raised` (float): Total round size in millions
- `pre_money_valuation` (float): Pre-money valuation in millions
- `rv_investment` (float): Your investment in millions
- `fund_size` (float): Fund size in millions (default: 70)
- `scenarios` (List[Tuple]): List of (description, exit_value_M, probability_%, years_to_exit)

**Returns:**
pandas DataFrame with columns:
- `$ Million`: Scenario description
- `Exit Value`: Company exit value (millions)
- `Probability`: Likelihood (0-1 or 0-100)
- `Year`: Years to exit
- `To RV`: Cash returned to RV (millions)
- `Multiple`: MOIC (Multiple of Invested Capital)
- `IRR %`: Internal Rate of Return
- `Cash Outflow`: Your investment (negative)
- `Cash Inflow`: Your return
- `Outflow Date`: Investment date
- `Inflow Date`: Exit date

Last row is the weighted average across all scenarios.

### display_scenario_table()

Displays formatted output to console.

**Parameters:**
- `df` (DataFrame): Output from `create_scenario_analysis()`
- `show_assumptions` (bool): Show assumptions section (default: True)
- `show_cash_flows` (bool): Show cash flow columns (default: False)

## Standalone Usage

```python
from datetime import date
from scenario_analysis.scenario_analysis import create_scenario_analysis, display_scenario_table

# Create analysis
df = create_scenario_analysis(
    company_name="Acme Corp",
    round_name="Series A",
    entry_date=date(2025, 1, 15),
    amount_raised=10.0,
    pre_money_valuation=90.0,
    rv_investment=2.0,
    fund_size=70.0,
    scenarios=[
        ("Company fails", 0, 0.15, 1),
        ("Modest exit", 200, 0.40, 3),
        ("Good exit", 1000, 0.30, 4),
        ("Great exit", 3000, 0.10, 5),
        ("Exceptional", 8000, 0.05, 6),
    ]
)

# Display formatted table
display_scenario_table(df, show_assumptions=True)

# Access results programmatically
print(f"RV Ownership: {df.attrs['rv_ownership']*100:.2f}%")
print(f"Weighted MOIC: {df.iloc[-1]['Multiple']:.2f}x")
print(f"Weighted IRR: {df.iloc[-1]['IRR %']:.1f}%")
print(f"% of Fund Returned: {df.attrs['pct_of_fund_returned']:.2f}%")

# Export to Excel
df.to_excel('scenario_analysis.xlsx', index=False)
```

## Tips

1. **Probability Check**: Ensure scenario probabilities sum to 1.0 (or 100%)

2. **Scenario Structure**: Typical structure is:
   - 1 downside scenario (failure/disaster)
   - 1-2 base/median scenarios
   - 2-3 upside scenarios (varying degrees of success)

3. **Years to Exit**:
   - Early-stage: 3-7 years typical
   - Later-stage: 2-4 years typical
   - Adjust based on company stage and exit expectations

4. **Exit Values**: Consider comparable company exits and valuation multiples:
   - Revenue multiples (ARR × multiple)
   - FDV/TVL ratios for DeFi
   - P/S ratios for traditional SaaS

5. **Metadata Access**: All assumptions stored in `df.attrs`:
   ```python
   df.attrs['rv_ownership']         # RV ownership %
   df.attrs['post_money_valuation'] # Post-money valuation
   df.attrs['pct_of_fund_returned'] # % of fund returned
   ```

6. **DataFrame Export**: Save analysis for later:
   ```python
   df.to_excel('analysis.xlsx', index=False)
   df.to_csv('analysis.csv', index=False)
   ```

## Matching Excel Template

This module is designed to match `250512_Scenario-Analysis_Felix_Seed_vF.xlsx`:
- Assumptions section matches exactly
- Calculations use identical formulas
- Output format mirrors Excel layout
- All metrics computed the same way (MOIC, IRR, weighted averages)

You can validate calculations by comparing Python output against Excel for the same inputs.
