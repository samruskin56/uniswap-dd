from datetime import datetime, date
from typing import List, Tuple
import textwrap
import pandas as pd
import numpy as np
from utils.helper_functions import report_errors


def create_scenario_analysis(
    # Company info
    company_name: str,
    round_name: str,  # e.g., "Seed", "Series A"
    entry_date: date,

    # Round details
    amount_raised: float,  # millions
    post_money_valuation: float,  # millions
    rv_investment: float,  # millions

    # Fund info
    fund_size: float = 50.0,  # millions

    # Scenarios: (description, exit_value, probability, years_to_exit, dilution)
    scenarios: List[Tuple[str, float, float, float, float]] = None
) -> pd.DataFrame:
    """
    Create a scenario analysis.

    Parameters:
    -----------
    company_name : str
        Name of the company
    round_name : str
        Round name (e.g., "Seed", "Series A")
    entry_date : date
        Date of investment
    amount_raised : float
        Total round size in millions
    post_money_valuation : float
        Post-money valuation in millions
    rv_investment : float
        RV's investment in millions
    fund_size : float
        Fund size in millions
    scenarios : List[Tuple[str, float, float, float, float]]
        List of (description, exit_value_M, probability, years_to_exit, dilution)
        dilution: fraction of ownership lost to future rounds (e.g., 0.30 = 30% diluted)

    Returns:
    --------
    pd.DataFrame with scenario analysis
    """

    # Calculate ownership
    pre_money_valuation = post_money_valuation - amount_raised
    rv_ownership = rv_investment / post_money_valuation

    # Default scenarios if none provided
    if scenarios is None:
        scenarios = [
            ("Disaster. Company implodes.", 0, 0.2, 1, 0.10),
            ("Base case", 100, 0.35, 2, 0.30),
            ("Success case", 600, 0.30, 2, 0.40),
            ("Strong success", 1250, 0.10, 3, 0.50),
            ("Exceptional outcome", 4000, 0.05, 3, 0.60),
        ]

    # Build scenario analysis
    scenario_results = []

    for desc, exit_value, prob, years, dilution in scenarios:
        effective_ownership = rv_ownership * (1 - dilution)
        to_rv = exit_value * effective_ownership
        multiple = to_rv / rv_investment if rv_investment > 0 else 0

        # IRR calculation
        if years > 0 and multiple > 0:
            irr = (multiple ** (1 / years) - 1) * 100
        else:
            irr = -100  # Total loss

        # Exit date
        exit_date = entry_date.replace(year=entry_date.year + int(years))

        scenario_results.append({
            '$ Million': desc,
            'Exit Value': exit_value,
            'Probability': prob,
            'Dilution': dilution,
            'Year': years,
            'To RV': to_rv,
            'Multiple': multiple,
            'IRR %': irr,
            'Cash Outflow': -rv_investment,
            'Cash Inflow': to_rv,
            'Outflow Date': entry_date,
            'Inflow Date': exit_date
        })

    # Create DataFrame
    df = pd.DataFrame(scenario_results)

    # Calculate weighted averages
    weighted_to_rv = (df['To RV'] * df['Probability']).sum()
    weighted_years = (df['Year'] * df['Probability']).sum()
    weighted_exit_value = (df['Exit Value'] * df['Probability']).sum()
    weighted_dilution = (df['Dilution'] * df['Probability']).sum()

    weighted_years_rounded = round(weighted_years)
    weighted_exit_date = entry_date.replace(year=entry_date.year + weighted_years_rounded)

    # Weighted multiple and IRR derived from weighted cash flows
    weighted_multiple = weighted_to_rv / rv_investment if rv_investment > 0 else 0
    if weighted_years_rounded > 0 and weighted_multiple > 0:
        weighted_irr = (weighted_multiple ** (1 / weighted_years_rounded) - 1) * 100
    else:
        weighted_irr = -100

    # Add weighted average row (use '' and np.nan so pandas can infer dtypes cleanly)
    weighted_row = pd.DataFrame([{
        '$ Million': '',
        'Exit Value': np.nan,
        'Probability': np.nan,
        'Dilution': weighted_dilution,
        'Year': weighted_years,
        'To RV': weighted_to_rv,
        'Multiple': weighted_multiple,
        'IRR %': weighted_irr,
        'Cash Outflow': -rv_investment,
        'Cash Inflow': weighted_to_rv,
        'Outflow Date': entry_date,
        'Inflow Date': weighted_exit_date
    }])

    df = pd.concat([df, weighted_row], ignore_index=True)

    # Store metadata
    df.attrs['company_name'] = company_name
    df.attrs['round_name'] = round_name
    df.attrs['entry_date'] = entry_date
    df.attrs['amount_raised'] = amount_raised
    df.attrs['pre_money_valuation'] = pre_money_valuation
    df.attrs['post_money_valuation'] = post_money_valuation
    df.attrs['rv_investment'] = rv_investment
    df.attrs['rv_ownership'] = rv_ownership
    df.attrs['fund_size'] = fund_size
    df.attrs['pct_of_fund'] = (rv_investment / fund_size) * 100
    df.attrs['pct_of_fund_returned'] = (weighted_to_rv / fund_size) * 100
    df.attrs['weighted_exit_value'] = weighted_exit_value

    return df


@report_errors()
def display_scenario_table(df: pd.DataFrame, show_assumptions: bool = True, show_cash_flows: bool = False):
    """Display the scenario analysis as formatted plain text."""
    attrs = df.attrs
    n = len(df)

    # --- Column widths ---
    w_s   = 52   # Scenario
    w_ev  = 13   # Exit Value
    w_p   = 6    # Prob
    w_d   = 10   # Dilution
    w_y   = 7    # Years
    w_rv  = 10   # To RV
    w_m   = 10   # Multiple
    w_irr = 7    # IRR

    cf_cols = 10 + 10 + 12 + 12 if show_cash_flows else 0
    line_width = w_s + w_ev + w_p + w_d + w_y + w_rv + w_m + w_irr + cf_cols

    # --- Assumptions ---
    if show_assumptions:
        print()
        print("=" * line_width)
        print(f"RV Scenario Analysis — {attrs['company_name']}")
        print("=" * line_width)
        print()
        print(f"Round: {attrs['round_name']}")
        print(f"Entry Date: {attrs['entry_date'].strftime('%B %d, %Y')}")
        print(f"Amount Raised: ${attrs['amount_raised']:.1f}m")
        print(f"Pre-Money Valuation: ${attrs['pre_money_valuation']:,.1f}m")
        print(f"Post-Money Valuation: ${attrs['post_money_valuation']:,.1f}m")
        print(f"RV Investment: ${attrs['rv_investment']:.1f}m")
        print(f"RV Ownership: {attrs['rv_ownership'] * 100:.2f}%")
        print(f"Fund Size: ${attrs['fund_size']:,.0f}m")
        print(f"% of Fund Deployed: {attrs['pct_of_fund']:.1f}%")
        print(f"% of Fund Returned (Weighted): {attrs['pct_of_fund_returned']:.1f}%")
        print()

    

    # --- Header ---
    header = (
        f"{'Scenario':<{w_s}}"
        f"{'Exit Value':>{w_ev}}"
        f"{'Prob':>{w_p}}"
        f"{'Dilution':>{w_d}}"
        f"{'Years':>{w_y}}"
        f"{'To RV':>{w_rv}}"
        f"{'Multiple':>{w_m}}"
        f"{'IRR':>{w_irr}}"
    )
    if show_cash_flows:
        header += f"{'Cash Out':>10}{'Cash In':>10}{'Out Date':>12}{'In Date':>12}"

    print("-" * line_width)
    print(header)
    print("-" * line_width)

    # --- Rows ---
    for i, row in df.iterrows():
        is_weighted = (i == n - 1)
        is_first_scenario = (i == 0)

        if is_weighted:
            print("-" * line_width)
        elif not is_first_scenario:
            print("." * line_width)

        scenario_label = 'Weighted Avg.' if is_weighted else row['$ Million']

        exit_val = '--'     if is_weighted else f"${row['Exit Value']:,.0f}m"
        prob     = '--'     if is_weighted else f"{row['Probability'] * 100:.0f}%"
        dilution = f"{row['Dilution'] * 100:.0f}%"
        years    = f"{row['Year']:.1f}"
        to_rv    = f"${row['To RV']:.1f}m"
        multiple = f"{row['Multiple']:.1f}x"
        irr      = f"{row['IRR %']:.0f}%"

        numbers = (
            f"{exit_val:>{w_ev}}"
            f"{prob:>{w_p}}"
            f"{dilution:>{w_d}}"
            f"{years:>{w_y}}"
            f"{to_rv:>{w_rv}}"
            f"{multiple:>{w_m}}"
            f"{irr:>{w_irr}}"
        )
        if show_cash_flows:
            outflow_str = f"(${abs(row['Cash Outflow']):.1f}m)"
            inflow_str  = f"${row['Cash Inflow']:.1f}m"
            out_date    = row['Outflow Date'].strftime('%Y-%m-%d')
            in_date     = row['Inflow Date'].strftime('%Y-%m-%d')
            numbers += f"{outflow_str:>10}{inflow_str:>10}{out_date:>12}{in_date:>12}"

        wrapped = textwrap.wrap(scenario_label, w_s - 1) or ['']
        for wrap_line in wrapped[:-1]:
            print(f"{wrap_line:<{w_s}}")
        print(f"{wrapped[-1]:<{w_s}}{numbers}")

    print("-" * line_width)
    print()
    print(f"Note: IRR and multiple are probability-weighted and derived from a single cash outflow of "
          f"${attrs['rv_investment']:.1f}m at entry and a "
          "\nprobability-weighted inflow across all scenarios.")
    print()
