import pandas as pd
import os
import datetime
import json
import sys
import numpy as np
import math
import re

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from data import defillama
from utils.helper_functions import ensure_directory, report_errors, save_json, get_session, llama_url, has_pro_key

# GENERAL

# ---------------------------------------------------------------------------
# Derived series: earnings = revenue - incentives (DefiLlama's Earnings definition)
# ---------------------------------------------------------------------------
EARNINGS_RULE = 'earnings = revenue - incentives; a day with no incentives point counts as 0; ' \
                'only days present in the revenue series get a point'


def _load_chart(path):
    """{day_ts: value} for a saved per-protocol file's totalDataChart (numeric values only),
    keyed by the UTC day the timestamp falls in, plus the raw dict. ({}, None) if absent."""
    if not os.path.exists(path):
        return {}, None
    with open(path) as f:
        data = json.load(f)
    chart = {}
    for ts, v in data.get('totalDataChart') or []:
        if isinstance(ts, (int, float)) and isinstance(v, (int, float)):
            chart[int(ts) // 86400 * 86400] = chart.get(int(ts) // 86400 * 86400, 0.0) + v
    return chart, data


@report_errors(default=[])
def build_earnings_series(assets, defillama_output_dir, pull_missing=True):
    """Derive data/raw/defillama/earnings/{Name}.json from the revenue and incentives files.

    Rule (EARNINGS_RULE): for every UTC day in the revenue series, earnings = revenue -
    incentives, with a day missing from the incentives chart counted as 0. This is
    DefiLlama's own Earnings definition (Revenue - Incentives) and convention (no tracked
    incentives -> earnings = revenue), so a protocol whose incentives file has an empty
    chart gets earnings equal to revenue. Days absent from the revenue series get no
    point even if incentives exist (the financial statement behaves the same way).
    Negative values are kept.

    ``assets`` is {defillama_slug: standard_name}. With ``pull_missing`` a missing revenue
    or incentives file is pulled first (incentives need DEFILLAMA_API_KEY; without it no
    earnings file is written, so the freshness check reports it as missing rather than
    silently equating earnings to revenue).

    Returns the standard names whose earnings equal revenue because no incentives are
    tracked, so the caller can say so.
    """
    no_incentives = []
    for asset_slug, asset_name in assets.items():
        rev_path = f"{defillama_output_dir}/revenue/{asset_name}.json"
        inc_path = f"{defillama_output_dir}/incentives/{asset_name}.json"
        if pull_missing and not os.path.exists(rev_path):
            defillama.pull_metric_data_by_asset({'dailyRevenue': 'revenue'}, {asset_slug: asset_name},
                                                defillama_output_dir, report=False)
        if pull_missing and not os.path.exists(inc_path):
            defillama.pull_metric_data_by_asset({'dailyIncentives': 'incentives'}, {asset_slug: asset_name},
                                                defillama_output_dir, report=False)
        if not os.path.exists(rev_path):
            print(f"  earnings: no revenue file for {asset_name}, skipped")
            continue
        if not os.path.exists(inc_path):
            print(f"  earnings: no incentives file for {asset_name} (needs DEFILLAMA_API_KEY), skipped")
            continue
        revenue, _ = _load_chart(rev_path)
        incentives, inc_data = _load_chart(inc_path)
        chart = [[ts, revenue[ts] - incentives.get(ts, 0.0)] for ts in sorted(revenue)]
        out = {
            'totalDataChart': chart,
            'derived': {
                'rule': EARNINGS_RULE,
                'inputs': [f"revenue/{asset_name}.json", f"incentives/{asset_name}.json"],
                'incentivesStatus': (inc_data or {}).get('status', 'ok'),
                'incentivesEmissionSlug': (inc_data or {}).get('emissionSlug'),
                'builtAt': defillama._today_utc_ts(),
            },
        }
        save_json(out, f"{defillama_output_dir}/earnings/{asset_name}.json")
        if not incentives:
            no_incentives.append(asset_name)
    if no_incentives:
        print(f"  earnings = revenue (no incentives tracked on DefiLlama): {', '.join(no_incentives)}")
    return no_incentives


@report_errors(default=pd.DataFrame())
def check_incentives_tieout(assets, defillama_output_dir, months=12, tolerance_usd=0.5):
    """Tie each saved incentives file to DefiLlama's financial statement, month by month.

    For every complete month in the last ``months`` where either side has a value, the
    incentives file's monthly sum is compared with the statement's monthly
    ``Incentives.value`` (Pro /v2/metrics/financial-statement). Statuses:
    ``match`` (within ``tolerance_usd``), ``mismatch``, ``statement-missing-month`` (the
    statement has no row for a month the chart covers, e.g. the fee adapter had no data),
    ``chart-missing-month``. Protocols with no incentives on either side produce no rows.
    Needs DEFILLAMA_API_KEY; returns an empty frame without it.

    Returns a DataFrame: protocol, emission_slug, month, chart_usd, statement_usd, diff, status.
    """
    if not has_pro_key():
        print('check_incentives_tieout: needs DEFILLAMA_API_KEY')
        return pd.DataFrame()
    current = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m')
    rows = []
    for asset_slug, asset_name in assets.items():
        inc_path = f"{defillama_output_dir}/incentives/{asset_name}.json"
        if not os.path.exists(inc_path):
            continue
        with open(inc_path) as f:
            inc = json.load(f)
        chart_m = {}
        for ts, v in inc.get('totalDataChart') or []:
            m = datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime('%Y-%m')
            if m < current and isinstance(v, (int, float)):
                chart_m[m] = chart_m.get(m, 0.0) + v
        r = get_session().get(llama_url(f'/v2/metrics/financial-statement/protocol/{asset_slug}', pro_only=True), timeout=90)
        monthly = ((r.json() if r.status_code == 200 else {}).get('aggregates') or {}).get('monthly') or {}
        stmt = {m: (v.get('Incentives') or {}).get('value') for m, v in monthly.items() if m < current}
        stmt = {m: v for m, v in stmt.items() if isinstance(v, (int, float))}
        for m in sorted(set(stmt) | {m for m, v in chart_m.items() if v})[-months:]:
            a, b = chart_m.get(m), stmt.get(m)
            if b is None:
                status = 'statement-missing-month'
            elif a is None:
                status = 'chart-missing-month'
            else:
                status = 'match' if abs(a - b) <= tolerance_usd else 'mismatch'
            rows.append(dict(protocol=asset_name, emission_slug=inc.get('emissionSlug'), month=m,
                             chart_usd=a, statement_usd=b,
                             diff=(None if a is None or b is None else a - b), status=status))
    df = pd.DataFrame(rows)
    if not df.empty:
        bad = df[df.status != 'match']
        print(f"incentives tie-out: {(df.status == 'match').sum()} month(s) match, {len(bad)} do not"
              + (f" ({', '.join(sorted(bad.protocol.unique()))})" if len(bad) else ''))
    return df

# Function to establish parent protocol categories using fees data
def establish_parent_protocols(defillama_output_dir, window='totalAllTime'):
    """
    Establishes parent protocol categories using fees data.
    Iterates over fees, revenue, and TVL files to capture all sub-protocols linked to each parent.
    If a parent has 2+ sub-protocols with different categories (with >0 fees), it's categorized as 'Super App'.
    Calculates total fees, revenue, and TVL for each parent protocol.
    Records each parent's own display name (`name`): DefiLlama's parent name from the
    children's `linkedProtocols[0]` (e.g. "Uniswap" for uniswap-v1..v4), not a child's name.
    Returns and saves a mapping of
    {parent_slug: {name, category, total_fees, total_revenue, total_holders_revenue, total_tvl, sub_protocols: [...]}}
    """

    print("Establishing parent protocol categories using fees data...")

    # First, collect all parent-child relationships and metrics from all three data sources
    all_parent_child_relationships = {}  # {parent_slug: {sub_slug: {name, category, fees, revenue, tvl}}}
    parent_names = {}  # {parent_slug: parent display name from linkedProtocols[0]}

    metrics_to_check = [
        ('fees', 'protocols', window),
        ('revenue', 'protocols', window),
        ('holders-revenue', 'protocols', window),
        ('tvl', None, 'tvl')
    ]
    
    print("Step 1: Collecting all parent-child relationships and metrics from fees, revenue, and TVL data...")
    
    for metric_name, protocols_key, value_key in metrics_to_check:
        print(f"  Checking {metric_name} data...")
        with open(f'{defillama_output_dir}/{metric_name}/all-protocols.json', 'r') as f:
            data = json.load(f)
        
        if protocols_key:
            protocols_list = data.get(protocols_key, [])
        else:
            protocols_list = data
        
        for entry in protocols_list:
            name = entry.get('name')
            slug = entry.get('slug')
            category = entry.get('category')
            parent_protocol = entry.get('parentProtocol')
            value = entry.get(value_key)

            if not slug:
                continue

            # Determine the parent slug (remove 'parent#' prefix if present)
            if parent_protocol:
                if parent_protocol.startswith('parent#'):
                    parent_slug = parent_protocol.replace('parent#', '', 1)
                else:
                    parent_slug = parent_protocol
                # Child entries in the fees/revenue overviews carry
                # linkedProtocols = [<parent display name>, ...siblings]; that is
                # the only place the parent's own name appears (the /protocols
                # TVL list has no such field). Uniswap V1-V4 -> "Uniswap".
                linked = entry.get('linkedProtocols')
                if isinstance(linked, list) and linked and linked[0]:
                    parent_names.setdefault(parent_slug, linked[0])
            else:
                # No parent, so this protocol is its own parent
                parent_slug = slug

            # Initialize parent entry if needed
            if parent_slug not in all_parent_child_relationships:
                all_parent_child_relationships[parent_slug] = {}
            
            # Initialize or update sub-protocol entry
            if slug not in all_parent_child_relationships[parent_slug]:
                all_parent_child_relationships[parent_slug][slug] = {
                    'name': name,
                    'category': category,
                    'fees': 0,
                    'revenue': 0,
                    'holders-revenue': 0,
                    'tvl': 0
                }
            
            # Update the specific metric value
            if value is not None:
                all_parent_child_relationships[parent_slug][slug][metric_name] = value
    
    print(f"  Found {len(all_parent_child_relationships)} unique parent protocols")
    
    # Now analyze fees data to determine categories
    print("\nStep 2: Analyzing fees data to determine parent categories...")
    parent_categories = {}  # {parent_slug: {category, total_fees, total_revenue, total_tvl, sub_protocols}}
    
    with open(f'{defillama_output_dir}/fees/all-protocols.json', 'r') as f:
        fees_data = json.load(f)
    
    # Aggregate by parent protocol using fees for category determination
    parent_aggregates_fees = {}
    
    for entry in fees_data.get('protocols', []):
        slug = entry.get('slug')
        category = entry.get('category')
        parent_protocol = entry.get('parentProtocol')
        value = entry.get(window)
        
        if value is None or value == 0:
            continue
        
        # Determine the parent slug (remove 'parent#' prefix if present)
        if parent_protocol:
            if parent_protocol.startswith('parent#'):
                parent_slug = parent_protocol.replace('parent#', '', 1)
            else:
                parent_slug = parent_protocol
        else:
            parent_slug = slug
        
        # Initialize parent entry if needed
        if parent_slug not in parent_aggregates_fees:
            parent_aggregates_fees[parent_slug] = []
        
        # Track category and value for this sub-protocol
        parent_aggregates_fees[parent_slug].append({
            'category': category,
            'value': value
        })
    
    # Determine category for each parent and calculate totals
    for parent_slug, sub_protocols_fees in parent_aggregates_fees.items():
        # Get categories of sub-protocols with >0 value
        categories_with_value = {}
        for sub in sub_protocols_fees:
            if sub['value'] > 0 and sub['category']:
                if sub['category'] not in categories_with_value:
                    categories_with_value[sub['category']] = 0
                categories_with_value[sub['category']] += sub['value']
        
        # Determine parent category
        if len(categories_with_value) == 0:
            continue
        elif len(categories_with_value) == 1:
            parent_category = list(categories_with_value.keys())[0]
        else:
            parent_category = 'Super App'
        
        # Get all sub-protocols for this parent (from all metrics)
        all_subs = all_parent_child_relationships.get(parent_slug, {})
        
        # Calculate totals and build sub-protocols list
        total_fees = 0
        total_revenue = 0
        total_holders_revenue = 0
        total_tvl = 0
        sub_protocols_list = []
        
        for sub_slug, sub_info in all_subs.items():
            sub_fees = sub_info.get('fees', 0)
            sub_revenue = sub_info.get('revenue', 0)
            sub_holders_revenue = sub_info.get('holders-revenue',0)
            sub_tvl = sub_info.get('tvl', 0)
            
            total_fees += sub_fees
            total_revenue += sub_revenue
            total_holders_revenue += sub_holders_revenue
            total_tvl += sub_tvl
            
            sub_protocols_list.append({
                'slug': sub_slug,
                'name': sub_info['name'],
                'category': sub_info['category'],
                'fees': sub_fees,
                'revenue': sub_revenue,
                'holders-revenue': sub_holders_revenue,
                'tvl': sub_tvl
            })
        
        # Parent display name. Prefer the name DefiLlama gives the parent
        # (linkedProtocols[0] on its children), then the sub-protocol that IS
        # the parent (standalone protocol, slug == parent_slug), and only as a
        # last resort the largest child by fees -- a child's name ("Uniswap V3")
        # must not stand in for the parent ("Uniswap").
        parent_name = parent_names.get(parent_slug)
        if not parent_name:
            self_sub = next((s for s in sub_protocols_list if s['slug'] == parent_slug), None)
            if self_sub and self_sub.get('name'):
                parent_name = self_sub['name']
        if not parent_name and sub_protocols_list:
            parent_name = max(sub_protocols_list, key=lambda s: s['fees'] or 0)['name']

        # Store name, category, totals, and sub-protocols
        parent_categories[parent_slug] = {
            'name': parent_name,
            'category': parent_category,
            'total_fees': total_fees,
            'total_revenue': total_revenue,
            'total_holders_revenue': total_holders_revenue,
            'total_tvl': total_tvl,
            'sub_protocols': sub_protocols_list
        }
    
    # Save parent categories mapping
    parent_categories_output = 'data/processed/parent_protocols.json'
    ensure_directory(os.path.dirname(parent_categories_output))
    with open(parent_categories_output, 'w') as f:
        json.dump(parent_categories, f, indent=2)
    print(f"\nParent protocols saved to: {parent_categories_output}")
    print(f"Total parent protocols: {len(parent_categories)}")
    print('Done')
    
    return parent_categories

def calculate_total_fees_by_category(parent_protocols_path, output_path='data/processed/total_fees_by_category.json'):
    """
    Reads parent protocols file and calculates total fees for each category.
    Returns and saves a dictionary of categories sorted by total fees.
    
    Args:
        parent_protocols_path: Path to the parent protocols JSON file
        output_path: Path where to save the output JSON file
        
    Returns:
        dict: Categories sorted by total fees in descending order
    """
    
    print(f"Loading parent protocols from {parent_protocols_path}...")
    
    # Load parent protocols
    with open(parent_protocols_path, 'r') as f:
        parent_protocols_data = json.load(f)
    
    # Aggregate fees by category
    category_fees = {}
    
    for parent_slug, parent_data in parent_protocols_data.items():
        category = parent_data.get('category')
        total_fees = parent_data.get('total_fees', 0)
        
        if category and total_fees:
            if category not in category_fees:
                category_fees[category] = 0
            category_fees[category] += total_fees
    
    # Sort categories by total fees (descending)
    sorted_categories = dict(sorted(
        category_fees.items(), 
        key=lambda x: x[1], 
        reverse=True
    ))
    
    print(f"\nFound {len(sorted_categories)} categories")
    print("\nTop 10 categories by total fees:")
    for i, (category, fees) in enumerate(list(sorted_categories.items())[:10], 1):
        print(f"  {i}. {category}: ${fees:,.0f}")
    
    # Save to file
    ensure_directory(os.path.dirname(output_path))
    with open(output_path, 'w') as f:
        json.dump(sorted_categories, f, indent=2)

    print(f"\nTotal fees by category saved to: {output_path}")
    print('Done')
    
    return sorted_categories


# Function to process all metrics using established parent categories
def top_n_protocols_by_category_by_metric(metrics, n, categories, defillama_output_dir, parent_categories_path='data/processed/parent_protocols.json', window='totalAllTime'):
    """
    Processes all metrics using pre-established parent protocol categories.
    Aggregates protocols by parent, sums metric, and organizes by category.
    
    Args:
        parent_categories_path: Path to the parent protocol categories JSON file.
                               If not provided, will use 'data/processed/parent_protocols.json'
    """
    
    # Load parent categories
    print(f"Loading parent protocol categories from {parent_categories_path}...")
    with open(parent_categories_path, 'r') as f:
        parent_categories_data = json.load(f)
    
    # Extract just the category mapping for easier lookup
    parent_categories = {
        parent_slug: data['category']
        for parent_slug, data in parent_categories_data.items()
    }
    # Parent display names as established by establish_parent_protocols
    # (DefiLlama's own parent name, e.g. "Uniswap" rather than "Uniswap V3").
    # Missing on a parent_protocols.json written before the name was recorded.
    parent_names = {
        parent_slug: data.get('name')
        for parent_slug, data in parent_categories_data.items()
    }

    print(f"Loaded {len(parent_categories)} parent protocol categories")
    
    # Process all metrics
    print("\nProcessing all metrics using established categories...")
    results = {}
    
    for metric_slug, metric_name in metrics.items():
        print(f"Processing {metric_name}...")
        with open(f'{defillama_output_dir}/{metric_name}/all-protocols.json', 'r') as f:
            data = json.load(f)

        # Aggregate by parent protocol
        parent_aggregates = {}  # {parent_slug: {total_value, sub_protocols}}
        
        if metric_name == 'fees' or metric_name == 'revenue' or metric_name == 'holders-revenue':
            protocols_list = data.get('protocols', [])
        elif metric_name == 'tvl':
            protocols_list = data
        else:
            print(f"Error: metric {metric_name} not recognized")
            continue
            
        for entry in protocols_list:
            name = entry.get('name')
            slug = entry.get('slug')
            parent_protocol = entry.get('parentProtocol')
            
            # Get the value based on metric type
            if metric_name in ['fees', 'revenue', 'holders-revenue']:
                value = entry.get(window)
            else:  # tvl
                value = entry.get('tvl')
                
            if value is None or value == 0:
                continue
            
            # Determine the parent slug (remove 'parent#' prefix if present)
            if parent_protocol:
                if parent_protocol.startswith('parent#'):
                    parent_slug = parent_protocol.replace('parent#', '', 1)
                else:
                    parent_slug = parent_protocol
            else:
                # No parent, so this protocol is its own parent
                parent_slug = slug
            
            # Skip if this parent doesn't have a category (wasn't in fees data)
            if parent_slug not in parent_categories:
                continue
            
            # Initialize parent entry if needed
            if parent_slug not in parent_aggregates:
                parent_aggregates[parent_slug] = {
                    'total_value': 0,
                    'sub_protocols': []
                }
            
            # Add to parent's total and track sub-protocol info
            parent_aggregates[parent_slug]['total_value'] += value
            parent_aggregates[parent_slug]['sub_protocols'].append({
                'value': value,
                'name': name,
                'slug': slug
            })
        
        # Determine parent names: the parent's own name from parent_protocols.json,
        # falling back to the largest sub-protocol only if no name was recorded
        for parent_slug, parent_data in parent_aggregates.items():
            sub_protocols = parent_data['sub_protocols']
            parent_name = parent_names.get(parent_slug)

            if not parent_name:
                largest_sub = max(sub_protocols, key=lambda x: x['value'])
                parent_name = largest_sub['name']
                # If this is a protocol without a parent, use its own name
                if len(sub_protocols) == 1 and sub_protocols[0]['slug'] == parent_slug:
                    parent_name = sub_protocols[0]['name']

            parent_aggregates[parent_slug]['parent_name'] = parent_name
        
        # Organize by category using the parent_categories mapping
        # Only include categories that are in the specified list
        category_results = {cat: {} for cat in categories}
        
        for parent_slug, parent_data in parent_aggregates.items():
            # Get the category from parent_categories mapping
            parent_category = parent_categories[parent_slug]
            
            # Only include if this category is in the specified categories list
            if parent_category not in categories:
                continue
            
            if parent_category not in category_results:
                category_results[parent_category] = {}
            
            # Only add each parent once, to the category specified in parent_categories
            category_results[parent_category][parent_slug] = {
                'name': parent_data['parent_name'],
                'total_value': parent_data['total_value']
            }
        
        # Get top N for each category
        results[metric_slug] = {}
        for category, parents in category_results.items():
            if not parents:
                continue
                
            # Sort by total_value and get top N
            sorted_parents = sorted(parents.items(), key=lambda x: x[1]['total_value'], reverse=True)
            top_n_parents = sorted_parents[:n]
            
            # Create the final dict with just slug: name
            results[metric_slug][category] = {slug: data['name'] for slug, data in top_n_parents}
    
    # Save final results
    results_output_dir = 'data/processed/top_protocols_by_category_by_metric.json'
    ensure_directory(os.path.dirname(results_output_dir))
    with open(results_output_dir, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nFinal output saved to: {results_output_dir}")
    print('Done')
    return results


# Function to pull time-series data for the top parent protocols ranked by a metric
@report_errors()
def pull_top_parent_protocols_timeseries(
    metrics,
    defillama_output_dir,
    parent_categories_path='data/processed/parent_protocols.json',
    top_protocols_path='data/processed/top_protocols_by_category_by_metric.json',
    rank_metric='dailyFees',
    max_stale_days=defillama.DEFAULT_MAX_STALE_DAYS,
    max_workers=defillama.DEFAULT_MAX_WORKERS,
):
    """Pull DefiLlama time-series data for the top parent protocols.

    Loads the established parent-protocol categories and the top-protocols-by-category
    file, collects the unique parent slugs ranked by `rank_metric`, and pulls the
    requested `metrics` for each at the parent level.

    Args:
        metrics: Dict of {api_slug: metric_name} to pull (e.g. {'dailyFees': 'fees'}).
        defillama_output_dir: Directory to save raw DefiLlama JSON.
        parent_categories_path: Path to parent_protocols.json.
        top_protocols_path: Path to top_protocols_by_category_by_metric.json.
        rank_metric: Which metric's top-protocols ranking to use (default 'dailyFees').
        max_stale_days: Skip files whose newest point is within this many days (None = repull all).
        max_workers: Requests in flight at once; see defillama.pull_metric_data_by_asset.
    """
    with open(parent_categories_path) as f:
        parent_categories_data = json.load(f)
    with open(top_protocols_path) as f:
        top_protocols_data = json.load(f)

    # Collect unique parent slugs from the chosen ranking metric
    top_parent_slugs = set()
    for protocols in top_protocols_data.get(rank_metric, {}).values():
        top_parent_slugs.update(protocols.keys())

    # Keep only slugs present in the parent categories; the slug doubles as the name
    parent_assets = {}
    for slug in sorted(top_parent_slugs):
        if slug in parent_categories_data:
            parent_assets[slug] = slug
        else:
            print(f"Warning: {slug} not found in parent categories, skipping")

    print(f"Pulling {len(metrics)} metric(s) for {len(parent_assets)} parent protocols (top by {rank_metric})")
    failures = defillama.pull_metric_data_by_asset(
        metrics, parent_assets, defillama_output_dir, report=False,
        max_stale_days=max_stale_days, max_workers=max_workers,
    )

    # Report missing data per metric (each metric is attempted for every protocol)
    total = len(parent_assets)
    missing_by_metric = {}
    for metric_name, item in failures:
        missing_by_metric.setdefault(metric_name, []).append(item)

    print("\nMissing data by metric:")
    for metric_name in metrics.values():
        missing = sorted(missing_by_metric.get(metric_name, []))
        print(f"  {metric_name}: {len(missing)} of {total} protocols missing")
        for item in missing:
            print(f"    - {item}")


# FINANCIAL METRICS

# Function for plotting gross margin for parent protocols by parent category
def category_gross_margin_comparison(start_date, parent_categories_path, top_protocols_path, defillama_output_dir):
    # Load the parent protocol categories mapping
    with open(parent_categories_path, 'r') as f:
        parent_categories_data = json.load(f)
    
    # Load the top protocols by category
    with open(top_protocols_path, 'r') as f:
        top_protocols_data = json.load(f)
    
    # Extract unique parent slugs from top protocols file for fees metric
    top_parent_slugs = set()
    if 'dailyFees' in top_protocols_data:
        for category, protocols in top_protocols_data['dailyFees'].items():
            for slug in protocols.keys():
                top_parent_slugs.add(slug)
    
    print(f"Found {len(top_parent_slugs)} top parent protocols with fees data")
    
    # Extract unique categories from top protocols
    categories = set()
    if 'dailyFees' in top_protocols_data:
        categories = set(top_protocols_data['dailyFees'].keys())
    
    print(f"Found categories: {categories}")
    
    master_df = None
    
    # Process each category
    for category in categories:
        print(f"\nProcessing category: {category}")
        
        # Get parent protocols in this category from top protocols file
        category_parent_slugs = set(top_protocols_data['dailyFees'].get(category, {}).keys())
        
        # Filter to only include parents that exist in parent_categories_data
        parents_in_category = {
            parent_slug: parent_categories_data[parent_slug]
            for parent_slug in category_parent_slugs
            if parent_slug in parent_categories_data
        }
        
        print(f"  Found {len(parents_in_category)} parent protocols in {category}")
        
        category_df = None
        
        # Process each parent protocol
        for parent_slug, parent_data in parents_in_category.items():
            print(f"  Processing parent: {parent_slug}")
            
            # Load fees and revenue for this parent protocol directly
            skip_parent = False
            parent_dfs = {'fees': None, 'revenue': None}
            
            for metric in ['fees', 'revenue']:
                path = f"data/raw/defillama/{metric}/{parent_slug}.json"
                
                if not os.path.exists(path):
                    print(f"    Warning: {path} not found, attempting to pull data")
                    metrics = {'dailyFees':'fees', 'dailyRevenue':'revenue'}
                    for metric_slug, metric_name in metrics.items():
                        if metric == metric_name:
                            interim_metrics = {metric_slug: metric_name}
                            interim_assets = {parent_slug: parent_slug}  # FIXED: Use parent_slug instead of parent_data['name']
                            failures = defillama.pull_metric_data_by_asset(interim_metrics, interim_assets, defillama_output_dir)
                            if failures:
                                print(f"    Failed to pull {metric} data for {parent_slug}, skipping this parent")
                                skip_parent = True
                                break
                    if skip_parent:
                        break
                
                if skip_parent:
                    break
                
                try:
                    with open(path, 'r') as f:
                        obj = json.load(f)
                    
                    df = pd.DataFrame(obj["totalDataChart"], columns=["timestamp", metric])
                    df["date"] = pd.to_datetime(df["timestamp"], unit="s")
                    df = df[["date", metric]]
                    df[metric] = pd.to_numeric(df[metric], errors='coerce').fillna(0)
                    parent_dfs[metric] = df
                except Exception as e:
                    print(f"    Error loading {path}: {e}")
                    skip_parent = True
                    break
            
            if skip_parent or parent_dfs['fees'] is None or parent_dfs['revenue'] is None:
                print(f"    Skipping {parent_slug} - incomplete data")
                continue
            
            # Merge fees and revenue for this parent
            parent_df = parent_dfs['fees'].merge(parent_dfs['revenue'], on='date', how='outer').sort_values('date')
            parent_df['fees'] = parent_df['fees'].fillna(0)
            parent_df['revenue'] = parent_df['revenue'].fillna(0)
            
            # Calculate and print parent-level average gross margin
            parent_total_fees = parent_df['fees'].sum()
            parent_total_revenue = parent_df['revenue'].sum()
            if parent_total_fees > 0:
                parent_gross_margin = parent_total_revenue / parent_total_fees
                print(f"    {parent_slug} avg gross margin: {parent_gross_margin:.2%}")
            else:
                print(f"    {parent_slug} has no fees data")
            
            # Rename columns to include parent slug
            parent_df = parent_df.rename(columns={
                'fees': f'{parent_slug}-fees',
                'revenue': f'{parent_slug}-revenue'
            })
            
            # Merge into category_df
            if category_df is None:
                category_df = parent_df
            else:
                category_df = category_df.merge(parent_df, on='date', how='outer')
                num_cols = category_df.columns.difference(['date'])
                category_df[num_cols] = category_df[num_cols].fillna(0)
        
        # Skip category if no data was loaded
        if category_df is None:
            print(f"  No data for category {category}, skipping")
            continue
        
        # Get all fee and revenue columns for this category
        fee_cols = [c for c in category_df.columns if c.endswith('-fees')]
        rev_cols = [c for c in category_df.columns if c.endswith('-revenue')]
        
        print(f"  Found {len(fee_cols)} parent fee columns and {len(rev_cols)} parent revenue columns")
        
        # Calculate category-level gross margin as sum(revenue) / sum(fees)
        total_fees = category_df[fee_cols].sum(axis=1)
        total_revenue = category_df[rev_cols].sum(axis=1)
        
        # Calculate gross margin: total revenue / total fees
        category_df[f'{category}-gross-margin'] = total_revenue / total_fees.replace(0, float('nan'))
        category_df[f'{category}-gross-margin'] = category_df[f'{category}-gross-margin'].fillna(0)
        
        # Apply rolling averages to the weighted gross margin
        category_df[f'{category}-gross-margin-7d'] = (
            category_df[f'{category}-gross-margin'].rolling(window=7, min_periods=1).mean()
        )
        category_df[f'{category}-gross-margin-30d'] = (
            category_df[f'{category}-gross-margin'].rolling(window=30, min_periods=1).mean()
        )
        
        print(f"  Category {category} processed successfully")
        
        # Save both category aggregate and individual parent protocol data
        # Category aggregate with 30d rolling average
        category_agg = category_df[['date', f'{category}-gross-margin-30d']].copy()
        
        # Individual parent protocols with their gross margins
        for parent_slug in parents_in_category.keys():
            fee_col = f'{parent_slug}-fees'
            rev_col = f'{parent_slug}-revenue'
            
            if fee_col in category_df.columns and rev_col in category_df.columns:
                # Calculate parent-level gross margin
                parent_margin = category_df[rev_col] / category_df[fee_col].replace(0, float('nan'))
                parent_margin = parent_margin.fillna(0)
                
                # Apply 30d rolling average
                parent_margin_30d = parent_margin.rolling(window=30, min_periods=1).mean()
                category_agg[f'{parent_slug}-gross-margin-30d'] = parent_margin_30d
        
        # Merge into the master_df
        if master_df is None:
            master_df = category_agg
        else:
            master_df = master_df.merge(category_agg, on="date", how="outer")
    
    # Cut off master_df to only be after start date
    master_df = master_df[master_df['date'] >= start_date].sort_values('date')
    
    # Forward fill NaN values
    master_df = master_df.ffill()
    
    # Create output directory if it doesn't exist
    output_dir = 'data/processed/financial-metrics'
    ensure_directory(output_dir)

    # Restructure the data to nest parent protocols under categories
    output_data = []
    for _, row in master_df.iterrows():
        date_entry = {'date': row['date'].isoformat()}

        # Group by category
        for category in categories:
            category_col = f'{category}-gross-margin-30d'
            if category_col in row:
                category_entry = {
                    'category_aggregate': row[category_col]
                }
                
                # Add parent protocols for this category
                parent_protocols = {}
                for col in master_df.columns:
                    if col.endswith('-gross-margin-30d') and col != category_col:
                        # Extract parent slug from column name
                        parent_slug = col.replace('-gross-margin-30d', '')
                        # Check if this parent belongs to this category
                        if parent_slug in parent_categories_data and parent_categories_data[parent_slug]['category'] == category:
                            parent_protocols[parent_slug] = row[col]
                
                if parent_protocols:
                    category_entry['parent_protocols'] = parent_protocols
                
                date_entry[category] = category_entry
        
        output_data.append(date_entry)
    
    output_path = os.path.join(output_dir, 'gross-margins-by-category.json')
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f'\nGross margins by parent category saved to: {output_path}')
    
    # Print head of category-level aggregates
    print('\nCategory-level aggregate gross margins (first 5 dates):')
    for i, entry in enumerate(output_data[:5]):
        print(f"\n{entry['date']}:")
        for category in categories:
            if category in entry:
                print(f"  {category}: {entry[category]['category_aggregate']:.4f}")
    
    # Create a simplified DataFrame with only category aggregates for display
    category_only_df = master_df[['date'] + [f'{cat}-gross-margin-30d' for cat in categories if f'{cat}-gross-margin-30d' in master_df.columns]].copy()
    
    return category_only_df


# Function for calculating fees/TVL for parent protocols by parent category
def category_fees_to_tvl_comparison(start_date, parent_categories_path, top_protocols_path, defillama_output_dir):
    
    # Load the parent protocol categories mapping
    with open(parent_categories_path, 'r') as f:
        parent_categories_data = json.load(f)
    
    # Load the top protocols by category
    with open(top_protocols_path, 'r') as f:
        top_protocols_data = json.load(f)
    
    # Extract unique parent slugs from top protocols file for fees metric
    top_parent_slugs = set()
    if 'dailyFees' in top_protocols_data:
        for category, protocols in top_protocols_data['dailyFees'].items():
            for slug in protocols.keys():
                top_parent_slugs.add(slug)
    
    print(f"Found {len(top_parent_slugs)} top parent protocols with fees data")
    
    # Extract unique categories from top protocols
    categories = set()
    if 'dailyFees' in top_protocols_data:
        categories = set(top_protocols_data['dailyFees'].keys())
    
    print(f"Found categories: {categories}")
    
    master_df = None
    
    # Process each category
    for category in categories:
        print(f"\nProcessing category: {category}")
        
        # Get parent protocols in this category from top protocols file
        category_parent_slugs = set(top_protocols_data['dailyFees'].get(category, {}).keys())
        
        # Filter to only include parents that exist in parent_categories_data
        parents_in_category = {
            parent_slug: parent_categories_data[parent_slug]
            for parent_slug in category_parent_slugs
            if parent_slug in parent_categories_data
        }
        
        print(f"  Found {len(parents_in_category)} parent protocols in {category}")
        
        category_df = None
        
        # Process each parent protocol
        for parent_slug, parent_data in parents_in_category.items():
            print(f"  Processing parent: {parent_slug}")
            
            # Load fees and TVL for this parent protocol directly
            parent_dfs = {'fees': None, 'tvl': None}
            
            # Try to load fees
            fees_path = f"data/raw/defillama/fees/{parent_slug}.json"
            if not os.path.exists(fees_path):
                print(f"    Warning: {fees_path} not found, attempting to pull data")
                interim_metrics = {'dailyFees': 'fees'}
                interim_assets = {parent_slug: parent_slug}  # Use parent_slug instead of parent_data['name']
                failures = defillama.pull_metric_data_by_asset(interim_metrics, interim_assets, defillama_output_dir)
                if failures:
                    print(f"    Failed to pull fees data for {parent_slug}")
            
            if os.path.exists(fees_path):
                try:
                    with open(fees_path, 'r') as f:
                        obj = json.load(f)
                    df = pd.DataFrame(obj["totalDataChart"], columns=["timestamp", "fees"])
                    df["date"] = pd.to_datetime(df["timestamp"], unit="s")
                    df = df[["date", "fees"]]
                    df["fees"] = pd.to_numeric(df["fees"], errors='coerce').fillna(0)
                    parent_dfs['fees'] = df
                except Exception as e:
                    print(f"    Error loading fees from {fees_path}: {e}")
            
            # Try to load TVL
            tvl_path = f"data/raw/defillama/tvl/{parent_slug}.json"
            if not os.path.exists(tvl_path):
                print(f"    Warning: {tvl_path} not found, attempting to pull data")
                interim_metrics = {'tvl': 'tvl'}
                interim_assets = {parent_slug: parent_slug}  # FIXED: Use parent_slug instead of parent_data['name']
                failures = defillama.pull_metric_data_by_asset(interim_metrics, interim_assets, defillama_output_dir)
                if failures:
                    print(f"    Failed to pull TVL data for {parent_slug}")
            
            if os.path.exists(tvl_path):
                try:
                    with open(tvl_path, 'r') as f:
                        obj = json.load(f)
                    
                    # Handle TVL data structure from chainTvls
                    if 'chainTvls' in obj:
                        # Aggregate TVL across ALL chains
                        all_chain_dfs = []
                        for chain_name, chain_data in obj['chainTvls'].items():
                            if isinstance(chain_data, dict) and 'tvl' in chain_data and isinstance(chain_data['tvl'], list):
                                chain_df = pd.DataFrame(chain_data['tvl'])
                                chain_df['date'] = pd.to_datetime(chain_df['date'], unit='s')
                                chain_df.rename(columns={'totalLiquidityUSD': 'tvl'}, inplace=True)
                                chain_df = chain_df[['date', 'tvl']]
                                chain_df['tvl'] = pd.to_numeric(chain_df['tvl'], errors='coerce').fillna(0)
                                all_chain_dfs.append(chain_df)
                        
                        if len(all_chain_dfs) == 0:
                            print(f"    Warning: No valid TVL data found in chainTvls for {parent_slug}")
                        else:
                            # Merge and sum TVL across all chains
                            df = all_chain_dfs[0]
                            for chain_df in all_chain_dfs[1:]:
                                df = df.merge(chain_df, on='date', how='outer', suffixes=('', '_new'))
                                tvl_cols = [c for c in df.columns if c.startswith('tvl')]
                                df['tvl'] = df[tvl_cols].fillna(0).sum(axis=1)
                                df = df[['date', 'tvl']]
                            parent_dfs['tvl'] = df
                    elif 'tvl' in obj and isinstance(obj['tvl'], list):
                        # Fallback: direct tvl array (older format)
                        tvl_list = obj['tvl']
                        df = pd.DataFrame(tvl_list)
                        df['date'] = pd.to_datetime(df['date'], unit='s')
                        df.rename(columns={'totalLiquidityUSD': 'tvl'}, inplace=True)
                        df = df[['date', 'tvl']]
                        df['tvl'] = pd.to_numeric(df['tvl'], errors='coerce').fillna(0)
                        parent_dfs['tvl'] = df
                    else:
                        print(f"    Warning: Unknown TVL structure for {parent_slug}")
                except Exception as e:
                    print(f"    Error loading TVL from {tvl_path}: {e}")
            
            # Skip parent if we have no fees data OR no TVL data
            if parent_dfs['fees'] is None:
                print(f"    No fees data for parent {parent_slug}, skipping")
                continue
            
            if parent_dfs['tvl'] is None:
                print(f"    No TVL data for parent {parent_slug}, skipping")
                continue
            
            # Merge fees and TVL for this parent
            parent_df = parent_dfs['fees'].merge(parent_dfs['tvl'], on='date', how='outer').sort_values('date')
            parent_df['fees'] = parent_df['fees'].fillna(0)
            parent_df['tvl'] = parent_df['tvl'].fillna(0)
            
            # Calculate and print parent-level average fees/TVL ratio
            parent_total_fees = parent_df['fees'].sum()
            parent_avg_tvl = parent_df['tvl'].mean()
            if parent_avg_tvl > 0:
                # Annualized fees / Average TVL
                parent_annualized_fees = parent_total_fees * 365 / len(parent_df)
                parent_fees_to_tvl = parent_annualized_fees / parent_avg_tvl
                print(f"    {parent_slug} avg fees/TVL: {parent_fees_to_tvl:.2%}")
            else:
                print(f"    {parent_slug} has no TVL data")
            
            # Rename columns to include parent slug
            parent_df = parent_df.rename(columns={
                'fees': f'{parent_slug}-fees',
                'tvl': f'{parent_slug}-tvl'
            })
            
            # Merge into category_df
            if category_df is None:
                category_df = parent_df
            else:
                category_df = category_df.merge(parent_df, on='date', how='outer')
                num_cols = category_df.columns.difference(['date'])
                category_df[num_cols] = category_df[num_cols].fillna(0)
        
        # Skip category if no data was loaded
        if category_df is None:
            print(f"  No data for category {category}, skipping")
            continue
        
        # Get all fee and TVL columns for this category
        fee_cols = [c for c in category_df.columns if c.endswith('-fees')]
        tvl_cols = [c for c in category_df.columns if c.endswith('-tvl')]
        
        print(f"  Found {len(fee_cols)} parent fee columns and {len(tvl_cols)} parent TVL columns")
        
        # Calculate category-level fees/TVL ratio as (sum(fees) * 365) / sum(tvl)
        # Annualize daily fees by multiplying by 365
        total_fees = category_df[fee_cols].sum(axis=1)
        total_tvl = category_df[tvl_cols].sum(axis=1)
        
        # Calculate fees/TVL ratio: (annualized fees) / TVL
        annualized_fees = total_fees * 365
        category_df[f'{category}-fees-to-tvl'] = annualized_fees / total_tvl.replace(0, float('nan'))
        category_df[f'{category}-fees-to-tvl'] = category_df[f'{category}-fees-to-tvl'].fillna(0)
        
        # Apply rolling averages to the fees/TVL ratio
        category_df[f'{category}-fees-to-tvl-7d'] = (
            category_df[f'{category}-fees-to-tvl'].rolling(window=7, min_periods=1).mean()
        )
        category_df[f'{category}-fees-to-tvl-30d'] = (
            category_df[f'{category}-fees-to-tvl'].rolling(window=30, min_periods=1).mean()
        )
        
        print(f"  Category {category} processed successfully")
        
        # Save both category aggregate and individual parent protocol data
        # Category aggregate with 30d rolling average
        category_agg = category_df[['date', f'{category}-fees-to-tvl-30d']].copy()
        
        # Individual parent protocols with their fees/TVL ratios
        for parent_slug in parents_in_category.keys():
            fee_col = f'{parent_slug}-fees'
            tvl_col = f'{parent_slug}-tvl'
            
            if fee_col in category_df.columns and tvl_col in category_df.columns:
                # Calculate parent-level fees/TVL ratio (annualized)
                parent_annualized_fees = category_df[fee_col] * 365
                parent_fees_to_tvl = parent_annualized_fees / category_df[tvl_col].replace(0, float('nan'))
                parent_fees_to_tvl = parent_fees_to_tvl.fillna(0)
                
                # Apply 30d rolling average
                parent_fees_to_tvl_30d = parent_fees_to_tvl.rolling(window=30, min_periods=1).mean()
                category_agg[f'{parent_slug}-fees-to-tvl-30d'] = parent_fees_to_tvl_30d
        
        # Merge into the master_df
        if master_df is None:
            master_df = category_agg
        else:
            master_df = master_df.merge(category_agg, on="date", how="outer")
    
    # Cut off master_df to only be after start date
    master_df = master_df[master_df['date'] >= start_date].sort_values('date')
    
    # Forward fill NaN values
    master_df = master_df.ffill()
    
    # Create output directory if it doesn't exist
    output_dir = 'data/processed/financial-metrics'
    ensure_directory(output_dir)

    # Restructure the data to nest parent protocols under categories
    output_data = []
    for _, row in master_df.iterrows():
        date_entry = {'date': row['date'].isoformat()}

        # Group by category
        for category in categories:
            category_col = f'{category}-fees-to-tvl-30d'
            if category_col in row:
                category_entry = {
                    'category_aggregate': row[category_col]
                }
                
                # Add parent protocols for this category
                parent_protocols = {}
                for col in master_df.columns:
                    if col.endswith('-fees-to-tvl-30d') and col != category_col:
                        # Extract parent slug from column name
                        parent_slug = col.replace('-fees-to-tvl-30d', '')
                        # Check if this parent belongs to this category
                        if parent_slug in parent_categories_data and parent_categories_data[parent_slug]['category'] == category:
                            parent_protocols[parent_slug] = row[col]
                
                if parent_protocols:
                    category_entry['parent_protocols'] = parent_protocols
                
                date_entry[category] = category_entry
        
        output_data.append(date_entry)
    
    output_path = os.path.join(output_dir, 'fees-to-tvl-by-category.json')
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f'\nFees to TVL ratios by parent category saved to: {output_path}')
    
    # Print head of category-level aggregates
    print('\nCategory-level aggregate fees/TVL ratios (first 5 dates):')
    for i, entry in enumerate(output_data[:5]):
        print(f"\n{entry['date']}:")
        for category in categories:
            if category in entry:
                print(f"  {category}: {entry[category]['category_aggregate']:.4f}")
    
    # Create a simplified DataFrame with only category aggregates for display
    category_only_df = master_df[['date'] + [f'{cat}-fees-to-tvl-30d' for cat in categories if f'{cat}-fees-to-tvl-30d' in master_df.columns]].copy()
    
    return category_only_df


# Function for calculating revenue/TVL for parent protocols by parent category
def category_revenue_to_tvl_comparison(start_date, parent_categories_path, top_protocols_path, defillama_output_dir):
    # Load the parent protocol categories mapping
    with open(parent_categories_path, 'r') as f:
        parent_categories_data = json.load(f)
    
    # Load the top protocols by category
    with open(top_protocols_path, 'r') as f:
        top_protocols_data = json.load(f)
    
    # Extract unique parent slugs from top protocols file for revenue metric
    # Use dailyRevenue instead of dailyFees
    top_parent_slugs = set()
    if 'dailyRevenue' in top_protocols_data:
        for category, protocols in top_protocols_data['dailyRevenue'].items():
            for slug in protocols.keys():
                top_parent_slugs.add(slug)
    
    print(f"Found {len(top_parent_slugs)} top parent protocols with revenue data")
    
    # Extract unique categories from top protocols
    categories = set()
    if 'dailyRevenue' in top_protocols_data:
        categories = set(top_protocols_data['dailyRevenue'].keys())
    
    print(f"Found categories: {categories}")
    
    master_df = None
    
    # Process each category
    for category in categories:
        print(f"\nProcessing category: {category}")
        
        # Get parent protocols in this category from top protocols file
        category_parent_slugs = set(top_protocols_data['dailyRevenue'].get(category, {}).keys())
        
        # Filter to only include parents that exist in parent_categories_data
        parents_in_category = {
            parent_slug: parent_categories_data[parent_slug]
            for parent_slug in category_parent_slugs
            if parent_slug in parent_categories_data
        }
        
        print(f"  Found {len(parents_in_category)} parent protocols in {category}")
        
        category_df = None
        
        # Process each parent protocol
        for parent_slug, parent_data in parents_in_category.items():
            print(f"  Processing parent: {parent_slug}")
            
            # Load revenue and TVL for this parent protocol directly
            parent_dfs = {'revenue': None, 'tvl': None}
            
            # Try to load revenue
            revenue_path = f"data/raw/defillama/revenue/{parent_slug}.json"
            if not os.path.exists(revenue_path):
                print(f"    Warning: {revenue_path} not found, attempting to pull data")
                interim_metrics = {'dailyRevenue': 'revenue'}
                interim_assets = {parent_slug: parent_slug}
                failures = defillama.pull_metric_data_by_asset(interim_metrics, interim_assets, defillama_output_dir)
                if failures:
                    print(f"    Failed to pull revenue data for {parent_slug}")
            
            if os.path.exists(revenue_path):
                try:
                    with open(revenue_path, 'r') as f:
                        obj = json.load(f)
                    df = pd.DataFrame(obj["totalDataChart"], columns=["timestamp", "revenue"])
                    df["date"] = pd.to_datetime(df["timestamp"], unit="s")
                    df = df[["date", "revenue"]]
                    df["revenue"] = pd.to_numeric(df["revenue"], errors='coerce').fillna(0)
                    parent_dfs['revenue'] = df
                except Exception as e:
                    print(f"    Error loading revenue from {revenue_path}: {e}")
            
            # Try to load TVL
            tvl_path = f"data/raw/defillama/tvl/{parent_slug}.json"
            if not os.path.exists(tvl_path):
                print(f"    Warning: {tvl_path} not found, attempting to pull data")
                interim_metrics = {'tvl': 'tvl'}
                interim_assets = {parent_slug: parent_slug}
                failures = defillama.pull_metric_data_by_asset(interim_metrics, interim_assets, defillama_output_dir)
                if failures:
                    print(f"    Failed to pull TVL data for {parent_slug}")
            
            if os.path.exists(tvl_path):
                try:
                    with open(tvl_path, 'r') as f:
                        obj = json.load(f)
                    
                    # Handle TVL data structure from chainTvls
                    if 'chainTvls' in obj:
                        # Aggregate TVL across ALL chains
                        all_chain_dfs = []
                        for chain_name, chain_data in obj['chainTvls'].items():
                            if isinstance(chain_data, dict) and 'tvl' in chain_data and isinstance(chain_data['tvl'], list):
                                chain_df = pd.DataFrame(chain_data['tvl'])
                                chain_df['date'] = pd.to_datetime(chain_df['date'], unit='s')
                                chain_df.rename(columns={'totalLiquidityUSD': 'tvl'}, inplace=True)
                                chain_df = chain_df[['date', 'tvl']]
                                chain_df['tvl'] = pd.to_numeric(chain_df['tvl'], errors='coerce').fillna(0)
                                all_chain_dfs.append(chain_df)
                        
                        if len(all_chain_dfs) == 0:
                            print(f"    Warning: No valid TVL data found in chainTvls for {parent_slug}")
                        else:
                            # Merge and sum TVL across all chains
                            df = all_chain_dfs[0]
                            for chain_df in all_chain_dfs[1:]:
                                df = df.merge(chain_df, on='date', how='outer', suffixes=('', '_new'))
                                tvl_cols = [c for c in df.columns if c.startswith('tvl')]
                                df['tvl'] = df[tvl_cols].fillna(0).sum(axis=1)
                                df = df[['date', 'tvl']]
                            parent_dfs['tvl'] = df
                    elif 'tvl' in obj and isinstance(obj['tvl'], list):
                        # Fallback: direct tvl array (older format)
                        tvl_list = obj['tvl']
                        df = pd.DataFrame(tvl_list)
                        df['date'] = pd.to_datetime(df['date'], unit='s')
                        df.rename(columns={'totalLiquidityUSD': 'tvl'}, inplace=True)
                        df = df[['date', 'tvl']]
                        df['tvl'] = pd.to_numeric(df['tvl'], errors='coerce').fillna(0)
                        parent_dfs['tvl'] = df
                    else:
                        print(f"    Warning: Unknown TVL structure for {parent_slug}")
                except Exception as e:
                    print(f"    Error loading TVL from {tvl_path}: {e}")
            
            # Skip parent if we have no revenue data OR no TVL data
            if parent_dfs['revenue'] is None:
                print(f"    No revenue data for parent {parent_slug}, skipping")
                continue
            
            if parent_dfs['tvl'] is None:
                print(f"    No TVL data for parent {parent_slug}, skipping")
                continue
            
            # Merge revenue and TVL for this parent
            parent_df = parent_dfs['revenue'].merge(parent_dfs['tvl'], on='date', how='outer').sort_values('date')
            parent_df['revenue'] = parent_df['revenue'].fillna(0)
            parent_df['tvl'] = parent_df['tvl'].fillna(0)
            
            # Calculate and print parent-level average revenue/TVL ratio
            parent_total_revenue = parent_df['revenue'].sum()
            parent_avg_tvl = parent_df['tvl'].mean()
            if parent_avg_tvl > 0:
                # Annualized revenue / Average TVL
                parent_annualized_revenue = parent_total_revenue * 365 / len(parent_df)
                parent_revenue_to_tvl = parent_annualized_revenue / parent_avg_tvl
                print(f"    {parent_slug} avg revenue/TVL: {parent_revenue_to_tvl:.2%}")
            else:
                print(f"    {parent_slug} has no TVL data")
            
            # Rename columns to include parent slug
            parent_df = parent_df.rename(columns={
                'revenue': f'{parent_slug}-revenue',
                'tvl': f'{parent_slug}-tvl'
            })
            
            # Merge into category_df
            if category_df is None:
                category_df = parent_df
            else:
                category_df = category_df.merge(parent_df, on='date', how='outer')
                num_cols = category_df.columns.difference(['date'])
                category_df[num_cols] = category_df[num_cols].fillna(0)
        
        # Skip category if no data was loaded
        if category_df is None:
            print(f"  No data for category {category}, skipping")
            continue
        
        # Get all revenue and TVL columns for this category
        revenue_cols = [c for c in category_df.columns if c.endswith('-revenue')]
        tvl_cols = [c for c in category_df.columns if c.endswith('-tvl')]
        
        print(f"  Found {len(revenue_cols)} parent revenue columns and {len(tvl_cols)} parent TVL columns")
        
        # Calculate category-level revenue/TVL ratio as (sum(revenue) * 365) / sum(tvl)
        # Annualize daily revenue by multiplying by 365
        total_revenue = category_df[revenue_cols].sum(axis=1)
        total_tvl = category_df[tvl_cols].sum(axis=1)
        
        # Calculate revenue/TVL ratio: (annualized revenue) / TVL
        annualized_revenue = total_revenue * 365
        category_df[f'{category}-revenue-to-tvl'] = annualized_revenue / total_tvl.replace(0, float('nan'))
        category_df[f'{category}-revenue-to-tvl'] = category_df[f'{category}-revenue-to-tvl'].fillna(0)
        
        # Apply rolling averages to the revenue/TVL ratio
        category_df[f'{category}-revenue-to-tvl-7d'] = (
            category_df[f'{category}-revenue-to-tvl'].rolling(window=7, min_periods=1).mean()
        )
        category_df[f'{category}-revenue-to-tvl-30d'] = (
            category_df[f'{category}-revenue-to-tvl'].rolling(window=30, min_periods=1).mean()
        )
        
        print(f"  Category {category} processed successfully")
        
        # Save both category aggregate and individual parent protocol data
        # Category aggregate with 30d rolling average
        category_agg = category_df[['date', f'{category}-revenue-to-tvl-30d']].copy()
        
        # Individual parent protocols with their revenue/TVL ratios
        for parent_slug in parents_in_category.keys():
            revenue_col = f'{parent_slug}-revenue'
            tvl_col = f'{parent_slug}-tvl'
            
            if revenue_col in category_df.columns and tvl_col in category_df.columns:
                # Calculate parent-level revenue/TVL ratio (annualized)
                parent_annualized_revenue = category_df[revenue_col] * 365
                parent_revenue_to_tvl = parent_annualized_revenue / category_df[tvl_col].replace(0, float('nan'))
                parent_revenue_to_tvl = parent_revenue_to_tvl.fillna(0)
                
                # Apply 30d rolling average
                parent_revenue_to_tvl_30d = parent_revenue_to_tvl.rolling(window=30, min_periods=1).mean()
                category_agg[f'{parent_slug}-revenue-to-tvl-30d'] = parent_revenue_to_tvl_30d
        
        # Merge into the master_df
        if master_df is None:
            master_df = category_agg
        else:
            master_df = master_df.merge(category_agg, on="date", how="outer")
    
    # Cut off master_df to only be after start date
    master_df = master_df[master_df['date'] >= start_date].sort_values('date')
    
    # Forward fill NaN values
    master_df = master_df.ffill()
    
    # Create output directory if it doesn't exist
    output_dir = 'data/processed/financial-metrics'
    ensure_directory(output_dir)

    # Restructure the data to nest parent protocols under categories
    output_data = []
    for _, row in master_df.iterrows():
        date_entry = {'date': row['date'].isoformat()}

        # Group by category
        for category in categories:
            category_col = f'{category}-revenue-to-tvl-30d'
            if category_col in row:
                category_entry = {
                    'category_aggregate': row[category_col]
                }
                
                # Add parent protocols for this category
                parent_protocols = {}
                for col in master_df.columns:
                    if col.endswith('-revenue-to-tvl-30d') and col != category_col:
                        # Extract parent slug from column name
                        parent_slug = col.replace('-revenue-to-tvl-30d', '')
                        # Check if this parent belongs to this category
                        if parent_slug in parent_categories_data and parent_categories_data[parent_slug]['category'] == category:
                            parent_protocols[parent_slug] = row[col]
                
                if parent_protocols:
                    category_entry['parent_protocols'] = parent_protocols
                
                date_entry[category] = category_entry
        
        output_data.append(date_entry)
    
    output_path = os.path.join(output_dir, 'revenue-to-tvl-by-category.json')
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f'\nRevenue to TVL ratios by parent category saved to: {output_path}')
    
    # Print head of category-level aggregates
    print('\nCategory-level aggregate revenue/TVL ratios (first 5 dates):')
    for i, entry in enumerate(output_data[:5]):
        print(f"\n{entry['date']}:")
        for category in categories:
            if category in entry:
                print(f"  {category}: {entry[category]['category_aggregate']:.4f}")
    
    # Create a simplified DataFrame with only category aggregates for display
    category_only_df = master_df[['date'] + [f'{cat}-revenue-to-tvl-30d' for cat in categories if f'{cat}-revenue-to-tvl-30d' in master_df.columns]].copy()
    
    return category_only_df

# Function to process and save annualized volatility data as JSON
def process_annualized_volatility_data(start_date, trailing_window=7):
    price_dir = 'data/raw/messari/price/'
    
    for filename in os.listdir(price_dir):
        if not filename.endswith('.json'):
            continue
            
        asset_slug = filename.replace('.json', '')
        price_path = os.path.join(price_dir, filename)
        
        with open(price_path) as price_f:
            price_data = json.load(price_f)
        
        points = price_data['data']['points']
        price_df = pd.DataFrame(points, columns=[
            'timestamp',
            'open',
            'high',
            'low',
            'close',
            'volume'
        ])
        price_df["date"] = pd.to_datetime(price_df["timestamp"], unit="s")
        price_df = price_df[['date', 'close']]

        # Fill blanks
        price_df = price_df.ffill()
        
        # Calculate volatility
        price_df['std'] = price_df['close'].rolling(window=trailing_window).std()

        # Annualize volatility
        price_df['annualized_std'] = price_df['std'] * math.sqrt(365)

        # Streamline df
        price_df = price_df[['date', 'annualized_std']]

        # Filter by date range
        price_df = price_df[price_df['date'] >= start_date]
        price_df = price_df.ffill()
        
        # Convert date to string for JSON serialization
        price_df['date'] = price_df['date'].dt.strftime('%Y-%m-%d')
        
        # Save to JSON
        ensure_directory('data/processed/annualized-volatility')
        
        output_data = {
            'asset_slug': asset_slug,
            'trailing_window': trailing_window,
            'data': price_df.to_dict(orient='records')
        }
        
        with open(f"data/processed/annualized-volatility/{asset_slug}.json", 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Saved {asset_slug} data to data/processed/annualized-volatility/{asset_slug}.json")
    
    print('Done')


# BLOCKWORKS BORROW/LEND METRICS

# Function to calculate borrow/lend efficiency ratios from Blockworks data
def calculate_borrow_lend_ratios(
    protocols,
    blockworks_output_dir='data/raw/blockworks',
    output_filepath='data/processed/borrow-lend/borrow-lend-ratios.json'
):
    """
    Calculate efficiency ratios for Blockworks borrow/lend protocols and save as a processed file.

    For each protocol this loads the daily Blockworks metrics (interest generated,
    net interest income, revenue, total borrows, total deposits) and computes the
    following daily ratios:
        - interest generated / total borrows
        - net interest income / total borrows
        - revenue / total borrows
        - interest generated / total deposits
        - net interest income / total deposits
        - revenue / total deposits
        - total borrows / total deposits

    Args:
        protocols: Iterable of Blockworks slugs (which match the saved filenames,
                   e.g. ['aave', 'morpho', 'kamino', 'spark', 'euler']). Can also be
                   a dict, in which case its keys are used.
        blockworks_output_dir: Directory holding the raw Blockworks JSON files.
        output_filepath: Path to save the processed ratios JSON file.

    Returns:
        dict: {protocol_slug: [{date, ...ratios}, ...]}
    """
    # Map raw Blockworks metric folders to the column names used here
    metric_files = {
        'interest_generated': 'interest-total-usd',
        'net_interest_income': 'nii-total-usd',
        'revenue': 'revenue-total-usd',
        'total_borrows': 'borrow-total-usd',
        'total_deposits': 'deposit-total-usd',
    }

    # Ratios to compute as (numerator_column, denominator_column)
    ratio_definitions = {
        'interest_generated_to_total_borrows': ('interest_generated', 'total_borrows'),
        'net_interest_income_to_total_borrows': ('net_interest_income', 'total_borrows'),
        'revenue_to_total_borrows': ('revenue', 'total_borrows'),
        'interest_generated_to_total_deposits': ('interest_generated', 'total_deposits'),
        'net_interest_income_to_total_deposits': ('net_interest_income', 'total_deposits'),
        'revenue_to_total_deposits': ('revenue', 'total_deposits'),
        'total_borrows_to_total_deposits': ('total_borrows', 'total_deposits'),
    }

    # Allow passing a dict (use its keys) or any iterable of slugs
    if isinstance(protocols, dict):
        protocols = list(protocols.keys())

    results = {}

    for slug in protocols:
        print(f"Processing {slug}...")

        # Load each metric time series and merge on date
        protocol_df = None
        missing_metrics = []
        for column, metric_folder in metric_files.items():
            path = f"{blockworks_output_dir}/{metric_folder}/{slug}.json"
            if not os.path.exists(path):
                missing_metrics.append(metric_folder)
                continue

            with open(path, 'r') as f:
                obj = json.load(f)

            # Files are keyed by the asset slug; fall back to the first value
            series = obj.get(slug)
            if series is None:
                series = next(iter(obj.values()), [])

            df = pd.DataFrame(series)
            if df.empty or 'date' not in df.columns:
                missing_metrics.append(metric_folder)
                continue

            df = df[['date', 'value']].rename(columns={'value': column})
            df[column] = pd.to_numeric(df[column], errors='coerce')

            if protocol_df is None:
                protocol_df = df
            else:
                protocol_df = protocol_df.merge(df, on='date', how='outer')

        if missing_metrics:
            print(f"  Warning: missing metric(s) for {slug}: {', '.join(missing_metrics)}")

        if protocol_df is None:
            print(f"  No data found for {slug}, skipping")
            continue

        protocol_df = protocol_df.sort_values('date').reset_index(drop=True)

        # Ensure all expected metric columns exist before computing ratios
        for column in metric_files:
            if column not in protocol_df.columns:
                protocol_df[column] = np.nan

        # Compute ratios, treating non-positive denominators as undefined (NaN)
        for ratio_name, (numerator, denominator) in ratio_definitions.items():
            denom = protocol_df[denominator].where(protocol_df[denominator] > 0, np.nan)
            protocol_df[ratio_name] = protocol_df[numerator] / denom

        # Keep date plus computed ratios; replace NaN/inf with None for clean JSON
        # (cast to object first so None survives instead of being coerced back to NaN)
        output_columns = ['date'] + list(ratio_definitions.keys())
        protocol_out = protocol_df[output_columns].replace([np.inf, -np.inf], np.nan)
        protocol_out = protocol_out.astype(object).where(pd.notnull(protocol_out), None)

        results[slug] = protocol_out.to_dict(orient='records')
        print(f"  Computed ratios for {len(results[slug])} dates")

    # Save processed file
    ensure_directory(os.path.dirname(output_filepath))
    with open(output_filepath, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nBorrow/lend ratios saved to: {output_filepath}")
    print(f"Protocols processed: {len(results)}")
    print('Done')


# VALUATION MULTIPLES

def replace_multiple_outliers(asset_data, outlier_factor=100, outlier_window=90, min_periods=30):
    """Replace outliers in one protocol's rolling-window multiples with the prior good value.

    Applies to keys such as 'FDV-to-annualized-fees-multiple-7d' (every rolling window).
    The instant daily multiple and the daily input series are left as computed, because
    single lumpy fee days are normal for many protocols and the rolling averages already
    smooth them. A value is an outlier when it is more than ``outlier_factor`` times above
    or below the median of that multiple over the prior ``outlier_window`` days (at least
    ``min_periods`` of them must exist). Outliers are overwritten with the most recent
    non-outlier value so nothing shows up as NaN or 0 downstream; a leading outlier with
    no prior value is dropped.

    Returns:
        {multiple_key: ['YYYY-MM-DD', ...] of days replaced}
    """
    flagged = {}
    if not outlier_factor or not asset_data:
        return flagged
    dates = sorted(asset_data)
    keys = {k for d in dates for k in asset_data[d] if re.search(r'-multiple-\d+d$', k)}
    for key in sorted(keys):
        s = pd.Series([asset_data[d].get(key) for d in dates], index=dates, dtype='float64')
        ref = s.shift(1).rolling(outlier_window, min_periods=min_periods).median()
        mask = ref.notna() & s.notna() & (ref > 0) & ((s > ref * outlier_factor) | (s < ref / outlier_factor))
        if not mask.any():
            continue
        cleaned = s.mask(mask).ffill()
        flagged[key] = [d for d, m in zip(dates, mask) if m]
        for d in flagged[key]:
            v = cleaned[d]
            if pd.isna(v):
                asset_data[d].pop(key, None)
            else:
                asset_data[d][key] = float(v)
    return flagged


def process_asset_metrics_data(categories, output_filepath, outlier_factor=100, outlier_window=90):
    """
    Process all metric combinations for all categories and save to a single JSON file.
    
    Output format:
    {
        "date": {
            "category_name": {
                "asset_slug": {
                    "metric": value
                }
            }
        }
    }
    
    Parameters:
    -----------
    categories : dict
        Dictionary mapping category names to asset dictionaries
        e.g., {'perps_dexs': {'hyperliquid': 'hyperliquid', ...}, ...}
    output_filepath : str
        Path to save the output JSON file
    outlier_factor : float or None
        Outlier rule applied to each protocol's rolling-window multiples (e.g.
        'FDV-to-annualized-fees-multiple-7d') after they are computed; see
        replace_multiple_outliers(). A value more than ``outlier_factor`` times above or
        below that multiple's median over the prior ``outlier_window`` days is replaced by
        the previous good value, so nothing shows up as NaN or 0 downstream, and the day is
        listed under ``outliers-replaced``. Daily inputs are not touched, so lumpy single-day
        fees survive. Default 100. Catches data breaks such as a fee adapter going dark,
        which otherwise produce multiples in the hundreds of thousands. None disables.
    outlier_window : int
        Trailing window (days) for the median in the outlier rule. Default 90; no flagging
        until 30 prior days exist.
    
    Returns:
    --------
    None (saves nested JSON structure to file)
    """
    all_dates = set()
    category_data = {}
    annualized_bottom_metrics = ['fees', 'revenue', 'holders-revenue']
    non_annualized_bottom_metrics = ['tvl']
    bottom_metrics = annualized_bottom_metrics + non_annualized_bottom_metrics
    top_metrics = ['FDV', 'MCAP']
    rolling_windows = [1, 7, 30, 90]

    # Blockworks borrow/lend bottom metrics (sourced from data/raw/blockworks/<metric>/<slug>.json).
    # Flows (interest, NII, revenue) are annualized like the DefiLlama flow metrics; total deposits
    # is a point-in-time stock and is treated like TVL (not annualized). These are named distinctly
    # so they do not collide with the DefiLlama 'revenue' bottom metric. Only the Blockworks
    # borrow/lend protocols have these files; for every other asset the loads are skipped silently.
    blockworks_output_dir = 'data/raw/blockworks'
    blockworks_annualized_metrics = {
        'interest-generated': 'interest-total-usd',
        'net-interest-income': 'nii-total-usd',
        'total-revenue': 'revenue-total-usd',
    }
    blockworks_non_annualized_metrics = {
        'total-deposits': 'deposit-total-usd',
        'total-borrows': 'borrow-total-usd',
    }

    # First pass to determine global date range
    print("\nDetermining global date range...")
    global_min_date = None
    global_max_date = None

    for category_name, assets in categories.items():
        for asset_slug, asset_name in assets.items():
            # Check annualized bottom metrics
            for bottom_metric in annualized_bottom_metrics:
                bottom_path = f"data/raw/defillama/{bottom_metric}/{asset_slug}.json"
                try:
                    with open(bottom_path) as f:
                        bottom_data = json.load(f)
                    temp_df = pd.DataFrame(bottom_data["totalDataChart"], columns=["timestamp", bottom_metric])
                    temp_df["date"] = pd.to_datetime(temp_df["timestamp"], unit="s")
                    
                    if global_min_date is None or temp_df["date"].min() < global_min_date:
                        global_min_date = temp_df["date"].min()
                    if global_max_date is None or temp_df["date"].max() > global_max_date:
                        global_max_date = temp_df["date"].max()
                except:
                    pass
            
            # Check top metrics (marketcap)
            top_path = f'data/raw/messari/marketcap/{asset_slug}.json'
            try:
                with open(top_path) as f:
                    top_data = json.load(f)
                points = top_data['data']['points']
                top_df = pd.DataFrame(points, columns=['timestamp', 'circulating-marketcap', 'circulating-marketcap-dominance', 'fully-diluted-marketcap'])
                top_df["date"] = pd.to_datetime(top_df["timestamp"], unit="s")
                
                if global_min_date is None or top_df["date"].min() < global_min_date:
                    global_min_date = top_df["date"].min()
                if global_max_date is None or top_df["date"].max() > global_max_date:
                    global_max_date = top_df["date"].max()
            except:
                pass
            
            # Check TVL data
            tvl_path = f'data/raw/defillama/tvl/{asset_slug}.json'
            try:
                with open(tvl_path) as f:
                    tvl_data = json.load(f)
                tvl_df = pd.DataFrame(tvl_data["tvl"])
                tvl_df["date"] = pd.to_datetime(tvl_df["date"], unit="s")
                
                if global_min_date is None or tvl_df["date"].min() < global_min_date:
                    global_min_date = tvl_df["date"].min()
                if global_max_date is None or tvl_df["date"].max() > global_max_date:
                    global_max_date = tvl_df["date"].max()
            except:
                pass

            # Check Blockworks borrow/lend metrics (interest, NII, revenue, deposits)
            for folder in list(blockworks_annualized_metrics.values()) + list(blockworks_non_annualized_metrics.values()):
                bw_path = f"{blockworks_output_dir}/{folder}/{asset_slug}.json"
                if not os.path.exists(bw_path):
                    continue
                try:
                    with open(bw_path) as f:
                        bw_data = json.load(f)
                    series = bw_data.get(asset_slug)
                    if series is None:
                        series = next(iter(bw_data.values()), [])
                    bw_df = pd.DataFrame(series)
                    if bw_df.empty or 'date' not in bw_df.columns:
                        continue
                    bw_df["date"] = pd.to_datetime(bw_df["date"])

                    if global_min_date is None or bw_df["date"].min() < global_min_date:
                        global_min_date = bw_df["date"].min()
                    if global_max_date is None or bw_df["date"].max() > global_max_date:
                        global_max_date = bw_df["date"].max()
                except:
                    pass

    # Create global date range
    global_date_range = pd.date_range(start=global_min_date, end=global_max_date, freq='D')
    print(f"Global date range: {global_min_date.date()} to {global_max_date.date()} ({len(global_date_range)} days)")
    
    # Process each category
    for category_name, assets in categories.items():
        print(f"\nProcessing category: {category_name}")
        category_data[category_name] = {}
        
        # Load data for each asset
        for asset_slug, asset_name in assets.items():
            print(f"  Processing asset: {asset_name}")
            asset_data = {}
            available_annualized_metrics = []
            available_non_annualized_metrics = []
            
            # Load annualized bottom metrics (skip if file doesn't exist)
            for bottom_metric in annualized_bottom_metrics:
                bottom_path = f"data/raw/defillama/{bottom_metric}/{asset_slug}.json"
                
                try:
                    with open(bottom_path) as bottom_f:
                        bottom_data = json.load(bottom_f)
                
                    temp_df = pd.DataFrame(bottom_data["totalDataChart"], columns=["timestamp", bottom_metric])
                    temp_df["date"] = pd.to_datetime(temp_df["timestamp"], unit="s")
                    
                    # Forward fill the bottom metric to handle missing days
                    temp_df = temp_df.set_index('date')
                    temp_df = temp_df.sort_index()
                    temp_df = temp_df.reindex(global_date_range)
                    temp_df = temp_df.ffill()
                    temp_df = temp_df.reset_index()
                    temp_df = temp_df.rename(columns={'index': 'date'})
                    temp_df['date'] = temp_df['date'].dt.strftime('%Y-%m-%d')
                    
                    temp_df[f"annualized-{bottom_metric}"] = temp_df[bottom_metric] * 365
                    
                    # Store in asset_data dictionary
                    for _, row in temp_df.iterrows():
                        date = row["date"]
                        all_dates.add(date)
                        if date not in asset_data:
                            asset_data[date] = {}
                        asset_data[date][bottom_metric] = row[bottom_metric]
                        asset_data[date][f"annualized-{bottom_metric}"] = row[f"annualized-{bottom_metric}"]
                    
                    available_annualized_metrics.append(bottom_metric)
                    
                except FileNotFoundError:
                    print(f"    Warning: {bottom_metric} file not found for {asset_name}, skipping this metric")
                    continue
                except Exception as e:
                    print(f"    Error loading {bottom_metric} for {asset_name}: {str(e)}")
                    continue
            
            # Load TVL data (non-annualized bottom metric)
            try:
                tvl_path = f'data/raw/defillama/tvl/{asset_slug}.json'
                with open(tvl_path) as tvl_f:
                    tvl_data = json.load(tvl_f)
                
                tvl_df = pd.DataFrame(tvl_data["tvl"])
                tvl_df["date"] = pd.to_datetime(tvl_df["date"], unit="s")
                tvl_df.rename(columns={'totalLiquidityUSD': 'TVL'}, inplace=True)
                
                # Forward fill TVL data
                tvl_df = tvl_df.set_index('date')
                tvl_df = tvl_df.sort_index()
                tvl_df = tvl_df[['TVL']].reindex(global_date_range)
                tvl_df = tvl_df.ffill()
                tvl_df = tvl_df.reset_index()
                tvl_df = tvl_df.rename(columns={'index': 'date'})
                tvl_df['date'] = tvl_df['date'].dt.strftime('%Y-%m-%d')
                
                # Store in asset_data dictionary
                for _, row in tvl_df.iterrows():
                    date = row["date"]
                    all_dates.add(date)
                    if date not in asset_data:
                        asset_data[date] = {}
                    asset_data[date]['TVL'] = row['TVL']
                
                available_non_annualized_metrics.append('TVL')
                
            except FileNotFoundError:
                print(f"    Warning: TVL file not found for {asset_name}, skipping TVL metric")
            except Exception as e:
                print(f"    Error loading TVL for {asset_name}: {str(e)}")

            # Load Blockworks borrow/lend annualized metrics (interest generated, NII, revenue)
            for bottom_metric, folder in blockworks_annualized_metrics.items():
                bw_path = f"{blockworks_output_dir}/{folder}/{asset_slug}.json"
                if not os.path.exists(bw_path):
                    continue
                try:
                    with open(bw_path) as bw_f:
                        bw_obj = json.load(bw_f)
                    series = bw_obj.get(asset_slug)
                    if series is None:
                        series = next(iter(bw_obj.values()), [])
                    temp_df = pd.DataFrame(series)
                    if temp_df.empty or 'date' not in temp_df.columns:
                        continue
                    temp_df = temp_df[['date', 'value']].rename(columns={'value': bottom_metric})
                    temp_df['date'] = pd.to_datetime(temp_df['date'])
                    temp_df[bottom_metric] = pd.to_numeric(temp_df[bottom_metric], errors='coerce')

                    # Forward fill the bottom metric onto the global date range
                    temp_df = temp_df.set_index('date').sort_index()
                    temp_df = temp_df[~temp_df.index.duplicated(keep='last')]
                    temp_df = temp_df.reindex(global_date_range)
                    temp_df = temp_df.ffill()
                    temp_df = temp_df.reset_index()
                    temp_df = temp_df.rename(columns={'index': 'date'})
                    temp_df['date'] = temp_df['date'].dt.strftime('%Y-%m-%d')

                    temp_df[f"annualized-{bottom_metric}"] = temp_df[bottom_metric] * 365

                    for _, row in temp_df.iterrows():
                        date = row["date"]
                        all_dates.add(date)
                        if date not in asset_data:
                            asset_data[date] = {}
                        asset_data[date][bottom_metric] = row[bottom_metric]
                        asset_data[date][f"annualized-{bottom_metric}"] = row[f"annualized-{bottom_metric}"]

                    available_annualized_metrics.append(bottom_metric)

                except Exception as e:
                    print(f"    Error loading {bottom_metric} (Blockworks) for {asset_name}: {str(e)}")
                    continue

            # Load Blockworks borrow/lend non-annualized metrics (total deposits)
            for bottom_metric, folder in blockworks_non_annualized_metrics.items():
                bw_path = f"{blockworks_output_dir}/{folder}/{asset_slug}.json"
                if not os.path.exists(bw_path):
                    continue
                try:
                    with open(bw_path) as bw_f:
                        bw_obj = json.load(bw_f)
                    series = bw_obj.get(asset_slug)
                    if series is None:
                        series = next(iter(bw_obj.values()), [])
                    temp_df = pd.DataFrame(series)
                    if temp_df.empty or 'date' not in temp_df.columns:
                        continue
                    temp_df = temp_df[['date', 'value']].rename(columns={'value': bottom_metric})
                    temp_df['date'] = pd.to_datetime(temp_df['date'])
                    temp_df[bottom_metric] = pd.to_numeric(temp_df[bottom_metric], errors='coerce')

                    temp_df = temp_df.set_index('date').sort_index()
                    temp_df = temp_df[~temp_df.index.duplicated(keep='last')]
                    temp_df = temp_df.reindex(global_date_range)
                    temp_df = temp_df.ffill()
                    temp_df = temp_df.reset_index()
                    temp_df = temp_df.rename(columns={'index': 'date'})
                    temp_df['date'] = temp_df['date'].dt.strftime('%Y-%m-%d')

                    for _, row in temp_df.iterrows():
                        date = row["date"]
                        all_dates.add(date)
                        if date not in asset_data:
                            asset_data[date] = {}
                        asset_data[date][bottom_metric] = row[bottom_metric]

                    available_non_annualized_metrics.append(bottom_metric)

                except Exception as e:
                    print(f"    Error loading {bottom_metric} (Blockworks) for {asset_name}: {str(e)}")
                    continue

            # Skip asset entirely if no bottom metrics were found
            if not available_annualized_metrics and not available_non_annualized_metrics:
                print(f"    Warning: No bottom metrics found for {asset_name}, skipping this asset")
                continue
            
            # Load top metrics (FDV and MCAP)
            try:
                top_path = f'data/raw/messari/marketcap/{asset_slug}.json'
                with open(top_path) as top_f:
                    top_data = json.load(top_f)
                
                points = top_data['data']['points']
                top_df = pd.DataFrame(points, columns=[
                    'timestamp',
                    'circulating-marketcap',
                    'circulating-marketcap-dominance',
                    'fully-diluted-marketcap'
                ])
                top_df["date"] = pd.to_datetime(top_df["timestamp"], unit="s")
                top_df.rename(columns={
                    'circulating-marketcap': 'MCAP',
                    'fully-diluted-marketcap': 'FDV'
                }, inplace=True)
                
                # Forward fill market cap data as well
                top_df = top_df.set_index('date')
                top_df = top_df.sort_index()
                top_df = top_df[['MCAP', 'FDV']].reindex(global_date_range)
                top_df = top_df.ffill()
                top_df = top_df.reset_index()
                top_df = top_df.rename(columns={'index': 'date'})
                top_df['date'] = top_df['date'].dt.strftime('%Y-%m-%d')
                
                # Store in asset_data dictionary
                for _, row in top_df.iterrows():
                    date = row["date"]
                    all_dates.add(date)
                    if date not in asset_data:
                        asset_data[date] = {}
                    asset_data[date]['MCAP'] = row['MCAP']
                    asset_data[date]['FDV'] = row['FDV']
                
            except FileNotFoundError:
                print(f"    Warning: Market cap file not found for {asset_name}, skipping this asset")
                continue
            except Exception as e:
                print(f"    Error loading market cap for {asset_name}: {str(e)}, skipping this asset")
                continue
            
            # Calculate rolling averages for TOP and BOTTOM metrics for all window sizes
            sorted_dates = sorted(asset_data.keys())
            
            for window in rolling_windows:
                for i, date in enumerate(sorted_dates):
                    window_start = max(0, i - (window - 1))
                    window_dates = sorted_dates[window_start:i+1]
                    
                    # Calculate rolling averages for top metrics
                    for top_metric in top_metrics:
                        values = [
                            asset_data[d][top_metric]
                            for d in window_dates
                            if top_metric in asset_data[d] and not pd.isna(asset_data[d][top_metric])
                        ]
                        if values:
                            asset_data[date][f"{top_metric}-{window}d"] = sum(values) / len(values)
                    
                    # Calculate rolling averages for annualized bottom metrics
                    for bottom_metric in available_annualized_metrics:
                        annualized_key = f"annualized-{bottom_metric}"
                        values = [
                            asset_data[d][annualized_key]
                            for d in window_dates
                            if annualized_key in asset_data[d] and not pd.isna(asset_data[d][annualized_key])
                        ]
                        if values:
                            asset_data[date][f"{annualized_key}-{window}d"] = sum(values) / len(values)
                    
                    # Calculate rolling averages for non-annualized bottom metrics (TVL)
                    for bottom_metric in available_non_annualized_metrics:
                        values = [
                            asset_data[d][bottom_metric]
                            for d in window_dates
                            if bottom_metric in asset_data[d] and not pd.isna(asset_data[d][bottom_metric])
                        ]
                        if values:
                            asset_data[date][f"{bottom_metric}-{window}d"] = sum(values) / len(values)
            
            # Now calculate multiples using both instant and smoothed data for all window sizes
            for date in asset_data:
                for top_metric in top_metrics:
                    # Multiples for annualized bottom metrics
                    for bottom_metric in available_annualized_metrics:
                        annualized_key = f"annualized-{bottom_metric}"
                        
                        # Calculate instant multiple
                        if (top_metric in asset_data[date] and 
                            annualized_key in asset_data[date] and
                            asset_data[date][annualized_key] != 0 and
                            asset_data[date][annualized_key] is not None and
                            not pd.isna(asset_data[date][annualized_key])):
                            
                            multiple_key = f"{top_metric}-to-annualized-{bottom_metric}-multiple"
                            asset_data[date][multiple_key] = (
                                asset_data[date][top_metric] / asset_data[date][annualized_key]
                            )
                        
                        # Calculate smoothed multiples for all window sizes
                        for window in rolling_windows:
                            annualized_window_key = f"{annualized_key}-{window}d"
                            top_window_key = f"{top_metric}-{window}d"
                            
                            if (top_window_key in asset_data[date] and
                                annualized_window_key in asset_data[date] and
                                asset_data[date][annualized_window_key] != 0 and
                                asset_data[date][annualized_window_key] is not None and
                                not pd.isna(asset_data[date][annualized_window_key])):
                                
                                multiple_key_window = f"{top_metric}-to-annualized-{bottom_metric}-multiple-{window}d"
                                asset_data[date][multiple_key_window] = (
                                    asset_data[date][top_window_key] / asset_data[date][annualized_window_key]
                                )
                    
                    # Multiples for non-annualized bottom metrics (TVL)
                    for bottom_metric in available_non_annualized_metrics:
                        # Calculate instant multiple
                        if (top_metric in asset_data[date] and 
                            bottom_metric in asset_data[date] and
                            asset_data[date][bottom_metric] != 0 and
                            asset_data[date][bottom_metric] is not None and
                            not pd.isna(asset_data[date][bottom_metric])):
                            
                            multiple_key = f"{top_metric}-to-{bottom_metric}-multiple"
                            asset_data[date][multiple_key] = (
                                asset_data[date][top_metric] / asset_data[date][bottom_metric]
                            )
                        
                        # Calculate smoothed multiples for all window sizes
                        for window in rolling_windows:
                            bottom_window_key = f"{bottom_metric}-{window}d"
                            top_window_key = f"{top_metric}-{window}d"
                            
                            if (top_window_key in asset_data[date] and
                                bottom_window_key in asset_data[date] and
                                asset_data[date][bottom_window_key] != 0 and
                                asset_data[date][bottom_window_key] is not None and
                                not pd.isna(asset_data[date][bottom_window_key])):
                                
                                multiple_key_window = f"{top_metric}-to-{bottom_metric}-multiple-{window}d"
                                asset_data[date][multiple_key_window] = (
                                    asset_data[date][top_window_key] / asset_data[date][bottom_window_key]
                                )
            
            # Replace outliers in the rolling-window multiples with the prior good value
            replaced = replace_multiple_outliers(asset_data, outlier_factor, outlier_window)
            if replaced:
                total = sum(len(v) for v in replaced.values())
                print(f"    Replaced {total} outlier multiple values for {asset_name} "
                      f"(>{outlier_factor}x from trailing {outlier_window}d median) across {len(replaced)} multiples")
                for key, days in replaced.items():
                    for d in days:
                        asset_data[d].setdefault('outliers-replaced', []).append(key)

            # Store asset data in category
            category_data[category_name][asset_name] = asset_data
    
    # Restructure data by date
    output_data = {}
    for date in sorted(all_dates):
        output_data[date] = {}
        for category_name in category_data:
            output_data[date][category_name] = {}
            for asset_name in category_data[category_name]:
                if date in category_data[category_name][asset_name]:
                    output_data[date][category_name][asset_name] = category_data[category_name][asset_name][date]
    
    # Create directory if it doesn't exist
    ensure_directory(os.path.dirname(output_filepath))

    # Save to JSON
    with open(output_filepath, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nData saved to {output_filepath}")

def weighted_average_multiple(valid_assets, outlier_factor=100):
    """Top-metric-weighted average of per-protocol multiples, with cross-sectional outliers removed.

    ``valid_assets`` is a list of {'slug', 'top_value' (FDV or MCAP), 'multiple'} for one
    category on one date. Entries with a missing or non-positive top value or multiple are
    ignored. When at least three entries remain and ``outlier_factor`` is set, any multiple
    more than ``outlier_factor`` times above or below the median multiple of the group is
    excluded before weighting. The default of 100 keeps any genuine dispersion within a DeFi
    category and only drops values produced by data breaks, e.g. a protocol whose fee
    adapter has gone dark and whose FDV / near-zero fees multiple would otherwise dominate
    an FDV-weighted average. It backs up the per-protocol rule in process_asset_metrics_data.
    Pass ``outlier_factor=None`` to disable the filter.

    Returns:
        (weighted_multiple or None, list of excluded slugs)
    """
    clean = [
        a for a in valid_assets
        if a.get('top_value') is not None and a.get('multiple') is not None
        and not pd.isna(a['top_value']) and not pd.isna(a['multiple'])
        and a['top_value'] > 0 and a['multiple'] > 0
    ]
    excluded = []
    if outlier_factor and len(clean) >= 3:
        median = float(np.median([a['multiple'] for a in clean]))
        lower, upper = median / outlier_factor, median * outlier_factor
        excluded = sorted(a['slug'] for a in clean if not (lower <= a['multiple'] <= upper))
        clean = [a for a in clean if lower <= a['multiple'] <= upper]

    total_top = sum(a['top_value'] for a in clean)
    if not clean or total_top <= 0:
        return None, excluded
    return sum((a['top_value'] / total_top) * a['multiple'] for a in clean), excluded


def calculate_weighted_multiples(input_filepath, categories, breakout_assets, start_date, output_filepath, outlier_factor=100):
    """Aggregate per-protocol multiples into top-metric-weighted category multiples.

    Outliers are removed per date with weighted_average_multiple(); protocols excluded that
    way are recorded under ``outliers-excluded`` -> {weighted_key: [slugs]} next to
    ``protocols-included`` and ``weighted-averages`` in the output.
    """
    # Load the data
    with open(input_filepath, 'r') as f:
        data = json.load(f)
    
    # Define metrics
    top_metrics = ['FDV', 'MCAP']
    annualized_bottom_metrics = ['fees', 'revenue', 'holders-revenue']
    non_annualized_bottom_metrics = ['TVL']  # Add TVL here
    rolling_windows = [7, 30, 90]
    
    output_data = {}
    
    # Process each date
    for date in sorted(data.keys()):
        if date < start_date:
            continue
        
        output_data[date] = {}
        
        # Process each category
        for category_name, assets in categories.items():
            if category_name not in data[date]:
                continue
            
            output_data[date][category_name] = {}
            category_date_data = data[date][category_name]
            
            # Get breakout asset for this category
            breakout_asset = breakout_assets.get(category_name, None)
            
            # Determine which assets to include
            if breakout_asset:
                candidate_slugs = [slug for slug in assets.values() if slug != breakout_asset]
            else:
                candidate_slugs = list(assets.values())
            
            # Store the list of protocols included in this category
            protocols_included = sorted([slug for slug in candidate_slugs if slug in category_date_data])
            output_data[date][category_name]['protocols-included'] = protocols_included
            
            # Calculate weighted multiples for each combination
            weighted_data = {}
            excluded_data = {}
            
            for top_metric in top_metrics:
                # Process annualized bottom metrics (fees, revenue, holders-revenue)
                for bottom_metric in annualized_bottom_metrics:
                    # Calculate instant weighted multiple
                    valid_assets = []
                    
                    for slug in candidate_slugs:
                        if slug not in category_date_data:
                            continue
                        
                        asset_data = category_date_data[slug]
                        top_col = top_metric
                        mult_col = f"{top_metric}-to-annualized-{bottom_metric}-multiple"
                        
                        if top_col in asset_data and mult_col in asset_data:
                            valid_assets.append({
                                'slug': slug,
                                'top_value': asset_data[top_col],
                                'multiple': asset_data[mult_col]
                            })
                    
                    # Calculate instant weighted average
                    weighted_multiple, excluded = weighted_average_multiple(valid_assets, outlier_factor)
                    if weighted_multiple is not None:
                        weighted_key = f"{top_metric}-weighted-{top_metric}-to-annualized-{bottom_metric}-multiple"
                        weighted_data[weighted_key] = weighted_multiple
                        if excluded:
                            excluded_data[weighted_key] = excluded
                    
                    # Calculate smoothed weighted multiples for all rolling windows
                    for window in rolling_windows:
                        valid_assets_window = []
                        
                        for slug in candidate_slugs:
                            if slug not in category_date_data:
                                continue
                            
                            asset_data = category_date_data[slug]
                            top_col = top_metric
                            mult_col = f"{top_metric}-to-annualized-{bottom_metric}-multiple-{window}d"
                            
                            if top_col in asset_data and mult_col in asset_data:
                                valid_assets_window.append({
                                    'slug': slug,
                                    'top_value': asset_data[top_col],
                                    'multiple': asset_data[mult_col]
                                })
                        
                        # Calculate smoothed weighted average
                        weighted_multiple, excluded = weighted_average_multiple(valid_assets_window, outlier_factor)
                        if weighted_multiple is not None:
                            weighted_key = f"{top_metric}-weighted-{top_metric}-to-annualized-{bottom_metric}-multiple-{window}d"
                            weighted_data[weighted_key] = weighted_multiple
                            if excluded:
                                excluded_data[weighted_key] = excluded
                
                # Process non-annualized bottom metrics (tvl)
                for bottom_metric in non_annualized_bottom_metrics:
                    # Calculate instant weighted multiple
                    valid_assets = []
                    
                    for slug in candidate_slugs:
                        if slug not in category_date_data:
                            continue
                        
                        asset_data = category_date_data[slug]
                        top_col = top_metric
                        mult_col = f"{top_metric}-to-{bottom_metric}-multiple"
                        
                        if top_col in asset_data and mult_col in asset_data:
                            valid_assets.append({
                                'slug': slug,
                                'top_value': asset_data[top_col],
                                'multiple': asset_data[mult_col]
                            })
                    
                    # Calculate instant weighted average
                    weighted_multiple, excluded = weighted_average_multiple(valid_assets, outlier_factor)
                    if weighted_multiple is not None:
                        weighted_key = f"{top_metric}-weighted-{top_metric}-to-{bottom_metric}-multiple"
                        weighted_data[weighted_key] = weighted_multiple
                        if excluded:
                            excluded_data[weighted_key] = excluded
                    
                    # Calculate smoothed weighted multiples for all rolling windows
                    for window in rolling_windows:
                        valid_assets_window = []
                        
                        for slug in candidate_slugs:
                            if slug not in category_date_data:
                                continue
                            
                            asset_data = category_date_data[slug]
                            top_col = top_metric
                            mult_col = f"{top_metric}-to-{bottom_metric}-multiple-{window}d"
                            
                            if top_col in asset_data and mult_col in asset_data:
                                valid_assets_window.append({
                                    'slug': slug,
                                    'top_value': asset_data[top_col],
                                    'multiple': asset_data[mult_col]
                                })
                        
                        # Calculate smoothed weighted average
                        weighted_multiple, excluded = weighted_average_multiple(valid_assets_window, outlier_factor)
                        if weighted_multiple is not None:
                            weighted_key = f"{top_metric}-weighted-{top_metric}-to-{bottom_metric}-multiple-{window}d"
                            weighted_data[weighted_key] = weighted_multiple
                            if excluded:
                                excluded_data[weighted_key] = excluded
            
            output_data[date][category_name]['weighted-averages'] = weighted_data
            output_data[date][category_name]['outliers-excluded'] = excluded_data
    
    # Create directory if it doesn't exist
    ensure_directory(os.path.dirname(output_filepath))

    # Save to JSON
    with open(output_filepath, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Weighted multiples data saved to {output_filepath}")


# ---------------------------------------------------------------------------
# Universe earnings: every DefiLlama protocol, rolled up to parents
# ---------------------------------------------------------------------------
UNIVERSE_EARNINGS_PATH = 'data/processed/earnings/all-protocols-earnings.json'
# Trailing windows DefiLlama reports as totals in the /overview/fees entries and the Pro
# /emissionsBreakdownAggregated snapshot: days -> (overview key, emissions key). These are the
# only windows the universe can be built on without a daily series per protocol (the overviews'
# daily breakdown omits chains and ~750 other protocols, and incentives have no universe-level
# daily series), so build_universe_earnings stores all of them and load_universe_earnings picks one.
UNIVERSE_WINDOWS = {1: ('total24h', 'emission24h'), 7: ('total7d', 'emission7d'),
                    30: ('total30d', 'emission30d'), 365: ('total1y', 'emissions1y')}
UNIVERSE_EARNINGS_RULES = (
    "One row per parent protocol (children rolled up via parentProtocol; a protocol without a parent "
    "is its own row). fees / revenue / holders_rev = sum of the children's DefiLlama overview totals "
    "for each trailing window (total24h, total7d, total30d, total1y -> the 1d, 7d, 30d, 365d fields). "
    "incentives = sum of the emissions snapshot's matching window (emission24h, emission7d, "
    "emission30d, emissions1y) for every emissions entry whose defillamaId or linked ids belong to the "
    "group, else whose name matches the group's display name (chains); no entry -> 0, DefiLlama's "
    "convention. Values are stored as reported; load_universe_earnings(window_days) derives "
    "earnings = revenue - incentives, annualized = value x 365 / window_days, earnings_yield = "
    "annualized earnings / mcap and earnings_minus_holders_rev (annualized). mcap = the largest mcap "
    "reported by any child in /protocols (children share one token), else the token mcap in the "
    "/emissions list (matched on protocolSlug / protocolId); None when neither has one (Tether, "
    "Circle, most chains), so those rows have no earnings yield. Negative earnings are kept. "
    "Concentration: revenue_max_day_share_{d}d / holders_rev_max_day_share_{d}d = the group's largest "
    "single day as a share of its total over the last d complete UTC days (7, 30, 365), from the "
    "overviews' daily breakdown (children summed to the parent); the day is recorded alongside. A "
    "ranking that annualizes a window multiplies one day by ~365/d, so a share above ~50% of a 30d "
    "window means one day drives the figure (regular weekly payers top out near 33%). The 1d window "
    "has no concentration measure (the day is the window)."
)


def _max_day_share(breakdown, name_group, days=30):
    """{group: (share, 'YYYY-MM-DD')} of the largest single day in the last ``days`` complete UTC
    days of an overview's totalDataChartBreakdown ([ts, {protocol display name: value}]), with
    children summed into their parent group via ``name_group``."""
    today = int(datetime.datetime.now(datetime.timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
    window = [p for p in (breakdown or []) if isinstance(p, list) and p[0] < today][-days:]
    daily = {}
    for ts, values in window:
        for name, v in (values or {}).items():
            g = name_group.get(name)
            if g is None or not isinstance(v, (int, float)):
                continue
            daily.setdefault(g, {})
            daily[g][ts] = daily[g].get(ts, 0.0) + v
    out = {}
    for g, ser in daily.items():
        total = sum(ser.values())
        if total <= 0:
            continue
        ts, mx = max(ser.items(), key=lambda kv: kv[1])
        out[g] = (mx / total, datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime('%Y-%m-%d'))
    return out


def _group_key(entry):
    """Parent slug for an overview / protocols-list entry ('parent#aave' -> 'aave'), else its own slug."""
    parent = entry.get('parentProtocol')
    if isinstance(parent, str) and parent.startswith('parent#'):
        return parent.split('#', 1)[1]
    return entry.get('slug')


def _derive_universe_window(raw, window_days):
    """The analysis-ready universe for one trailing window, from the raw rows build_universe_earnings
    writes (a DataFrame indexed by parent slug holding every window's totals as reported).

    Columns: name, category, protocol_type, n_children, mcap, mcap_source, incentives_status,
    window_days, fees, revenue, holders_rev, incentives, earnings (window totals; earnings =
    revenue - incentives), ann_fees, ann_revenue, ann_holders_rev, ann_earnings (x 365 /
    window_days), earnings_yield (ann_earnings / mcap; None without mcap),
    earnings_minus_holders_rev (ann_earnings - ann_holders_rev; negative = paying token holders
    more than the protocol earns), revenue_max_day_share / revenue_max_day and
    holders_rev_max_day_share / holders_rev_max_day (largest single day's share of the window
    and that day; None for the 1d window or when the group had no revenue in it). Flag a row as
    one-day-driven with e.g. ``df.revenue_max_day_share >= 0.5``.
    """
    if window_days not in UNIVERSE_WINDOWS:
        raise ValueError(f"window_days must be one of {sorted(UNIVERSE_WINDOWS)} (DefiLlama's trailing windows), got {window_days!r}")
    d = window_days
    df = raw[['name', 'category', 'protocol_type', 'n_children', 'mcap', 'mcap_source', 'incentives_status']].copy()
    df['window_days'] = d
    for k in ('fees', 'revenue', 'holders_rev', 'incentives'):
        df[k] = raw[f'{k}_{d}d'].astype(float)
    df['earnings'] = df.revenue - df.incentives
    for k in ('fees', 'revenue', 'holders_rev', 'earnings'):
        df[f'ann_{k}'] = df[k] * 365.0 / d
    df['earnings_yield'] = df.ann_earnings / df.mcap.where(df.mcap > 0)
    df['earnings_minus_holders_rev'] = df.ann_earnings - df.ann_holders_rev
    for k in ('revenue_max_day_share', 'revenue_max_day', 'holders_rev_max_day_share', 'holders_rev_max_day'):
        if f'{k}_{d}d' in raw.columns:
            df[k] = raw[f'{k}_{d}d']
        else:                                   # 1d window: no concentration measure
            df[k] = float('nan') if k.endswith('_share') else None
    return df


@report_errors(default=pd.DataFrame())
def build_universe_earnings(defillama_output_dir='data/raw/defillama', output_path=UNIVERSE_EARNINGS_PATH):
    """Fees, revenue, holders revenue, incentives and market cap for EVERY DefiLlama protocol over
    each of DefiLlama's trailing windows (UNIVERSE_WINDOWS: 1, 7, 30, 365 days), from the universe
    snapshots already on disk: ``fees/``, ``revenue/`` and ``holders-revenue/all-protocols.json``
    (the /overview/fees payloads), ``incentives/all-protocols.json`` (Pro
    /emissionsBreakdownAggregated, see defillama.pull_emissions_breakdown) and
    ``tvl/all-protocols.json`` (the /protocols list, for mcap and the parent map). Rules:
    UNIVERSE_EARNINGS_RULES.

    Writes ``output_path`` as {"asof", "windows", "rules", "protocols": [row, ...]} where each row
    carries the window totals as reported (``fees_30d``, ``revenue_7d``, ``incentives_365d``, ...)
    and the per-window concentration fields, with no derived values; pick a window with
    load_universe_earnings(window_days). Returns the 30-day view (see _derive_universe_window).
    """
    def load(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)

    overviews = {k: load(f"{defillama_output_dir}/{name}/all-protocols.json")
                 for k, name in (('fees', 'fees'), ('revenue', 'revenue'), ('holders_rev', 'holders-revenue'))}
    protos = load(f"{defillama_output_dir}/tvl/all-protocols.json")
    inc_path = f"{defillama_output_dir}/incentives/all-protocols.json"
    if os.path.exists(inc_path):
        emissions = load(inc_path).get('protocols') or []
    else:
        emissions = []
        print(f"  {inc_path} missing (Pro pull): incentives treated as 0 for every protocol")
    tokens_path = f"{defillama_output_dir}/incentives/all-emissions.json"
    token_mcap = {}
    if os.path.exists(tokens_path):
        for t in load(tokens_path) or []:
            if isinstance(t.get('mcap'), (int, float)) and t['mcap'] > 0:
                slug = t.get('protocolSlug') or str(t.get('protocolId') or '').replace('parent#', '')
                if slug:
                    token_mcap[slug] = max(token_mcap.get(slug, 0), t['mcap'])

    entries = [e for ov in overviews.values() for e in ov['protocols']]
    # id -> group (parent slug) from the /protocols list; the overview entries carry the same ids
    id_group = {str(p['id']): _group_key(p) for p in protos if p.get('id')}
    for e in entries:
        id_group.setdefault(str(e.get('defillamaId')), _group_key(e))

    rows = {}

    def row(group, entry):
        r = rows.get(group)
        if r is None:
            r = rows[group] = dict(slug=group, name=None, category=entry.get('category'),
                                   protocol_type=entry.get('protocolType', 'protocol'), n_children=0,
                                   mcap=None, mcap_source=None, incentives_status='none', children=set())
            for k in ('fees', 'revenue', 'holders_rev', 'incentives'):
                for d in UNIVERSE_WINDOWS:
                    r[f'{k}_{d}d'] = 0.0
        return r

    for k, ov in overviews.items():
        for e in ov['protocols']:
            r = row(_group_key(e), e)
            for d, (total_key, _) in UNIVERSE_WINDOWS.items():
                r[f'{k}_{d}d'] += float(e.get(total_key) or 0)
            r['children'].add(e.get('slug'))
            if e.get('parentProtocol') is None:
                r['name'] = r['name'] or e.get('displayName') or e.get('name')
            elif not r['name']:
                r['name'] = (e.get('linkedProtocols') or [None])[0]

    # mcap: largest child-reported mcap in the group (one token per group)
    for p in protos:
        g = _group_key(p)
        if g in rows and isinstance(p.get('mcap'), (int, float)) and p['mcap'] > 0:
            rows[g]['mcap'] = max(rows[g]['mcap'] or 0, p['mcap'])
            rows[g]['mcap_source'] = 'protocols'
            if not rows[g]['name'] and p.get('parentProtocol') is None:
                rows[g]['name'] = p.get('name')
    for g, r in rows.items():
        if r['mcap'] is None and g in token_mcap:
            r['mcap'], r['mcap_source'] = token_mcap[g], 'emissions-list'

    # incentives: attach each emissions entry to the first group its ids resolve to
    # chains carry 'chain#<slug>' ids in the overviews but numeric / blank ids in the emissions
    # snapshot, so fall back to the display name when no id resolves
    name_group = {str(r['name']).lower(): g for g, r in rows.items() if r.get('name')}
    unmatched = []
    for x in emissions:
        ids = [x.get('defillamaId')] + list(x.get('linked') or [])
        g = next((id_group[str(i)] for i in ids if str(i) in id_group), None)
        if g is None:
            g = name_group.get(str(x.get('name') or '').lower())
        if g is None or g not in rows:
            if (x.get('emission30d') or 0) > 0:        # only worth flagging when it carries incentives
                unmatched.append(x.get('name'))
            continue
        for d, (_, emission_key) in UNIVERSE_WINDOWS.items():
            rows[g][f'incentives_{d}d'] += float(x.get(emission_key) or 0)
        rows[g]['incentives_status'] = 'tracked'

    # single-day concentration of each window (7d and longer), per parent, from the daily breakdowns
    name_group = {}
    for e in entries:
        for key in ('displayName', 'name'):
            if e.get(key):
                name_group.setdefault(e[key], _group_key(e))
    conc = {}
    for k, ov in (('revenue', overviews['revenue']), ('holders_rev', overviews['holders_rev'])):
        for d in UNIVERSE_WINDOWS:
            if d > 1:
                conc[(k, d)] = _max_day_share(ov.get('totalDataChartBreakdown'), name_group, days=d)

    out = []
    for g, r in rows.items():
        r['n_children'] = len(r.pop('children'))
        r['name'] = r['name'] or g.replace('-', ' ').title()
        for (k, d), shares in conc.items():
            r[f'{k}_max_day_share_{d}d'], r[f'{k}_max_day_{d}d'] = shares.get(g, (None, None))
        out.append(r)
    out.sort(key=lambda r: r['slug'])

    ensure_directory(os.path.dirname(output_path))
    with open(output_path, 'w') as f:
        json.dump({'asof': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d'),
                   'windows': sorted(UNIVERSE_WINDOWS), 'rules': UNIVERSE_EARNINGS_RULES,
                   'protocols': out}, f, indent=1)
    raw = pd.DataFrame(out).set_index('slug')
    tracked = int((raw.incentives_status == 'tracked').sum())
    print(f"Universe earnings: {len(raw)} parent protocols ({tracked} with tracked incentives, "
          f"{int(raw.mcap.notna().sum())} with mcap), windows {sorted(UNIVERSE_WINDOWS)}d -> {output_path}")
    if unmatched:
        print(f"  {len(unmatched)} emissions entries with incentives matched no fee-tracked protocol and were ignored: "
              f"{unmatched[:8]}{' ...' if len(unmatched) > 8 else ''}")
    return _derive_universe_window(raw, 30)


def load_universe_earnings(path=UNIVERSE_EARNINGS_PATH, window_days=30):
    """The universe written by build_universe_earnings for one trailing window (1, 7, 30 or 365
    days; see _derive_universe_window for the columns), indexed by parent slug, plus the file's
    'asof' date."""
    with open(path, encoding='utf-8') as f:
        payload = json.load(f)
    raw = pd.DataFrame(payload['protocols']).set_index('slug')
    return _derive_universe_window(raw, window_days), payload.get('asof')
