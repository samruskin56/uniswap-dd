"""
Protocol Index Module
Centralized management of protocol and equity name mappings across different API sources.
Supports both crypto protocols (DefiLlama, Messari) and public equities (Yahoo Finance).
"""

import json
import os
from typing import Dict, List, Optional, Any
from utils.helper_functions import report_errors

# Repo root (src/data/protocol_index.py -> repo root)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Default path to the protocol index file
DEFAULT_INDEX_PATH = os.path.join(_REPO_ROOT, 'config', 'protocol_index.json')

# Default paths to the cached "all-*" files the populate helper reads from.
# These are produced by the normal data pulls (messari.pull_all_assets and the
# Blockworks project list), so the helper can run offline against fresh data.
DEFAULT_MESSARI_ALL_ASSETS_PATH = os.path.join(
    _REPO_ROOT, 'data', 'raw', 'messari', 'all-assets.json'
)
DEFAULT_BLOCKWORKS_ALL_PROJECTS_PATH = os.path.join(
    _REPO_ROOT, 'data', 'raw', 'blockworks', 'all-projects.json'
)
# Produced by data_processing.establish_parent_protocols; maps each DefiLlama
# parent slug to its own display name, category, totals and sub-protocols.
DEFAULT_PARENT_PROTOCOLS_PATH = os.path.join(
    _REPO_ROOT, 'data', 'processed', 'parent_protocols.json'
)

# Canonical key order within a single protocol entry, matching the existing
# config/protocol_index.json style: blockworks_slug sits between messari_slug
# and category, and is only written once it has a value.
_ENTRY_KEY_ORDER = ['display_name', 'defillama_slug', 'messari_slug', 'blockworks_slug', 'category']


def _ordered_entry(entry: Dict) -> Dict:
    """Return a protocol entry with keys in the canonical index order."""
    ordered = {k: entry[k] for k in _ENTRY_KEY_ORDER if k in entry}
    # Preserve any unexpected extra keys rather than dropping them.
    for k, v in entry.items():
        if k not in ordered:
            ordered[k] = v
    return ordered


def load_index(index_path: str = None) -> Dict:
    """Load the protocol index from JSON file."""
    path = index_path or DEFAULT_INDEX_PATH
    with open(path, 'r') as f:
        return json.load(f)


def save_index(data: Dict, index_path: str = None) -> None:
    """Save the protocol index to JSON file."""
    path = index_path or DEFAULT_INDEX_PATH
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def get_protocol(standard_name: str, index_path: str = None) -> Optional[Dict]:
    """
    Get full protocol info by standardized name.

    Args:
        standard_name: The standardized protocol name (e.g., 'ether-fi')
        index_path: Optional path to index file

    Returns:
        Dict with display_name, defillama_slug, messari_slug, category
        or None if not found
    """
    index = load_index(index_path)
    return index.get('protocols', {}).get(standard_name)


def get_defillama_slug(standard_name: str, index_path: str = None) -> Optional[str]:
    """
    Get the DefiLlama API slug for a standardized protocol name.

    Args:
        standard_name: The standardized protocol name (e.g., 'ether-fi')

    Returns:
        DefiLlama slug (e.g., 'ether.fi') or None if not found
    """
    protocol = get_protocol(standard_name, index_path)
    return protocol.get('defillama_slug') if protocol else None


def get_messari_slug(standard_name: str, index_path: str = None) -> Optional[str]:
    """
    Get the Messari API slug for a standardized protocol name.

    Args:
        standard_name: The standardized protocol name (e.g., 'ether-fi')

    Returns:
        Messari slug (e.g., 'ether-fi') or None if not found
    """
    protocol = get_protocol(standard_name, index_path)
    return protocol.get('messari_slug') if protocol else None


def get_blockworks_slug(standard_name: str, index_path: str = None) -> Optional[str]:
    """
    Get the Blockworks API slug for a standardized protocol name.

    Args:
        standard_name: The standardized protocol name (e.g., 'solana')

    Returns:
        Blockworks slug (e.g., 'solana') or None if not found
    """
    protocol = get_protocol(standard_name, index_path)
    return protocol.get('blockworks_slug') if protocol else None


def get_display_name(standard_name: str, index_path: str = None) -> Optional[str]:
    """
    Get the human-readable display name for a protocol.

    Args:
        standard_name: The standardized protocol name (e.g., 'ether-fi')

    Returns:
        Display name (e.g., 'EtherFi') or None if not found
    """
    protocol = get_protocol(standard_name, index_path)
    return protocol.get('display_name') if protocol else None


def get_protocols_by_category(category: str, index_path: str = None) -> List[str]:
    """
    Get all standardized protocol names in a category.

    Args:
        category: Category name (e.g., 'liquid_staking')

    Returns:
        List of standardized protocol names
    """
    index = load_index(index_path)
    protocols = index.get('protocols', {})
    return [
        name for name, info in protocols.items()
        if info.get('category') == category
    ]


def get_defillama_assets_dict(
    standard_names: List[str] = None,
    category: str = None,
    index_path: str = None
) -> Dict[str, str]:
    """
    Build assets dict for DefiLlama API calls.
    Format: {defillama_slug: standard_name}

    Args:
        standard_names: List of standardized protocol names. If None, uses category.
        category: Category to get all protocols from. Ignored if standard_names provided.

    Returns:
        Dict mapping defillama_slug -> standard_name

    Example:
        >>> get_defillama_assets_dict(category='liquid_staking')
        {'ether.fi': 'ether-fi', 'lido': 'lido', ...}
    """
    if standard_names is None and category is None:
        raise ValueError("Must provide either standard_names or category")

    if standard_names is None:
        standard_names = get_protocols_by_category(category, index_path)

    result = {}
    for name in standard_names:
        slug = get_defillama_slug(name, index_path)
        if slug:
            result[slug] = name
    return result


def get_messari_assets_dict(
    standard_names: List[str] = None,
    category: str = None,
    index_path: str = None
) -> Dict[str, str]:
    """
    Build assets dict for Messari API calls.
    Format: {messari_slug: standard_name}

    Args:
        standard_names: List of standardized protocol names. If None, uses category.
        category: Category to get all protocols from. Ignored if standard_names provided.

    Returns:
        Dict mapping messari_slug -> standard_name

    Example:
        >>> get_messari_assets_dict(category='liquid_staking')
        {'ether-fi': 'ether-fi', 'lido-dao': 'lido', ...}
    """
    if standard_names is None and category is None:
        raise ValueError("Must provide either standard_names or category")

    if standard_names is None:
        standard_names = get_protocols_by_category(category, index_path)

    result = {}
    for name in standard_names:
        slug = get_messari_slug(name, index_path)
        if slug:  # Skip protocols without messari_slug
            result[slug] = name
    return result


def get_blockworks_assets_dict(
    standard_names: List[str] = None,
    category: str = None,
    index_path: str = None
) -> Dict[str, str]:
    """
    Build assets dict for Blockworks API calls.
    Format: {blockworks_slug: standard_name}

    Args:
        standard_names: List of standardized protocol names. If None, uses category.
        category: Category to get all protocols from. Ignored if standard_names provided.

    Returns:
        Dict mapping blockworks_slug -> standard_name

    Example:
        >>> get_blockworks_assets_dict(category='smart_contract_chains')
        {'solana': 'solana', 'ethereum': 'ethereum', ...}
    """
    if standard_names is None and category is None:
        raise ValueError("Must provide either standard_names or category")

    if standard_names is None:
        standard_names = get_protocols_by_category(category, index_path)

    result = {}
    for name in standard_names:
        slug = get_blockworks_slug(name, index_path)
        if slug:  # Skip protocols without blockworks_slug
            result[slug] = name
    return result


def get_all_categories(index_path: str = None) -> List[str]:
    """Get list of all unique categories in the index."""
    index = load_index(index_path)
    protocols = index.get('protocols', {})
    categories = set(info.get('category') for info in protocols.values() if info.get('category'))
    return sorted(list(categories))


def get_all_protocols(index_path: str = None) -> List[str]:
    """Get list of all standardized protocol names in the index."""
    index = load_index(index_path)
    return sorted(list(index.get('protocols', {}).keys()))


def add_protocol(
    standard_name: str,
    display_name: str,
    defillama_slug: str,
    messari_slug: Optional[str],
    category: str,
    blockworks_slug: Optional[str] = None,
    overwrite: bool = True,
    index_path: str = None
) -> bool:
    """
    Add a single protocol to the index.

    Args:
        standard_name: The standardized name to use as the key
        display_name: Human-readable name
        defillama_slug: Slug used by DefiLlama API
        messari_slug: Slug used by Messari API (can be None)
        category: Category for the protocol
        blockworks_slug: Slug used by Blockworks API (optional; only written
            to the entry when not None, matching the existing index style)
        overwrite: When False, leave an already-present protocol untouched
            (preserves manual edits). Defaults to True for backwards
            compatibility.

    Returns:
        True if the entry was written, False if it was skipped because it
        already existed and overwrite was False.
    """
    index = load_index(index_path)

    if 'protocols' not in index:
        index['protocols'] = {}

    if standard_name in index['protocols'] and not overwrite:
        print(f"Protocol already present, skipping: {standard_name}")
        return False

    entry = {
        'display_name': display_name,
        'defillama_slug': defillama_slug,
        'messari_slug': messari_slug,
        'category': category,
    }
    if blockworks_slug is not None:
        entry['blockworks_slug'] = blockworks_slug

    index['protocols'][standard_name] = _ordered_entry(entry)

    save_index(index, index_path)
    print(f"Added protocol: {standard_name}")
    return True


def add_defillama_protocols(
    protocols: Any,
    category: str,
    overwrite: bool = False,
    index_path: str = None,
) -> Dict[str, List[str]]:
    """
    Add DefiLlama protocols to the index if they are not already present.

    Designed to be fed straight from data_processing.top_n_protocols_by_category_by_metric,
    whose per-category output is a {defillama_slug: display_name} mapping. Only
    the protocols passed in are added -- there is no scan of the full DefiLlama
    protocol universe. The slug becomes the standardized-name key, the display
    name is taken from the mapping, and messari_slug is initialized to None so it
    can be filled later by populate_remaining_slugs().

    Non-destructive by design: a slug already in the index (as a key OR as an
    existing defillama_slug value) is skipped, so manual edits to
    protocol_index.json are never clobbered. Pass overwrite=True only if you
    explicitly want existing entries rewritten.

    Args:
        protocols: Either a {defillama_slug: display_name} dict (as produced by
            top_n_protocols_by_category_by_metric for a single category) or a
            list/iterable of slugs (the display name then defaults to the slug).
        category: Standardized category key to assign to all added protocols
            (e.g. 'physical_tcg').
        overwrite: When True, rewrite entries that already exist. Defaults to
            False (preserve existing/manual entries).
        index_path: Optional path to the index file.

    Returns:
        Summary dict: {'added': [...], 'skipped': [...]}.
    """
    items = list(protocols.items()) if isinstance(protocols, dict) else [(s, s) for s in protocols]

    index = load_index(index_path)
    index_protocols = index.setdefault('protocols', {})
    existing_slugs = {info.get('defillama_slug') for info in index_protocols.values()}

    added, skipped = [], []

    for slug, display_name in items:
        if not overwrite and (slug in index_protocols or slug in existing_slugs):
            skipped.append(slug)
            continue

        index_protocols[slug] = _ordered_entry({
            'display_name': display_name or slug,
            'defillama_slug': slug,
            'messari_slug': None,
            'category': category,
        })
        existing_slugs.add(slug)
        added.append(slug)

    save_index(index, index_path)

    print(f"add_defillama_protocols: added {len(added)}, skipped {len(skipped)}")
    if added:
        print(f"  added:   {added}")
    if skipped:
        print(f"  skipped: {skipped} (already in index)")

    return {'added': added, 'skipped': skipped}


def repair_parent_display_names(
    category: str = None,
    standard_names: List[str] = None,
    dry_run: bool = False,
    parent_protocols_path: str = None,
    index_path: str = None,
) -> Dict[str, List]:
    """
    Replace a display_name that is really the name of one of the parent's
    sub-protocols with the DefiLlama parent's own name.

    Before parent_protocols.json recorded parent names, add_defillama_protocols
    was fed the largest child's name, so e.g. index key `uniswap` (DefiLlama
    parent slug `uniswap`, aggregating V1-V4) was stored as "Uniswap V3". This
    fixes those entries in place.

    Deliberately narrow, to respect hand edits in protocol_index.json:
      - only entries whose defillama_slug is a parent in parent_protocols.json
        with a recorded `name`;
      - only when the current display_name is exactly one of that parent's
        sub-protocol names -- a name that is not a child's is treated as a
        manual choice and kept (e.g. "Compound" for compound-finance);
      - only within `category` / `standard_names` when given. Pass the
        category that add_defillama_protocols just synced; a child name can be
        a legitimate manual choice elsewhere ("Maple" is a child of Maple Finance).

    Args:
        category: Restrict to entries with this standardized category.
        standard_names: Restrict to these index keys.
        dry_run: When True, report what would change without writing.
        parent_protocols_path: Path to parent_protocols.json.
        index_path: Optional path to the index file.

    Returns:
        {'changed': [(standard_name, old_display_name, new_display_name)],
         'skipped': [(standard_name, reason)]}
    """
    parents_path = parent_protocols_path or DEFAULT_PARENT_PROTOCOLS_PATH
    if not os.path.exists(parents_path):
        print(f"repair_parent_display_names: {parents_path} not found; "
              f"run data_processing.establish_parent_protocols first. Nothing changed.")
        return {'changed': [], 'skipped': []}
    with open(parents_path, 'r') as f:
        parents = json.load(f)

    index = load_index(index_path)
    protocols = index.setdefault('protocols', {})

    changed, skipped = [], []
    for std, info in protocols.items():
        if standard_names is not None and std not in standard_names:
            continue
        if category is not None and info.get('category') != category:
            continue

        slug = info.get('defillama_slug')
        parent = parents.get(slug) if slug else None
        parent_name = (parent or {}).get('name')
        if not parent_name:
            skipped.append((std, 'not a DefiLlama parent with a recorded name'))
            continue

        current = info.get('display_name')
        if current == parent_name:
            skipped.append((std, 'already correct'))
            continue

        child_names = {s.get('name') for s in parent.get('sub_protocols', [])}
        if current not in child_names:
            skipped.append((std, f"manual name kept ('{current}' is not a sub-protocol of '{parent_name}')"))
            continue

        changed.append((std, current, parent_name))
        if not dry_run:
            info['display_name'] = parent_name
            protocols[std] = _ordered_entry(info)

    if changed and not dry_run:
        save_index(index, index_path)

    verb = 'would change' if dry_run else 'changed'
    print(f"repair_parent_display_names{' (dry run)' if dry_run else ''}: "
          f"{verb} {len(changed)}, kept {len(skipped)}")
    for std, old, new in changed:
        print(f"  {std}: '{old}' -> '{new}'")
    for std, reason in skipped:
        if reason.startswith('manual name kept'):
            print(f"  {std}: {reason}")

    return {'changed': changed, 'skipped': skipped}


def _load_messari_slugs(messari_assets_path: str) -> set:
    """Load the set of valid Messari slugs from the cached all-assets file."""
    with open(messari_assets_path, 'r') as f:
        data = json.load(f)
    assets = data.get('data', data) if isinstance(data, dict) else data
    return {a.get('slug') for a in assets if a.get('slug')}


@report_errors()
def _load_blockworks_slugs(api_key: str = None, cache_path: str = None, timeout: int = 60) -> Optional[set]:
    """Return the set of valid Blockworks project slugs, or None if unavailable.

    Reads the cached all-projects file if present; otherwise fetches the full
    project list live from /v1/projects (requires api_key) and caches it to
    disk, mirroring messari.pull_all_assets. A single metric is unreliable for
    detecting whether a project exists (different projects expose different
    metrics), so membership in the project list is used instead. Imports
    requests lazily so the rest of this module stays free of a hard network
    dependency.
    """
    path = cache_path or DEFAULT_BLOCKWORKS_ALL_PROJECTS_PATH

    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
    elif api_key:
        import requests
        headers = {
            "x-api-key": api_key,
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
        }
        try:
            resp = requests.get(
                'https://api.blockworks.com/v1/projects',
                headers=headers,
                params={'limit': 10000},
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.exceptions.RequestException, ValueError):
            return None
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    else:
        return None

    projects = data.get('data', data) if isinstance(data, dict) else data
    return {p.get('slug') for p in projects if p.get('slug')}


@report_errors(default={})
def populate_remaining_slugs(
    standard_names: List[str] = None,
    sources: List[str] = ('messari', 'blockworks'),
    overwrite: bool = False,
    messari_assets_path: str = None,
    blockworks_api_key: str = None,
    index_path: str = None,
) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Fill in missing source slugs for index entries that have a DefiLlama slug.

    For each protocol, the DefiLlama slug (and, as a fallback, the standardized
    name key) is checked for membership in each requested source's slug list:
      - 'messari':    kept if it appears in the cached Messari all-assets file.
      - 'blockworks': kept if it appears in the Blockworks project list (read
                      from the cached all-projects file, or fetched live and
                      cached when blockworks_api_key is provided).
    A source that yields no match is left as-is (None), never guessed.

    Non-destructive by design: a slug that is already set is left untouched
    unless overwrite=True, so manual edits to protocol_index.json are preserved.

    Args:
        standard_names: Protocols to process. If None, processes every protocol
            that has a defillama_slug.
        sources: Which sources to fill. Subset of {'messari', 'blockworks'}.
        overwrite: When True, re-probe and replace slugs that are already set.
            Defaults to False (only fill missing slugs).
        messari_assets_path: Path to the cached Messari all-assets JSON.
            Defaults to data/raw/messari/all-assets.json.
        blockworks_api_key: API key used to fetch the Blockworks project list
            when no cached all-projects file exists. If omitted and no cache is
            present, Blockworks is skipped.
        index_path: Optional path to the index file.

    Returns:
        Per-protocol dict of what was filled this run, e.g.
        {'courtyard': {'messari_slug': None, 'blockworks_slug': None}, ...}
        (a value of None means probed but no match; absent means not probed).
    """
    sources = list(sources)
    index = load_index(index_path)
    protocols = index.get('protocols', {})

    targets = standard_names if standard_names is not None else [
        name for name, info in protocols.items() if info.get('defillama_slug')
    ]

    messari_slugs = None
    if 'messari' in sources:
        path = messari_assets_path or DEFAULT_MESSARI_ALL_ASSETS_PATH
        if os.path.exists(path):
            messari_slugs = _load_messari_slugs(path)
        else:
            print(f"  Messari all-assets file not found ({path}); run messari.pull_all_assets first. Skipping Messari.")

    blockworks_slugs = None
    if 'blockworks' in sources:
        blockworks_slugs = _load_blockworks_slugs(api_key=blockworks_api_key)
        if blockworks_slugs is None:
            print("  No cached Blockworks project list and no blockworks_api_key; skipping Blockworks.")

    results: Dict[str, Dict[str, Optional[str]]] = {}

    for name in targets:
        info = protocols.get(name)
        if not info or not info.get('defillama_slug'):
            continue

        # Candidate slugs to probe, in priority order, de-duplicated.
        candidates = list(dict.fromkeys([info.get('defillama_slug'), name]))
        result: Dict[str, Optional[str]] = {}

        if 'messari' in sources and messari_slugs is not None:
            if overwrite or not info.get('messari_slug'):
                match = next((c for c in candidates if c in messari_slugs), None)
                if match:
                    info['messari_slug'] = match
                result['messari_slug'] = match

        if 'blockworks' in sources and blockworks_slugs is not None:
            if overwrite or not info.get('blockworks_slug'):
                match = next((c for c in candidates if c in blockworks_slugs), None)
                if match:
                    info['blockworks_slug'] = match
                result['blockworks_slug'] = match

        protocols[name] = _ordered_entry(info)
        if result:
            results[name] = result

    save_index(index, index_path)

    filled_messari = sum(1 for r in results.values() if r.get('messari_slug'))
    filled_bw = sum(1 for r in results.values() if r.get('blockworks_slug'))
    print(f"populate_remaining_slugs: processed {len(results)} protocol(s); "
          f"filled {filled_messari} messari_slug, {filled_bw} blockworks_slug")

    return results


def remove_protocol(standard_name: str, index_path: str = None) -> bool:
    """
    Remove a protocol from the index.

    Returns:
        True if removed, False if not found
    """
    index = load_index(index_path)

    if standard_name in index.get('protocols', {}):
        del index['protocols'][standard_name]
        save_index(index, index_path)
        print(f"Removed protocol: {standard_name}")
        return True
    return False


def get_categories_dict(
    categories: List[str] = None,
    source: str = 'defillama',
    index_path: str = None
) -> Dict[str, Dict[str, str]]:
    """
    Build the full categories dict for API calls, matching the format
    used in example-pull-raw-data.ipynb.

    Args:
        categories: List of categories to include. If None, includes all.
        source: 'defillama', 'messari', or 'blockworks' - determines which slugs to use

    Returns:
        Dict of format:
        {
            'category_name': {api_slug: standard_name, ...},
            ...
        }

    Example:
        >>> get_categories_dict(['liquid_staking'], source='defillama')
        {'liquid_staking': {'ether.fi': 'ether-fi', 'lido': 'lido', ...}}

        >>> get_categories_dict(['liquid_staking'], source='messari')
        {'liquid_staking': {'ether-fi': 'ether-fi', 'lido-dao': 'lido', ...}}

        >>> get_categories_dict(['smart_contract_chains'], source='blockworks')
        {'smart_contract_chains': {'solana': 'solana', 'ethereum': 'ethereum', ...}}
    """
    if categories is None:
        categories = get_all_categories(index_path)

    result = {}
    for cat in categories:
        if source == 'defillama':
            result[cat] = get_defillama_assets_dict(category=cat, index_path=index_path)
        elif source == 'messari':
            result[cat] = get_messari_assets_dict(category=cat, index_path=index_path)
        elif source == 'blockworks':
            result[cat] = get_blockworks_assets_dict(category=cat, index_path=index_path)
        else:
            raise ValueError(f"Unknown source: {source}. Use 'defillama', 'messari', or 'blockworks'")

    return result


def get_standard_names_dict(
    categories: List[str] = None,
    index_path: str = None
) -> Dict[str, Dict[str, str]]:
    """
    Build a categories dict using standardized names only.
    Format: {category: {standard_name: standard_name, ...}}

    This is useful for functions like process_asset_metrics_data that
    expect standardized names as both keys and values.

    Args:
        categories: List of categories to include. If None, includes all.

    Returns:
        Dict of format:
        {
            'category_name': {standard_name: standard_name, ...},
            ...
        }

    Example:
        >>> get_standard_names_dict(['liquid_staking'])
        {'liquid_staking': {'ether-fi': 'ether-fi', 'lido': 'lido', ...}}
    """
    if categories is None:
        categories = get_all_categories(index_path)

    result = {}
    for cat in categories:
        protocols = get_protocols_by_category(cat, index_path)
        result[cat] = {name: name for name in protocols}

    return result


def get_display_names_dict(
    standard_names: List[str] = None,
    category: str = None,
    index_path: str = None
) -> Dict[str, str]:
    """
    Build assets dict with display names for visualization functions.
    Format: {standard_name: display_name}

    Args:
        standard_names: List of standardized protocol names. If None, uses category.
        category: Category to get all protocols from. Ignored if standard_names provided.

    Returns:
        Dict mapping standard_name -> display_name

    Example:
        >>> get_display_names_dict(category='liquid_staking')
        {'ether-fi': 'EtherFi', 'lido': 'Lido', ...}

        >>> get_display_names_dict(standard_names=['maple-finance', 'euler'])
        {'maple-finance': 'Maple', 'euler': 'Euler'}
    """
    if standard_names is None and category is None:
        raise ValueError("Must provide either standard_names or category")

    if standard_names is None:
        standard_names = get_protocols_by_category(category, index_path)

    result = {}
    for name in standard_names:
        display = get_display_name(name, index_path)
        if display:
            result[name] = display
    return result


def print_index_summary(index_path: str = None) -> None:
    """Print a summary of the protocol index."""
    index = load_index(index_path)
    protocols = index.get('protocols', {})

    print(f"Protocol Index Summary")
    print(f"=" * 50)
    print(f"Total protocols: {len(protocols)}")
    print()

    # Group by category
    by_category = {}
    for name, info in protocols.items():
        cat = info.get('category', 'uncategorized')
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(name)

    for cat in sorted(by_category.keys()):
        print(f"{cat}: {len(by_category[cat])} protocols")
        for name in sorted(by_category[cat]):
            info = protocols[name]
            print(f"  - {name}")
            print(f"      display: {info.get('display_name')}")
            print(f"      defillama: {info.get('defillama_slug')}")
            print(f"      messari: {info.get('messari_slug')}")
        print()


# =============================================================================
# EQUITY INDEX FUNCTIONS
# =============================================================================

def get_equity(ticker: str, index_path: str = None) -> Optional[Dict]:
    """
    Get full equity info by ticker symbol.

    Args:
        ticker: The stock ticker symbol (e.g., 'COIN')
        index_path: Optional path to index file

    Returns:
        Dict with display_name, ticker, category
        or None if not found
    """
    index = load_index(index_path)
    return index.get('equities', {}).get(ticker)


def get_equity_display_name(ticker: str, index_path: str = None) -> Optional[str]:
    """
    Get the human-readable display name for an equity.

    Args:
        ticker: The stock ticker symbol (e.g., 'COIN')

    Returns:
        Display name (e.g., 'Coinbase') or None if not found
    """
    equity = get_equity(ticker, index_path)
    return equity.get('display_name') if equity else None


def get_equities_by_category(category: str, index_path: str = None) -> List[str]:
    """
    Get all ticker symbols in a category.

    Args:
        category: Category name (e.g., 'crypto_equities')

    Returns:
        List of ticker symbols
    """
    index = load_index(index_path)
    equities = index.get('equities', {})
    return [
        ticker for ticker, info in equities.items()
        if info.get('category') == category
    ]


def get_all_equity_categories(index_path: str = None) -> List[str]:
    """Get list of all unique equity categories in the index."""
    index = load_index(index_path)
    equities = index.get('equities', {})
    categories = set(info.get('category') for info in equities.values() if info.get('category'))
    return sorted(list(categories))


def get_all_equities(index_path: str = None) -> List[str]:
    """Get list of all ticker symbols in the index."""
    index = load_index(index_path)
    return sorted(list(index.get('equities', {}).keys()))


def get_equity_display_names_dict(
    tickers: List[str] = None,
    category: str = None,
    index_path: str = None
) -> Dict[str, str]:
    """
    Build assets dict with display names for equity visualization.
    Format: {ticker: display_name}

    Args:
        tickers: List of ticker symbols. If None, uses category.
        category: Category to get all equities from. Ignored if tickers provided.

    Returns:
        Dict mapping ticker -> display_name

    Example:
        >>> get_equity_display_names_dict(category='crypto_equities')
        {'COIN': 'Coinbase', 'HOOD': 'Robinhood', ...}
    """
    if tickers is None and category is None:
        raise ValueError("Must provide either tickers or category")

    if tickers is None:
        tickers = get_equities_by_category(category, index_path)

    result = {}
    for ticker in tickers:
        display = get_equity_display_name(ticker, index_path)
        if display:
            result[ticker] = display
    return result


def get_equity_categories_dict(
    categories: List[str] = None,
    index_path: str = None
) -> Dict[str, Dict[str, str]]:
    """
    Build the full categories dict for equity data, similar to protocol categories.

    Args:
        categories: List of equity categories to include. If None, includes all.

    Returns:
        Dict of format:
        {
            'category_name': {ticker: display_name, ...},
            ...
        }

    Example:
        >>> get_equity_categories_dict(['crypto_equities'])
        {'crypto_equities': {'COIN': 'Coinbase', 'HOOD': 'Robinhood', ...}}
    """
    if categories is None:
        categories = get_all_equity_categories(index_path)

    result = {}
    for cat in categories:
        result[cat] = get_equity_display_names_dict(category=cat, index_path=index_path)

    return result


def add_equity(
    ticker: str,
    display_name: str,
    category: str,
    index_path: str = None
) -> None:
    """
    Add a new equity to the index.

    Args:
        ticker: The stock ticker symbol (e.g., 'COIN')
        display_name: Human-readable name (e.g., 'Coinbase')
        category: Category for the equity (e.g., 'crypto_equities')
    """
    index = load_index(index_path)

    if 'equities' not in index:
        index['equities'] = {}

    index['equities'][ticker] = {
        'display_name': display_name,
        'ticker': ticker,
        'category': category
    }

    save_index(index, index_path)
    print(f"Added equity: {ticker}")


def remove_equity(ticker: str, index_path: str = None) -> bool:
    """
    Remove an equity from the index.

    Returns:
        True if removed, False if not found
    """
    index = load_index(index_path)

    if ticker in index.get('equities', {}):
        del index['equities'][ticker]
        save_index(index, index_path)
        print(f"Removed equity: {ticker}")
        return True
    return False


def print_equity_summary(index_path: str = None) -> None:
    """Print a summary of the equity index."""
    index = load_index(index_path)
    equities = index.get('equities', {})

    print(f"Equity Index Summary")
    print(f"=" * 50)
    print(f"Total equities: {len(equities)}")
    print()

    # Group by category
    by_category = {}
    for ticker, info in equities.items():
        cat = info.get('category', 'uncategorized')
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(ticker)

    for cat in sorted(by_category.keys()):
        print(f"{cat}: {len(by_category[cat])} equities")
        for ticker in sorted(by_category[cat]):
            info = equities[ticker]
            print(f"  - {ticker}: {info.get('display_name')}")
        print()


def print_full_index_summary(index_path: str = None) -> None:
    """Print a summary of both protocols and equities."""
    print_index_summary(index_path)
    print()
    print_equity_summary(index_path)
