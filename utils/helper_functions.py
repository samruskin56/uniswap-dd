import os
import re
import functools
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
import pandas as pd
from datetime import datetime, timezone
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
from matplotlib.dates import MO, WeekdayLocator, MonthLocator, YearLocator
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from tqdm import tqdm

# API keys live in API_KEYS.env at the repo root (git-ignored). Loading it here
# means every module that builds a URL sees DEFILLAMA_API_KEY, not only the
# valuation notebooks that load it themselves.
try:
    from dotenv import load_dotenv
    load_dotenv('API_KEYS.env')
except Exception:
    pass

def ensure_directory(path):
    """Ensure a directory exists, creating it (and any parents) if needed."""
    os.makedirs(path, exist_ok=True)

def add_directories(top_root, parent_root, child_roots):
    """
    Create a parent directory and multiple child subdirectories within it.
    
    Args:
        parent_root (str): Name of the parent directory
        child_roots (list): List of child directory names
    """
    # Create the parent directory
    parent_path = os.path.join(top_root, parent_root)
    
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
        print(f'Directory made: {parent_path}')
    else:
        print(f'Directory already exists: {parent_path}')
    
    # Create child directories
    for child in child_roots:
        child_path = os.path.join(top_root, parent_root, child)
        
        if not os.path.exists(child_path):
            os.makedirs(child_path)
            print(f'Directory made: {child_path}')
        else:
            print(f'Directory already exists: {child_path}')

def delete_directories(directory_path):
    """
    Recursively delete all files in a directory and its subdirectories.
    Directories themselves are preserved (not deleted).
    
    Args:
        directory_path (str): Path to the root directory
    
    Returns:
        dict: Summary with counts of deleted files and any errors
    """
    directory_path = os.path.normpath(directory_path)
    
    deleted_count = 0
    deleted_files = []
    errors = []
    
    # Walk through directory tree and collect files
    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            deleted_files.append(file_path)
    
    # Show what will be deleted
    print(f"\nFound {len(deleted_files)} files to delete")
    
    # Confirm action
    confirm = input(f"\nDelete all {len(deleted_files)} files? (yes/no): ")
    
    if confirm.lower() == 'yes':
        for file_path in deleted_files:
            try:
                os.remove(file_path)
                deleted_count += 1
            except Exception as e:
                error_msg = f"Failed to delete {file_path}: {str(e)}"
                errors.append(error_msg)
                print(error_msg)
    else:
        print("Operation cancelled")
    
    # Return summary
    summary = {
        'total_found': len(deleted_files),
        'deleted': deleted_count,
        'errors': errors
    }
    
    return summary

def error_code(e):
    """Return an ``'HTTP 429 '`` style prefix if the exception carries a status code, else ``''``.

    Handles ``requests`` exceptions (``e.response.status_code``) and
    ``urllib.error.HTTPError`` (``e.code``).
    """
    resp = getattr(e, 'response', None)
    code = getattr(resp, 'status_code', None) or getattr(e, 'code', None)
    return f"HTTP {code} " if isinstance(code, int) else ""


# ---- DefiLlama hosts -------------------------------------------------------
# Free hosts, keyed by the section name the Pro host uses as a path prefix:
# https://pro-api.llama.fi/{key}/{section}{path} mirrors https://{section}.llama.fi{path}.
_LLAMA_FREE_HOST = {'api': 'https://api.llama.fi',
                    'coins': 'https://coins.llama.fi',
                    'stablecoins': 'https://stablecoins.llama.fi',
                    'yields': 'https://yields.llama.fi',
                    'bridges': 'https://bridges.llama.fi'}
_LLAMA_PRO_HOST = 'https://pro-api.llama.fi'


def defillama_key():
    """The DeFiLlama Pro API key from ``DEFILLAMA_API_KEY`` (``''`` when unset)."""
    return (os.getenv('DEFILLAMA_API_KEY') or '').strip()


def has_pro_key():
    """True when a DeFiLlama Pro key is configured, so pulls run on pro-api.llama.fi."""
    return bool(defillama_key())


def llama_url(path, section='api', pro_only=False):
    """Build a DeFiLlama URL for ``path`` (``'/protocol/aave'``), routed by key.

    With ``DEFILLAMA_API_KEY`` set the request goes to the Pro host
    (``pro-api.llama.fi/{key}/{section}{path}``: same payloads as the free
    endpoints, 1,000 req/min instead of the free tier's throttling, and the only
    place Pro-only endpoints exist). Without a key it goes to the free host for
    ``section`` (``api`` / ``coins`` / ``stablecoins`` / ``yields`` / ``bridges``),
    or returns None when ``pro_only`` is True so callers can skip quietly.
    redact_url() masks the key in any logged URL.
    """
    key = defillama_key()
    if key:
        return f"{_LLAMA_PRO_HOST}/{key}/{section}{path}"
    if pro_only:
        return None
    if section not in _LLAMA_FREE_HOST:
        raise ValueError(f"unknown DeFiLlama section {section!r}")
    return f"{_LLAMA_FREE_HOST[section]}{path}"


_KEY_QUERY = re.compile(r"(api[_-]?key=)[^&]+", re.IGNORECASE)
_KEY_PATH = re.compile(r"/[0-9a-fA-F]{32,}(?=/|$)")


def redact_url(url):
    """Mask API keys embedded in a URL (``api_key=`` query values and long hex path segments)."""
    url = _KEY_QUERY.sub(lambda m: m.group(1) + "***", str(url))
    return _KEY_PATH.sub("/***", url)


def report_errors(default=None, reraise=False):
    """Decorator: catch any exception from the wrapped function and print the error code.

    On failure prints ``[fn_name] failed: HTTP 429 HTTPError: <message>`` (the
    HTTP prefix only when the exception carries a status code) and returns
    ``default``. With ``reraise=True`` the exception is re-raised after
    printing, for functions whose callers unpack or index the return value
    and would otherwise fail one line later with a confusing TypeError.
    """
    def wrap(fn):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                print(f"[{fn.__name__}] failed: {error_code(e)}{type(e).__name__}: {e}")
                if reraise:
                    raise
                return default
        return inner
    return wrap


_SESSION = None
_SESSION_LOCK = threading.Lock()


def get_session(pool_maxsize=16):
    """Return the shared ``requests.Session`` used by every pull.

    Reusing one session keeps TCP/TLS connections open across requests
    (the biggest per-request cost against the DeFiLlama hosts) and mounts a retry
    policy: up to 4 retries with exponential backoff on 429 and 5xx, honoring
    ``Retry-After`` when the server sends one. Sessions are safe to share
    between the worker threads used by run_parallel().
    """
    global _SESSION
    if _SESSION is None:
        with _SESSION_LOCK:
            if _SESSION is None:
                retry = Retry(total=4, backoff_factor=1,
                              status_forcelist=[429, 500, 502, 503, 504],
                              allowed_methods=["GET"], respect_retry_after_header=True,
                              raise_on_status=False)
                adapter = HTTPAdapter(max_retries=retry, pool_connections=pool_maxsize, pool_maxsize=pool_maxsize)
                session = requests.Session()
                session.mount("https://", adapter)
                session.mount("http://", adapter)
                _SESSION = session
    return _SESSION


def save_json(data, path):
    """Write ``data`` to ``path`` as compact JSON, creating parent directories.

    Raw pulls are only ever read back with ``json.load``, so the compact form
    (no indent, no spaces after separators) roughly halves file size and write
    time, which matters for multi-MB DefiLlama payloads in a synced folder.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, separators=(',', ':'))


def fetch_and_save_json(url, output_path, error_message=None, headers=None, params=None, timeout=60, transform=None):
    """Fetch JSON from an API and save it to file.

    Returns the parsed JSON on success, or None on failure. On failure a
    single line with the HTTP status code (when available), exception type
    and URL is printed; callers report progress via pull_progress() and
    summarize results (including failures) via report_pull().

    Args:
        transform: Optional callable applied to the parsed JSON before it is
            written, so trimming (e.g. dropping the partial current day) costs
            one write instead of two. It must return the object to save.
    """
    try:
        response = get_session().get(url, headers=headers, params=params, timeout=timeout)
        response.raise_for_status()
        data = response.json()

        if transform is not None:
            data = transform(data)

        save_json(data, output_path)
        return data
    except (requests.exceptions.RequestException, ValueError) as e:
        print(f"  request failed: {error_code(e)}{type(e).__name__}: {redact_url(url)}")
        return None


def last_timestamp(path, extract):
    """Return the last data timestamp (unix seconds) in a saved JSON file, or None.

    ``extract`` maps the parsed JSON to a unix timestamp for the newest point.
    Any read/parse/shape failure returns None so the caller treats the file as
    missing and repulls it.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        ts = extract(data)
        return int(ts) if ts is not None else None
    except Exception:
        return None


def is_fresh(path, extract, max_stale_days):
    """True when ``path`` exists and its newest point is within ``max_stale_days`` of today (UTC).

    ``max_stale_days=None`` disables the check and always returns False, so a
    caller can force a full repull.
    """
    if max_stale_days is None:
        return False
    ts = last_timestamp(path, extract)
    if ts is None:
        return False
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    last = datetime.fromtimestamp(ts, tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return (today - last).days <= max_stale_days


def run_parallel(tasks, fn, max_workers, desc, total=None):
    """Run ``fn(task)`` over ``tasks`` on a thread pool with the standard progress bar.

    Returns a list of ``fn`` results in the same order as ``tasks``. An
    exception raised inside ``fn`` is printed and recorded as None for that
    task, matching the print-and-continue behaviour of the sequential loops.
    ``max_workers`` <= 1 runs the tasks sequentially in the calling thread.
    """
    tasks = list(tasks)
    results = [None] * len(tasks)
    if not tasks:
        return results

    def guarded(i):
        try:
            return fn(tasks[i])
        except Exception as e:
            print(f"  task failed: {error_code(e)}{type(e).__name__}: {e}")
            return None

    if max_workers is None or max_workers <= 1:
        for i in pull_progress(range(len(tasks)), desc, total=total or len(tasks)):
            results[i] = guarded(i)
        return results

    with ThreadPoolExecutor(max_workers=min(max_workers, len(tasks))) as pool:
        futures = {pool.submit(guarded, i): i for i in range(len(tasks))}
        for future in pull_progress(as_completed(futures), desc, total=total or len(tasks)):
            results[futures[future]] = future.result()
    return results


def pull_progress(items, desc, total=None):
    """Standardized tqdm progress bar for data-pull loops.

    Wrap the loop over the primary entities being pulled (assets, protocols,
    networks, queries, ...) so every pull function reports progress the same
    way — a single bar counting those entities, e.g. ``Dune queries: 3/5``.

    Args:
        items: Iterable of primary entities to loop over.
        desc: Short label shown to the left of the bar.
        total: Optional explicit length, for iterables without ``len()``.

    Returns:
        A tqdm iterator to use directly in a ``for`` loop.
    """
    return tqdm(items, desc=desc, total=total, unit="pull")


def report_pull(output_dir, metric_dirs, failures):
    """Print a standardized summary after a data pull.

    Prints one line per metric subdirectory showing where its files were
    saved (individual file paths are intentionally not printed), then lists
    any pulls that were attempted but failed.

    Args:
        output_dir: Base directory the data was saved under.
        metric_dirs: Iterable of metric subdirectory names. Pass an empty
            iterable when files are saved directly under output_dir.
        failures: List of (metric, item) tuples for pulls that failed. Items
            sharing a metric are grouped and bulleted under it; use None as
            the metric for failures with no metric grouping (listed flat).
    """
    print()
    metric_dirs = list(metric_dirs)
    if metric_dirs:
        for metric_dir in metric_dirs:
            print(f"All {metric_dir} data saved to {output_dir}/{metric_dir}/")
    else:
        print(f"All data saved to {output_dir}/")
    if failures:
        print(f"\nFailed to pull ({len(failures)}):")
        grouped = {}
        for metric, item in failures:
            grouped.setdefault(metric, []).append(item)
        for metric in sorted(grouped, key=lambda m: (m is None, str(m))):
            items = sorted(grouped[metric])
            if metric:
                print(f"  {metric}:")
                for item in items:
                    print(f"    - {item}")
            else:
                for item in items:
                    print(f"  - {item}")

def format_date_with_suffix(date_obj):
    # 1) turn str or datetime/date into a pd.Timestamp
    ts = pd.to_datetime(date_obj)

    # 2) get the day and compute the ordinal suffix
    day = ts.day
    if 10 <= day % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')

    # 3) build the final string
    #    Month name: ts.strftime('%B')
    #    Day + suffix: f'{day}{suffix}'
    #    Year: ts.year
    return f"{ts.strftime('%B')} {day}{suffix}, {ts.year}"

def to_unix_mmddyy(date_str: str) -> int:
    """
    Convert a date in U.S. **M/D/YY** format (e.g. '1/1/25') to a Unix timestamp.
    Interprets two‑digit years as 2000‑2099.

    >>> to_unix_mmddyy("1/1/25")
    1735689600
    """
    # Parse with strptime, then convert the naive datetime to UTC
    dt = datetime.strptime(date_str, "%m/%d/%y").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())

# Function to format dates on x-axis
def space_x_axis_dates(ax, df_name):
    today = datetime.now()
    delta = (today - df_name['date'].iloc[0]).days
    
    if delta < 90:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=MO, interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    elif delta < 600:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    elif delta < 1200:
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    else:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))


# Function to format values on the y-axis
def format_large_numbers(data_max, symbol='$'):
    """
    Returns a formatter function based on the maximum value in the data.
    
    Parameters:
    -----------
    data_max : float
        The maximum value in the dataset
    symbol : str
        The currency/unit symbol to use (default: '$')
    """
    if data_max >= 1e13:
        return mtick.FuncFormatter(lambda x, _: f'{symbol}{x/1e12:.0f}t')
    elif data_max >= 1e12:
        return mtick.FuncFormatter(lambda x, _: f'{symbol}{x/1e12:.1f}t')
    elif data_max >= 1e10:
        return mtick.FuncFormatter(lambda x, _: f'{symbol}{x/1e9:.0f}b')
    elif data_max >= 1e9:
        return mtick.FuncFormatter(lambda x, _: f'{symbol}{x/1e9:.1f}b')
    elif data_max >= 1e7:
        return mtick.FuncFormatter(lambda x, _: f'{symbol}{x/1e6:.0f}m')
    elif data_max >= 1e6:
        return mtick.FuncFormatter(lambda x, _: f'{symbol}{x/1e6:.1f}m')
    elif data_max >= 1e4:
        return mtick.FuncFormatter(lambda x, _: f'{symbol}{x/1e3:.0f}k')
    elif data_max >= 1e3:
        return mtick.FuncFormatter(lambda x, _: f'{symbol}{x/1e3:.1f}k')
    elif data_max >= 1:
        return mtick.FuncFormatter(lambda x, _: f'{symbol}{x:.0f}')
    else:
        return mtick.FuncFormatter(lambda x, _: f'{symbol}{x:.2f}')

# Function to format multiples on the y-axis
def format_multiples(data_max):
    """
    Returns a formatter function based on the maximum value in the data.

    Parameters:
    -----------
    data_max : float
        The maximum value in the dataset
    symbol : str
        The currency/unit symbol to use (default: '$')
    """
    if data_max <= 2:
        return mtick.FuncFormatter(lambda x, _: f'{x:.2f}x')
    elif data_max <= 20:
        return mtick.FuncFormatter(lambda x, _: f'{x:.1f}x')
    else:
        return mtick.FuncFormatter(lambda x, _: f'{x:,.0f}x')


def format_percentages(data_max):
    """
    Returns a percentage formatter with decimal precision scaled to the data's magnitude,
    so every y-axis tick shares a consistent number of decimal places. Mirrors
    format_multiples / format_large_numbers.

    Parameters:
    -----------
    data_max : float
        The maximum value in the dataset (as a fraction, e.g. 0.05 for 5%)
    """
    if data_max is None or pd.isna(data_max) or data_max >= 0.1:
        decimals = 0
    elif data_max >= 0.01:
        decimals = 1
    elif data_max >= 0.001:
        decimals = 2
    else:
        decimals = 3
    return mtick.FuncFormatter(lambda x, _: f'{x*100:,.{decimals}f}%')


def format_multiple_value(value):
    """
    Format a single multiple value based on its own magnitude.
    Useful for tables where different columns may have very different ranges.

    Parameters:
    -----------
    value : float
        The multiple value to format

    Returns:
    --------
    str : Formatted string with 'x' suffix
    """
    if value is None or pd.isna(value):
        return ""
    if value <= 2:
        return f'{value:.2f}x'
    elif value <= 20:
        return f'{value:.1f}x'
    else:
        return f'{value:,.0f}x'
    
def format_list_with_oxford_comma(items):
    """
    Formats a list of items into a comma-separated string with an Oxford comma.
    
    Args:
        items: A list of items to format
        
    Returns:
        A formatted string with proper comma usage
    """
    if not items:
        return ""
    elif len(items) == 1:
        return str(items[0].lower())
    elif len(items) == 2:
        return f"{items[0].lower()} and {items[1].lower()}"
    else:
        # Join all items except the last with commas, then add Oxford comma and last item
        return ", ".join(str(item.lower()) for item in items[:-1]) + ", and " + str(items[-1].lower())
    
def ffill_chart_breakdown(data):
    """
    Forward-fill missing protocol values in totalDataChartBreakdown.
    
    For each protocol, after its first appearance, any subsequent missing 
    entries are filled with the last known value.
    
    Parameters:
    -----------
    data : dict
        The raw JSON data from DefiLlama API
        
    Returns:
    --------
    dict
        The data with forward-filled values in totalDataChartBreakdown
    """
    if 'totalDataChartBreakdown' not in data:
        return data
    
    breakdown = data['totalDataChartBreakdown']
    
    # Track last known value for each protocol
    last_values = {}
    
    # Track which protocols have appeared (to not fill before first value)
    seen_protocols = set()
    
    for entry in breakdown:
        timestamp = entry[0]
        protocols = entry[1]
        
        # Update seen_protocols and last_values with current data
        for protocol, value in protocols.items():
            seen_protocols.add(protocol)
            last_values[protocol] = value
        
        # Forward-fill: add missing protocols that we've seen before
        for protocol in seen_protocols:
            if protocol not in protocols and protocol in last_values:
                protocols[protocol] = last_values[protocol]
    
    return data

# Set a default logo file path (relative or absolute)
default_logo_file = os.path.join(os.path.dirname(__file__), 'rv_lockup_sec.png')

def add_logo(ax, logo_file=None, zoom=0.1, position=(0.075, -0.15)):
    """
    Adds a logo to the given matplotlib axis.

    Parameters:
    - ax: The matplotlib axis where the logo will be added.
    - logo_file: Filepath to the logo image. If not provided, the default is used.
    - zoom: Scaling factor for the logo image.
    - position: Position of the logo within the plot, specified as (x, y) in axes fraction.
    """
    if logo_file is None:
        logo_file = default_logo_file  # Use default if no logo path is provided

    try:
        logo = plt.imread(logo_file)
        imagebox = OffsetImage(logo, zoom=zoom)
        ab = AnnotationBbox(
            imagebox, 
            position, 
            xycoords='axes fraction', 
            frameon=False,
            box_alignment=(0, 1)  # Align left edge and top of logo to position
        )
        ax.add_artist(ab)
    except Exception as e:
        print(f"Failed to add logo: {e}")

def format_logo_and_note(ax, note_text, logo_file=None, zoom=0.1, is_table=False):
    """
    Add logo and note aligned to the left y-axis, positioned below the x-axis labels.
    
    Parameters:
    -----------
    ax : matplotlib axis
        The axis to add the logo and note to
    note_text : str
        The note text to display
    logo_file : str, optional
        Path to logo file. Uses default_logo_file if None.
    zoom : float, optional
        Zoom level for the logo image
    is_table : bool, optional
        If True, uses fixed positioning suitable for table figures.
        If False (default), calculates position based on x-axis labels.
    """
    fig = ax.get_figure()
    fig.canvas.draw()
    
    if is_table:
        # For tables, position below the table (table bottom is at y=0.35 in axes fraction)
        logo_y_position = 0.30  # Just below the table
        note_y_position = 0.20   # Below the logo
        logo_offset = 0
        note_offset = 0
    else:
        # Original behavior: calculate based on x-axis labels
        logo_y_position = 0
        note_y_position = 0
        logo_offset = -55
        note_offset = -80
    
    if logo_file is None:
        logo_file = default_logo_file
    
    try:
        logo = plt.imread(logo_file)
        imagebox = OffsetImage(logo, zoom=zoom)
        
        ab = AnnotationBbox(
            imagebox, 
            (0, logo_y_position),  # Use y_position for tables
            xycoords='axes fraction',
            xybox=(0, logo_offset),
            boxcoords='offset points',
            frameon=False,
            box_alignment=(0, 1),
            clip_on=False
        )
        ax.add_artist(ab)
        
        # Add note below the logo
        ax.annotate(
            note_text,
            xy=(0, note_y_position),  # Use y_position for tables
            xycoords='axes fraction',
            xytext=(0, note_offset),
            textcoords='offset points',
            ha='left',
            va='top',
            fontsize=10,
            style='italic',
            annotation_clip=False
        )
        
    except Exception as e:
        print(f"Failed to add logo: {e}")

def get_last_data_date(df, date_column='date', formatted=True):
    """
    Get the last (most recent) date from a DataFrame and optionally format it.
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None")
    
    if date_column not in df.columns:
        raise ValueError(f"Column '{date_column}' not found in DataFrame")
    
    last_date = df[date_column].max()
    
    if formatted:
        return format_date_with_suffix(last_date)
    else:
        return last_date
    
def load_metric_data(asset_slug, metric, trailing_window):
    """
    Load and process metric data from either Messari or DefiLlama.
    
    Returns: DataFrame with columns ['date', metric, f'{metric}-{trailing_window}d']
    """
    # Messari metrics (price, mcap, fdv)
    messari_metrics = {
        'price': {'file': 'price', 'column': 'close', 'display': 'Price'},
        'mcap': {'file': 'marketcap', 'column': 'circulating-marketcap', 'display': 'MCAP'},
        'fdv': {'file': 'marketcap', 'column': 'fully-diluted-marketcap', 'display': 'FDV'}
    }
    
    if metric in messari_metrics:
        config = messari_metrics[metric]
        with open(f'data/raw/messari/{config["file"]}/{asset_slug}.json', 'r') as f:
            data = json.load(f)
        
        points = data['data']['points']
        
        if config['file'] == 'price':
            df = pd.DataFrame(points, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])
        else:  # marketcap file
            df = pd.DataFrame(points, columns=[
                'timestamp',
                'circulating-marketcap',
                'circulating-marketcap-dominance',
                'fully-diluted-marketcap'
            ])
        
        df["date"] = pd.to_datetime(df["timestamp"], unit="s")
        df = df[['date', config['column']]]
        df = df.rename(columns={config['column']: metric})
        df[f'{metric}-{trailing_window}d'] = df[metric].rolling(window=trailing_window).mean()
        
        return df, config['display']
    
    else:
        # DefiLlama metrics (fees, revenue, volume, tvl, etc.)
        with open(f'data/raw/defillama/{metric}/{asset_slug}.json', "r") as f:
            data = json.load(f)
        
        df = pd.DataFrame(data["totalDataChart"], columns=["timestamp", metric])
        df["date"] = pd.to_datetime(df["timestamp"], unit="s")
        df = df[["date", metric]]
        df[f"{metric}-{trailing_window}d"] = df[metric].rolling(window=trailing_window).mean()
        
        return df, metric.capitalize()