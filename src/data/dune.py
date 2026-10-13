import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.helper_functions import fetch_and_save_json, pull_progress, report_pull, report_errors


@report_errors()
def fetch_and_save_dune_data(dune_api_key, query_ids, dune_output_dir):
    """
    Fetch results from the Dune API for each query ID and save them to JSON.

    :param dune_api_key: str - The Dune API key.
    :param query_ids: dict - Mapping of {query_id: query_title}.
    :param dune_output_dir: str - Directory to save the result files in.
    """
    failures = []
    for query_id, query_title in pull_progress(query_ids.items(), "Dune queries"):
        url = f"https://api.dune.com/api/v1/query/{query_id}/results"
        headers = {"X-DUNE-API-KEY": dune_api_key}
        output_path = f"{dune_output_dir}/{query_title}.json"

        if fetch_and_save_json(url, output_path, headers=headers) is None:
            failures.append((None, query_title))

    report_pull(dune_output_dir, [], failures)
