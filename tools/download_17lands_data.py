
import argparse
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_TYPES = ["draft_data", "game_data", "replay_data"]

def get_set_codes() -> List[str]:
    """
    Fetches the list of all available set codes from the 17Lands API.

    Returns:
        A list of set codes (e.g., ["NEO", "KHM"]).
    """
    url = "https://www.17lands.com/data/expansions"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # The API returns a JSON list of set codes.
        # We need to filter out the sets that are not in the standard 3-letter format.
        set_codes = [code for code in response.json() if re.match(r"^[A-Z0-9]{3}$", code)]

        # Remove duplicates and sort the list
        unique_sets = sorted(list(set(set_codes)))
        logging.info(f"Found {len(unique_sets)} unique set codes.")
        return unique_sets

    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch or parse set codes: {e}")
        return []

def download_17lands_data(set_code: str, data_type: str, output_dir: Path, retries: int = 3, backoff_factor: float = 0.5):
    """
    Downloads a single 17Lands dataset with retry logic.

    Args:
        set_code: The set code to download (e.g., "NEO").
        data_type: The type of data to download (e.g., "draft_data").
        output_dir: The directory to save the file in.
        retries: The number of times to retry the download.
        backoff_factor: The factor to use for exponential backoff.
    """
    if data_type not in DATA_TYPES:
        logging.error(f"Invalid data type: {data_type}. Must be one of {DATA_TYPES}")
        return

    # Construct the URL based on the data type
    if data_type == "replay_data":
        url = f"https://17lands-public.s3.amazonaws.com/replay_data/replay_data_public.{set_code}.PremierDraft.csv.gz"
    else:
        url = f"https://17lands-public.s3.amazonaws.com/analysis_data/{data_type}/{data_type}_public.{set_code}.PremierDraft.csv.gz"

    output_path = output_dir / f"{set_code}_{data_type}.csv.gz"

    # Skip download if the file already exists
    if output_path.exists():
        logging.info(f"Skipping download for {set_code} {data_type}, file already exists.")
        return

    for i in range(retries):
        try:
            logging.info(f"Downloading {data_type} for {set_code}...")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()  # Raise an exception for bad status codes

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logging.info(f"Successfully downloaded {output_path}")
            return  # Exit the loop on success

        except requests.exceptions.RequestException as e:
            logging.warning(f"Attempt {i+1} failed for {set_code} {data_type}: {e}")
            if i < retries - 1:
                time.sleep(backoff_factor * (2 ** i))  # Exponential backoff
            else:
                logging.error(f"Failed to download {url} after {retries} attempts.")

def main():
    """Main function to parse arguments and initiate downloads."""
    parser = argparse.ArgumentParser(description="Download 17Lands public datasets.")
    parser.add_argument("--set", type=str, help="Specific set code to download (e.g., 'NEO'). If not provided, all sets will be downloaded.")
    parser.add_argument("--data_types", nargs="+", default=DATA_TYPES, help=f"A list of data types to download. Defaults to all: {', '.join(DATA_TYPES)}")
    parser.add_argument("--output_dir", type=Path, default=Path("data/17lands_data"), help="Directory to save the downloaded files.")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of concurrent downloads.")
    args = parser.parse_args()

    # Ensure the output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.set:
        sets_to_download = [args.set]
    else:
        sets_to_download = get_set_codes()

    if not sets_to_download:
        logging.error("No set codes found or provided. Exiting.")
        return

    logging.info(f"Preparing to download data for the following sets: {', '.join(sets_to_download)}")

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        for set_code in sets_to_download:
            for data_type in args.data_types:
                futures.append(executor.submit(download_17lands_data, set_code, data_type, args.output_dir))

        for future in as_completed(futures):
            try:
                future.result()  # Raise any exceptions that occurred during download
            except Exception as e:
                logging.error(f"An error occurred during download: {e}")

    logging.info("All downloads completed.")

if __name__ == "__main__":
    main()
