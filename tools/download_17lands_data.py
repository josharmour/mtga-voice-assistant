import logging
import requests
import gzip
import shutil
import time
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_17lands_data(set_code: str, draft_type: str, data_type: str, output_dir: Path, retries: int = 3, backoff_factor: float = 0.5):
    """
    Downloads and decompresses 17Lands data for a specific set, draft type, and data type.

    Args:
        set_code (str): The MTG set code (e.g., 'MKM').
        draft_type (str): The draft type (e.g., 'PremierDraft').
        data_type (str): The data type to download (e.g., 'draft_data', 'game_data', 'replay_data').
        output_dir (Path): The directory to save the downloaded file.
        retries (int): The number of times to retry the download on failure.
        backoff_factor (float): The factor to use for exponential backoff between retries.
    """
    # Validate data_type
    valid_data_types = ['draft_data', 'game_data', 'replay_data']
    if data_type not in valid_data_types:
        logging.error(f"Invalid data_type: {data_type}. Must be one of {valid_data_types}")
        return

    # Create a versioned directory structure
    date_str = datetime.now().strftime('%Y-%m-%d')
    versioned_dir = output_dir / set_code / date_str
    versioned_dir.mkdir(parents=True, exist_ok=True)

    url = f"https://17lands-public.s3.amazonaws.com/analysis_data/{data_type}/{set_code}_{draft_type}.csv.gz"
    gz_filename = f"{set_code}_{draft_type}.csv.gz"
    csv_filename = f"{set_code}_{draft_type}.csv"

    gz_filepath = versioned_dir / gz_filename
    csv_filepath = versioned_dir / csv_filename

    logging.info(f"Downloading data for {set_code} ({draft_type}, {data_type}) from {url}")

    for i in range(retries):
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(gz_filepath, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

            logging.info(f"Successfully downloaded {gz_filepath}")

            logging.info(f"Decompressing {gz_filepath} to {csv_filepath}")
            with gzip.open(gz_filepath, 'rb') as f_in:
                with open(csv_filepath, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            logging.info(f"Successfully decompressed to {csv_filepath}")

            # Clean up the compressed file
            gz_filepath.unlink()
            logging.info(f"Removed compressed file: {gz_filepath}")
            return

        except requests.exceptions.RequestException as e:
            logging.warning(f"Attempt {i + 1} of {retries} failed: {e}")
            if i < retries - 1:
                sleep_time = backoff_factor * (2 ** i)
                logging.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                logging.error(f"Failed to download data for {set_code} ({draft_type}, {data_type}) after {retries} attempts.")
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            break
