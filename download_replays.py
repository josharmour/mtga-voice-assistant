
import argparse
import logging
import time
from pathlib import Path
import requests
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def update_registry(registry_path: Path, set_code: str, draft_type: str, file_path: Path):
    """
    Updates the data registry with information about the downloaded file.
    """
    registry_dir = registry_path.parent
    registry_dir.mkdir(parents=True, exist_ok=True)

    if registry_path.exists():
        with open(registry_path, "r") as f:
            try:
                registry = json.load(f)
            except json.JSONDecodeError:
                registry = {}
    else:
        registry = {}

    file_key = f"{set_code}_{draft_type}"
    registry[file_key] = {
        "set_code": set_code,
        "draft_type": draft_type,
        "download_date": datetime.now().isoformat(),
        "file_size": file_path.stat().st_size,
        "version": datetime.now().strftime("%Y%m%d"),
        "path": str(file_path.relative_to(registry_dir))
    }

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=4)

    logging.info(f"Updated data registry at {registry_path}")

def download_17lands_replay_data(set_code: str, draft_type: str, output_dir: Path, registry_path: Path):
    """
    Downloads 17Lands replay data for a given set and draft type.
    """
    url = f"https://17lands-public.s3.amazonaws.com/analysis_data/replay_data/replay_data_public.{set_code}.{draft_type}.csv.gz"

    output_path = output_dir / f"replay_data_public.{set_code}.{draft_type}.csv.gz"

    retries = 3
    delay = 5

    for attempt in range(retries):
        try:
            logging.info(f"Downloading from {url}...")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logging.info(f"Successfully downloaded to {output_path}")

            update_registry(registry_path, set_code, draft_type, output_path)

            return True

        except requests.exceptions.RequestException as e:
            if e.response and e.response.status_code == 403:
                logging.warning(f"Data for {set_code} - {draft_type} not available (403 Forbidden). Skipping.")
                return False

            logging.warning(f"Attempt {attempt + 1}/{retries} failed: {e}")
            if attempt < retries - 1:
                logging.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
            else:
                logging.error(f"All download attempts failed for {url}.")
                return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download 17Lands replay data.")
    parser.add_argument("--set", type=str, required=True, help="The set code(s) to download (e.g., 'MKM'). Can be a comma-separated list.")
    parser.add_argument("--format", type=str, default="PremierDraft", help="The draft format(s) to download. Use 'all' for all known formats, or provide a comma-separated list (e.g., 'PremierDraft,QuickDraft').")
    parser.add_argument("--base-dir", type=Path, default=Path("data/replay_data"), help="The base directory to save the data.")

    args = parser.parse_args()

    KNOWN_FORMATS = ["PremierDraft", "QuickDraft", "TradDraft", "Sealed", "TradSealed"]

    sets_to_download = [s.strip().upper() for s in args.set.split(',')]

    formats_to_download = []
    if args.format.lower() == 'all':
        formats_to_download = KNOWN_FORMATS
    else:
        formats_to_download = [f.strip() for f in args.format.split(',')]

    registry_path = args.base_dir / "registry.json"

    total_success = 0
    total_fail = 0

    for set_code in sets_to_download:
        logging.info(f"===== Processing set: {set_code} =====")
        set_output_dir = args.base_dir / set_code

        for draft_format in formats_to_download:
            logging.info(f"--- Downloading {set_code} - {draft_format} ---")
            if download_17lands_replay_data(set_code, draft_format, set_output_dir, registry_path):
                total_success += 1
            else:
                total_fail += 1
            time.sleep(1)

    logging.info("===== Download Summary =====")
    logging.info(f"Successfully downloaded: {total_success} file(s)")
    logging.info(f"Failed to download or skipped: {total_fail} file(s)")
