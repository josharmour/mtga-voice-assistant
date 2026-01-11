"""Scryfall API integration for fetching Magic: The Gathering card data."""

import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests


class ScryfallLoader:
    """Handles downloading and loading card data from Scryfall's bulk data API."""

    BULK_DATA_URL = "https://api.scryfall.com/bulk-data"
    CACHE_DIR = Path("data")
    ORACLE_CACHE_FILE = CACHE_DIR / "oracle-cards.json"

    def __init__(self):
        """Initialize the Scryfall loader and ensure cache directory exists."""
        self.CACHE_DIR.mkdir(exist_ok=True)

    def _retry_request(self, url: str, max_retries: int = 5) -> requests.Response:
        """
        Make HTTP request with exponential backoff retry logic.

        Args:
            url: The URL to fetch
            max_retries: Maximum number of retry attempts

        Returns:
            The successful response

        Raises:
            Exception: If all retries fail
        """
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to fetch {url} after {max_retries} attempts: {e}")

                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s, 8s, 16s
                print(f"Request failed (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
                time.sleep(wait_time)

    def fetch_data(self, force_download: bool = False) -> None:
        """
        Download Oracle card data from Scryfall's bulk data API.

        Args:
            force_download: If True, re-download even if cache exists
        """
        if self.ORACLE_CACHE_FILE.exists() and not force_download:
            print(f"Using cached data: {self.ORACLE_CACHE_FILE}")
            return

        print("Fetching bulk data list from Scryfall...")
        response = self._retry_request(self.BULK_DATA_URL)
        bulk_data = response.json()

        # Find the Oracle Cards bulk data entry
        oracle_entry = None
        for entry in bulk_data["data"]:
            if entry["type"] == "oracle_cards":
                oracle_entry = entry
                break

        if not oracle_entry:
            raise Exception("Oracle cards bulk data not found in Scryfall API response")

        download_url = oracle_entry["download_uri"]
        print(f"Downloading Oracle cards from: {download_url}")
        print(f"Size: ~{oracle_entry['size'] / 1024 / 1024:.1f} MB")

        # Download the bulk data file
        response = self._retry_request(download_url)

        # Save to cache
        print(f"Saving to {self.ORACLE_CACHE_FILE}...")
        with open(self.ORACLE_CACHE_FILE, 'wb') as f:
            f.write(response.content)

        print(f"Download complete. Updated: {oracle_entry['updated_at']}")

    def load_cards(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load card data from cached Oracle cards file.

        Args:
            limit: Optional limit on number of cards to load (for testing)

        Returns:
            List of card dictionaries

        Raises:
            FileNotFoundError: If cache file doesn't exist
        """
        if not self.ORACLE_CACHE_FILE.exists():
            raise FileNotFoundError(
                f"Cache file not found: {self.ORACLE_CACHE_FILE}. "
                "Run fetch_data() first."
            )

        print(f"Loading cards from {self.ORACLE_CACHE_FILE}...")
        with open(self.ORACLE_CACHE_FILE, 'r', encoding='utf-8') as f:
            cards = json.load(f)

        if limit:
            cards = cards[:limit]
            print(f"Loaded {len(cards)} cards (limited)")
        else:
            print(f"Loaded {len(cards)} cards")

        return cards
