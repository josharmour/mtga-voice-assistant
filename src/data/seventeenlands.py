"""17Lands integration for Limited (Draft/Sealed) format statistics."""

import json
import time
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from urllib.parse import urlencode
import requests


class SeventeenLandsClient:
    """
    Client for accessing 17Lands Limited format data and statistics.

    17Lands provides card performance data for Draft and Sealed formats,
    including win rates, pick rates, and other metrics.
    """

    API_BASE_URL = "https://api.17lands.com"
    WEB_BASE_URL = "https://www.17lands.com"
    CACHE_DIR = Path("data/17lands_cache")
    CACHE_DURATION = 43200  # 12 hours in seconds (data updates frequently)

    def __init__(self):
        """Initialize 17Lands client and ensure cache directory exists."""
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PlaneswalkerAgent/1.0 (Educational MTG AI Project)'
        })

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a given key."""
        # Sanitize key for filesystem
        safe_key = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in key)
        return self.CACHE_DIR / f"{safe_key}.json"

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cached data is still valid."""
        if not cache_path.exists():
            return False

        age = time.time() - cache_path.stat().st_mtime
        return age < self.CACHE_DURATION

    def _read_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Read data from cache if available and valid."""
        cache_path = self._get_cache_path(key)

        if self._is_cache_valid(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Failed to read cache for {key}: {e}")

        return None

    def _write_cache(self, key: str, data: Dict[str, Any]) -> None:
        """Write data to cache."""
        cache_path = self._get_cache_path(key)

        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to write cache for {key}: {e}")

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
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to fetch {url} after {max_retries} attempts: {e}")

                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Request failed (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
                time.sleep(wait_time)

    def get_card_ratings(
        self,
        expansion: str,
        format_type: str = "PremierDraft",
        force_refresh: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Fetch card ratings and performance data for a specific set.

        Args:
            expansion: Set code (e.g., 'MKM', 'LCI', 'WOE')
            format_type: Format type ('PremierDraft', 'QuickDraft', 'Sealed', 'TradDraft')
            force_refresh: If True, bypass cache

        Returns:
            List of card rating dictionaries with performance metrics
        """
        cache_key = f"card_ratings_{expansion}_{format_type}"

        # Check cache first
        if not force_refresh:
            cached_data = self._read_cache(cache_key)
            if cached_data:
                print(f"Using cached card ratings for {expansion} ({format_type})")
                return cached_data.get("card_ratings", [])

        # Try API endpoint
        url = f"{self.API_BASE_URL}/card_ratings/data"
        params = {
            "expansion": expansion,
            "format": format_type
        }

        print(f"Fetching 17Lands card ratings for {expansion} ({format_type})...")

        try:
            response = self._retry_request(url + "?" + "&".join(f"{k}={v}" for k, v in params.items()))

            # Try to parse as JSON
            try:
                card_data = response.json()
            except json.JSONDecodeError:
                # If not JSON, might be CSV or other format
                print("Warning: Response not in JSON format, using fallback parsing")
                card_data = []

            data = {
                "card_ratings": card_data if isinstance(card_data, list) else [card_data],
                "expansion": expansion,
                "format": format_type,
                "fetched_at": time.time()
            }

            # Cache the result
            self._write_cache(cache_key, data)

            return data["card_ratings"]

        except Exception as e:
            print(f"Error fetching card ratings from API: {e}")
            return []

    def get_card_data(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Fetch general card data from 17Lands.

        Args:
            force_refresh: If True, bypass cache

        Returns:
            Dictionary containing card data
        """
        cache_key = "card_data_general"

        # Check cache first
        if not force_refresh:
            cached_data = self._read_cache(cache_key)
            if cached_data:
                print("Using cached general card data")
                return cached_data

        url = f"{self.API_BASE_URL}/card_data"

        print("Fetching general card data from 17Lands...")

        try:
            response = self._retry_request(url)

            try:
                data = response.json()
            except json.JSONDecodeError:
                print("Warning: Card data response not in JSON format")
                data = {"raw_response": response.text[:1000]}

            data["fetched_at"] = time.time()

            # Cache the result
            self._write_cache(cache_key, data)

            return data

        except Exception as e:
            print(f"Error fetching general card data: {e}")
            return {"error": str(e)}

    def get_set_stats(
        self,
        expansion: str,
        format_type: str = "PremierDraft",
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Fetch overall statistics for a set.

        Args:
            expansion: Set code (e.g., 'MKM', 'LCI', 'WOE')
            format_type: Format type ('PremierDraft', 'QuickDraft', 'Sealed', 'TradDraft')
            force_refresh: If True, bypass cache

        Returns:
            Dictionary containing set-wide statistics
        """
        cache_key = f"set_stats_{expansion}_{format_type}"

        # Check cache first
        if not force_refresh:
            cached_data = self._read_cache(cache_key)
            if cached_data:
                print(f"Using cached set stats for {expansion}")
                return cached_data

        # Build stats from card ratings
        card_ratings = self.get_card_ratings(expansion, format_type, force_refresh)

        if not card_ratings:
            return {"error": "No data available", "expansion": expansion}

        # Calculate aggregate statistics
        stats = {
            "expansion": expansion,
            "format": format_type,
            "total_cards": len(card_ratings),
            "fetched_at": time.time()
        }

        # Cache the result
        self._write_cache(cache_key, stats)

        return stats

    def get_color_pair_data(
        self,
        expansion: str,
        format_type: str = "PremierDraft",
        force_refresh: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Fetch color pair win rates and statistics.

        Args:
            expansion: Set code (e.g., 'MKM', 'LCI', 'WOE')
            format_type: Format type ('PremierDraft', 'QuickDraft', 'Sealed')
            force_refresh: If True, bypass cache

        Returns:
            List of color pair statistics
        """
        cache_key = f"color_pairs_{expansion}_{format_type}"

        # Check cache first
        if not force_refresh:
            cached_data = self._read_cache(cache_key)
            if cached_data:
                print(f"Using cached color pair data for {expansion}")
                return cached_data.get("color_pairs", [])

        print(f"Fetching color pair data for {expansion}...")

        # Mapping of color codes to guild names
        guild_names = {
            "WU": "Azorius",
            "UB": "Dimir",
            "BR": "Rakdos",
            "RG": "Gruul",
            "GW": "Selesnya",
            "WG": "Selesnya",  # API sometimes returns WG
            "WB": "Orzhov",
            "UR": "Izzet",
            "BG": "Golgari",
            "RW": "Boros",
            "WR": "Boros",     # API sometimes returns WR
            "GU": "Simic",
            "UG": "Simic",     # API sometimes returns UG
        }

        # Fetch from 17Lands API
        url = f"{self.WEB_BASE_URL}/color_ratings/data"

        # Use a safe start date that covers recent history
        start_date = "2019-01-01"
        end_date = datetime.date.today().strftime("%Y-%m-%d")

        params = {
            "expansion": expansion,
            "event_type": format_type,
            "start_date": start_date,
            "end_date": end_date,
            "combine_splash": "false"
        }

        color_pairs = []

        try:
            # Use safe URL construction
            query_string = urlencode(params)
            response = self._retry_request(f"{url}?{query_string}")

            try:
                raw_data = response.json()
            except json.JSONDecodeError:
                print("Warning: Response not in JSON format")
                raw_data = []

            for item in raw_data:
                # Filter out summaries and check for 2-color pairs
                if item.get("is_summary"):
                    continue

                colors = item.get("short_name")

                # Check if this is one of our guild pairs
                if colors in guild_names:
                    wins = item.get("wins", 0)
                    games = item.get("games", 0)

                    win_rate = (wins / games) if games > 0 else 0.0

                    # Normalize color code to standard (e.g., WG -> GW) if needed
                    canonical_colors = {
                        "WG": "GW",
                        "WR": "RW",
                        "UG": "GU"
                    }.get(colors, colors)

                    color_pairs.append({
                        "colors": canonical_colors,
                        "name": guild_names[colors],
                        "win_rate": win_rate,
                        "wins": wins,
                        "games": games
                    })

        except Exception as e:
            print(f"Error fetching color pair data from API: {e}")
            # Return empty list or fall back to previous behavior if needed,
            # but for now we return what we found (or empty)

        # If API fails or returns no data, we might want to return the structure with 0s
        # But let's only do that if we got nothing
        if not color_pairs:
            # Use unique canonical pairs for fallback to avoid duplicates
            unique_pairs = {}
            for code, name in guild_names.items():
                canonical = {
                    "WG": "GW",
                    "WR": "RW",
                    "UG": "GU"
                }.get(code, code)
                unique_pairs[canonical] = name

            color_pairs = [
                {"colors": code, "name": name, "win_rate": 0.0}
                for code, name in unique_pairs.items()
            ]

        data = {
            "color_pairs": color_pairs,
            "expansion": expansion,
            "format": format_type,
            "fetched_at": time.time()
        }

        # Cache the result
        self._write_cache(cache_key, data)

        return color_pairs
