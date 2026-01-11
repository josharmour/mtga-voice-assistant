"""EDHREC integration for Commander metagame statistics using pyedhrec."""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import requests
from bs4 import BeautifulSoup
from pyedhrec import EDHRec


class EDHRECClient:
    """
    Client for accessing EDHREC Commander metagame data.
    Uses pyedhrec for detailed commander data and custom scraping for trending commanders.
    """

    BASE_URL = "https://edhrec.com"
    CACHE_DIR = Path("data/edhrec_cache")
    CACHE_DURATION = 86400  # 24 hours in seconds

    def __init__(self):
        """Initialize EDHREC client and ensure cache directory exists."""
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.edhrec_lib = EDHRec()
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

    def get_commander_page(self, commander_name: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Fetch EDHREC page data for a specific commander using pyedhrec.
        """
        cache_key = f"commander_{commander_name.lower().replace(' ', '-')}"

        if not force_refresh:
            cached_data = self._read_cache(cache_key)
            if cached_data:
                print(f"Using cached data for {commander_name}")
                return cached_data

        print(f"Fetching EDHREC data for {commander_name} via pyedhrec...")

        try:
            # Use pyedhrec to get the raw data structure
            raw_data = self.edhrec_lib.get_commander_data(commander_name)
            
            if not raw_data:
                raise Exception(f"No data returned for {commander_name}")

            # Extract relevant parts from the complex structure
            container = raw_data.get("container", {})
            json_dict = container.get("json_dict", {})
            
            data = {
                "commander": commander_name,
                "url": f"{self.BASE_URL}/commanders/{commander_name.lower().replace(' ', '-').replace(',', '')}",
                "fetched_at": time.time(),
                "cards": self._parse_cardlists(json_dict.get("cardlists", [])),
                "themes": self._parse_themes(raw_data.get("panels", {}).get("taglinks", [])),
                "meta": self._parse_meta(json_dict.get("card", {}))
            }

            self._write_cache(cache_key, data)
            return data

        except Exception as e:
            print(f"Error fetching EDHREC data for {commander_name}: {e}")
            return {
                "commander": commander_name,
                "error": str(e)
            }

    def _parse_cardlists(self, cardlists: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse cardlists from raw EDHREC JSON."""
        cards = []
        for cl in cardlists:
            category = cl.get("header", "Unknown")
            if category == "New Cards":
                continue
            for card in cl.get("cardviews", []):
                cards.append({
                    "name": card.get("name"),
                    "synergy": card.get("synergy"),
                    "inclusion": card.get("inclusion"),
                    "num_decks": card.get("num_decks"),
                    "category": category
                })
        return cards

    def _parse_themes(self, taglinks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse themes from raw EDHREC JSON."""
        return [
            {
                "name": tag.get("value"),
                "slug": tag.get("slug"),
                "url": f"{self.BASE_URL}/themes/{tag.get('slug', '')}"
            }
            for tag in taglinks
        ]

    def _parse_meta(self, card_info: Dict[str, Any]) -> Dict[str, Any]:
        """Parse metadata from raw EDHREC JSON."""
        return {
            "rank": card_info.get("rank"),
            "total_decks": card_info.get("num_decks"),
            "salt_score": card_info.get("salt")
        }

    def get_top_commanders(self, timeframe: str = "week", force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Get list of top commanders using structured data from the page.
        """
        cache_key = f"top_commanders_{timeframe}"

        if not force_refresh:
            cached_data = self._read_cache(cache_key)
            if cached_data:
                print(f"Using cached top commanders for {timeframe}")
                return cached_data.get("commanders", [])

        url = f"{self.BASE_URL}/commanders"
        print(f"Fetching top commanders from {url}...")

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract __NEXT_DATA__
            script_tag = soup.find("script", id="__NEXT_DATA__", type="application/json")
            if not script_tag:
                print("Warning: Could not find __NEXT_DATA__ on commanders page.")
                return []

            next_data = json.loads(script_tag.string)
            
            commanders = []
            seen_names = set()

            # The structure for top commanders is usually in props.pageProps.data.container.json_dict.cardlists
            try:
                data_section = next_data.get("props", {}).get("pageProps", {}).get("data", {})
                container = data_section.get("container", {})
                json_dict = container.get("json_dict", {})
                cardlists = json_dict.get("cardlists", [])

                for cl in cardlists:
                    # We want actual commanders, usually in "Top Commanders" list
                    for card in cl.get("cardviews", []):
                        name = card.get("name")
                        if name and name not in seen_names:
                            commanders.append({
                                "name": name,
                                "url": f"{self.BASE_URL}{card.get('url', '')}"
                            })
                            seen_names.add(name)
            except Exception as e:
                print(f"Error parsing NEXT_DATA for commanders: {e}")

            # Fallback if NEXT_DATA parsing failed
            if not commanders:
                print("Falling back to simple link scraping...")
                links = soup.find_all('a', href=lambda x: x and '/commanders/' in x if x else False)
                for link in links:
                    href = link.get('href', '')
                    slug = href.split('/commanders/')[-1].split('/')[0]
                    if not slug or len(slug) <= 2 or any(x in slug for x in ['?', '#', 'theme', 'partner']):
                        continue
                    name = slug.replace('-', ' ').title()
                    if name not in seen_names:
                        commanders.append({"name": name})
                        seen_names.add(name)

            data = {
                "commanders": commanders[:50],
                "fetched_at": time.time()
            }
            self._write_cache(cache_key, data)
            return commanders[:50]

        except Exception as e:
            print(f"Error fetching top commanders: {e}")
            return []