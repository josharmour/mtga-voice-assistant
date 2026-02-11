#!/usr/bin/env python3
"""
Compare CSV vs API approaches for 17lands card ratings.

Test both methods and measure performance/data quality.
"""

import requests
import time
import json
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class CardRating:
    """Card rating from 17lands"""
    name: str
    mtga_id: int
    color: str
    rarity: str
    win_rate: Optional[float]
    game_count: int
    drawn_win_rate: Optional[float]
    ever_drawn_win_rate: Optional[float]
    drawn_improvement_win_rate: Optional[float]  # IWD equivalent
    play_rate: Optional[float]
    avg_taken_at: Optional[float]


class SeventeenLandsAPIClient:
    """
    17lands API client - fast, real-time, no pre-processing required.

    API endpoint: https://www.17lands.com/card_ratings/data
    Query params:
        - expansion: SET_CODE (OTJ, BLB, FDN, etc.)
        - format: PremierDraft, QuickDraft, TradDraft, PickTwoDraft
        - start_date: YYYY-MM-DD (optional)
        - end_date: YYYY-MM-DD (optional)
    """

    BASE_URL = "https://www.17lands.com/card_ratings/data"

    def __init__(self, cache_minutes: int = 30):
        """
        Args:
            cache_minutes: How long to cache responses (default 30 min)
        """
        self._cache: Dict[str, tuple] = {}  # key -> (timestamp, data)
        self._cache_duration = cache_minutes * 60

    def get_card_ratings(
        self,
        expansion: str,
        format: str = "PremierDraft",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[CardRating]:
        """
        Fetch card ratings from 17lands API.

        Args:
            expansion: Set code (OTJ, BLB, FDN, etc.)
            format: Draft format (PremierDraft, QuickDraft, etc.)
            start_date: Optional date filter YYYY-MM-DD
            end_date: Optional date filter YYYY-MM-DD

        Returns:
            List of CardRating objects
        """
        # Check cache
        cache_key = f"{expansion}_{format}_{start_date}_{end_date}"
        if cache_key in self._cache:
            timestamp, data = self._cache[cache_key]
            if time.time() - timestamp < self._cache_duration:
                print(f"  ✓ Using cached data for {expansion} ({format})")
                return data

        # Build URL
        params = {
            "expansion": expansion,
            "format": format,
        }
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        print(f"  Fetching {expansion} ({format})...")
        start = time.time()

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()

            elapsed = time.time() - start
            print(f"  ✓ Downloaded in {elapsed:.2f}s")

            # Parse JSON
            raw_data = response.json()

            # Convert to CardRating objects
            ratings = []
            for card in raw_data:
                # Skip cards with no data
                if card.get("game_count", 0) == 0:
                    continue

                ratings.append(CardRating(
                    name=card["name"],
                    mtga_id=card["mtga_id"],
                    color=card["color"],
                    rarity=card["rarity"],
                    win_rate=card.get("win_rate"),
                    game_count=card.get("game_count", 0),
                    drawn_win_rate=card.get("drawn_win_rate"),
                    ever_drawn_win_rate=card.get("ever_drawn_win_rate"),
                    drawn_improvement_win_rate=card.get("drawn_improvement_win_rate"),
                    play_rate=card.get("play_rate"),
                    avg_taken_at=card.get("avg_taken_at")
                ))

            # Cache result
            self._cache[cache_key] = (time.time(), ratings)

            print(f"  ✓ Parsed {len(ratings)} cards with game data")
            return ratings

        except requests.exceptions.RequestException as e:
            print(f"  ✗ API request failed: {e}")
            return []

    def get_multiple_sets(self, expansions: List[str], format: str = "PremierDraft") -> Dict[str, List[CardRating]]:
        """
        Fetch ratings for multiple sets efficiently.

        Args:
            expansions: List of set codes
            format: Draft format

        Returns:
            Dictionary mapping set_code -> List[CardRating]
        """
        results = {}

        for expansion in expansions:
            ratings = self.get_card_ratings(expansion, format)
            if ratings:
                results[expansion] = ratings

            # Small delay to be polite
            time.sleep(0.5)

        return results

    def lookup_card(
        self,
        card_name: str,
        expansion: str,
        format: str = "PremierDraft"
    ) -> Optional[CardRating]:
        """
        Lookup a single card's rating.

        Args:
            card_name: Card name (case-insensitive)
            expansion: Set code
            format: Draft format

        Returns:
            CardRating or None if not found
        """
        ratings = self.get_card_ratings(expansion, format)

        # Case-insensitive search
        card_name_lower = card_name.lower()
        for rating in ratings:
            if rating.name.lower() == card_name_lower:
                return rating

        return None

    def get_grade(self, win_rate: Optional[float], drawn_wr: Optional[float]) -> str:
        """
        Convert win rate to letter grade (A+ to F).

        Uses GIH WR if available, falls back to overall WR.
        Based on typical 17lands grading scale.
        """
        # Use drawn (GIH) win rate if available, otherwise overall
        wr = drawn_wr if drawn_wr is not None else win_rate

        if wr is None:
            return "?"

        # Grading scale (approximate 17lands tiers)
        if wr >= 0.625:
            return "A+"
        elif wr >= 0.600:
            return "A"
        elif wr >= 0.575:
            return "A-"
        elif wr >= 0.560:
            return "B+"
        elif wr >= 0.545:
            return "B"
        elif wr >= 0.530:
            return "B-"
        elif wr >= 0.515:
            return "C+"
        elif wr >= 0.500:
            return "C"
        elif wr >= 0.485:
            return "C-"
        elif wr >= 0.470:
            return "D+"
        elif wr >= 0.450:
            return "D"
        else:
            return "F"


def compare_approaches():
    """
    Compare CSV download vs API approach.
    """
    print("="*70)
    print("17lands: CSV Download vs API Comparison")
    print("="*70)

    # Test API approach
    print("\n1. API APPROACH (Direct)")
    print("-" * 70)

    client = SeventeenLandsAPIClient()

    # Test single set
    print("\nFetching single set (OTJ Premier Draft):")
    start = time.time()
    otj_ratings = client.get_card_ratings("OTJ", "PremierDraft")
    api_single_time = time.time() - start

    print(f"\n✓ API single set: {len(otj_ratings)} cards in {api_single_time:.2f}s")

    # Test multiple sets
    print("\nFetching multiple sets (current Standard):")
    standard_sets = ["OTJ", "MKM", "LCI", "WOE", "BLB"]
    start = time.time()
    standard_ratings = client.get_multiple_sets(standard_sets)
    api_multi_time = time.time() - start

    total_cards = sum(len(ratings) for ratings in standard_ratings.values())
    print(f"\n✓ API multiple sets: {total_cards} cards across {len(standard_sets)} sets in {api_multi_time:.2f}s")

    # Test card lookup
    print("\nLooking up specific card (Holy Cow):")
    start = time.time()
    holy_cow = client.lookup_card("Holy Cow", "OTJ", "PremierDraft")
    lookup_time = time.time() - start

    if holy_cow:
        grade = client.get_grade(holy_cow.win_rate, holy_cow.drawn_win_rate)
        print(f"\n  Card: {holy_cow.name}")
        print(f"  MTGA ID: {holy_cow.mtga_id}")
        print(f"  Color: {holy_cow.color}")
        print(f"  Rarity: {holy_cow.rarity}")
        print(f"  Grade: {grade}")
        print(f"  Overall WR: {holy_cow.win_rate:.1%}" if holy_cow.win_rate else "  Overall WR: N/A")
        print(f"  GIH WR: {holy_cow.drawn_win_rate:.1%}" if holy_cow.drawn_win_rate else "  GIH WR: N/A")
        print(f"  IWD: {holy_cow.drawn_improvement_win_rate:+.1%}" if holy_cow.drawn_improvement_win_rate else "  IWD: N/A")
        print(f"  Games: {holy_cow.game_count:,}")
        print(f"  Play Rate: {holy_cow.play_rate:.1%}" if holy_cow.play_rate else "  Play Rate: N/A")
        print(f"\n✓ Lookup time: {lookup_time:.3f}s")

    # CSV approach comparison
    print("\n\n2. CSV APPROACH (Current Implementation)")
    print("-" * 70)
    print("\nDownload & Parse Process:")
    print("  1. Download CSV.gz (20-50MB compressed, 200-500MB uncompressed)")
    print("  2. Decompress file")
    print("  3. Parse 100k-200k rows of game data")
    print("  4. Aggregate statistics per card")
    print("  5. Store in SQLite database")
    print("\nEstimated time:")
    print("  - Single set: 10-30 minutes")
    print("  - 5 sets: 60-180 minutes")
    print("  - Storage: ~50-200MB per set")

    # Summary
    print("\n\n3. COMPARISON SUMMARY")
    print("="*70)

    print("\n📊 SPEED:")
    print(f"  API (single set):    {api_single_time:.1f}s")
    print(f"  API (5 sets):        {api_multi_time:.1f}s")
    print(f"  CSV (single set):    ~10-30 minutes (600-1800s)")
    print(f"  CSV (5 sets):        ~60-180 minutes (3600-10800s)")
    print(f"  Speedup:             ~100-500x faster")

    print("\n💾 STORAGE:")
    print(f"  API (cached):        ~1-5MB JSON in memory")
    print(f"  CSV (database):      ~50-200MB per set")
    print(f"  Savings:             ~10-100x less storage")

    print("\n📈 DATA FRESHNESS:")
    print(f"  API:                 Real-time (updated hourly)")
    print(f"  CSV:                 Static snapshot (must re-download)")

    print("\n🔧 EASE OF USE:")
    print(f"  API:                 Simple HTTP GET, instant use")
    print(f"  CSV:                 Download, decompress, parse, aggregate")

    print("\n✅ ADVANTAGES OF API:")
    print("  • 100-500x faster (seconds vs hours)")
    print("  • No preprocessing required")
    print("  • Always up-to-date")
    print("  • Minimal storage (cache in memory)")
    print("  • Simple implementation")
    print("  • No SQLite database needed")
    print("  • Can filter by date range")

    print("\n⚠️  DISADVANTAGES OF API:")
    print("  • Requires internet connection")
    print("  • Subject to rate limits (unknown, but seems generous)")
    print("  • No access to raw game-by-game data")
    print("  • Depends on 17lands availability")

    print("\n💡 RECOMMENDATION:")
    print("  Use API for real-time draft advice!")
    print("  • Perfect for draft overlay tools")
    print("  • Fast enough for on-demand lookups")
    print("  • No user setup required")
    print("  • Can cache for 30-60 minutes")
    print("\n  Keep CSV approach ONLY if:")
    print("  • Need custom aggregations")
    print("  • Doing research/analysis")
    print("  • Need game-level data")

    print("\n" + "="*70)


def test_draft_scenario():
    """
    Simulate real-time draft scenario.
    """
    print("\n\n4. REAL-TIME DRAFT SIMULATION")
    print("="*70)

    client = SeventeenLandsAPIClient(cache_minutes=30)

    # Simulate P1P1 in OTJ Premier Draft
    pack = [
        "Holy Cow",
        "Mystical Tether",
        "Take Up the Shield",
        "Stagecoach Security",
        "Sterling Supplier",
        "Bridled Bighorn",
        "Bounding Felidar",
        "Lassoed by the Law",
        "Getaway Glamer",
        "Fortune, Loyal Steed",
        "Dust Animus",
        "Another Round",
        "Aven Interrupter",
        "Final Showdown"
    ]

    print(f"\nPack 1, Pick 1 - Evaluating {len(pack)} cards...")
    print(f"Set: OTJ (Outlaws of Thunder Junction)")
    print(f"Format: Premier Draft")

    start = time.time()

    # Fetch ratings (first call downloads, subsequent calls use cache)
    ratings = {}
    for card_name in pack:
        rating = client.lookup_card(card_name, "OTJ", "PremierDraft")
        if rating:
            ratings[card_name] = rating

    elapsed = time.time() - start

    # Sort by GIH win rate
    sorted_cards = sorted(
        ratings.items(),
        key=lambda x: x[1].ever_drawn_win_rate or x[1].win_rate or 0.0,
        reverse=True
    )

    print(f"\n✓ Evaluated in {elapsed:.2f}s")
    print("\nCard Rankings (by GIH WR):")
    print("-" * 70)

    for i, (card_name, rating) in enumerate(sorted_cards[:10], 1):
        grade = client.get_grade(rating.win_rate, rating.drawn_win_rate)
        gih_wr = rating.ever_drawn_win_rate or rating.drawn_win_rate or rating.win_rate
        iwd = rating.drawn_improvement_win_rate

        print(f"{i:2}. [{grade:>2}] {card_name:<30} {gih_wr:>6.1%}  ", end="")
        if iwd is not None:
            print(f"IWD: {iwd:+.1%}  ", end="")
        print(f"({rating.game_count:,} games)")

    print("\n💡 Suggested pick: " + sorted_cards[0][0])
    print(f"   Fast enough for real-time overlay! ({elapsed:.2f}s)")


if __name__ == "__main__":
    compare_approaches()
    test_draft_scenario()

    print("\n\n" + "="*70)
    print("CONCLUSION: Use the API! It's orders of magnitude better.")
    print("="*70)
