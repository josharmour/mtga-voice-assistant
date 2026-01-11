"""Build the synergy graph from card database."""

import sys
import argparse
from src.data.scryfall import ScryfallLoader
from src.cognitive import SynergyGraph


def main():
    """Build synergy graph from ingested card data."""
    parser = argparse.ArgumentParser(description="Build card synergy graph.")
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of cards to process (for testing)"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild graph even if cache exists"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SYNERGY GRAPH BUILDER")
    print("=" * 60)
    print()

    # Initialize synergy graph
    graph = SynergyGraph()

    # Try to load existing graph
    if not args.rebuild and graph.load():
        print("\nSynergy graph loaded from cache.")
        stats = graph.stats()
        print(f"\nGraph Statistics:")
        print(f"  Cards: {stats['num_cards']}")
        print(f"  Synergies: {stats['num_synergies']}")
        print(f"  Avg synergies per card: {stats['avg_synergies_per_card']:.1f}")
        print(f"  Graph density: {stats['density']:.4f}")
        print("\nUse --rebuild to rebuild the graph.")
        return

    # Load card data
    print("Loading card data...")
    loader = ScryfallLoader()

    try:
        cards = loader.load_cards(limit=args.limit)
    except FileNotFoundError:
        print("Error: Card data not found.")
        print("Please run 'python ingest_data.py' first to download card data.")
        sys.exit(1)

    print(f"Loaded {len(cards)} cards")
    print()

    # Add cards to graph
    print("Adding cards to synergy graph...")
    for i, card in enumerate(cards):
        if i % 100 == 0 and i > 0:
            print(f"  Processed {i}/{len(cards)} cards...")
        graph.add_card(card)

    print(f"✓ Added {len(cards)} cards to graph")
    print()

    # Build synergies
    print("Analyzing card relationships...")
    graph.build_synergies()
    print("✓ Synergy analysis complete")
    print()

    # Display statistics
    stats = graph.stats()
    print("Graph Statistics:")
    print(f"  Cards: {stats['num_cards']}")
    print(f"  Synergies: {stats['num_synergies']}")
    print(f"  Avg synergies per card: {stats['avg_synergies_per_card']:.1f}")
    print(f"  Graph density: {stats['density']:.4f}")
    print()

    # Save graph
    print("Saving synergy graph...")
    graph.save()
    print("✓ Build complete!")


if __name__ == "__main__":
    main()
