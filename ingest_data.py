import sys
import argparse
from src.data.scryfall import ScryfallLoader
from src.data.chroma import VectorStore

def main():
    parser = argparse.ArgumentParser(description="Ingest Magic: The Gathering card data.")
    parser.add_argument("--limit", type=int, help="Limit the number of cards to process (for testing).")
    parser.add_argument("--force-download", action="store_true", help="Force re-download of Scryfall data.")
    args = parser.parse_args()

    # 1. Load Data
    print("Initializing Scryfall Loader...")
    loader = ScryfallLoader()
    try:
        print("Fetching card data...")
        loader.fetch_data(force_download=args.force_download)
        
        print(f"Loading cards (Limit: {args.limit})...")
        cards = loader.load_cards(limit=args.limit)
        print(f"Loaded {len(cards)} cards.")
    except Exception as e:
        print(f"Failed to load data: {e}")
        sys.exit(1)

    # 2. Vector Database
    try:
        print("Initializing Vector Store...")
        store = VectorStore()
        print("Upserting cards into ChromaDB...")
        store.upsert_cards(cards)
    except Exception as e:
        print(f"Failed to ingest into Vector DB: {e}")
        sys.exit(1)

    print("Ingestion complete.")

if __name__ == "__main__":
    main()
