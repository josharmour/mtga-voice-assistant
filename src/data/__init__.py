"""Data layer for The Planeswalker Agent."""

def get_scryfall_loader():
    from .scryfall import ScryfallLoader
    return ScryfallLoader

def get_vector_store():
    from .chroma import VectorStore
    return VectorStore

def get_edhrec_client():
    from .edhrec import EDHRECClient
    return EDHRECClient

def get_seventeenlands_client():
    from .seventeenlands import SeventeenLandsClient
    return SeventeenLandsClient

__all__ = [
    "get_scryfall_loader",
    "get_vector_store",
    "get_edhrec_client",
    "get_seventeenlands_client",
]
