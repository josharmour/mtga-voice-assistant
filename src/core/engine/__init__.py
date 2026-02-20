# Unified ArenaMCP Engine
# Lazy loaders to avoid slow startup

def get_log_parser():
    from .parser import LogParser
    return LogParser

def get_game_state():
    from .gamestate import GameState
    return GameState

def get_coach_engine():
    from .coach import CoachEngine
    return CoachEngine

def get_rules_engine():
    from .rules_engine import RulesEngine
    return RulesEngine

def get_draft_state():
    from .draftstate import DraftState
    return DraftState

def get_create_draft_handler():
    from .draftstate import create_draft_handler
    return create_draft_handler

def get_create_game_state_handler():
    from .gamestate import create_game_state_handler
    return create_game_state_handler

def get_scryfall_cache():
    from .scryfall import ScryfallCache
    return ScryfallCache

def get_mtga_database():
    from .mtgadb import MTGADatabase
    return MTGADatabase

def get_fetch_proxy_models():
    from .coach import fetch_proxy_models
    return fetch_proxy_models
