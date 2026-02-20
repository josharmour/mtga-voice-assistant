# Unified ArenaMCP Engine
# Supports both direct and lazy imports for robustness

from .parser import LogParser
from .watcher import MTGALogWatcher
from .gamestate import GameState, create_game_state_handler
from .draftstate import DraftState, create_draft_handler
from .coach import CoachEngine, fetch_proxy_models
from .rules_engine import RulesEngine
from .scryfall import ScryfallCache
from .mtgadb import MTGADatabase
from .synergy import SynergyGraph, get_synergy_graph

# Lazy loaders for new app.py architecture
def get_log_parser(): return LogParser
def get_game_state(): return GameState
def get_coach_engine(): return CoachEngine
def get_rules_engine(): return RulesEngine
def get_draft_state(): return DraftState
def get_create_draft_handler(): return create_draft_handler
def get_create_game_state_handler(): return create_game_state_handler
def get_scryfall_cache(): return ScryfallCache
def get_mtga_database(): return MTGADatabase
def get_fetch_proxy_models(): return fetch_proxy_models
