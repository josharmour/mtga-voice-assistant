"""
Mana utility functions for Magic: The Gathering simulation.
Handles mana cost parsing, mana source identification, and payment resolution.
"""

import re
from typing import Dict, List, Any, Optional, Set, Tuple

# Regex to find mana symbols in braces: {1}, {U}, {2/W}, etc.
MANA_SYMBOL_RE = re.compile(r"\{([0-9A-Z/]+)\}")

# Regex to find mana production abilities in oracle text.
# Matches "Add " followed by mana symbols.
MANA_PRODUCTION_RE = re.compile(r"Add\s*((?:\{[0-9A-Z/]+\})+)")

def parse_mana_cost(cost_str: str) -> Dict[str, int]:
    """
    Parses a mana cost string (e.g., "{1}{U}{U}") into a dictionary.

    Args:
        cost_str: The mana cost string.

    Returns:
        Dict with keys for colors ('W', 'U', 'B', 'R', 'G', 'C') and 'generic'.
    """
    cost = {
        'W': 0, 'U': 0, 'B': 0, 'R': 0, 'G': 0, 'C': 0,
        'generic': 0
    }

    if not cost_str:
        return cost

    symbols = MANA_SYMBOL_RE.findall(cost_str)

    for symbol in symbols:
        if symbol.isdigit():
            cost['generic'] += int(symbol)
        elif symbol == 'X':
            # X is usually 0 for casting requirements check, or handled elsewhere.
            pass
        elif '/' in symbol:
            # Hybrid mana (e.g., "2/W", "R/G", "G/P")
            parts = symbol.split('/')
            if parts[0] == '2':
                 # {2/W} - specific handling needed.
                 # For now, treat as the colored part for simplicity, or we need a complex cost object.
                 # Let's treat it as the colored requirement as a heuristic for greedy simulator.
                 # A better solver would treat it as (2 Generic OR 1 Color).
                 # To keep simple dict structure, we might need a compromise.
                 # Let's just count the color.
                 if parts[1] in cost:
                     cost[parts[1]] += 1
            elif parts[1] == 'P':
                 # Phyrexian mana: {G/P} -> 1 Green (or 2 life).
                 # Simulator assumes paying mana.
                 if parts[0] in cost:
                     cost[parts[0]] += 1
            else:
                 # Hybrid color/color {R/G}.
                 # This is tricky with simple dict.
                 # Let's count it as 'hybrid': [('R', 'G')]
                 # For now, simplistic approach: pick first color (very greedy/wrong).
                 # IMPROVEMENT: Add a 'hybrid' list to the cost dict.
                 if 'hybrid' not in cost:
                     cost['hybrid'] = []
                 cost['hybrid'].append(set(parts))
        else:
            if symbol in cost:
                cost[symbol] += 1

    return cost

class ManaSource:
    """Represents a permanent's ability to produce mana."""

    def __init__(self, card: Dict[str, Any], tapped: bool = False, entered_turn: int = 0, current_turn: int = 0):
        self.card = card
        self.tapped = tapped
        self.entered_turn = entered_turn
        self.current_turn = current_turn
        self.production_options = self._parse_production()

    def _parse_production(self) -> List[Dict[str, int]]:
        """
        Determines what mana this source can produce.
        Returns a list of options. Each option is a dict of {color: amount}.
        """
        # 1. Check if it's a land (basic or non-basic) with produced_mana
        # Scryfall 'produced_mana' is a list of colors. It doesn't give quantity or combinations.
        # But for basics and simple duals, it's usually "tap for one of these".

        type_line = self.card.get("type_line", "")
        oracle_text = self.card.get("oracle_text", "")
        produced_colors = self.card.get("produced_mana", [])

        options = []

        # Parse explicit "Add {X}" text
        matches = MANA_PRODUCTION_RE.findall(oracle_text)
        if matches:
            for match in matches:
                # match is like "{C}{G}" or "{G}"
                symbols = MANA_SYMBOL_RE.findall(match)
                production = {}
                for s in symbols:
                    if s in ['W', 'U', 'B', 'R', 'G', 'C']:
                        production[s] = production.get(s, 0) + 1
                    elif s == 'Any': # "Add one mana of any color" -> Scryfall doesn't use {Any} symbol usually?
                        pass
                if production:
                    options.append(production)

        # If no explicit "Add" text found (e.g. Basic Lands often have no text in Scryfall Oracle data?),
        # or if we want to trust 'produced_mana' as a fallback for simple tap abilities.
        if not options and produced_colors:
            # Assume it produces 1 of any listed color (choice).
            for color in produced_colors:
                options.append({color: 1})

        # Fallback for Basic Lands if produced_mana is missing (rare but possible in some data subsets)
        if not options and "Land" in type_line:
            if "Forest" in type_line: options.append({'G': 1})
            elif "Island" in type_line: options.append({'U': 1})
            elif "Mountain" in type_line: options.append({'R': 1})
            elif "Swamp" in type_line: options.append({'B': 1})
            elif "Plains" in type_line: options.append({'W': 1})

        return options

    def is_usable(self) -> bool:
        """Check if source can be used (untapped, not summoning sick)."""
        if self.tapped:
            return False

        # Summoning Sickness
        # Creatures cannot tap for abilities with {T} unless they have Haste.
        # We assume mana abilities use {T} (standard).
        if "Creature" in self.card.get("type_line", ""):
            # Check for Haste
            keywords = [k.lower() for k in self.card.get("keywords", [])]
            # Also check oracle text for "Haste" if not in keywords (unlikely for Scryfall)
            if "haste" not in keywords:
                if self.entered_turn == self.current_turn:
                    return False

        return True

def can_pay_cost(cost: Dict[str, Any], sources: List[ManaSource]) -> Tuple[bool, List[ManaSource]]:
    """
    Determines if the cost can be paid by the sources.
    Returns (True, [used_sources]) or (False, []).

    Uses a backtracking algorithm.
    """
    # Filter usable sources
    available_sources = [s for s in sources if s.is_usable()]

    # Extract requirements
    req_colored = {k: v for k, v in cost.items() if k in "WUBRGC" and v > 0}
    req_generic = cost.get("generic", 0)
    req_hybrid = cost.get("hybrid", []) # List of sets

    # Optimization: Check if total mana available < total mana needed
    total_needed = sum(req_colored.values()) + req_generic + len(req_hybrid)
    # Estimate max potential of sources (most produce at least 1)
    if len(available_sources) * 3 < total_needed: # Heuristic factor 3 for crazy rocks
         # If simple check fails (count of sources < needed), quick fail?
         # No, some sources produce 2 or 3. Sol Ring produces 2.
         pass

    solution = []

    def solve(current_req_colored, current_req_generic, current_req_hybrid, used_indices):
        # Base case: All costs satisfied
        if sum(current_req_colored.values()) == 0 and current_req_generic == 0 and not current_req_hybrid:
            return True, []

        # Try to satisfy requirements

        # 1. Satisfy specific colors first (most restrictive)
        target_color = None
        for c in "WUBRGC":
            if current_req_colored.get(c, 0) > 0:
                target_color = c
                break

        if target_color:
            # Find a source that can produce this color
            for i, src in enumerate(available_sources):
                if i in used_indices:
                    continue

                # Check options
                for opt in src.production_options:
                    if opt.get(target_color, 0) > 0:
                        # Use this source for this color
                        # Decrement requirement
                        new_req = current_req_colored.copy()
                        new_req[target_color] -= 1

                        # Handle leftover mana from this source?
                        # Complex: "Add {G}{G}". If we need {G}, we use one {G}, does the other {G} float?
                        # The simulator doesn't handle mana pools/phases yet.
                        # For now, we assume a source is fully consumed for the spell (simplification).
                        # Or, we can subtract the whole production from requirements.

                        # Better approach: Subtract what the source produces from total requirements.
                        # Be careful not to go negative in a way that helps other colors (unless we track pool).
                        # Let's stick to "One source pays for part of the cost".
                        # If source produces {G}{G} and we need {G}, we use the source.
                        # The extra {G} is lost (mana burn is gone, but we don't float it to next spell).
                        # This is a safe conservative assumption for Goldfish.

                        # Subtract everything this option provides from requirements
                        next_req_colored = current_req_colored.copy()
                        next_req_generic = current_req_generic

                        # We apply the full production of this option to the cost
                        satisfied_something = False

                        # Temporary accounting for this option
                        prod = opt.copy()

                        # Pay specific colors
                        for c, amt in prod.items():
                            needed = next_req_colored.get(c, 0)
                            paid = min(needed, amt)
                            next_req_colored[c] -= paid
                            prod[c] -= paid
                            if paid > 0: satisfied_something = True

                        # Pay generic with remaining
                        remaining_mana = sum(prod.values())
                        if next_req_generic > 0 and remaining_mana > 0:
                             paid_gen = min(next_req_generic, remaining_mana)
                             next_req_generic -= paid_gen
                             remaining_mana -= paid_gen
                             satisfied_something = True

                        # If we didn't satisfy the target color specifically, this wasn't a good move for this branch
                        # (We picked this branch because target_color was needed)
                        # Wait, we already checked `opt.get(target_color, 0) > 0`. So we definitely paid at least 1.

                        success, path = solve(next_req_colored, next_req_generic, current_req_hybrid, used_indices | {i})
                        if success:
                            return True, [src] + path

            return False, []

        # 2. Satisfy Hybrid
        if current_req_hybrid:
            next_hybrid = current_req_hybrid[1:]
            options_needed = current_req_hybrid[0] # Set of colors e.g. {'R', 'G'}

            for i, src in enumerate(available_sources):
                if i in used_indices: continue

                for opt in src.production_options:
                    # Check if option provides any of the needed colors
                    provides = set(opt.keys())
                    overlap = provides.intersection(options_needed)
                    if overlap:
                        # Use it
                         # (Logic for leftover mana similar to above, simplified here)
                        success, path = solve(current_req_colored, current_req_generic, next_hybrid, used_indices | {i})
                        if success:
                            return True, [src] + path
            return False, []

        # 3. Satisfy Generic
        if current_req_generic > 0:
            # Use any remaining source
            for i, src in enumerate(available_sources):
                if i in used_indices: continue

                # Any option produces mana?
                for opt in src.production_options:
                    if sum(opt.values()) > 0:
                        # Assuming source produces at least 1 mana
                        amt = sum(opt.values())
                        new_gen = max(0, current_req_generic - amt)
                        success, path = solve(current_req_colored, new_gen, current_req_hybrid, used_indices | {i})
                        if success:
                            return True, [src] + path
            return False, []

        return False, []

    success, sources_used = solve(req_colored, req_generic, req_hybrid, set())
    return success, sources_used
