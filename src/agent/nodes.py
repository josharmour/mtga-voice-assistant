from typing import Dict, Any, List
from src.agent.state import AgentState
from src.data.chroma import get_vector_store
from src.data.edhrec import EDHRECClient
from src.data.seventeenlands import SeventeenLandsClient
from src.data.mtggoldfish import MTGGoldfishClient
from src.cognitive import get_synergy_graph


def router_node(state: AgentState) -> AgentState:
    """
    Route the query to the appropriate metagame source.

    Classifies queries as:
    - "limited": Draft, Sealed -> 17Lands
    - "commander": Commander/EDH -> EDHREC
    - "standard", "modern", "pioneer", "legacy", "pauper" -> MTGGoldfish
    - "constructed": Generic constructed -> Default to Commander (EDHREC)
    """
    query = state["user_query"].lower()

    # Keywords for Limited format
    limited_keywords = [
        "draft",
        "sealed",
        "limited",
        "pick",
        "pack",
        "color pair",
        "archetype",
        "premier draft",
        "quick draft",
        "arena",
        "mtga",
    ]

    # Explicit format keywords
    formats = {
        "standard": ["standard"],
        "modern": ["modern"],
        "pioneer": ["pioneer"],
        "legacy": ["legacy"],
        "pauper": ["pauper"],
        "commander": ["commander", "edh"],
        "limited": limited_keywords,
    }

    # Check for specific format mentions first
    query_type = "constructed"  # Default

    # Check simple formats first
    for fmt, keywords in formats.items():
        if any(kw in query for kw in keywords):
            query_type = fmt
            break

    print(f"[Router] Classified query as: {query_type.upper()}")

    state["query_type"] = query_type
    if "metadata" not in state:
        state["metadata"] = {}

    return state


def oracle_node(state: AgentState) -> AgentState:
    """
    Perform semantic card search using ChromaDB.
    Detects set-based queries and filters appropriately.
    Also extracts card names from the tracked board state for exact lookups.
    """
    import re
    query = state["user_query"]
    query_lower = query.lower()
    print(f"[Oracle] Processing query context...")

    try:
        store = get_vector_store()

        # 1. Extract detailed entities from the prompt (Hand, Board, etc.)
        # The Advisor formats lines like "My Hand: Card A, Card B"
        extracted_names = []
        patterns = [
            r"My Hand: (.*?)(?:\n|$)",
            r"My Board: (.*?)(?:\n|$)",
            r"Opponent Board: (.*?)(?:\n|$)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                # Split by comma, strip whitespace, ignore "Empty"
                items = [item.strip() for item in match.group(1).split(',')]
                extracted_names.extend([i for i in items if i and i != "Empty"])

        # Deduplicate
        extracted_names = list(set(extracted_names))
        
        exact_results = []
        if extracted_names:
            print(f"[Oracle] Identifying cards from board state: {extracted_names}")
            exact_results = store.get_cards_by_names(extracted_names)
            print(f"[Oracle] Found {len(exact_results)} exact matches for board state cards")

        # 2. Detect set-based queries
        set_mappings = {
            "avatar": "tla",
            "avatar the last airbender": "tla",
            "last airbender": "tla",
            "atla": "tla",
            "tla": "tla",
            "murders at karlov manor": "mkm",
            "karlov manor": "mkm",
            "mkm": "mkm",
            "lost caverns": "lci",
            "ixalan": "lci",
            "lci": "lci",
            "wilds of eldraine": "woe",
            "eldraine": "woe",
            "woe": "woe",
            "lord of the rings": "ltr",
            "lotr": "ltr",
            "ltr": "ltr",
            "march of the machine": "mom",
            "mom": "mom",
            "foundations": "fdn",
            "fdn": "fdn",
        }

        detected_set = None
        for set_name, set_code in set_mappings.items():
            if set_name in query_lower:
                detected_set = set_code
                print(f"[Oracle] Detected set query: {set_code.upper()}")
                break

        # 3. Perform Semantic Search (for the user's specific question part)
        # We try to isolate the 'User Question' or just search the whole thing if short
        search_query = query
        user_q_match = re.search(r"User Question: (.*)", query, re.DOTALL)
        if user_q_match:
            search_query = user_q_match.group(1).strip()
        
        semantic_results = []
        if detected_set:
            # For set queries, get cards from that set
            results = store.query_by_set(detected_set, n_results=10)
            if results and "ids" in results and results["ids"]:
                ids = results["ids"]
                documents = results.get("documents", [])
                metadatas = results.get("metadatas", [])

                for i in range(len(ids)):
                    semantic_results.append({
                        "id": ids[i],
                        "name": metadatas[i].get("name", "Unknown") if metadatas else "Unknown",
                        "type_line": metadatas[i].get("type_line", "") if metadatas else "",
                        "text": documents[i] if documents else "",
                        "metadata": metadatas[i] if metadatas else {},
                    })
        else:
            # Standard semantic search on the specific question part
            # If the query is just the board state dump, this might return random stuff, 
            # but exact_results should cover the board.
            results = store.query_similar(search_query, n_results=5)
            if results and "ids" in results and results["ids"] and len(results["ids"]) > 0:
                ids = results["ids"][0]
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]

                for i in range(len(ids)):
                    semantic_results.append({
                        "id": ids[i],
                        "name": metadatas[i].get("name", "Unknown"),
                        "type_line": metadatas[i].get("type_line", ""),
                        "text": documents[i],
                        "metadata": metadatas[i],
                    })

        # 4. Merge Results (Exact matches first, then Semantic)
        # Use a dict by ID to deduplicate
        merged_map = {}
        for card in exact_results:
            merged_map[card["id"]] = card
        
        for card in semantic_results:
            if card["id"] not in merged_map:
                merged_map[card["id"]] = card
        
        final_results = list(merged_map.values())

        if final_results:
            print(f"[Oracle] Providing {len(final_results)} relevant cards (Exact: {len(exact_results)}, Semantic: {len(semantic_results)})")
        else:
            print("[Oracle] No cards found")

        state["oracle_results"] = final_results

    except Exception as e:
        print(f"[Oracle] Error: {e}")
        state["oracle_results"] = []

    return state


def constructed_metagame_node(state: AgentState) -> AgentState:
    """
    Fetch Constructed metagame data.

    Supports:
    - Commander -> EDHREC
    - Standard, Modern, Pioneer -> MTGGoldfish
    """
    query_type = state.get("query_type", "constructed")
    user_query = state.get("user_query", "").lower()

    # If limited, skip
    if query_type == "limited":
        return state

    print(f"[{query_type.title()}] Fetching metagame data...")
    state["metagame_results"] = {}

    try:
        # CASE 1: Commander / Generic Constructed -> EDHREC
        if query_type in ["commander", "constructed"]:
            client = EDHRECClient()

            # Check if query mentions a specific commander
            oracle_results = state.get("oracle_results", [])
            commanders_found = []

            for card in oracle_results[:3]:
                type_line = card.get("type_line", "").lower()
                if "legendary" in type_line and "creature" in type_line:
                    commanders_found.append(card["name"])

            if commanders_found:
                print(f"[Commander] Found potential commanders: {commanders_found}")
                commander_name = commanders_found[0]
                commander_data = client.get_commander_page(commander_name)
                state["metagame_results"]["commander_recommendations"] = commander_data
            else:
                top_commanders = client.get_top_commanders(timeframe="week")
                state["metagame_results"]["top_commanders"] = top_commanders[:10]

        # CASE 2: Competitive Formats -> MTGGoldfish
        elif query_type in ["standard", "modern", "pioneer", "legacy", "pauper"]:
            client = MTGGoldfishClient()
            decks = client.get_metagame(query_type)
            state["metagame_results"]["top_decks"] = decks
            print(f"[{query_type.title()}] Found {len(decks)} top decks")

            # --- CONTEXT AWARENESS FOR DECKS ---
            target_deck = None

            # Scenario A: "What is the best deck?"
            if "best" in user_query or "top" in user_query:
                if decks:
                    target_deck = decks[0]
                    print(
                        f"[{query_type.title()}] 'Best' deck requested. Targeting: {target_deck['name']}"
                    )

            # Scenario B: "Tell me about [Deck Name]"
            # We check if any deck name from our results is in the user query
            else:
                for deck in decks:
                    # Simple fuzzy matching: check if prominent parts of deck name appear in query
                    # e.g., "Dimir Midrange" -> check "dimir" and "midrange" or exact string
                    clean_name = deck["name"].lower()
                    if clean_name in user_query:
                        target_deck = deck
                        print(
                            f"[{query_type.title()}] Specific deck requested: {deck['name']}"
                        )
                        break

            # Fetch Decklist if we have a target
            if target_deck and "url" in target_deck:
                print(
                    f"[{query_type.title()}] Fetching deck list for {target_deck['name']}..."
                )
                deck_list = client.get_deck_list(target_deck["url"])
                state["metagame_results"]["focus_deck"] = {
                    "info": target_deck,
                    "list": deck_list,
                }

    except Exception as e:
        print(f"[{query_type.title()}] Error: {e}")
        state["metagame_results"] = {"error": str(e)}

    return state


def limited_metagame_node(state: AgentState) -> AgentState:
    """
    Fetch Limited (Draft/Sealed) metagame data from 17Lands.
    """
    if state.get("query_type") != "limited":
        return state

    print("[Limited] Fetching 17Lands data...")

    try:
        client = SeventeenLandsClient()
        query = state["user_query"].upper()
        # Simple expansion detection
        set_codes = ["MKM", "LCI", "WOE", "LTR", "MOM"]
        expansion = next((code for code in set_codes if code in query), "MKM")

        metagame_data = {}
        metagame_data["color_pairs"] = client.get_color_pair_data(
            expansion=expansion, format_type="PremierDraft"
        )
        metagame_data["set_stats"] = client.get_set_stats(
            expansion=expansion, format_type="PremierDraft"
        )

        state["metagame_results"] = metagame_data

    except Exception as e:
        print(f"[Limited] Error: {e}")
        state["metagame_results"] = {"error": str(e)}

    return state


def synergy_node(state: AgentState) -> AgentState:
    """
    Find card synergies using the synergy graph.
    """
    oracle_results = state.get("oracle_results", [])

    if not oracle_results:
        state["synergy_results"] = None
        return state

    print("[Synergy] Analyzing card interactions...")

    try:
        graph = get_synergy_graph()
        if graph is None:
            print("[Synergy] Synergy graph not available.")
            state["synergy_results"] = None
            return state

        synergy_results = {}
        card_names = [card["name"] for card in oracle_results[:3]]

        for card_name in card_names:
            synergies = graph.find_synergies_for_card(card_name, top_n=5)
            if synergies:
                synergy_results[card_name] = [
                    {"card": syn_card, "score": score, "types": syn_types}
                    for syn_card, score, syn_types in synergies
                ]

        state["synergy_results"] = synergy_results

    except Exception as e:
        print(f"[Synergy] Error: {e}")
        state["synergy_results"] = None

    return state


def synthesizer_node(state: AgentState) -> AgentState:
    """
    Synthesize results into a final response.
    """
    print("[Synthesizer] Combining results...")

    query = state["user_query"]
    query_type = state.get("query_type", "unknown")
    oracle_results = state.get("oracle_results", [])
    synergy_results = state.get("synergy_results", {})
    metagame_results = state.get("metagame_results", {})

    response_parts = []
    # response_parts.append(f"Query: {query}") # Removed for Voice
    # response_parts.append(f"Type: {query_type.title()}") # Removed for Voice
    # response_parts.append("")

    # Check if we have a "Focus Deck" (Detailed View)
    focus_deck = metagame_results.get("focus_deck")

    # 1. Oracle Results (Context)
    if oracle_results and not focus_deck:
        response_parts.append("I found these relevant cards:")
        for i, card in enumerate(oracle_results[:3], 1):
            response_parts.append(f"{card['name']} ({card['type_line']})")
        response_parts.append("")

    # 2. Metagame Results (The Meat)
    if metagame_results and "error" not in metagame_results:
        # COMMANDER RECS
        if "commander_recommendations" in metagame_results:
            cmd_data = metagame_results["commander_recommendations"]
            response_parts.append(
                f"Commander Recommendation: {cmd_data.get('commander', 'Unknown')}"
            )

            themes = cmd_data.get("themes", [])
            if themes:
                theme_names = [t.get('name', str(t)) if isinstance(t, dict) else str(t) for t in themes[:3]]
                response_parts.append(f"Themes include: {', '.join(theme_names)}")

            cards = cmd_data.get("cards", [])
            if cards:
                response_parts.append("Top Synergies:")
                for card in cards[:5]:
                    response_parts.append(f"  {card.get('name', 'Unknown')}")

        # TOP COMMANDERS
        elif "top_commanders" in metagame_results:
            response_parts.append("Trending Commanders:")
            for cmd in metagame_results["top_commanders"][:5]:
                response_parts.append(f"  {cmd.get('name', 'Unknown')}")

        # FOCUS DECK (Detailed List)
        elif focus_deck:
            info = focus_deck.get("info", {})
            deck_list = focus_deck.get("list", {})

            response_parts.append(f"Deck Focus: {info.get('name', 'Unknown')}")
            response_parts.append(f"Meta Share: {info.get('meta_share', 'N/A')}")

            response_parts.append("Mainboard highlights:")
            if deck_list.get("mainboard"):
                for line in deck_list["mainboard"][:20]:  # Limit to top 20 lines for brevity in UI
                    response_parts.append(f"  {line}")

        # META DECKS LIST (General View)
        elif "top_decks" in metagame_results:
            response_parts.append(f"Top {query_type.title()} Decks:")
            decks = metagame_results["top_decks"]
            if decks:
                for i, deck in enumerate(decks[:5], 1): # Limit to 5 for voice
                    share = deck.get("meta_share", "").replace("\n", "").strip()
                    response_parts.append(f"{deck['name']} ({share})")
            else:
                response_parts.append("No active meta decks found.")

        # LIMITED
        elif "color_pairs" in metagame_results:
            response_parts.append("Limited Win Rates:")
            pairs = sorted(
                metagame_results["color_pairs"],
                key=lambda x: x.get("win_rate", 0),
                reverse=True,
            )
            for pair in pairs[:5]:
                response_parts.append(
                    f"  {pair['colors']}: {pair.get('win_rate', 0):.1%}"
                )

    # 3. Synergy (Bonus)
    if synergy_results:
        response_parts.append("Card Interactions:")
        for card_name, synergies in synergy_results.items():
            if synergies:
                response_parts.append(f"For {card_name}:")
                for syn in synergies[:2]:
                    response_parts.append(f"  combos with {syn['card']}")
        response_parts.append("")

    # response_parts.append("---")
    # response_parts.append(f"Powered by: Scryfall, {query_type.title()} Sources")

    state["final_response"] = "\n".join(response_parts)
    print("[Synthesizer] Response generated")

    return state
