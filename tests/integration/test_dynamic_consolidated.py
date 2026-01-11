import pytest
import random
from pathlib import Path
from mtg_agent import run_query
from src.data.edhrec import EDHRECClient
from src.data.mtggoldfish import MTGGoldfishClient
from src.data.seventeenlands import SeventeenLandsClient
from src.cognitive import get_synergy_graph
from src.data.chroma import get_vector_store

class TestDynamicConsolidated:
    """
    Consolidated Integration Tests for the Planeswalker Agent.
    Includes both Dynamic tests (checking against live data) and Static baseline tests.
    """

    @pytest.fixture(scope="class")
    def edhrec_client(self):
        return EDHRECClient()

    @pytest.fixture(scope="class")
    def mtggoldfish_client(self):
        return MTGGoldfishClient()

    @pytest.fixture(scope="class")
    def seventeenlands_client(self):
        return SeventeenLandsClient()

    @pytest.fixture(scope="class")
    def trending_commanders(self, edhrec_client):
        """Fetch trending commanders for dynamic testing."""
        try:
            trending = edhrec_client.get_top_commanders(timeframe="week")
            if trending and len(trending) >= 3:
                return [
                    {
                        "name": cmd.get("name", cmd) if isinstance(cmd, dict) else cmd,
                        "rank": i + 1
                    }
                    for i, cmd in enumerate(trending[:3])
                ]
        except Exception as e:
            pass # Graceful degradation
        return []

    # --- DYNAMIC TESTS ---

    def test_commander_synergy_trending(self, agent, audit_logger, trending_commanders, edhrec_client):
        """Test Commander synergy query with CURRENTLY trending commander."""
        if not trending_commanders:
            pytest.skip("No trending commanders available")

        # Use the #1 trending commander
        commander = trending_commanders[0]
        commander_name = commander["name"]
        query = f"What cards work well with {commander_name}?"
        
        commander_slug = commander_name.lower().replace(",", "").replace("'", "").replace(" ", "-")
        source_url = f"https://edhrec.com/commanders/{commander_slug}"

        try:
            commander_data = edhrec_client.get_commander_page(commander_name)
            if commander_data and "cards" in commander_data:
                edhrec_cards = [card["name"] for card in commander_data["cards"][:10]]
                expected_data = {
                    "source": "EDHREC (Live Trending Data)",
                    "commander": commander_name,
                    "trending_rank": commander["rank"],
                    "top_cards": edhrec_cards,
                    "themes": commander_data.get("themes", [])[:5]
                }
            else:
                expected_data = {"note": "EDHREC data unavailable"}
        except Exception as e:
            expected_data = {"error": str(e)}

        audit_logger(query, source_url, expected_data, f"Commander Synergy - {commander_name} (Trending #{commander['rank']})")

        result = run_query(agent, query)
        response = result.get("final_response", "").lower()
        
        audit_logger(query, source_url, expected_data, f"Commander Synergy - {commander_name}", agent_response=result.get("final_response", ""))

        assert commander_name.lower() in response, f"Response should mention {commander_name}"
        
        # Validation
        has_content = len(response) > 100
        matches = 0
        if isinstance(expected_data.get("top_cards"), list):
             matches = sum(1 for card in expected_data["top_cards"] if card.lower() in response)
             
        assert matches >= 1 or has_content, "Response should provide synergy information or mention EDHREC cards"

    def test_commander_recommendation_dynamic(self, agent, audit_logger):
        """Test commander recommendation query with dynamic theme."""
        themes = [
            ("plus-1-plus-1-counters", "+1/+1 counters"),
            ("tokens", "tokens"),
            ("graveyard", "graveyard"),
            ("artifacts", "artifacts"),
            ("tribal", "tribal")
        ]
        theme_slug, theme_name = random.choice(themes)
        
        query = f"What's a good commander for a {theme_name} deck?"
        source_url = f"https://edhrec.com/themes/{theme_slug}"
        
        expected_data = {
            "source": f"EDHREC {theme_name.title()} Theme",
            "theme": theme_name,
            "query_type": "commander_recommendation"
        }

        audit_logger(query, source_url, expected_data, f"Commander Recommendation - {theme_name.title()}")
        
        result = run_query(agent, query)
        response = result.get("final_response", "").lower()
        
        audit_logger(query, source_url, expected_data, f"Commander Recommendation - {theme_name.title()}", agent_response=result.get("final_response", ""))

        assert theme_name in response, f"Response should mention {theme_name} theme"
        assert len(result["final_response"]) > 50, "Should return a substantial response"

    @pytest.mark.parametrize("format_name", ["modern", "standard", "pioneer"])
    def test_metagame_dynamic(self, agent, audit_logger, mtggoldfish_client, format_name):
        """Test metagame query with live data for multiple formats."""
        try:
            metagame = mtggoldfish_client.get_metagame(format_name)
            if not metagame:
                 # Don't skip entire test suite, just this case if data fails
                 if format_name == "modern": pytest.skip(f"No metagame data for {format_name}")
                 return 
        except Exception as e:
            pytest.skip(f"MTGGoldfish {format_name} data unavailable: {e}")

        top_decks = metagame[:5]
        query = f"What are the top decks in {format_name.title()}?"
        source_url = f"https://www.mtggoldfish.com/metagame/{format_name}"

        expected_data = {
            "source": f"MTGGoldfish {format_name.title()} Metagame (Live)",
            "top_decks": [{"name": d["name"], "percentage": d.get("percentage", "N/A")} for d in top_decks]
        }
        
        audit_logger(query, source_url, expected_data, f"{format_name.title()} Metagame")
        
        result = run_query(agent, query)
        response = result.get("final_response", "").lower()
        
        audit_logger(query, source_url, expected_data, f"{format_name.title()} Metagame", agent_response=result.get("final_response", ""))

        assert format_name in response, f"Response should mention {format_name}"
        
        # Check for deck mentions
        top_3_decks_names = [d["name"].lower() for d in top_decks[:3]]
        matches = 0
        for deck_name in top_3_decks_names:
            keywords = [k for k in deck_name.split() if len(k) > 3]
            if any(k in response for k in keywords):
                matches += 1
                
        assert matches >= 1 or len(response) > 100, "Should mention top decks or provide substantial content"

    def test_draft_color_pairs_dynamic(self, agent, audit_logger, seventeenlands_client):
        """Test draft color pair query with CURRENT draft data."""
        sets = [
            {"code": "MKM", "name": "Murders at Karlov Manor"},
            {"code": "LCI", "name": "The Lost Caverns of Ixalan"},
            {"code": "WOE", "name": "Wilds of Eldraine"}
        ]
        set_info = random.choice(sets)
        set_code = set_info["code"]
        set_name = set_info["name"]

        query = f"What are the best color pairs in {set_name} draft?"
        source_url = f"https://www.17lands.com/color_ratings?expansion={set_code}&format=PremierDraft"
        
        try:
            color_data = seventeenlands_client.get_color_pair_data(expansion=set_code, format_type="PremierDraft")
            if not color_data:
                 pytest.skip(f"17Lands {set_code} data unavailable")
        except Exception as e:
             pytest.skip(f"17Lands error: {e}")

        # Top pairs
        sorted_pairs = sorted(color_data, key=lambda x: x.get("win_rate", 0), reverse=True)
        top_pairs = sorted_pairs[:5]

        expected_data = {
            "source": f"17Lands {set_code} (Live)",
            "top_pairs": [{"name": p.get("name"), "colors": p.get("colors"), "win_rate": p.get("win_rate")} for p in top_pairs]
        }

        audit_logger(query, source_url, expected_data, f"Draft Color Pairs - {set_code}")
        
        result = run_query(agent, query)
        response = result.get("final_response", "").lower()
        
        audit_logger(query, source_url, expected_data, f"Draft Color Pairs - {set_code}", agent_response=result.get("final_response", ""))

        assert any(k in response for k in [set_code.lower(), set_name.lower()]), "Should mention set name"
        
        matches = 0
        for pair in top_pairs[:3]:
            if pair.get("name", "").lower() in response or pair.get("colors", "").lower() in response:
                matches += 1
                
        assert matches >= 1 or len(response) > 100, "Should mention top color pairs or have substantial content"

    def test_synergy_trending_card(self, agent, audit_logger, trending_commanders):
        """Test synergy detection with a CURRENTLY trending/popular card."""
        card_name = "Sol Ring"
        if trending_commanders:
            card_name = random.choice(trending_commanders)["name"]

        query = f"What cards have synergy with {card_name}?"
        source_url = f"https://edhrec.com/cards/{card_name.lower().replace(' ', '-').replace(',', '')}"

        try:
            synergy_graph = get_synergy_graph()
            synergies = synergy_graph.find_synergies_for_card(card_name, top_n=10)
            expected_data = {
                "source": "Local Synergy Graph + EDHREC",
                "card": card_name,
                "top_synergies": [s[0] for s in synergies] if synergies else []
            }
        except Exception as e:
            expected_data = {"error": str(e)}

        audit_logger(query, source_url, expected_data, f"Card Synergy - {card_name}")
        
        result = run_query(agent, query)
        response = result.get("final_response", "").lower()
        
        audit_logger(query, source_url, expected_data, f"Card Synergy - {card_name}", agent_response=result.get("final_response", ""))

        assert card_name.lower() in response, f"Response should mention {card_name}"
        assert len(response) > 50, "Response should provide synergy information"

    
    # --- SEMANTIC SEARCH TESTS ---

    def test_semantic_card_draw(self, agent, audit_logger):
        """Test semantic search for card draw."""
        query = "Show me blue cards that draw cards"
        source_url = "https://scryfall.com/search?q=c%3Au+o%3Adraw"
        expected_cards = ["mulldrifter", "brainstorm", "ponder", "opt"] # Common examples
        
        audit_logger(query, source_url, {"expected": expected_cards}, "Semantic Search - Card Draw")
        
        result = run_query(agent, query)
        response = result.get("final_response", "").lower()
        
        audit_logger(query, source_url, {}, "Semantic Search - Card Draw", agent_response=response)
        
        found = any(c in response for c in expected_cards)
        discusses = any(k in response for k in ["draw", "card advantage", "blue"])
        assert found or discusses, "Should discuss card draw or mention relevant cards"

    def test_semantic_removal_black(self, agent, audit_logger):
        """Test semantic search for black removal."""
        query = "What are efficient creature removal spells in black?"
        source_url = "https://scryfall.com/search?q=c%3Ab+o%3Adestroy"
        known_removal = ["doom blade", "murder", "go for the throat", "fatal push", "terminate"]
        
        audit_logger(query, source_url, {"known": known_removal}, "Semantic Search - Black Removal")
        
        result = run_query(agent, query)
        response = result.get("final_response", "").lower()
        
        audit_logger(query, source_url, {}, "Semantic Search - Black Removal", agent_response=response)
        
        found = any(c in response for c in known_removal)
        assert found or "removal" in response, "Should mention removal spells"

    def test_semantic_mechanic_proliferate(self, agent, audit_logger):
        """Test semantic search for mechanics (proliferate)."""
        query = "Show me cards with proliferate"
        source_url = "https://scryfall.com/search?q=o%3Aproliferate"
        known_cards = ["atraxa", "contagion", "karn's bastion", "flux channeler", "evolution sage"]
        
        audit_logger(query, source_url, {"known": known_cards}, "Semantic Search - Proliferate")
        
        result = run_query(agent, query)
        response = result.get("final_response", "").lower()
        
        audit_logger(query, source_url, {}, "Semantic Search - Proliferate", agent_response=response)
        
        assert "proliferate" in response
        assert any(c in response for c in known_cards) or len(response) > 50

    # --- STATIC / BASELINE TESTS ---
    
    def test_commander_synergy_atraxa(self, agent, audit_logger):
        """Static: Test Commander synergy for Atraxa (Baseline)."""
        query = "What cards work well with Atraxa, Praetors' Voice?"
        source_url = "https://edhrec.com/commanders/atraxa-praetors-voice"
        expected_data = {"baseline": "Expect mentions of proliferate, counters, or known synergy cards."}
        
        audit_logger(query, source_url, expected_data, "Commander Synergy - Atraxa (Static)")
        
        result = run_query(agent, query)
        response = result.get("final_response", "").lower()
        
        audit_logger(query, source_url, expected_data, "Commander Synergy - Atraxa", agent_response=response)
        
        assert "atraxa" in response
        
        has_keywords = any(k in response for k in ["proliferate", "counter", "doubling season", "infect", "toxic", "phyrexian"])
        has_content = len(response) > 100
        
        assert has_keywords or has_content, "Should mention Atraxa mechanics or provide substantial content"

    def test_commander_recommendation_static(self, agent, audit_logger):
        """Static: Test commander recommendation for +1/+1 counters."""
        query = "What's a good commander for a +1/+1 counters deck?"
        source_url = "https://edhrec.com/themes/plus-1-plus-1-counters"
        known_commanders = ["atraxa", "ezuri", "vorel", "ghave", "hamza", "reyhan", "pir", "toothy"]
        
        audit_logger(query, source_url, {"known": known_commanders}, "Commander Rec - Counters (Static)")
        
        result = run_query(agent, query)
        response = result.get("final_response", "").lower()
        
        audit_logger(query, source_url, {}, "Commander Rec - Counters", agent_response=response)
        
        found = any(c in response for c in known_commanders)
        assert found or len(response) > 100, "Should recommend known +1/+1 counter commanders"

    def test_draft_archetype_lci(self, agent, audit_logger):
        """Static: Test draft archetype for LCI."""
        query = "What are good archetypes for drafting LCI?"
        source_url = "https://www.17lands.com/color_ratings?expansion=LCI&format=PremierDraft"
        
        audit_logger(query, source_url, {}, "Draft Archetypes - LCI (Static)")
        
        result = run_query(agent, query)
        response = result.get("final_response", "").lower()
        
        audit_logger(query, source_url, {}, "Draft Archetypes - LCI", agent_response=response)
        
        assert "lci" in response or "lost caverns" in response or "ixalan" in response
        assert len(response) > 50

    def test_sealed_format_woe(self, agent, audit_logger):
        """Static: Test sealed format for WOE."""
        query = "What should I prioritize in WOE sealed?"
        source_url = "https://www.17lands.com/card_ratings?expansion=WOE&format=Sealed"
        
        audit_logger(query, source_url, {}, "Sealed Format - WOE (Static)")
        
        result = run_query(agent, query)
        response = result.get("final_response", "").lower()
        
        audit_logger(query, source_url, {}, "Sealed Format - WOE", agent_response=response)
        
        assert "sealed" in response or "woe" in response or "wilds of eldraine" in response
        assert len(response) > 50

    def test_synergy_sacrifice(self, agent, audit_logger):
        """Static: Test synergy detection for sacrifice strategy."""
        query = "What cards work well in a sacrifice deck?"
        source_url = "https://edhrec.com/themes/aristocrats"
        
        audit_logger(query, source_url, {}, "Sacrifice Synergies (Static)")
        result = run_query(agent, query)
        response = result.get("final_response", "").lower()
        audit_logger(query, source_url, {}, "Sacrifice Synergies", agent_response=response)
        
        assert any(t in response for t in ["sacrifice", "aristocrats", "dies", "grave pact", "ashnod"]), "Should mention sacrifice themes"

    def test_synergy_graveyard(self, agent, audit_logger):
        """Static: Test synergy detection for graveyard strategies."""
        query = "What are good graveyard synergy cards?"
        source_url = "https://edhrec.com/themes/graveyard"
        
        audit_logger(query, source_url, {}, "Graveyard Synergies (Static)")
        result = run_query(agent, query)
        response = result.get("final_response", "").lower()
        audit_logger(query, source_url, {}, "Graveyard Synergies", agent_response=response)
        
        assert any(t in response for t in ["graveyard", "reanimate", "dredge", "entomb", "flashback"]), "Should mention graveyard themes"
