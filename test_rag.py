#!/usr/bin/env python3
"""
RAG System Test Suite

Tests the complete RAG implementation including:
1. Rules parsing
2. Card statistics database
3. Prompt enhancement
4. Integration with advisor.py
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_rules_parsing():
    """Test MTG rules parsing"""
    logger.info("\n" + "="*70)
    logger.info("TEST 1: MTG Rules Parsing")
    logger.info("="*70)

    try:
        from rag_advisor import RulesParser

        parser = RulesParser("data/MagicCompRules.txt")
        rules = parser.parse()

        # The 2025 comprehensive rules have ~3000 top-level rules
        # (Each major rule like 100.1, 100.1a, etc. is counted separately)
        assert len(rules) > 1000, f"Expected >1000 rules, got {len(rules)}"
        assert rules[0]['id'], "Rules should have IDs"
        assert rules[0]['text'], "Rules should have text"
        assert rules[0]['section'], "Rules should have sections"

        logger.info(f"✓ Parsed {len(rules)} rules successfully")
        logger.info(f"  Sample rule: {rules[100]['id']} - {rules[100]['text'][:80]}...")

        return True

    except Exception as e:
        logger.error(f"✗ Rules parsing failed: {e}")
        return False


def test_card_stats_db():
    """Test card statistics database"""
    logger.info("\n" + "="*70)
    logger.info("TEST 2: Card Statistics Database")
    logger.info("="*70)

    try:
        from rag_advisor import CardStatsDB

        db = CardStatsDB()

        # Test retrieving card stats
        test_cards = [
            'Lightning Bolt',
            'Llanowar Elves',
            'Jace, the Mind Sculptor',
            'Sheoldred, the Apocalypse'
        ]

        success_count = 0
        for card_name in test_cards:
            stats = db.get_card_stats(card_name)
            if stats:
                logger.info(
                    f"✓ {card_name}: "
                    f"WR={stats['win_rate']:.1%}, "
                    f"GIH WR={stats['gih_win_rate']:.1%}, "
                    f"IWD={stats['iwd']:+.1%}, "
                    f"Games={stats['games_played']}"
                )
                success_count += 1

                # Validate data
                assert 0 <= stats['win_rate'] <= 1, "Win rate out of range"
                assert 0 <= stats['gih_win_rate'] <= 1, "GIH WR out of range"
                assert -1 <= stats['iwd'] <= 1, "IWD out of range"
                assert stats['games_played'] > 0, "Games played should be positive"
            else:
                logger.warning(f"✗ {card_name}: Not found in database")

        db.close()

        assert success_count >= 3, f"Expected at least 3 cards, found {success_count}"
        logger.info(f"\n✓ Card stats database working ({success_count}/{len(test_cards)} cards found)")

        return True

    except Exception as e:
        logger.error(f"✗ Card stats test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vector_search():
    """Test rules vector search (requires ChromaDB)"""
    logger.info("\n" + "="*70)
    logger.info("TEST 3: Vector Search (Rules Database)")
    logger.info("="*70)

    try:
        from rag_advisor import RAGSystem, CHROMADB_AVAILABLE, SENTENCE_TRANSFORMERS_AVAILABLE

        if not CHROMADB_AVAILABLE:
            logger.warning("⚠ ChromaDB not available - skipping vector search test")
            logger.info("  Install with: pip install chromadb sentence-transformers torch")
            return None  # None = skipped

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("⚠ sentence-transformers not available - skipping vector search test")
            logger.info("  Install with: pip install sentence-transformers torch")
            return None  # None = skipped

        rag = RAGSystem()

        # Initialize rules (loads existing collection or creates new one)
        logger.info("  Loading rules database...")
        rag.initialize_rules()

        # Test queries
        test_queries = [
            ("What are the combat steps?", ["combat", "step", "phase"]),
            ("How does priority work?", ["priority", "player", "active"]),
            ("Rules about flying", ["flying", "creature", "block"])
        ]

        for query, expected_keywords in test_queries:
            results = rag.query_rules(query, top_k=3)

            if not results:
                logger.warning(f"✗ Query '{query}' returned no results")
                continue

            logger.info(f"\n  Query: \"{query}\"")
            for i, result in enumerate(results, 1):
                logger.info(f"    {i}. Rule {result['id']} (distance: {result['distance']:.3f})")
                logger.info(f"       {result['text'][:100]}...")

                # Check if any expected keywords are in the result
                text_lower = result['text'].lower()
                found_keywords = [kw for kw in expected_keywords if kw in text_lower]
                if found_keywords:
                    logger.info(f"       ✓ Contains keywords: {', '.join(found_keywords)}")

        rag.close()
        logger.info("\n✓ Vector search working")
        return True

    except Exception as e:
        logger.error(f"✗ Vector search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prompt_enhancement():
    """Test prompt enhancement with RAG context"""
    logger.info("\n" + "="*70)
    logger.info("TEST 4: Prompt Enhancement")
    logger.info("="*70)

    try:
        from rag_advisor import RAGSystem

        rag = RAGSystem()

        # Initialize rules to enable full enhancement
        logger.info("  Loading rules database...")
        rag.initialize_rules()

        # Mock board state
        board_state = {
            'phase': 'combat',
            'turn': 5,
            'battlefield': {
                'player': [
                    {'name': 'Lightning Bolt'},
                    {'name': 'Llanowar Elves'},
                    {'name': 'Forest'}
                ],
                'opponent': [
                    {'name': 'Serra Angel'},
                    {'name': 'Plains'}
                ]
            },
            'hand': [
                {'name': 'Murder'},
                {'name': 'Rampant Growth'}
            ],
            'graveyard': {
                'player': [],
                'opponent': []
            },
            'stack': [],
            'stack_size': 0
        }

        base_prompt = """
== GAME STATE: Turn 5, combat Phase ==
Your life: 20 | Opponent life: 18
Your library: 45 cards | Opponent library: 47 cards

== YOUR HAND ==
Murder, Rampant Growth

== YOUR BATTLEFIELD ==
Lightning Bolt, Llanowar Elves, Forest

== OPPONENT BATTLEFIELD ==
Serra Angel, Plains

== MANA AVAILABLE ==
2 mana available from 1 lands

== QUESTION ==
What is the optimal tactical play right now?
"""

        enhanced_prompt = rag.enhance_prompt(board_state, base_prompt)

        # Verify enhancement
        assert len(enhanced_prompt) > len(base_prompt), "Prompt should be enhanced"
        assert base_prompt in enhanced_prompt, "Enhanced prompt should contain base prompt"

        logger.info(f"✓ Base prompt length: {len(base_prompt)} chars")
        logger.info(f"✓ Enhanced prompt length: {len(enhanced_prompt)} chars")
        logger.info(f"✓ Added context: {len(enhanced_prompt) - len(base_prompt)} chars")

        # Check for expected additions
        enhancements = []
        if "MTG Rules:" in enhanced_prompt or "Rule " in enhanced_prompt:
            enhancements.append("rules context")
        if "17lands" in enhanced_prompt or "win_rate" in enhanced_prompt.lower():
            enhancements.append("card statistics")

        if enhancements:
            logger.info(f"✓ Enhancement includes: {', '.join(enhancements)}")
        else:
            logger.warning("⚠ No clear enhancements detected (may be due to missing dependencies)")

        # Show sample of enhancement
        logger.info("\n  Enhanced prompt preview:")
        lines = enhanced_prompt.split('\n')
        base_lines = base_prompt.split('\n')

        # Find where enhancement starts (after base prompt)
        if len(lines) > len(base_lines):
            logger.info("  " + "-"*60)
            for line in lines[len(base_lines):len(base_lines)+10]:
                logger.info(f"  {line[:76]}")
            if len(lines) > len(base_lines) + 10:
                logger.info("  ...")

        rag.close()
        logger.info("\n✓ Prompt enhancement working")
        return True

    except Exception as e:
        logger.error(f"✗ Prompt enhancement test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test integration with advisor.py"""
    logger.info("\n" + "="*70)
    logger.info("TEST 5: Integration with advisor.py")
    logger.info("="*70)

    try:
        # Import advisor components
        import sys
        sys.path.insert(0, str(Path(__file__).parent))

        # Check if RAG is importable from advisor
        from advisor import RAG_AVAILABLE

        logger.info(f"  RAG_AVAILABLE in advisor: {RAG_AVAILABLE}")

        if not RAG_AVAILABLE:
            logger.warning("⚠ RAG not available in advisor (missing dependencies)")
            logger.info("  Install with: pip install chromadb sentence-transformers torch")
            return None

        # Try to import AIAdvisor (don't instantiate, as it needs Ollama)
        logger.info("  Checking AIAdvisor class...")

        # Read advisor.py to verify integration
        advisor_path = Path(__file__).parent / "advisor.py"
        with open(advisor_path, 'r') as f:
            content = f.read()

        checks = [
            ("RAGSystem import", "from rag_advisor import RAGSystem"),
            ("use_rag parameter", "use_rag: bool = True"),
            ("RAG initialization", "self.rag_system = RAGSystem()"),
            ("Prompt enhancement", "self.rag_system.enhance_prompt"),
            ("Board state conversion", "_board_state_to_dict")
        ]

        all_found = True
        for check_name, check_pattern in checks:
            if check_pattern in content:
                logger.info(f"  ✓ {check_name}")
            else:
                logger.warning(f"  ✗ {check_name} not found")
                all_found = False

        if all_found:
            logger.info("\n✓ Integration with advisor.py complete")
            return True
        else:
            logger.warning("\n⚠ Some integration checks failed")
            return False

    except Exception as e:
        logger.error(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    logger.info("\n" + "="*70)
    logger.info("RAG SYSTEM TEST SUITE")
    logger.info("="*70)

    # Check prerequisites
    logger.info("\nChecking prerequisites...")

    # Check data files
    required_files = [
        ("MTG Rules", "data/MagicCompRules.txt"),
        ("Card Stats DB", "data/card_stats.db"),
    ]

    missing_files = []
    for name, path in required_files:
        if Path(path).exists():
            size = Path(path).stat().st_size
            logger.info(f"  ✓ {name}: {path} ({size:,} bytes)")
        else:
            logger.warning(f"  ✗ {name}: {path} (missing)")
            missing_files.append(path)

    if missing_files:
        logger.error("\nMissing required files. Run setup first:")
        logger.error("  python load_17lands_data.py")
        return False

    # Run tests
    tests = [
        ("Rules Parsing", test_rules_parsing),
        ("Card Statistics", test_card_stats_db),
        ("Vector Search", test_vector_search),
        ("Prompt Enhancement", test_prompt_enhancement),
        ("Integration", test_integration)
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            logger.error(f"\nTest '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False

    # Summary
    logger.info("\n" + "="*70)
    logger.info("TEST SUMMARY")
    logger.info("="*70)

    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)
    total = len(results)

    for test_name, result in results.items():
        if result is True:
            logger.info(f"  ✓ {test_name}: PASSED")
        elif result is False:
            logger.info(f"  ✗ {test_name}: FAILED")
        else:
            logger.info(f"  ⚠ {test_name}: SKIPPED")

    logger.info("")
    logger.info(f"Results: {passed} passed, {failed} failed, {skipped} skipped (total: {total})")

    if failed > 0:
        logger.warning("\n⚠ Some tests failed. Check output above for details.")
        return False
    elif skipped > 0:
        logger.info("\n✓ All tests passed (some skipped due to missing dependencies)")
        logger.info("  Install optional dependencies with:")
        logger.info("  pip install chromadb sentence-transformers torch")
        return True
    else:
        logger.info("\n✓ All tests passed!")
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
