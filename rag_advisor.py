"""
RAG (Retrieval-Augmented Generation) System for MTGA Voice Advisor

This module provides semantic search over MTG Comprehensive Rules and
card statistics from 17lands.com to enhance AI advisor recommendations.

Dependencies:
- chromadb: Vector database for semantic search
- sentence-transformers: Embedding generation
- torch: Required by sentence-transformers

Install with: pip install chromadb sentence-transformers torch
"""

import json
import re
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def clean_card_name(name: str) -> str:
    """
    Remove HTML tags from card names.

    Some card names from Arena's database contain HTML tags like <nobr> and </nobr>
    that need to be stripped for proper display and matching.

    Args:
        name: Raw card name potentially containing HTML tags

    Returns:
        Clean card name with all HTML tags removed
    """
    if not name:
        return name
    clean_name = re.sub(r'<[^>]+>', '', name)
    return clean_name


# Try to import optional RAG dependencies
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("ChromaDB not available. Install with: pip install chromadb")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available. Install with: pip install sentence-transformers")


class RulesParser:
    """Parses the MTG Comprehensive Rules text file into structured chunks.

    This class reads the official rules text file and processes it into a list
    of individual rule entries. It uses hierarchical chunking to combine sub-rules
    with their parent rules, which provides better context for semantic search.

    Attributes:
        rules_path: The path to the rules text file.
        rules: A list of dictionaries, where each dictionary represents a parsed rule.
    """

    def __init__(self, rules_path: str):
        """Initializes the RulesParser.

        Args:
            rules_path: The file path to the Magic Comprehensive Rules text file.
        """
        self.rules_path = Path(rules_path)
        self.rules: List[Dict[str, str]] = []

    def parse(self) -> List[Dict[str, str]]:
        """
        Parse rules file into chunks with hierarchical context.
        Sub-rules (e.g., "100.1a") are combined with their parent rule ("100.1")
        to provide better context for semantic search.

        Returns:
            List of dicts with 'id', 'text', 'section' keys
        """
        if not self.rules_path.exists():
            logger.error(f"Rules file not found: {self.rules_path}")
            return []

        with open(self.rules_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        rule_pattern = re.compile(r'^(\d+\.\d+[a-z]*)\.\s+(.+)$')
        section_pattern = re.compile(r'^(\d+)\.\s+(.+)$')

        sections = {}
        rules = []
        current_section = "Unknown"
        parent_rule_text_cache = {}  # Cache for parent rule text

        for line in lines:
            line = line.strip()
            if not line:
                continue

            section_match = section_pattern.match(line)
            if section_match:
                section_num = section_match.group(1)
                section_name = section_match.group(2).strip()
                if len(section_num) == 3:
                    sections[section_num] = section_name
                    current_section = section_name
                continue

            rule_match = rule_pattern.match(line)
            if rule_match:
                rule_id = rule_match.group(1)
                rule_text = rule_match.group(2).strip()

                section_num = rule_id.split('.')[0]
                if section_num in sections:
                    current_section = sections[section_num]

                # Hierarchical chunking
                parent_id = rule_id.rstrip('a-z')
                is_sub_rule = parent_id != rule_id and parent_id in parent_rule_text_cache
                
                final_text = rule_text
                if is_sub_rule:
                    parent_text = parent_rule_text_cache[parent_id]
                    final_text = f"{parent_text} // {rule_text}"

                # Cache parent rule text (rules without a letter suffix)
                if not re.search('[a-z]$', rule_id):
                    parent_rule_text_cache[rule_id] = rule_text

                rules.append({
                    'id': rule_id,
                    'text': final_text,
                    'section': current_section
                })

        logger.info(f"Parsed {len(rules)} rules from {self.rules_path} with hierarchical context")
        self.rules = rules
        return rules


class RulesVectorDB:
    """Manages a vector database for performing semantic searches over MTG rules.

    This class handles the creation of a ChromaDB collection, the generation of
    text embeddings using a sentence-transformer model, and the execution of
    similarity-based queries on the rules.

    Attributes:
        db_path: The directory where the ChromaDB database is stored.
        collection_name: The name of the collection holding the rules.
        client: The ChromaDB client instance.
        collection: The ChromaDB collection object.
        embedding_model: The sentence-transformer model used for embeddings.
    """

    def __init__(self, db_path: str = "data/chromadb", collection_name: str = "mtg_rules"):
        """Initializes the vector database for MTG rules.

        This constructor sets up the ChromaDB client and loads the sentence-
        transformer model required for generating embeddings.

        Args:
            db_path: The directory to store the vector database files.
            collection_name: The name of the collection within the database.
        """
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.embedding_model = None

        if not CHROMADB_AVAILABLE:
            logger.warning("ChromaDB not available. Vector search disabled.")
            return

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("sentence-transformers not available. Vector search disabled.")
            return

        try:
            # Initialize ChromaDB
            self.db_path.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(anonymized_telemetry=False)
            )

            # Initialize embedding model
            logger.info("Loading sentence-transformers model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded")

        except Exception as e:
            logger.error(f"Failed to initialize vector DB: {e}")
            self.client = None

    def initialize_collection(self, rules: List[Dict[str, str]], force_recreate: bool = False):
        """
        Create or load the rules collection and add embeddings.

        Args:
            rules: Parsed rules from RulesParser
            force_recreate: If True, delete existing collection and recreate
        """
        if not self.client or not self.embedding_model:
            logger.warning("Vector DB not available")
            return

        try:
            # Check if collection exists
            collections = self.client.list_collections()
            collection_exists = any(c.name == self.collection_name for c in collections)

            if collection_exists and force_recreate:
                logger.info(f"Deleting existing collection: {self.collection_name}")
                self.client.delete_collection(self.collection_name)
                collection_exists = False

            if collection_exists:
                logger.info(f"Loading existing collection: {self.collection_name}")
                self.collection = self.client.get_collection(self.collection_name)
                logger.info(f"Collection has {self.collection.count()} documents")
            else:
                logger.info(f"Creating new collection: {self.collection_name}")
                self.collection = self.client.create_collection(self.collection_name)

                # Add rules in batches
                batch_size = 100
                for i in range(0, len(rules), batch_size):
                    batch = rules[i:i+batch_size]

                    ids = [rule['id'] for rule in batch]
                    texts = [f"Rule {rule['id']} ({rule['section']}): {rule['text']}" for rule in batch]
                    metadatas = [{'section': rule['section'], 'rule_id': rule['id']} for rule in batch]

                    # Generate embeddings
                    embeddings = self.embedding_model.encode(texts, show_progress_bar=False).tolist()

                    self.collection.add(
                        ids=ids,
                        documents=texts,
                        embeddings=embeddings,
                        metadatas=metadatas
                    )

                    if (i + batch_size) % 500 == 0:
                        logger.info(f"Indexed {min(i + batch_size, len(rules))}/{len(rules)} rules")

                logger.info(f"Indexed {len(rules)} rules successfully")

        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")
            self.collection = None

    def query(self, query_text: str, top_k: int = 3) -> List[Dict[str, any]]:
        """
        Query the rules database with semantic search.

        Args:
            query_text: Natural language query
            top_k: Number of results to return

        Returns:
            List of dicts with 'id', 'text', 'section', 'distance' keys
        """
        if not self.collection or not self.embedding_model:
            logger.warning("Vector DB not available for query")
            return []

        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query_text], show_progress_bar=False).tolist()[0]

            # Query collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )

            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'section': results['metadatas'][0][i]['section'],
                    'distance': results['distances'][0][i]
                })

            return formatted_results

        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []


class CardStatsDB:
    """Manages an SQLite database for storing and retrieving 17lands card statistics.

    This class provides an interface for inserting, updating, and querying card
    performance data, such as win rates and average pick positions.

    Attributes:
        db_path: The file path to the SQLite database.
        conn: The active SQLite database connection.
    """

    def __init__(self, db_path: str = "data/card_stats.db"):
        """Initializes the CardStatsDB.

        Args:
            db_path: The file path for the SQLite database.
        """
        self.db_path = Path(db_path)
        self.conn = None
        self._initialize_db()

    def _initialize_db(self):
        """Create database and tables if they don't exist."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            cursor = self.conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS card_stats (
                    card_name TEXT PRIMARY KEY,
                    set_code TEXT,
                    color TEXT,
                    rarity TEXT,
                    games_played INTEGER,
                    win_rate REAL,
                    avg_taken_at REAL,
                    games_in_hand INTEGER,
                    gih_win_rate REAL,
                    opening_hand_win_rate REAL,
                    drawn_win_rate REAL,
                    ever_drawn_win_rate REAL,
                    never_drawn_win_rate REAL,
                    alsa REAL,
                    ata REAL,
                    iwd REAL,
                    last_updated TEXT
                )
            """)

            self.conn.commit()
            logger.info(f"Initialized card stats database: {self.db_path}")

        except Exception as e:
            logger.error(f"Failed to initialize card stats DB: {e}")
            self.conn = None

    def insert_card_stats(self, stats: List[Dict[str, any]]):
        """
        Insert or update card statistics.

        Args:
            stats: List of card stat dictionaries
        """
        if not self.conn:
            logger.warning("Database not available")
            return

        cursor = self.conn.cursor()

        for stat in stats:
            cursor.execute("""
                INSERT OR REPLACE INTO card_stats VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                stat.get('card_name'),
                stat.get('set_code'),
                stat.get('color'),
                stat.get('rarity'),
                stat.get('games_played', 0),
                stat.get('win_rate', 0.0),
                stat.get('avg_taken_at', 0.0),
                stat.get('games_in_hand', 0),
                stat.get('gih_win_rate', 0.0),
                stat.get('opening_hand_win_rate', 0.0),
                stat.get('drawn_win_rate', 0.0),
                stat.get('ever_drawn_win_rate', 0.0),
                stat.get('never_drawn_win_rate', 0.0),
                stat.get('alsa', 0.0),
                stat.get('ata', 0.0),
                stat.get('iwd', 0.0),
                stat.get('last_updated', '')
            ))

        self.conn.commit()
        logger.info(f"Inserted/updated {len(stats)} card statistics")

    def get_card_stats(self, card_name: str) -> Optional[Dict[str, any]]:
        """
        Get statistics for a specific card.

        Args:
            card_name: Name of the card

        Returns:
            Dictionary of card stats or None
        """
        if not self.conn:
            return None

        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM card_stats WHERE card_name = ?", (card_name,))
        row = cursor.fetchone()

        if not row:
            return None

        return {
            'card_name': row[0],
            'set_code': row[1],
            'color': row[2],
            'rarity': row[3],
            'games_played': row[4],
            'win_rate': row[5],
            'avg_taken_at': row[6],
            'games_in_hand': row[7],
            'gih_win_rate': row[8],
            'opening_hand_win_rate': row[9],
            'drawn_win_rate': row[10],
            'ever_drawn_win_rate': row[11],
            'never_drawn_win_rate': row[12],
            'alsa': row[13],
            'ata': row[14],
            'iwd': row[15],
            'last_updated': row[16]
        }

    def delete_set_data(self, set_code: str):
        """
        Delete all statistics for a given set.

        Args:
            set_code: The set code to delete (e.g., 'M21')
        """
        if not self.conn:
            logger.warning("Database not available")
            return

        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM card_stats WHERE set_code = ?", (set_code,))
        self.conn.commit()
        logger.info(f"Deleted old data for set: {set_code}")

    def search_by_name(self, pattern: str, limit: int = 10) -> List[Dict[str, any]]:
        """Searches for cards by a name pattern.

        Args:
            pattern: An SQL LIKE pattern for the card name (e.g., "%Bolt%").
            limit: The maximum number of results to return.

        Returns:
            A list of dictionaries, where each dictionary contains the stats
            for a matching card.
        """
        if not self.conn:
            return []

        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM card_stats
            WHERE card_name LIKE ?
            ORDER BY games_played DESC
            LIMIT ?
        """, (pattern, limit))

        results = []
        for row in cursor.fetchall():
            results.append({
                'card_name': row[0],
                'set_code': row[1],
                'color': row[2],
                'rarity': row[3],
                'games_played': row[4],
                'win_rate': row[5],
                'gih_win_rate': row[8],
                'iwd': row[15]
            })

        return results

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


class CardMetadataDB:
    """Manages an SQLite database for basic card metadata.

    This class provides a simple interface to query a database of card
    metadata, such as color identity, mana value, and types.

    Attributes:
        db_path: The file path to the SQLite database.
        conn: The active SQLite database connection.
    """

    def __init__(self, db_path: str = "data/card_metadata.db"):
        """Initializes the CardMetadataDB.

        Args:
            db_path: The file path for the SQLite database.
        """
        self.db_path = Path(db_path)
        self.conn = None
        self._initialize_db()

    def _initialize_db(self):
        """Connect to existing metadata database."""
        try:
            if not self.db_path.exists():
                logger.warning(f"Card metadata DB not found: {self.db_path}")
                logger.warning("Run: python3 download_card_metadata.py")
                return

            self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            logger.info(f"Connected to card metadata database: {self.db_path}")

        except Exception as e:
            logger.error(f"Failed to connect to card metadata DB: {e}")
            self.conn = None

    def get_card_metadata(self, card_name: str) -> Optional[Dict[str, any]]:
        """
        Get metadata for a specific card.

        Args:
            card_name: Name of the card

        Returns:
            Dictionary with card attributes (color, mana_value, types, rarity)
        """
        if not self.conn:
            return None

        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT card_id, expansion, name, rarity, color_identity,
                   mana_value, types, is_booster
            FROM card_metadata
            WHERE name = ?
            LIMIT 1
        """, (card_name,))

        row = cursor.fetchone()
        if not row:
            return None

        return {
            'card_id': row[0],
            'expansion': row[1],
            'name': row[2],
            'rarity': row[3],
            'color_identity': row[4],
            'mana_value': row[5],
            'types': row[6],
            'is_booster': row[7]
        }

    def search_cards(self,
                     mana_value: Optional[int] = None,
                     color: Optional[str] = None,
                     types: Optional[str] = None,
                     rarity: Optional[str] = None,
                     limit: int = 10) -> List[Dict]:
        """
        Search cards by attributes.

        Args:
            mana_value: Filter by mana cost
            color: Filter by color (e.g., 'U', 'R', 'UB')
            types: Filter by type (e.g., 'Creature', 'Instant')
            rarity: Filter by rarity
            limit: Max results

        Returns:
            List of card metadata dictionaries
        """
        if not self.conn:
            return []

        query = "SELECT * FROM card_metadata WHERE 1=1"
        params = []

        if mana_value is not None:
            query += " AND mana_value = ?"
            params.append(mana_value)

        if color:
            query += " AND color_identity = ?"
            params.append(color)

        if types:
            query += " AND types LIKE ?"
            params.append(f"%{types}%")

        if rarity:
            query += " AND rarity = ?"
            params.append(rarity)

        query += f" LIMIT {limit}"

        cursor = self.conn.cursor()
        cursor.execute(query, params)

        results = []
        for row in cursor.fetchall():
            results.append({
                'card_id': row[0],
                'expansion': row[1],
                'name': row[2],
                'rarity': row[3],
                'color_identity': row[4],
                'mana_value': row[5],
                'types': row[6],
                'is_booster': row[7]
            })

        return results

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


class RAGSystem:
    """The main Retrieval-Augmented Generation (RAG) system.

    This class integrates the rules vector database and the card statistics
    database to provide a unified interface for augmenting AI prompts. It can
    query for relevant rules and fetch performance data for cards, then use
    this information to enhance the context provided to the LLM.

    Attributes:
        rules_db: An instance of `RulesVectorDB`.
        card_stats: An instance of `CardStatsDB`.
        card_metadata: An instance of `CardMetadataDB`.
    """

    def __init__(self,
                 rules_path: str = "data/MagicCompRules.txt",
                 db_path: str = "data/chromadb",
                 card_stats_db: str = "data/card_stats.db",
                 card_metadata_db: str = "data/card_metadata.db"):
        self.rules_db = RulesVectorDB(db_path)
        self.card_stats = CardStatsDB(card_stats_db)
        self.card_metadata = CardMetadataDB(card_metadata_db)
        self.rules_path = rules_path

        # Track initialization state
        self.rules_initialized = False

    def initialize_rules(self, force_recreate: bool = False):
        """
        Parse and index MTG rules.

        Args:
            force_recreate: If True, recreate the entire index
        """
        if not CHROMADB_AVAILABLE or not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("Cannot initialize rules: dependencies not available")
            return

        logger.info("Initializing rules database...")
        parser = RulesParser(self.rules_path)
        rules = parser.parse()

        if rules:
            self.rules_db.initialize_collection(rules, force_recreate=force_recreate)
            self.rules_initialized = True
            logger.info("Rules database initialized successfully")
        else:
            logger.error("Failed to parse rules")

    def query_rules(self, question: str, top_k: int = 3) -> List[Dict[str, any]]:
        """
        Query MTG rules database with semantic search.

        Args:
            question: Natural language question about rules
            top_k: Number of relevant rules to return

        Returns:
            List of relevant rule sections
        """
        if not self.rules_initialized:
            logger.warning("Rules not initialized. Call initialize_rules() first.")
            return []

        return self.rules_db.query(question, top_k=top_k)

    def get_card_stats(self, card_name: str) -> Optional[Dict[str, any]]:
        """
        Get 17lands statistics for a card.

        Args:
            card_name: Name of the card

        Returns:
            Dictionary of card statistics or None
        """
        return self.card_stats.get_card_stats(card_name)

    def enhance_prompt(self, board_state: Dict[str, any], base_prompt: str) -> str:
        """
        Enhance AI prompt with an expert persona, structured context, and goal-oriented queries.
        Uses concurrency to fetch data efficiently.

        Args:
            board_state: Current game state dictionary
            base_prompt: Base prompt to enhance

        Returns:
            Enhanced prompt with RAG context
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # 1. Define Expert Persona and Objectives
        persona = (
            "You are an expert Magic: The Gathering strategist providing tactical advice during a live match. "
            "Your goal is to help the player win. Your advice must be clear, direct, and actionable. "
            "Justify your recommendations by referencing the provided rules and card data. Explain the 'why'."
        )

        # 2. Goal-Oriented Prompting based on Game Phase
        phase = board_state.get('phase', 'main')
        if phase in ['combat', 'declare_attackers']:
            goal_prompt = "It is the declare attackers step. What is the optimal attack? Analyze potential blocks, combat tricks, and risks. How does this attack advance our plan to win?"
        elif phase == 'declare_blockers':
            goal_prompt = "The opponent has attacked. What is the optimal blocking strategy to minimize damage and preserve our board state? Should we take any damage to save a key creature?"
        else: # Main phase, etc.
            goal_prompt = base_prompt # Fallback to the original prompt

        # --- RAG Data Fetching (Concurrent) ---
        tasks = []
        rules_results = []
        stats_results = {}

        with ThreadPoolExecutor() as executor:
            situation_queries = self._detect_situation(board_state)
            if situation_queries and self.rules_initialized:
                for query in situation_queries[:2]:
                    tasks.append(executor.submit(self.query_rules, query, top_k=2))

            card_names_to_query = self._get_card_names_from_board(board_state)
            for card_name in card_names_to_query:
                tasks.append(executor.submit(self.get_card_stats, card_name))

            for future in as_completed(tasks):
                try:
                    result = future.result()
                    if not result: continue
                    if isinstance(result, list) and all(isinstance(i, dict) and 'text' in i for i in result):
                        rules_results.extend(result)
                    elif isinstance(result, dict) and 'card_name' in result:
                        stats_results[result['card_name']] = result
                except Exception as e:
                    logger.error(f"A RAG task failed: {e}")

        # 3. Assemble Prompt with Structured, Annotated Context
        enhanced_prompt = f"{persona}\n\n## Current Goal\n{goal_prompt}\n"

        if rules_results:
            enhanced_prompt += "\n## Relevant MTG Rules\n"
            for rule in rules_results:
                enhanced_prompt += f"- **Rule {rule.get('id', '')}**: {rule.get('text', '')}\n"

        if stats_results:
            enhanced_prompt += "\n## Tactical Analysis of Key Cards\n"
            for card_name, stats in stats_results.items():
                if stats and (stats.get('games_played') or 0) > 100:
                    # Tactical Annotation
                    win_rate = stats.get('gih_win_rate') or 0.0
                    tactical_note = "A key performer; prioritize it." if win_rate > 0.58 else "A solid role-player." if win_rate > 0.53 else "Below average performer."
                    
                    enhanced_prompt += (
                        f"- **{card_name}**: "
                        f"GIH Win Rate: {win_rate:.1%} ({stats.get('games_played', 0)} games). "
                        f"*Tactical Note: {tactical_note}*\n"
                    )
        
        return enhanced_prompt

    def _detect_situation(self, board_state: Dict[str, any]) -> List[str]:
        """
        Detect current game situation and generate relevant rule queries.

        Args:
            board_state: Current game state

        Returns:
            List of natural language queries for relevant rules
        """
        queries = []

        # Check for combat
        if board_state.get('phase') in ['combat', 'declare_attackers', 'declare_blockers']:
            queries.append("combat damage and blocking rules")
            queries.append("first strike and double strike")

        # Check for stack/spell resolution
        if board_state.get('stack_size', 0) > 0:
            queries.append("stack and priority rules")
            queries.append("resolving spells and abilities")

        # Check for keywords in card names (simplified)
        battlefield = board_state.get('battlefield', {})
        keywords = ['flying', 'trample', 'first strike', 'lifelink', 'deathtouch']

        for keyword in keywords:
            for zone in ['player', 'opponent']:
                if zone in battlefield:
                    for card in battlefield[zone]:
                        if keyword in card.get('name', '').lower():
                            queries.append(f"{keyword} rules and interactions")
                            break

        # Default query if nothing specific detected
        if not queries:
            queries.append("priority and phases")

        return queries

    def _get_board_card_stats(self, board_state: Dict[str, any]) -> Dict[str, Dict]:
        """
        Get statistics for cards currently on the battlefield.

        Args:
            board_state: Current game state

        Returns:
            Dictionary mapping card names to their statistics
        """
        stats_dict = {}
        battlefield = board_state.get('battlefield', {})

        for zone in ['player', 'opponent']:
            if zone in battlefield:
                for card in battlefield[zone]:
                    card_name = card.get('name', '')
                    if card_name and card_name not in stats_dict:
                        stats = self.get_card_stats(card_name)
                        if stats:
                            stats_dict[card_name] = stats

        return stats_dict

    def _get_card_names_from_board(self, board_state: Dict[str, any]) -> List[str]:
        """
        Get unique card names from the battlefield.

        Args:
            board_state: Current game state

        Returns:
            A list of unique card names.
        """
        card_names = set()
        battlefield = board_state.get('battlefield', {})

        for zone in ['player', 'opponent']:
            if zone in battlefield:
                for card in battlefield[zone]:
                    if name := card.get('name'):
                        card_names.add(name)
        return list(card_names)

    def close(self) -> None:
        """Closes database connections and cleans up resources."""
        self.card_stats.close()


# Sample data loader for 17lands (mock data)
def load_sample_17lands_data(db: CardStatsDB):
    """
    Load sample 17lands data for testing.
    In production, this would fetch from 17lands API or CSV exports.
    """
    sample_data = [
        {
            'card_name': 'Lightning Bolt',
            'set_code': 'M21',
            'color': 'R',
            'rarity': 'common',
            'games_played': 50000,
            'win_rate': 0.58,
            'gih_win_rate': 0.62,
            'iwd': 0.04,
            'last_updated': '2025-01-15'
        },
        {
            'card_name': 'Counterspell',
            'set_code': 'M21',
            'color': 'U',
            'rarity': 'common',
            'games_played': 45000,
            'win_rate': 0.59,
            'gih_win_rate': 0.61,
            'iwd': 0.03,
            'last_updated': '2025-01-15'
        },
        {
            'card_name': 'Llanowar Elves',
            'set_code': 'M21',
            'color': 'G',
            'rarity': 'common',
            'games_played': 55000,
            'win_rate': 0.61,
            'gih_win_rate': 0.68,
            'iwd': 0.09,
            'last_updated': '2025-01-15'
        },
        {
            'card_name': 'Jace, the Mind Sculptor',
            'set_code': 'A25',
            'color': 'U',
            'rarity': 'mythic',
            'games_played': 12000,
            'win_rate': 0.65,
            'gih_win_rate': 0.70,
            'iwd': 0.05,
            'last_updated': '2025-01-15'
        },
        {
            'card_name': 'Sheoldred, the Apocalypse',
            'set_code': 'DMU',
            'color': 'B',
            'rarity': 'mythic',
            'games_played': 80000,
            'win_rate': 0.68,
            'gih_win_rate': 0.72,
            'iwd': 0.04,
            'last_updated': '2025-01-15'
        },
        {
            'card_name': 'Grizzly Bears',
            'set_code': '10E',
            'color': 'G',
            'rarity': 'common',
            'games_played': 200,
            'win_rate': 0.50,
            'gih_win_rate': 0.51,
            'iwd': 0.01,
            'last_updated': '2025-01-15'
        },
        {
            'card_name': 'Jace, the Mind Sculptor',
            'set_code': 'A25',
            'color': 'U',
            'rarity': 'mythic',
            'games_played': 12000,
            'win_rate': 0.65,
            'gih_win_rate': 0.70,
            'iwd': 0.05,
            'last_updated': '2025-01-15'
        },
        {
            'card_name': 'Sheoldred, the Apocalypse',
            'set_code': 'DMU',
            'color': 'B',
            'rarity': 'mythic',
            'games_played': 80000,
            'win_rate': 0.68,
            'gih_win_rate': 0.72,
            'iwd': 0.04,
            'last_updated': '2025-01-15'
        },
        {
            'card_name': 'Grizzly Bears',
            'set_code': '10E',
            'color': 'G',
            'rarity': 'common',
            'games_played': 200,
            'win_rate': 0.50,
            'gih_win_rate': 0.51,
            'iwd': 0.01,
            'last_updated': '2025-01-15'
        },
        {
            'card_name': 'Jace, the Mind Sculptor',
            'set_code': 'A25',
            'color': 'U',
            'rarity': 'mythic',
            'games_played': 12000,
            'win_rate': 0.65,
            'gih_win_rate': 0.70,
            'iwd': 0.05,
            'last_updated': '2025-01-15'
        },
        {
            'card_name': 'Sheoldred, the Apocalypse',
            'set_code': 'DMU',
            'color': 'B',
            'rarity': 'mythic',
            'games_played': 80000,
            'win_rate': 0.68,
            'gih_win_rate': 0.72,
            'iwd': 0.04,
            'last_updated': '2025-01-15'
        },
        {
            'card_name': 'Grizzly Bears',
            'set_code': '10E',
            'color': 'G',
            'rarity': 'common',
            'games_played': 200,
            'win_rate': 0.50,
            'gih_win_rate': 0.51,
            'iwd': 0.01,
            'last_updated': '2025-01-15'
        }
    ]

    db.insert_card_stats(sample_data)
    logger.info(f"Loaded {len(sample_data)} sample cards")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize RAG system
    logger.info("Initializing RAG system...")
    rag = RAGSystem()

    # Initialize rules database (only needed once)
    if CHROMADB_AVAILABLE and SENTENCE_TRANSFORMERS_AVAILABLE:
        logger.info("Initializing rules database (this may take a few minutes)...")
        rag.initialize_rules(force_recreate=False)

        # Test rules query
        logger.info("\nTesting rules query...")
        results = rag.query_rules("What are the combat steps?", top_k=3)
        for result in results:
            logger.info(f"Rule {result['id']}: {result['text'][:100]}...")
    else:
        logger.warning("Skipping rules initialization (dependencies not available)")

    # Load sample card data
    logger.info("\nLoading sample card statistics...")
    load_sample_17lands_data(rag.card_stats)

    # Test card stats query
    logger.info("\nTesting card stats query...")
    stats = rag.get_card_stats("Lightning Bolt")
    if stats:
        logger.info(f"Lightning Bolt stats: WR={stats['win_rate']:.1%}, GIH WR={stats['gih_win_rate']:.1%}")

    # Test enhanced prompt
    logger.info("\nTesting prompt enhancement...")
    mock_board_state = {
        'phase': 'combat',
        'battlefield': {
            'player': [{'name': 'Lightning Bolt'}],
            'opponent': [{'name': 'Llanowar Elves'}]
        }
    }

    base_prompt = "Analyze the current board state and provide tactical advice."
    enhanced_prompt = rag.enhance_prompt(mock_board_state, base_prompt)
    logger.info(f"\nEnhanced prompt length: {len(enhanced_prompt)} chars")
    logger.info(f"Preview: {enhanced_prompt[:500]}...")

    rag.close()
    logger.info("\nRAG system test complete!")
