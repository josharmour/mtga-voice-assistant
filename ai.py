import logging
import requests
from typing import Dict, Optional, List
import json
import re
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Content of rag_advisor.py
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
    logging.warning("ChromaDB not available. Install with: pip install chromadb")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Install with: pip install sentence-transformers")

# RAG system is available if both ChromaDB and sentence-transformers are available
RAG_AVAILABLE = CHROMADB_AVAILABLE and SENTENCE_TRANSFORMERS_AVAILABLE


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
            logging.error(f"Rules file not found: {self.rules_path}")
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

        logging.info(f"Parsed {len(rules)} rules from {self.rules_path} with hierarchical context")
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
            logging.warning("ChromaDB not available. Vector search disabled.")
            return

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logging.warning("sentence-transformers not available. Vector search disabled.")
            return

        try:
            # Initialize ChromaDB
            self.db_path.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(anonymized_telemetry=False)
            )

            # Initialize embedding model
            logging.info("Loading sentence-transformers model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logging.info("Embedding model loaded")

        except Exception as e:
            logging.error(f"Failed to initialize vector DB: {e}")
            self.client = None

    def initialize_collection(self, rules: List[Dict[str, str]], force_recreate: bool = False):
        """
        Create or load the rules collection and add embeddings.

        Args:
            rules: Parsed rules from RulesParser
            force_recreate: If True, delete existing collection and recreate
        """
        if not self.client or not self.embedding_model:
            logging.warning("Vector DB not available")
            return

        try:
            # Check if collection exists
            collections = self.client.list_collections()
            collection_exists = any(c.name == self.collection_name for c in collections)

            if collection_exists and force_recreate:
                logging.info(f"Deleting existing collection: {self.collection_name}")
                self.client.delete_collection(self.collection_name)
                collection_exists = False

            if collection_exists:
                logging.info(f"Loading existing collection: {self.collection_name}")
                self.collection = self.client.get_collection(self.collection_name)
                logging.info(f"Collection has {self.collection.count()} documents")
            else:
                logging.info(f"Creating new collection: {self.collection_name}")
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
                        logging.info(f"Indexed {min(i + batch_size, len(rules))}/{len(rules)} rules")

                logging.info(f"Indexed {len(rules)} rules successfully")

        except Exception as e:
            logging.error(f"Failed to initialize collection: {e}")
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
            logging.warning("Vector DB not available for query")
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
            logging.error(f"Query failed: {e}")
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
            logging.info(f"Initialized card stats database: {self.db_path}")

        except Exception as e:
            logging.error(f"Failed to initialize card stats DB: {e}")
            self.conn = None

    def insert_card_stats(self, stats: List[Dict[str, any]]):
        """
        Insert or update card statistics.

        Args:
            stats: List of card stat dictionaries
        """
        if not self.conn:
            logging.warning("Database not available")
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
        logging.info(f"Inserted/updated {len(stats)} card statistics")

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
            logging.warning("Database not available")
            return

        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM card_stats WHERE set_code = ?", (set_code,))
        self.conn.commit()
        logging.info(f"Deleted old data for set: {set_code}")

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
                logging.warning(f"Card metadata DB not found: {self.db_path}")
                logging.warning("Run: python3 download_card_metadata.py")
                return

            self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            logging.info(f"Connected to card metadata database: {self.db_path}")

        except Exception as e:
            logging.error(f"Failed to connect to card metadata DB: {e}")
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

    def search_cards(
        self, 
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
            logging.warning("Cannot initialize rules: dependencies not available")
            return

        logging.info("Initializing rules database...")
        parser = RulesParser(self.rules_path)
        rules = parser.parse()

        if rules:
            self.rules_db.initialize_collection(rules, force_recreate=force_recreate)
            self.rules_initialized = True
            logging.info("Rules database initialized successfully")
        else:
            logging.error("Failed to parse rules")

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
            logging.warning("Rules not initialized. Call initialize_rules() first.")
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
                    logging.error(f"A RAG task failed: {e}")

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

        # CRITICAL: Always include the base_prompt with actual board state
        # This ensures opponent life, board state details, and all game information is preserved
        enhanced_prompt += f"\n## Game State and Board Information\n{base_prompt}"

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

# Content of prompt_builder.py
class GroundedPromptBuilder:
    """
    Builds LLM prompts with grounded, cited context.

    Uses CardRAG for card data and RulesVectorDB for rules,
    ensuring the LLM has factual information with sources.
    """

    def __init__(self, card_rag=None, rules_db=None, verbose: bool = False):
        """
        Initialize prompt builder.

        Args:
            card_rag: CardRagDatabase instance
            rules_db: RulesVectorDB instance
            verbose: Include more detailed information
        """
        self.card_rag = card_rag
        self.rules_db = rules_db
        self.verbose = verbose

    def build_board_state_section(
        self,
        player_hand: List[int],
        player_board: List[int],
        opponent_hand_count: int,
        opponent_board: List[int],
        format_type: str = "PremierDraft"
    ) -> str:
        """
        Build a grounded board state section.

        Args:
            player_hand: List of card grpIds in player's hand
            player_board: List of card grpIds on player's board
            opponent_hand_count: Number of cards opponent has
            opponent_board: List of card grpIds on opponent's board
            format_type: Draft format for statistics

        Returns:
            Formatted board state with full card information
        """
        if not self.card_rag:
            return ""

        lines = []
        lines.append("## YOUR BOARD STATE\n")

        # Your hand
        if player_hand:
            lines.append("### Your Hand")
            for grp_id in player_hand:
                card = self.card_rag.get_card_by_grpid(grp_id, format_type)
                if card:
                    # Format: Name (Cost) [Type] - Abilities
                    context = card.to_prompt_context()
                    lines.append(f"- {context}")
            lines.append("")

        # Your board
        if player_board:
            lines.append("### Your Board")
            for grp_id in player_board:
                card = self.card_rag.get_card_by_grpid(grp_id, format_type)
                if card:
                    context = card.to_prompt_context()
                    lines.append(f"- {context}")
            lines.append("")

        # Opponent's board
        if opponent_board:
            lines.append("### Opponent's Board")
            for grp_id in opponent_board:
                card = self.card_rag.get_card_by_grpid(grp_id, format_type)
                if card:
                    context = card.to_prompt_context()
                    lines.append(f"- {context}")
            lines.append("")

        lines.append(f"Opponent's hand: {opponent_hand_count} unknown cards\n")

        return "\n".join(lines)

    def build_card_abilities_section(
        self,
        card_grp_ids: List[int],
        format_type: str = "PremierDraft"
    ) -> str:
        """
        Build detailed card abilities section with citations.

        Args:
            card_grp_ids: List of card grpIds
            format_type: Draft format for statistics

        Returns:
            Formatted card details section
        """
        if not self.card_rag or not card_grp_ids:
            return ""

        lines = []
        lines.append("## CARD ABILITIES AND DETAILS\n")

        for grp_id in card_grp_ids:
            card = self.card_rag.get_card_by_grpid(grp_id, format_type)
            if card:
                lines.append(card.to_rag_citation(include_stats=self.verbose))

        return "\n".join(lines)

    def build_rules_section(self, situation_query: str, top_k: int = 3) -> str:
        """
        Build relevant rules section using semantic search.

        Args:
            situation_query: Description of game situation
            top_k: Number of rules to retrieve

        Returns:
            Formatted rules section with citations
        """
        if not self.rules_db:
            return ""

        try:
            results = self.rules_db.query(situation_query, top_k=top_k)
            if not results:
                return ""

            lines = []
            lines.append("## RELEVANT MTG RULES\n")

            for rule in results:
                rule_id = rule.get('id', '')
                section = rule.get('section', '')
                text = rule.get('text', '')
                lines.append(f"**Rule {rule_id}** ({section}):")
                lines.append(f"{text}\n")

            lines.append("[Source: MTG Comprehensive Rules]\n")
            return "\n".join(lines)

        except Exception as e:
            logging.warning(f"Could not retrieve rules: {e}")
            return ""

    def build_win_rate_context(
        self,
        card_names: List[str],
        set_code: str,
        format_type: str = "PremierDraft"
    ) -> str:
        """
        Build win rate context for cards.

        Args:
            card_names: List of card names
            set_code: Set code
            format_type: Draft format

        Returns:
            Formatted win rate context
        """
        if not self.card_rag:
            return ""

        lines = []
        high_winrate = []
        low_winrate = []

        for card_name in card_names:
            card = self.card_rag.get_card_by_name(card_name, set_code)
            if card and card.win_rate:
                if card.win_rate > 0.55:
                    high_winrate.append((card_name, card.win_rate, card.games_played))
                elif card.win_rate < 0.48:
                    low_winrate.append((card_name, card.win_rate, card.games_played))

        if high_winrate or low_winrate:
            lines.append("## PERFORMANCE CONTEXT\n")

            if high_winrate:
                lines.append("### Strong Cards (High Win Rate)")
                for name, wr, games in high_winrate:
                    if games and games >= 1000:
                        lines.append(f"- **{name}**: {wr*100:.1f}% WR ({games} games)")
                lines.append("")

            if low_winrate:
                lines.append("### Weaker Cards (Low Win Rate)")
                for name, wr, games in low_winrate:
                    if games and games >= 1000:
                        lines.append(f"- **{name}**: {wr*100:.1f}% WR ({games} games)")
                lines.append("")

            lines.append("[Source: 17lands.com]\n")

        return "\n".join(lines)

    def build_complete_prompt(
        self,
        base_objective: str,
        player_hand: List[int],
        player_board: List[int],
        opponent_hand_count: int,
        opponent_board: List[int],
        situation_query: str = None,
        card_names_for_stats: List[str] = None,
        set_code: str = None,
        format_type: str = "PremierDraft"
    ) -> str:
        """
        Build a complete grounded prompt with all RAG context.

        Args:
            base_objective: What the player needs to do (e.g., "What should I attack with?")
            player_hand: Player's hand grpIds
            player_board: Player's board creatures grpIds
            opponent_hand_count: Number of cards opponent has
            opponent_board: Opponent's board creatures grpIds
            situation_query: Query for rules search (e.g., "combat damage timing")
            card_names_for_stats: Card names to include performance stats for
            set_code: Set being played
            format_type: Draft format

        Returns:
            Complete prompt with grounded context and citations
        """
        prompt_sections = []

        # Expert persona
        persona = (
            "You are an expert Magic: The Gathering strategist. "
            "Your advice must be grounded in the card abilities, rules, and statistics provided. "
            "Only reference information that is cited. Explain your reasoning step-by-step."
        )
        prompt_sections.append(persona)
        prompt_sections.append("")

        # Objective
        prompt_sections.append(f"## OBJECTIVE: {base_objective}\n")

        # Board state with complete card information
        board_section = self.build_board_state_section(
            player_hand,
            player_board,
            opponent_hand_count,
            opponent_board,
            format_type
        )
        if board_section:
            prompt_sections.append(board_section)

        # Card abilities and details
        all_grp_ids = player_hand + player_board + opponent_board
        if all_grp_ids:
            abilities_section = self.build_card_abilities_section(all_grp_ids, format_type)
            if abilities_section:
                prompt_sections.append(abilities_section)

        # Relevant rules
        if situation_query:
            rules_section = self.build_rules_section(situation_query, top_k=2)
            if rules_section:
                prompt_sections.append(rules_section)

        # Win rate context
        if card_names_for_stats and set_code:
            stats_section = self.build_win_rate_context(
                card_names_for_stats,
                set_code,
                format_type
            )
            if stats_section:
                prompt_sections.append(stats_section)

        # Citation footer
        prompt_sections.append("\n" + "="*70)
        prompt_sections.append("IMPORTANT: Base your advice only on the information above.")
        prompt_sections.append("Do not reference cards, rules, or statistics not listed.")
        prompt_sections.append("="*70)

        return "\n\n".join(prompt_sections)


class RagContextCache:
    """
    Caches RAG context to avoid redundant lookups.

    Since board state doesn't change every turn, caching
    card information reduces database queries.
    """

    def __init__(self, max_size: int = 1000):
        """Initialize the cache."""
        self.cache: Dict[int, Dict] = {}  # grpId → card info
        self.max_size = max_size

    def get(self, grp_id: int) -> Optional[Dict]:
        """Get cached card info."""
        return self.cache.get(grp_id)

    def put(self, grp_id: int, card_info: Dict):
        """Cache card info."""
        if len(self.cache) >= self.max_size:
            # Simple LRU: remove oldest (not implemented for simplicity)
            pass
        self.cache[grp_id] = card_info

    def clear(self):
        """Clear the cache."""
        self.cache.clear()

# Content of ai_advisor.py
class OllamaClient:
    def __init__(self, host: str = "http://localhost:11434", model: str = "mistral:7b"):
        self.host = host
        self.model = model

    def is_running(self) -> bool:
        """Check if Ollama service is running"""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=2)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def start_ollama(self) -> bool:
        """Try to start Ollama service"""
        try:
            import subprocess
            # Try to start ollama serve in the background
            subprocess.Popen(
                ['ollama', 'serve'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            # Wait a moment for it to start
            import time
            time.sleep(2)
            return self.is_running()
        except Exception as e:
            logging.error(f"Failed to start Ollama: {e}")
            return False

    def generate(self, prompt: str) -> Optional[str]:
        logging.debug(f"Ollama prompt: {prompt[:500]}...")
        try:
            payload = {"model": self.model, "prompt": prompt, "stream": False}
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=30
            )
            response.raise_for_status()  # Raise an exception for bad status codes
            result = response.json()
            logging.debug(f"Ollama raw response: {result}")
            return result.get('response', '').strip()
        except requests.exceptions.RequestException as e:
            logging.error(f"Ollama error: {e}")
            return None

class AIAdvisor:
    SYSTEM_PROMPT = """You are an expert Magic: The Gathering tactical advisor.

CRITICAL RULES:
1. ONLY reference cards explicitly listed in "YOUR HAND" or "YOUR BATTLEFIELD"
2. You CANNOT destroy lands (Forest, Plains, Swamp, Mountain, Island) - lands are permanent
3. You can only cast spells from YOUR HAND during your main phase
4. Creatures can attack if they've been on battlefield since your last turn
5. If you see "Unknown" cards, say "Wait for card identification"

FORBIDDEN ACTIONS:
- Do NOT mention cards not listed in the board state
- Do NOT suggest destroying/removing lands
- Do NOT invent card names

Give ONLY tactical advice in 1-2 short sentences. Start directly with your recommendation."""

    def __init__(self, ollama_host: str = "http://localhost:11434", model: str = "mistral:7b", use_rag: bool = True, card_db: Optional["ArenaCardDatabase"] = None):
        self.client = OllamaClient(host=ollama_host, model=model)
        self.use_rag = use_rag and RAG_AVAILABLE
        self.rag_system = None
        self.card_db = card_db  # Store card database for oracle text lookups
        self.last_rag_references = None  # Track most recent RAG references used

        # Initialize RAG system if enabled
        if self.use_rag:
            try:
                logging.info("Initializing RAG system...")
                self.rag_system = RAGSystem()
                # Only initialize rules if ChromaDB and embeddings are available
                if hasattr(self.rag_system.rules_db, 'client') and self.rag_system.rules_db.client:
                    logging.info("RAG system initialized with rules database")
                else:
                    logging.info("RAG system initialized (rules search disabled - install chromadb and sentence-transformers)")
            except Exception as e:
                logging.error(f"Failed to initialize RAG system: {e}")
                self.rag_system = None
                self.use_rag = False
        else:
            if not RAG_AVAILABLE:
                logging.info("RAG system disabled (dependencies not installed)")
            else:
                logging.info("RAG system disabled by configuration")

    def _build_prompt(self, board_state: "BoardState") -> str:
        """Build comprehensive prompt with all zones and game history"""
        lines = [
            f"== GAME STATE: Turn {board_state.current_turn}, {board_state.current_phase} Phase ==",
            f"Your life: {board_state.your_life} | Opponent life: {board_state.opponent_life}",
            f"Your library: {board_state.your_library_count} cards | Opponent library: {board_state.opponent_library_count} cards",
            "",
        ]

        # Game History - what happened this turn
        if board_state.history and board_state.history.turn_number == board_state.current_turn:
            history = board_state.history
            if history.cards_played_this_turn or history.died_this_turn or history.lands_played_this_turn:
                lines.append("== THIS TURN ==")
                if history.cards_played_this_turn:
                    played_names = [c.name for c in history.cards_played_this_turn]
                    lines.append(f"Cards played: {', '.join(played_names)}")
                if history.lands_played_this_turn > 0:
                    lines.append(f"Lands played: {history.lands_played_this_turn}")
                if history.died_this_turn:
                    lines.append(f"Creatures died: {', '.join(history.died_this_turn)}")
                lines.append("")

        # Hand - with oracle text
        if board_state.your_hand:
            lines.append(f"== YOUR HAND ({len(board_state.your_hand)}) ==")
            for card in board_state.your_hand:
                card_info = f"• {card.name}"
                # Add oracle text if available
                if self.card_db and card.grp_id:
                    oracle_text = self.card_db.get_oracle_text(card.grp_id)
                    if oracle_text:
                        card_info += f"\n  ({oracle_text})"
                lines.append(card_info)
            lines.append("")
        else:
            lines.append("== YOUR HAND == (empty)")
            lines.append("")

        # Battlefield - with tapped/untapped status and oracle text
        if board_state.your_battlefield:
            lines.append(f"== YOUR BATTLEFIELD ({len(board_state.your_battlefield)}) ==")
            for card in board_state.your_battlefield:
                status_flags = []
                if card.is_tapped:
                    status_flags.append("TAPPED")
                if card.summoning_sick:
                    status_flags.append("summoning sick")
                if card.is_attacking:
                    status_flags.append("ATTACKING")
                if card.counters:
                    counter_str = ", ".join([f"{v} {k}" for k, v in card.counters.items()])
                    status_flags.append(f"counters: {counter_str}")
                if card.attached_to:
                    status_flags.append(f"attached to instance {card.attached_to}")

                status_text = f" ({ ', '.join(status_flags)})" if status_flags else ""
                power_toughness = f" [{card.power}/{card.toughness}]" if card.power is not None else ""
                card_line = f"• {card.name}{power_toughness}{status_text}"

                # Add oracle text if available
                if self.card_db and card.grp_id:
                    oracle_text = self.card_db.get_oracle_text(card.grp_id)
                    if oracle_text:
                        card_line += f"\n  ({oracle_text})"

                lines.append(card_line)
            lines.append("")
        else:
            lines.append("== YOUR BATTLEFIELD == (empty)")
            lines.append("")

        # Opponent's battlefield
        if board_state.opponent_battlefield:
            lines.append(f"== OPPONENT BATTLEFIELD ({len(board_state.opponent_battlefield)}) ==")
            for card in board_state.opponent_battlefield:
                status_flags = []
                if card.is_tapped:
                    status_flags.append("TAPPED")
                if card.is_attacking:
                    status_flags.append("ATTACKING")
                if card.counters:
                    counter_str = ", ".join([f"{v} {k}" for k, v in card.counters.items()])
                    status_flags.append(f"counters: {counter_str}")

                status_text = f" ({ ', '.join(status_flags)})" if status_flags else ""
                power_toughness = f" [{card.power}/{card.toughness}]" if card.power is not None else ""
                card_line = f"• {card.name}{power_toughness}{status_text}"

                # Add oracle text if available
                if self.card_db and card.grp_id:
                    oracle_text = self.card_db.get_oracle_text(card.grp_id)
                    if oracle_text:
                        card_line += f"\n  ({oracle_text})"

                lines.append(card_line)
            lines.append("")
        else:
            lines.append("== OPPONENT BATTLEFIELD == (empty)")
            lines.append("")

        # Graveyards
        if board_state.your_graveyard:
            graveyard_names = [c.name for c in board_state.your_graveyard]
            lines.append(f"== YOUR GRAVEYARD ({len(graveyard_names)}) ==")
            lines.append(", ".join(graveyard_names))
            lines.append("")

        if board_state.opponent_graveyard:
            opp_graveyard_names = [c.name for c in board_state.opponent_graveyard]
            lines.append(f"== OPPONENT GRAVEYARD ({len(opp_graveyard_names)}) ==")
            lines.append(", ".join(opp_graveyard_names))
            lines.append("")

        return "\n".join(lines)

    def get_tactical_advice(self, board_state: "BoardState") -> Optional[str]:
        prompt = self._build_prompt(board_state)

        # Debug: Log the board state values being used
        logging.debug(f"[TACTICAL ADVICE] Board state: Your life={board_state.your_life}, Opponent life={board_state.opponent_life}")
        logging.debug(f"[TACTICAL ADVICE] Your battlefield: {len(board_state.your_battlefield)} cards")
        logging.debug(f"[TACTICAL ADVICE] Opponent battlefield: {len(board_state.opponent_battlefield)} cards")
        logging.debug(f"[TACTICAL ADVICE] Your hand: {len(board_state.your_hand)} cards")

        # Debug: Log the prompt before RAG enhancement
        logging.debug(f"[TACTICAL ADVICE] Prompt before RAG:\n{prompt[:1000]}...")

        # Enhance prompt with RAG context if available
        if self.use_rag and self.rag_system:
            try:
                # Convert BoardState to dict format for RAG system
                board_dict = self._board_state_to_dict(board_state)

                # Use the enhanced method that returns references
                if hasattr(self.rag_system, 'enhance_prompt_with_references'):
                    prompt, self.last_rag_references = self.rag_system.enhance_prompt_with_references(board_dict, prompt)
                    logging.debug(f"Prompt enhanced with RAG context. References: {self.last_rag_references}")
                else:
                    # Fallback to old method if the new method isn't available
                    prompt = self.rag_system.enhance_prompt(board_dict, prompt)
                    logging.debug("Prompt enhanced with RAG context (references not tracked)")
            except Exception as e:
                logging.warning(f"Failed to enhance prompt with RAG: {e}")
                self.last_rag_references = None

        # Debug: Log the full prompt that will be sent to AI
        logging.info(f"[TACTICAL ADVICE] FULL PROMPT BEING SENT TO AI:\n{self.SYSTEM_PROMPT}\n\n{prompt}")

        advice = self.client.generate(f"{self.SYSTEM_PROMPT}\n\n{prompt}")
        if advice:
            logging.debug(f"AI generated advice: {advice[:500]}...")
        else:
            logging.debug("AI did not generate any advice.")
        return advice

    def _board_state_to_dict(self, board_state: "BoardState") -> dict:
        """Converts BoardState dataclass to a dictionary for RAG processing."""
        import dataclasses
        if not board_state:
            return {}
        return dataclasses.asdict(board_state)

    def get_last_rag_references(self) -> Optional[Dict]:
        """Get the RAG references from the last tactical advice generation."""
        return self.last_rag_references

    def check_important_updates(self, board_state: "BoardState", previous_board_state: Optional["BoardState"]) -> Optional[str]:
        """
        Check if there are important changes that warrant immediate notification.
        Returns advice if important, None if not worth speaking.
        """
        if not previous_board_state:
            return None

        # Build a prompt asking the model to evaluate importance
        evaluation_prompt = f"""You are a tactical advisor monitoring a Magic: The Gathering game in progress.

PREVIOUS STATE (just before):
- Turn {previous_board_state.current_turn}, {previous_board_state.current_phase}
- Your life: {previous_board_state.your_life} | Opponent: {previous_board_state.opponent_life}
- Your battlefield: {len(previous_board_state.your_battlefield)} | Opponent: {len(previous_board_state.opponent_battlefield)}

CURRENT STATE (right now):
- Turn {board_state.current_turn}, {board_state.current_phase}
- Your life: {board_state.your_life} | Opponent: {board_state.opponent_life}
- Your battlefield: {len(board_state.your_battlefield)} | Opponent: {len(board_state.opponent_battlefield)}

WHAT JUST HAPPENED:"""

        changes_detected = False
        # Add detected changes
        if board_state.history and board_state.history.turn_number == board_state.current_turn:
            history = board_state.history
            if history.cards_played_this_turn:
                evaluation_prompt += f"\n- Cards played: {', '.join([c.name for c in history.cards_played_this_turn])}"
                changes_detected = True
            if history.died_this_turn:
                evaluation_prompt += f"\n- Creatures died: {', '.join(history.died_this_turn)}"
                changes_detected = True

        # Check life total changes (only significant ones)
        life_change = board_state.your_life - previous_board_state.your_life
        if life_change < -5:  # Only care about losing 5+ life
            evaluation_prompt += f"\n- Your life dropped by {abs(life_change)}"
            changes_detected = True
        elif life_change != 0:
            # Track it but don't necessarily alert
            evaluation_prompt += f"\n- Your life: {life_change:+d}"
            changes_detected = True

        opponent_life_change = board_state.opponent_life - previous_board_state.opponent_life
        if opponent_life_change < -5:  # Only care if opponent losing significant life
            evaluation_prompt += f"\n- Opponent life dropped by {abs(opponent_life_change)}"
            changes_detected = True

        # If no significant changes, don't even query
        if not changes_detected:
            return None

        evaluation_prompt += """

IMPORTANT: Most game events are NOT critical. Only alert if this is URGENT and the player must act NOW.

Examples of NOT critical:
- Opponent played a creature (unless it's lethal next turn)
- Lost 3-4 life (that's normal)
- Opponent gained some life
- A single creature died

Examples of CRITICAL:
- Opponent has exact lethal damage on board ready to attack
- Opponent played a game-ending combo piece
- You're at 2 life and they have burn spell

Respond in EXACTLY this format:
- If critical: "ALERT: [one sentence warning]"
- If not critical: "NO"

Your response:"""

        response = self.client.generate(evaluation_prompt)
        if response:
            response = response.strip()
            logging.debug(f"Importance check response: {response}")

            if response.startswith("ALERT:"):
                # Extract the advice part after "ALERT:"
                advice = response[6:].strip()
                logging.info(f"Critical update detected: {advice}")
                return advice
            elif "ALERT:" in response:
                # Handle case where model adds extra text before ALERT:
                alert_start = response.find("ALERT:")
                advice = response[alert_start + 6:].strip()
                # Remove any trailing quotes or punctuation artifacts
                advice = advice.strip('"\',!?')
                logging.info(f"Critical update detected (extracted): {advice}")
                return advice

        return None
