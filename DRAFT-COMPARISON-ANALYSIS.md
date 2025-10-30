# Comparative Analysis: python-mtga-helper vs MTGA Voice Advisor

**Date**: 2025-10-29
**Purpose**: Analyze python-mtga-helper to guide drafting feature implementation in MTGA Voice Advisor

---

## 1. Feature Comparison

### python-mtga-helper Features

| Feature | Implementation | Status |
|---------|---------------|--------|
| **Draft Assistance** | Premier Draft & Quick Draft pick recommendations | ✅ Core feature |
| **Sealed Pool Analysis** | Color pair ranking with statistics | ✅ Core feature |
| **Play Assistance** | Basic game start/end logging only | ⚠️ Minimal |
| **Card Details** | 17lands win rates, grades (A+ to F) | ✅ Data-driven |
| **Deck Building** | Color pair recommendations, creature counts | ✅ During draft |

### MTGA Voice Advisor Features

| Feature | Implementation | Status |
|---------|---------------|--------|
| **Draft Assistance** | None | ❌ Missing |
| **Sealed Pool Analysis** | None | ❌ Missing |
| **Play Assistance** | Real-time tactical advice via LLM | ✅ Core feature |
| **Card Details** | Scryfall data + optional 17lands stats | ✅ Via RAG |
| **Deck Building** | None | ❌ Missing |

### Key Differences

**python-mtga-helper:**
- **Focus**: Limited formats (draft/sealed) only
- **Advice Type**: Data-driven grades from 17lands statistics
- **Output**: CLI tables with color-coded letter grades
- **Real-time**: Triggers on draft picks and pool events
- **Voice**: None

**MTGA Voice Advisor:**
- **Focus**: Constructed play only (currently)
- **Advice Type**: LLM-generated tactical recommendations
- **Output**: Voice + GUI/TUI/CLI
- **Real-time**: Triggers on priority changes during gameplay
- **Voice**: Multi-engine TTS with multiple voices

---

## 2. Technical Stack Analysis

### Card Data Resolution

#### python-mtga-helper Approach
```
Arena Card ID → 17lands API → Rankings with stats
                    ↓
        Cache in XDG_CACHE_HOME (~/.cache/python-mtga-helper/17lands/)
                    ↓
        Calculate grades using scipy.stats.norm.cdf()
```

**Data Sources:**
1. **17lands API**: `https://www.17lands.com/card_ratings/data`
   - Parameters: `expansion`, `format`, `end_date`
   - Returns: JSON array with card stats
   - Fields: `mtga_id`, `name`, `color`, `rarity`, `types`, `ever_drawn_win_rate`, etc.

2. **Caching Strategy**: XDG cache with URL parameter-based filenames
   - Example: `expansion=blb&format=PremierDraft&end_date=2025-10-29.json`

3. **Grading Algorithm**:
   ```python
   # Calculate mean and std dev of all card win rates
   mean = np.mean(all_win_rates)
   std = np.std(all_win_rates, ddof=1)

   # Convert win rate to percentile using CDF
   cdf = norm.cdf(card_win_rate, loc=mean, scale=std)
   score = cdf * 100  # 0-100 scale

   # Map to letter grade
   if score >= 99: grade = "A+"
   elif score >= 95: grade = "A"
   # ... etc
   ```

#### MTGA Voice Advisor Approach
```
Arena grpId → ArenaCardDatabase (Raw_CardDatabase_*.mtga)
              ↓ (fallback)
          Scryfall cache (card_cache.json)
              ↓ (fallback)
          Scryfall API
              ↓
          Optional: RAGSystem
              ↓
    [card_stats.db, card_metadata.db, chromadb]
```

**Data Sources:**
1. **Arena Database**: Local binary file (most accurate)
2. **Scryfall API**: REST API with caching
3. **17lands Data** (via RAG): Pre-downloaded CSV files → SQLite databases
4. **MTG Rules**: Comprehensive rules → ChromaDB vector store

### MTGA Log Parsing

#### Both Use Similar Event Detection

**python-mtga-helper Events:**
```python
# Start events (JSON in same line)
"[UnityCrossThreadLogger]==> LogBusinessEvents {...}"

# End events (JSON on next line)
"<== EventGetCoursesV2(uuid)"
"<== BotDraftDraftStatus(uuid)"
"<== BotDraftDraftPick(uuid)"

# Special format
"GreToClientEvent" (no start marker)
```

**MTGA Voice Advisor Events:**
```python
# Currently only parses:
"GreToClientEvent" → GameStateManager
```

**Critical Missing Events for Draft Support:**
- `EventGetCoursesV2` - Pool information for sealed/draft
- `LogBusinessEvents` - Premier draft picks
- `BotDraftDraftStatus` - Quick draft pool/picks
- `BotDraftDraftPick` - Quick draft pick confirmation

### Log Event Structures

#### EventGetCoursesV2 (Sealed/Draft Pools)
```json
{
  "Courses": [
    {
      "InternalEventName": "PremierDraft_BLB_20250815",
      "CardPool": [75001, 75002, ...],  // Arena IDs
      "CurrentWins": 0,
      "CurrentLosses": 0,
      "CourseDeckSummary": {...}
    }
  ]
}
```

#### LogBusinessEvents (Premier Draft Picks)
```json
{
  "EventId": "PremierDraft_BLB_20250815",
  "DraftId": "abc-123",
  "PackNumber": 1,
  "PickNumber": 3,
  "CardsInPack": ["75001", "75002", ...]
}
```

#### BotDraftDraftStatus (Quick Draft State)
```json
{
  "EventName": "QuickDraft_BLB_20250815",
  "PackNumber": 0,  // 0-indexed
  "PickNumber": 0,  // 0-indexed
  "DraftPack": ["75001", "75002", ...],
  "PickedCards": ["75010", "75011", ...]
}
```

---

## 3. Data Source Integration Analysis

### 17lands Integration

#### python-mtga-helper Method
```python
# Direct API query with caching
def query_17lands(expansion: str, format_name: str):
    params = {
        "expansion": expansion,
        "format": format_name,
        "end_date": datetime.now(timezone.utc).date().isoformat(),
    }
    res = requests.get("https://www.17lands.com/card_ratings/data", params=params)
    return res.json()
```

**Pros:**
- Always up-to-date data
- Minimal storage (cache only what's queried)
- Simple implementation

**Cons:**
- Requires network on first run
- API rate limiting concerns
- Dependent on 17lands uptime

#### MTGA Voice Advisor Method
```python
# Pre-downloaded CSV → SQLite database
# download_real_17lands_data.py downloads entire sets
# Stores in data/card_stats.db
```

**Pros:**
- Offline operation
- Fast queries (indexed SQLite)
- More metadata available

**Cons:**
- Manual updates required
- Larger storage footprint
- Initial download time (~60-180 min)

### Scryfall Integration

**python-mtga-helper:** Not used (relies on 17lands for all card data)

**MTGA Voice Advisor:**
- Primary fallback when Arena database unavailable
- Caches in `card_cache.json` (~2.8MB)
- Used for card names, types, oracle text

### MTGA.exe Analysis

**Neither application reverse-engineers the MTGA executable.**

Both rely exclusively on the log file (`Player.log`) which is designed for third-party integrations.

**Detailed Logs Must Be Enabled:**
```
MTGA → Options → Account → Detailed Logs (Plugin Support)
```

---

## 4. How to Enable Drafting in MTGA Voice Advisor

### Implementation Roadmap

#### Phase 1: Event Detection (1-2 hours)

**Add new event parsers to MatchScanner:**

```python
class MatchScanner:
    def __init__(self):
        # ... existing code ...
        self.draft_state = None
        self.pool_state = None

    def parse_log_line(self, line: str):
        # Existing GreToClientEvent parsing
        if "GreToClientEvent" in line:
            # ... existing code ...

        # NEW: Start events with JSON in line
        elif line.startswith("[UnityCrossThreadLogger]==>"):
            self._parse_start_event(line)

        # NEW: End events with JSON on next line
        elif line.startswith("<=="):
            self._parse_end_event_marker(line)

    def _parse_start_event(self, line: str):
        """Parse events like LogBusinessEvents"""
        match = re.search(r'\[UnityCrossThreadLogger\]==> (\w+) (.*)', line)
        if not match:
            return

        event_type = match.group(1)
        outer_json = match.group(2)

        if event_type == "LogBusinessEvents":
            data = json.loads(outer_json)
            inner_data = json.loads(data["request"])
            if "DraftId" in inner_data:
                self._handle_premier_draft_pick(inner_data)

    def _parse_end_event_marker(self, line: str):
        """Mark that next line should be parsed as end event"""
        match = re.search(r'<== (\w+)\(([a-f0-9-]+)\)', line)
        if match:
            self.next_line_event = match.group(1)

    def _handle_premier_draft_pick(self, data: dict):
        """Handle Premier Draft pick event"""
        pack = data.get("CardsInPack", [])
        pack_num = data.get("PackNumber", 0)
        pick_num = data.get("PickNumber", 0)

        logging.info(f"Premier Draft: Pack {pack_num}, Pick {pick_num}")
        logging.debug(f"Cards in pack: {pack}")

        # Trigger draft advice callback
        if self.draft_pick_callback:
            self.draft_pick_callback(pack, pack_num, pick_num, "PremierDraft")
```

#### Phase 2: Draft Advice Generation (2-3 hours)

**Create DraftAdvisor class:**

```python
class DraftAdvisor:
    """Generates draft pick recommendations using 17lands data + LLM"""

    def __init__(self, card_db, rag_system=None, ollama_client=None):
        self.card_db = card_db
        self.rag = rag_system
        self.ollama = ollama_client
        self.picked_cards = []

    def recommend_pick(self, pack_arena_ids: list, pack_num: int, pick_num: int) -> str:
        """Generate pick recommendation"""

        # 1. Resolve card names
        pack_cards = []
        for arena_id in pack_arena_ids:
            card_name = self.card_db.get_card_name(arena_id)
            if card_name:
                pack_cards.append({
                    "arena_id": arena_id,
                    "name": card_name
                })

        # 2. Get 17lands stats if available
        if self.rag and self.rag.card_stats_db:
            for card in pack_cards:
                stats = self.rag.card_stats_db.get_card_stats(card["name"])
                if stats:
                    card["win_rate"] = stats.get("win_rate", 0)
                    card["gih_win_rate"] = stats.get("gih_win_rate", 0)
                    card["iwd"] = stats.get("iwd", 0)

        # 3. Sort by stats
        pack_cards.sort(key=lambda c: c.get("gih_win_rate", 0), reverse=True)

        # 4. Generate LLM recommendation (optional)
        if self.ollama and pick_num < 5:  # Only for early picks
            prompt = self._build_draft_prompt(pack_cards, pack_num, pick_num)
            advice = self.ollama.query(prompt)
        else:
            # Simple stats-based recommendation
            top_card = pack_cards[0]
            advice = f"Pick {top_card['name']} ({top_card.get('gih_win_rate', 0):.1%} GIH WR)"

        return advice

    def _build_draft_prompt(self, pack_cards: list, pack_num: int, pick_num: int) -> str:
        """Build LLM prompt for draft advice"""

        prompt = f"You are a Magic: The Gathering draft expert.\n\n"
        prompt += f"Pack {pack_num}, Pick {pick_num}\n\n"

        if self.picked_cards:
            prompt += f"Already picked: {', '.join(self.picked_cards[:5])}\n\n"

        prompt += "Available cards:\n"
        for i, card in enumerate(pack_cards[:10], 1):  # Top 10 only
            stats = f"{card.get('gih_win_rate', 0):.1%} GIH WR" if 'gih_win_rate' in card else "No stats"
            prompt += f"{i}. {card['name']} - {stats}\n"

        prompt += "\nProvide a brief (1-2 sentence) pick recommendation."
        return prompt
```

#### Phase 3: UI Integration (1-2 hours)

**Add draft display to TUI:**

```python
class AdvisorTUI:
    def __init__(self):
        # ... existing code ...
        self.draft_mode = False
        self.current_pack = []

    def show_draft_pack(self, pack_cards: list, recommendation: str):
        """Display draft pack with recommendations"""
        self.draft_mode = True
        self.current_pack = pack_cards

        # Clear board state window, show pack instead
        self.board_window.clear()

        # Header
        self.board_window.addstr(0, 0, "═══ DRAFT PICK ═══",
                                curses.color_pair(2) | curses.A_BOLD)

        # Show cards with grades
        for i, card in enumerate(pack_cards[:15], 1):  # Top 15
            y = i + 1
            grade = card.get("grade", "")
            win_rate = card.get("gih_win_rate", 0)

            # Color code by grade
            if grade.startswith("A"):
                color = curses.color_pair(1)  # Green
            elif grade.startswith("B"):
                color = curses.color_pair(2)  # Cyan
            else:
                color = 0

            line = f"{i:2}. {card['name']:30} {grade:3} {win_rate:.1%}"
            self.board_window.addstr(y, 2, line, color)

        # Show recommendation
        self.add_message(f"Recommendation: {recommendation}", "advice")

        self.board_window.refresh()
```

**Add to CLI mode:**

```python
class CLIVoiceAdvisor:
    def on_draft_pick(self, pack: list, pack_num: int, pick_num: int, format: str):
        """Callback when draft pick detected"""

        logging.info(f"Draft pick detected: Pack {pack_num}, Pick {pick_num}")

        # Generate recommendation
        advice = self.draft_advisor.recommend_pick(pack, pack_num, pick_num)

        # Display in UI
        if self.tui:
            self.tui.show_draft_pack(pack_cards, advice)
        else:
            # CLI mode
            print(f"\n{'='*60}")
            print(f"Pack {pack_num}, Pick {pick_num}")
            print(f"{'='*60}")
            for i, card in enumerate(pack_cards[:10], 1):
                grade = card.get("grade", "")
                print(f"{i:2}. {card['name']:30} {grade}")
            print(f"\n{advice}")

        # Speak recommendation
        if self.tts and not self.voice_muted:
            # Only speak top pick, not full analysis
            top_pick = pack_cards[0]["name"]
            self.tts.speak(f"Pick {top_pick}")
```

#### Phase 4: 17lands Data Integration (2-3 hours)

**Option A: Use existing RAG system (recommended)**

The MTGA Voice Advisor already has `card_stats.db` with 17lands data. Just use it:

```python
class DraftAdvisor:
    def get_card_grades(self, card_names: list) -> dict:
        """Get grades for cards using existing card_stats.db"""

        grades = {}
        for name in card_names:
            stats = self.rag.card_stats_db.get_card_stats(name)
            if stats:
                # Calculate grade similar to python-mtga-helper
                gih_wr = stats.get("gih_win_rate", 0)
                grade = self._win_rate_to_grade(gih_wr)
                grades[name] = {
                    "grade": grade,
                    "win_rate": stats.get("win_rate", 0),
                    "gih_win_rate": gih_wr,
                    "iwd": stats.get("iwd", 0)
                }

        return grades

    def _win_rate_to_grade(self, win_rate: float) -> str:
        """Convert win rate to letter grade"""
        # Use percentile-based grading
        # This requires calculating mean/std of all cards in set
        # For simplicity, use fixed thresholds:
        if win_rate >= 0.60: return "A+"
        elif win_rate >= 0.58: return "A"
        elif win_rate >= 0.56: return "A-"
        elif win_rate >= 0.54: return "B+"
        elif win_rate >= 0.52: return "B"
        elif win_rate >= 0.50: return "B-"
        elif win_rate >= 0.48: return "C+"
        elif win_rate >= 0.46: return "C"
        else: return "C-"
```

**Option B: Implement 17lands API client (like python-mtga-helper)**

```python
class SeventeenLandsClient:
    """17lands.com API client with caching"""

    def __init__(self, cache_dir: Path = Path("data/17lands_cache")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)

    def get_set_ratings(self, set_code: str, format: str = "PremierDraft") -> dict:
        """Get card ratings for a set"""

        cache_file = self.cache_dir / f"{set_code}_{format}.json"

        # Check cache
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)

        # Query API
        params = {
            "expansion": set_code,
            "format": format,
            "end_date": datetime.now().date().isoformat()
        }

        response = requests.get(
            "https://www.17lands.com/card_ratings/data",
            params=params,
            timeout=30
        )
        response.raise_for_status()

        data = response.json()

        # Cache it
        with open(cache_file, 'w') as f:
            json.dump(data, f)

        return data
```

---

## 5. UI Comparison and Lessons Learned

### python-mtga-helper UI

**Strengths:**
- ✅ Clean tabular output using `tabulate` library
- ✅ Color-coded grades using `termcolor` and RGB calculations
- ✅ Simple, focused output (no complex UI state)
- ✅ Emoji indicators for colors and rarity (✨ for mythic, 💎 for rare, etc.)
- ✅ Clear visual hierarchy (headers, spacing)

**Example Output:**
```
== Pack #1 Pick #3 ==

   🔴 ⚫  Card Name                  A+   62.50  Creature - Dragon
   🔵     Another Card               B    55.30  Instant
   ⚪     Third Card                 C+   48.20  Enchantment

== Pool ==

Top 23 Mean Grade          B+
Top 23 Mean Score          58.42%
Top 23 Grade Range         A+ - C
Total Creatures            15
Total Non Creatures        8
```

**Implementation:**
```python
from tabulate import tabulate
from termcolor import colored

table = [
    (format_color_emoji(color), rarity_emoji(rarity), name,
     colored(grade, color=grade_color), f"{win_rate:.2f}", types)
]
print(tabulate(table, headers=("", "", "Card", "", "Win %", "Type")))
```

### MTGA Voice Advisor TUI

**Issues (as you noted: "garbage"):**
- ❌ Overly complex curses implementation
- ❌ Scrolling mechanics are janky
- ❌ Too many windows/panes for simple information
- ❌ Resize handling is fragile
- ❌ Non-blocking input is tricky and error-prone

**Your Assessment is Correct:**

The TUI in advisor.py is over-engineered for draft display. For draft picks:

**Simple Terminal Output (python-mtga-helper style) is better because:**
1. Draft picks are discrete events (not continuous like gameplay)
2. You need to see the pick, then it's done (no scrolling needed)
3. Color-coded tables are easier to read than curses windows
4. Less code, fewer bugs, faster development

### Recommendation: Hybrid Approach

**For Draft Mode:**
- Use simple `tabulate` + `termcolor` output (like python-mtga-helper)
- Print to stdout, let terminal scrollback handle history
- Add voice recommendations via existing TTS system

**For Gameplay Mode:**
- Keep GUI if user wants it
- Keep TUI as optional
- Make CLI the default (with colors)

**Implementation:**

```python
class CLIVoiceAdvisor:
    def __init__(self, args):
        # Determine mode
        if args.draft_only:
            self.mode = "draft"
            self.ui = None  # No TUI, just print
        elif args.tui:
            self.mode = "play"
            self.ui = AdvisorTUI()
        elif args.gui and TKINTER_AVAILABLE:
            self.mode = "play"
            self.ui = AdvisorGUI()
        else:
            self.mode = "play"
            self.ui = None  # CLI mode

    def display_draft_pick(self, pack_cards: list, advice: str):
        """Display draft pick using simple table output"""

        from tabulate import tabulate
        from termcolor import colored

        print("\n" + "="*70)
        print(f"Pack {self.pack_num}, Pick {self.pick_num}")
        print("="*70 + "\n")

        table = []
        for i, card in enumerate(pack_cards, 1):
            grade = card.get("grade", "")
            grade_str = self._colored_grade(grade)

            table.append([
                i,
                card.get("color_emoji", ""),
                card["name"],
                grade_str,
                f"{card.get('gih_win_rate', 0)*100:.1f}%",
                ", ".join(card.get("types", []))
            ])

        print(tabulate(table,
                      headers=["#", "", "Card", "Grade", "GIH WR", "Type"],
                      tablefmt="simple"))

        print(f"\n💡 {advice}\n")

    def _colored_grade(self, grade: str) -> str:
        """Return color-coded grade string"""
        if not grade:
            return ""

        if grade.startswith("A"):
            return colored(grade, "green", attrs=["bold"])
        elif grade.startswith("B"):
            return colored(grade, "cyan")
        elif grade.startswith("C"):
            return colored(grade, "yellow")
        else:
            return colored(grade, "red")
```

---

## Summary: Action Plan

### Minimal Viable Draft Support (4-6 hours)

1. **Add event detection** for `LogBusinessEvents` and `BotDraftDraftStatus` (~1 hour)
2. **Create DraftAdvisor class** with basic 17lands integration (~2 hours)
3. **Simple table output** using `tabulate` + `termcolor` (~1 hour)
4. **Voice recommendations** using existing TTS (~30 min)
5. **Testing** with actual drafts (~1-2 hours)

### Dependencies to Add

```bash
pip install tabulate termcolor scipy
```

### Files to Create/Modify

**New files:**
- `draft_advisor.py` - Draft pick recommendation logic
- `seventeen_lands_client.py` - 17lands API integration (or use existing RAG)

**Modify:**
- `advisor.py` - Add event detection in MatchScanner
- `requirements.txt` - Add tabulate, termcolor, scipy

### Expected Result

```
$ python advisor.py --draft

Waiting for MTGA draft to start...

===================================================================
Pack 1, Pick 3
===================================================================

#   Card Name                         Grade  GIH WR   Type
--  --------------------------------  -----  -------  ------------------
1   Lightning Strike                  A+     62.5%    Instant
2   Courageous Goblin                 B+     57.2%    Creature - Goblin
3   Scroll of Avacyn                  B      54.1%    Artifact
4   Forest                                   48.0%    Land
...

💡 Pick Lightning Strike (A+ removal, high win rate)

[Speaking: "Pick Lightning Strike"]
```

---

## Conclusion

The python-mtga-helper approach to drafting is **much simpler** than trying to shoehorn it into your existing TUI. Key takeaways:

1. **Don't use curses for draft display** - simple tables are better
2. **17lands data is essential** - use your existing RAG system or add API client
3. **Voice adds unique value** - python-mtga-helper has no voice, you do
4. **Event detection is straightforward** - just need to handle a few new event types
5. **LLM can enhance** - combine 17lands stats with GPT analysis for deeper insights

The "garbage TUI" you mentioned is indeed overcomplicated for this use case. Follow python-mtga-helper's lead: simple, clean, tabular output with color coding.
