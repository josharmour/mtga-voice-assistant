# 17lands Data Update Schedule

**Question:** How often should I update the card statistics database?

**Short answer:** Once per set release (~4 times per year) is usually sufficient.

---

## MTG Set Release Schedule

Magic sets are released on a quarterly schedule:

| Quarter | Typical Release | Example Sets |
|---------|----------------|--------------|
| Q1 (Jan-Mar) | Late January/Early Feb | MKM (Feb 2024), ONE (Feb 2023) |
| Q2 (Apr-Jun) | Late April | MOM (Apr 2023), OTJ (Apr 2024) |
| Q3 (Jul-Sep) | Late July/Early Aug | WOE (Sep 2023) |
| Q4 (Oct-Dec) | Late October | LCI (Nov 2023) |

**Update frequency: ~4 times per year** (when new sets release)

---

## When to Update

### ✅ Definitely Update:

1. **New Set Release** (~4 times/year)
   - Wait 2-3 weeks after release for data to stabilize
   - First 1-2 weeks have insufficient games for accurate stats
   - After 3-4 weeks, data is reliable

2. **Major Format Change**
   - Standard rotation (once/year in late summer)
   - Ban list updates (rare)

3. **Starting to use the advisor for the first time**
   - Download current Standard sets
   - Usually 4-6 sets are in Standard at any time

### 🤔 Optional Updates:

1. **During Draft Season** (first 6 weeks of new set)
   - Weekly if you want cutting-edge draft data
   - Useful for Limited (draft/sealed) players
   - Less relevant for Constructed players

2. **Meta Shifts** (1-2 times per set)
   - After major tournaments
   - After balance patches
   - Usually minor impact on card stats

### ❌ Don't Need to Update:

1. **Weekly/Daily** - Overkill, data doesn't change that fast
2. **Between set releases** - Old data remains valid
3. **For casual play** - Even year-old data is fine

---

## Recommended Update Schedule

### For Most Users (Casual/Ranked Play)

```
Update: Once per set release
Timing: 3-4 weeks after set goes live
Frequency: ~4 times per year
Effort: 5-10 minutes per update
```

**Why this works:**
- Card statistics stabilize after 3-4 weeks
- Standard format doesn't change between sets
- Old set data remains accurate
- Low maintenance burden

### For Competitive Players (Draft/Limited Focus)

```
Update: Weekly during draft season
Timing: First 6-8 weeks of new set
Frequency: ~24 times per year (6 weeks × 4 sets)
Effort: 5 minutes per week
```

**Why more frequent:**
- Draft meta evolves quickly
- Early picks change as data accumulates
- Competitive edge from latest trends
- Worth it for serious Limited grinders

### For Set-and-Forget Users

```
Update: Once per year
Timing: After major Standard rotation
Frequency: 1 time per year
Effort: 20 minutes once
```

**Why this works:**
- Card fundamentals don't change
- Historical data still valuable
- Advisor still provides good advice
- Minimal effort

---

## How to Check If You Need an Update

### Check Database Age

```bash
# Last update timestamp
sqlite3 data/card_stats.db "SELECT MAX(last_updated) FROM card_stats;"

# Count cards
sqlite3 data/card_stats.db "SELECT COUNT(*) FROM card_stats;"

# Check which sets you have
sqlite3 data/card_stats.db "SELECT DISTINCT set_code FROM card_stats;"
```

### Check Current Standard Format

Visit: https://whatsinstandard.com/

If you're missing sets that are in Standard, you should update.

---

## Practical Update Examples

### Example 1: Casual Player

**Scenario:** You play MTGA casually, mostly Constructed

**Schedule:**
- **February 2024:** Download MKM data (new set)
- **May 2024:** Download OTJ data (new set)
- **September 2024:** Download Bloomburrow data (new set)
- **November 2024:** Download Duskmourn data (new set)

**Total updates:** 4 per year, ~20 minutes total

### Example 2: Limited Grinder

**Scenario:** You draft 3-4 times per week

**Schedule:**
- **Week 1 of new set:** Wait (insufficient data)
- **Week 2:** Quick sample download (10k rows)
- **Week 3:** Full download (stable data)
- **Week 4-8:** Weekly updates (meta refinement)
- **Week 9+:** Monthly updates (meta stable)

**Total updates:** ~10 per set, ~40 per year

### Example 3: Set-and-Forget

**Scenario:** You installed advisor, want minimal maintenance

**Schedule:**
- **Initial setup:** Download all current Standard sets (1 hour)
- **Annual update:** Download all sets after rotation (1 hour)

**Total updates:** 1-2 per year, ~1-2 hours total

---

## Quick Update Command

For convenience, create an alias:

```bash
# Add to ~/.bashrc or ~/.zshrc
alias update-mtg-data='cd ~/logparser && python3 download_real_17lands_data.py'
```

Then just run:
```bash
update-mtg-data
```

---

## Automation Options

### Option 1: Cron Job (Monthly)

```bash
# Edit crontab
crontab -e

# Add line (runs first of every month at 3 AM)
0 3 1 * * cd /home/joshu/logparser && python3 download_real_17lands_data.py --auto-sample
```

### Option 2: Systemd Timer (After Set Release)

More complex but more flexible scheduling.

### Option 3: Manual Script

Create `update-if-needed.sh`:
```bash
#!/bin/bash
# Check if database is older than 60 days
if [ -f data/card_stats.db ]; then
    AGE=$(( ($(date +%s) - $(stat -c %Y data/card_stats.db)) / 86400 ))
    if [ $AGE -gt 60 ]; then
        echo "Database is $AGE days old, updating..."
        python3 download_real_17lands_data.py
    else
        echo "Database is only $AGE days old, no update needed"
    fi
else
    echo "No database found, downloading..."
    python3 download_real_17lands_data.py
fi
```

---

## Data Freshness Impact

### Impact on Advisor Quality

| Data Age | Impact on Advice | Recommended Action |
|----------|------------------|-------------------|
| **< 1 month** | ✅ Excellent - Current meta | Keep using |
| **1-3 months** | ✅ Good - Slightly dated | Fine for most users |
| **3-6 months** | ⚠️ Okay - Missing new cards | Update if new set released |
| **6-12 months** | ⚠️ Stale - Missing 2-3 sets | Update recommended |
| **> 12 months** | ❌ Outdated - Missing rotation | Update required |

### What Gets Stale

**Does NOT get stale:**
- Card mechanics (Lightning Bolt always deals 3 damage)
- Rules knowledge (RAG rules database)
- Basic strategy principles
- Card interactions

**DOES get stale:**
- New card statistics (cards released after your last update)
- Meta trends (what's popular now)
- Draft pick priorities (evolves with meta)
- Deck archetypes (new combos discovered)

---

## Recommendations by Use Case

### 🎮 Constructed (Standard/Historic)

**Update:** Once per set (4 times/year)
**Why:** Card pools stable between sets

### 🎲 Limited (Draft/Sealed)

**Update:** Weekly for first month of set, then monthly
**Why:** Draft meta evolves quickly

### 🏆 Competitive

**Update:** Weekly during tournament prep
**Why:** Edge matters, latest data helps

### 🎯 Casual

**Update:** Once or twice per year
**Why:** Advisor still helpful with older data

---

## Current Standard Sets (as of October 2024)

If you're starting fresh, download these:

1. **Bloomburrow** (August 2024) - Most recent
2. **Outlaws of Thunder Junction** (April 2024)
3. **Murders at Karlov Manor** (February 2024)
4. **The Lost Caverns of Ixalan** (November 2023)

**Rotation:** September 2025 will rotate out LCI

---

## Summary

**Recommended schedule for most users:**

1. **Initial setup:** Download current Standard sets (4-6 sets)
2. **Quarterly updates:** Download new set 3-4 weeks after release
3. **Annual check:** Verify you have all Standard-legal sets
4. **Ad-hoc:** Update if you notice missing cards in matches

**Time investment:**
- Setup: 30 minutes
- Maintenance: ~20 minutes per year (4 updates × 5 min)
- Total: <1 hour per year

**The advisor works fine even with 6-month-old data!** Updates improve accuracy but aren't critical for casual use.

---

## Quick Reference

```bash
# Check your data age
sqlite3 data/card_stats.db "SELECT MAX(last_updated) FROM card_stats;"

# Update with quick sample (when in doubt)
python3 download_real_17lands_data.py
# Choose option 1

# Full update (quarterly)
python3 download_real_17lands_data.py
# Choose option 2, select latest set code
```

---

END OF DOCUMENT
