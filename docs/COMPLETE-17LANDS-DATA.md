# Complete 17lands Data Access Guide

**Date**: 2025-10-28
**Status**: ✅ COMPLETE - All 29 sets verified and working

---

## Summary

17lands provides **29 sets** with publicly available game data. The download scripts now support **all 29 sets** including special formats like OM1's PickTwoDraft.

---

## Available Sets (29 Total)

### 2025 Sets (4 sets)
- **EOE** - Edge of Eternities
- **FIN** - Final Fantasy
- **TDM** - Tarkir: Dragonstorm
- **DFT** - Aetherdrift

### 2024-2025 Sets (7 sets)
- **OM1** - Through the Omenpaths (uses PickTwoDraft format)
- **FDN** - Foundations
- **DSK** - Duskmourn: House of Horror
- **BLB** - Bloomburrow
- **MH3** - Modern Horizons 3
- **OTJ** - Outlaws of Thunder Junction
- **MKM** - Murders at Karlov Manor

### 2023-2024 Sets (5 sets)
- **LCI** - The Lost Caverns of Ixalan
- **WOE** - Wilds of Eldraine
- **LTR** - The Lord of the Rings: Tales of Middle-earth
- **MOM** - March of the Machine
- **ONE** - Phyrexia: All Will Be One

### 2022-2023 Sets (6 sets)
- **BRO** - The Brothers' War
- **DMU** - Dominaria United
- **SNC** - Streets of New Capenna
- **NEO** - Kamigawa: Neon Dynasty
- **VOW** - Innistrad: Crimson Vow
- **MID** - Innistrad: Midnight Hunt

### 2021-2022 Sets (3 sets)
- **AFR** - Adventures in the Forgotten Realms
- **STX** - Strixhaven: School of Mages
- **KHM** - Kaldheim

### Special/Remastered/Masters Sets (4 sets)
- **SIR** - Shadows over Innistrad Remastered
- **PIO** - Pioneer Masters
- **HBG** - Alchemy Horizons: Baldur's Gate
- **KTK** - Khans of Tarkir

---

## Current Standard (8 sets)

As of October 2025, Standard includes:
- OM1 (Through the Omenpaths)
- FDN (Foundations)
- DSK (Duskmourn)
- BLB (Bloomburrow)
- OTJ (Outlaws of Thunder Junction)
- MKM (Murders at Karlov Manor)
- LCI (The Lost Caverns of Ixalan)
- WOE (Wilds of Eldraine)

---

## Data Formats

Each set typically has data in multiple formats:

### Draft Formats:
- **PremierDraft** - Standard ranked draft (most common)
- **TradDraft** - Traditional draft (BO3)
- **PickTwoDraft** - New pick-two method (OM1 only)
- **PickTwoTradDraft** - Traditional pick-two (OM1 only)

### Sealed Formats:
- **Sealed** - Standard sealed
- **TradSealed** - Traditional sealed (BO3)

### Data Types:
- **game_data** - Game outcomes, win rates, card performance
- **draft_data** - Draft pick data, ALSA, pick rates
- **replay_data** - Full game replays (not all sets)

---

## Special Format Handling

Most sets use **PremierDraft** as the default format, but some sets use special formats:

| Set | Default Format | Reason |
|-----|---------------|--------|
| OM1 | PickTwoDraft  | Uses new Pick-Two draft method |
| (others) | PremierDraft | Standard format |

The `download_real_17lands_data.py` script automatically uses the correct format for each set via the `SET_FORMATS` mapping.

---

## Download Scripts

### 1. download_real_17lands_data.py
**Main download script** - Downloads CSV files and parses into database.

```bash
python3 download_real_17lands_data.py
```

**Options:**
- 1: Quick sample (test - 1 minute)
- 2: Single set (10-30 minutes)
- 3: Current Standard (8 sets, 60-180 minutes) **← RECOMMENDED**
- 4: All 29 sets (several hours)

**Features:**
- ✅ Auto-detects correct format (PremierDraft vs PickTwoDraft)
- ✅ Downloads compressed CSV from S3
- ✅ Parses and aggregates card statistics
- ✅ Inserts into `data/card_stats.db`

### 2. update_card_data.py
**Smart updater** - Only downloads what's missing or outdated.

```bash
# Check status
python3 update_card_data.py --status

# Update outdated sets (interactive)
python3 update_card_data.py

# Auto-update (no prompts)
python3 update_card_data.py --auto

# Update sets older than 30 days
python3 update_card_data.py --max-age 30
```

**Features:**
- ✅ Checks which sets you already have
- ✅ Checks age of each set's data
- ✅ Only downloads missing or old data
- ✅ Perfect for cron jobs

### 3. check_all_sets_comprehensive.py
**Availability checker** - Verifies which sets have data.

```bash
python3 check_all_sets_comprehensive.py
```

Checks all 64 sets from 17lands page and reports which have public data.

---

## Example Usage

### Initial Setup (First Time)

Download current Standard sets:

```bash
python3 download_real_17lands_data.py
# Choose option 3 (Current Standard - 8 sets)
# Takes 60-180 minutes
# Results: ~6000-8000 cards in database
```

### Monthly Maintenance

Update outdated data:

```bash
python3 update_card_data.py --status
# Shows which sets need updating

python3 update_card_data.py --auto
# Downloads only what's needed
```

### Automatic Updates (Cron)

Set up monthly updates:

```bash
# Edit crontab
crontab -e

# Add this line (runs monthly on the 1st at 3 AM)
0 3 1 * * cd /home/joshu/logparser && python3 update_card_data.py --auto --max-age 60
```

---

## Database Schema

Cards are stored in `data/card_stats.db` with fields:

```sql
CREATE TABLE card_stats (
    card_name TEXT,
    set_code TEXT,
    games_played INTEGER,
    win_rate REAL,
    gih_win_rate REAL,      -- Games In Hand Win Rate
    opening_hand_win_rate REAL,
    drawn_win_rate REAL,
    iwd REAL,               -- Improvement When Drawn
    alsa REAL,              -- Average Last Seen At (draft pick)
    avg_taken_at REAL,
    last_updated TEXT,
    PRIMARY KEY (card_name, set_code)
);
```

---

## Data Quality Notes

### Minimum Thresholds

The parser uses these minimums:
- **Quick sample**: 10 games (for testing)
- **Full download**: 1000 games (production)

### Data Freshness

From `DATA-UPDATE-SCHEDULE.md`:
- Update **once per set** (4x/year) for most users
- Update **weekly** during draft season for competitive players
- Update **monthly** for maintenance

### 17lands Data Policy

Per 17lands:
- Win rates only shown for cards with 500+ games (as of Sept 2023)
- Data is aggregated and anonymized
- Rate limiting applies - don't spam requests

---

## Troubleshooting

### Issue: "403 Forbidden" for a set

**Cause**: Set doesn't have data in that format.

**Solution**: Check if set uses special format (e.g., OM1 uses PickTwoDraft, not PremierDraft).

```bash
# Check available formats for a set
for format in PremierDraft PickTwoDraft Sealed; do
  curl -s -I "https://17lands-public.s3.amazonaws.com/analysis_data/game_data/game_data_public.OM1.$format.csv.gz" | grep HTTP
done
```

### Issue: Download takes too long

**Solution**: Use quick sample first to test, then full download.

```bash
# Quick test (1 minute)
python3 download_real_17lands_data.py
# Choose option 1

# Full download later
python3 download_real_17lands_data.py
# Choose option 3
```

### Issue: Which sets should I download?

**Solution**: Depends on your use case:

| Use Case | Recommendation | Sets | Time |
|----------|---------------|------|------|
| Arena Standard player | Current Standard | 8 sets | 60-180 min |
| Historic player | All available | 29 sets | 3-5 hours |
| Limited/Draft only | Latest 3 sets | 3 sets | 30-60 min |
| Testing | Quick sample | 1 set | 1 min |

---

## URLs and Resources

- **17lands Public Datasets**: https://www.17lands.com/public_datasets
- **S3 Base URL**: https://17lands-public.s3.amazonaws.com/analysis_data/
- **Game Data Pattern**: `game_data/game_data_public.{SET}.{FORMAT}.csv.gz`
- **Draft Data Pattern**: `draft_data/draft_data_public.{SET}.{FORMAT}.csv.gz`

---

## Summary

✅ **29 sets available** (verified October 2025)
✅ **Scripts updated** to handle all sets and formats
✅ **OM1 working** via PickTwoDraft format
✅ **EOE added** (Edge of Eternities)
✅ **Current Standard** includes all 8 legal sets

**Next steps:**
1. Run `python3 download_real_17lands_data.py` (choose option 3)
2. Wait 60-180 minutes for Standard sets to download
3. Run `python3 update_card_data.py --status` monthly to check for updates

---

END OF DOCUMENT
