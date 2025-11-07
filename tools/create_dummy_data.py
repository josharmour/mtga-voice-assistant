import sqlite3
import pandas as pd

# Create a dummy database with more comprehensive sample data
conn = sqlite3.connect("data/card_stats.db")
cursor = conn.cursor()

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
        format TEXT,
        last_updated TEXT
    )
""")

# More comprehensive sample data
sample_data = [
    ('Card A', 'ONE', 'White', 'Common', 1000, 0.55, 7.5, 500, 0.60, 0.58, 0.57, 0.56, 0.54, 8.0, 7.5, 0.05, 'PremierDraft', '2023-01-01'),
    ('Card B', 'ONE', 'Blue', 'Uncommon', 1200, 0.60, 6.5, 600, 0.65, 0.63, 0.62, 0.61, 0.59, 7.0, 6.5, 0.06, 'PremierDraft', '2023-01-01'),
    ('Card C', 'MOM', 'Black', 'Rare', 800, 0.65, 5.5, 400, 0.70, 0.68, 0.67, 0.66, 0.64, 6.0, 5.5, 0.07, 'PremierDraft', '2023-01-01'),
    ('Card D', 'MOM', 'Red', 'Mythic', 500, 0.70, 4.5, 250, 0.75, 0.73, 0.72, 0.71, 0.69, 5.0, 4.5, 0.08, 'PremierDraft', '2023-01-01'),
    ('Card E', 'LTR', 'Green', 'Common', 1500, 0.50, 8.5, 750, 0.55, 0.53, 0.52, 0.51, 0.49, 9.0, 8.5, 0.04, 'PremierDraft', '2023-01-01'),
    ('Card F', 'LTR', 'White', 'Uncommon', 1100, 0.58, 7.0, 550, 0.63, 0.61, 0.60, 0.59, 0.57, 7.5, 7.0, 0.055, 'PremierDraft', '2023-01-01'),
    ('Card G', 'WOE', 'Blue', 'Rare', 900, 0.62, 6.0, 450, 0.67, 0.65, 0.64, 0.63, 0.61, 6.5, 6.0, 0.065, 'PremierDraft', '2023-01-01'),
    ('Card H', 'WOE', 'Black', 'Mythic', 600, 0.68, 5.0, 300, 0.73, 0.71, 0.70, 0.69, 0.67, 5.5, 5.0, 0.075, 'PremierDraft', '2023-01-01'),
    ('Card I', 'LCI', 'Red', 'Common', 1300, 0.53, 8.0, 650, 0.58, 0.56, 0.55, 0.54, 0.52, 8.5, 8.0, 0.045, 'PremierDraft', '2023-01-01'),
    ('Card J', 'LCI', 'Green', 'Uncommon', 1400, 0.56, 7.5, 700, 0.61, 0.59, 0.58, 0.57, 0.55, 8.0, 7.5, 0.05, 'PremierDraft', '2023-01-01'),
]

cursor.executemany("""
    INSERT OR REPLACE INTO card_stats VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""", sample_data)

conn.commit()
conn.close()
