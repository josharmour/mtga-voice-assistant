import sqlite3
import os

db_path = "data/unified_cards.db"
search_terms = ["Ozai", "Cruelty"]

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

for term in search_terms:
    print(f"--- Searching for '{term}' ---")
    cursor.execute("SELECT * FROM cards WHERE name LIKE ?", (f"%{term}%",))
    rows = cursor.fetchall()
    for row in rows:
        print(f"Found: {row['name']} (grpId: {row['grpId']})")

conn.close()
