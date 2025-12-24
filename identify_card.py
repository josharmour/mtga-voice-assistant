import sqlite3
import os

db_path = "data/unified_cards.db"
grp_id = 97381

if not os.path.exists(db_path):
    print(f"Error: {db_path} not found.")
    exit(1)

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()
cursor.execute("SELECT * FROM cards WHERE grpId = ?", (grp_id,))
row = cursor.fetchone()

if row:
    print(f"Card Found: {dict(row)}")
else:
    print(f"Card {grp_id} not found.")

conn.close()
