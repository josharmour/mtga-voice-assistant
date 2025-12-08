import sqlite3
import logging
from pathlib import Path

path = Path(r"C:\Program Files (x86)\Steam\steamapps\common\MTGA\MTGA_Data\Downloads\Raw\Raw_CardDatabase_22df27f4a70bcc34335164f2cb65717e.mtga")
conn = sqlite3.connect(str(path))
cursor = conn.cursor()
cursor.execute(f"PRAGMA table_info(Abilities)")
columns = [c[1] for c in cursor.fetchall()]
print("Abilities Columns:", columns)
cursor.execute("SELECT * FROM Abilities WHERE GrpId=192762")
row = cursor.fetchone()
if row:
    print("Row:", tuple(row))
conn.close()
