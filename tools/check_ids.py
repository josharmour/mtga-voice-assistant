import sqlite3
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ids_to_check = [192762, 192763, 189377, 1005, 192752, 133461, 1055, 193506, 192884]
db_path = Path("data/unified_cards.db")

def check_ids():
    if not db_path.exists():
        logger.error("DB not found")
        return

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    print(f"Checking {len(ids_to_check)} IDs in {db_path}...")
    
    found_count = 0
    for grp_id in ids_to_check:
        cursor.execute("SELECT name, set_code FROM cards WHERE grpId=?", (grp_id,))
        result = cursor.fetchone()
        if result:
            print(f"✅ Found {grp_id}: {result[0]} ({result[1]})")
            found_count += 1
        else:
            print(f"❌ Missing {grp_id}")
            
    print(f"\nFound {found_count}/{len(ids_to_check)} cards.")
    conn.close()

    # Check Raw DB
    raw_db_path = Path(r"C:\Program Files (x86)\Steam\steamapps\common\MTGA\MTGA_Data\Downloads\Raw\Raw_CardDatabase_22df27f4a70bcc34335164f2cb65717e.mtga")
    if raw_db_path.exists():
        print(f"\nChecking Raw DB: {raw_db_path}")
        try:
            conn = sqlite3.connect(str(raw_db_path))
            cursor = conn.cursor()
            
            # List tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            print("Tables:", [t[0] for t in tables])
            
            for grp_id in ids_to_check:
                found_in_any = False
                for table_row in tables:
                    table_name = table_row[0]
                    # Try to find column named 'GrpId' or similar
                    try:
                        cursor.execute(f"PRAGMA table_info({table_name})")
                        columns = [c[1] for c in cursor.fetchall()]
                        
                        start_col = None
                        if 'GrpId' in columns: start_col = 'GrpId'
                        elif 'Id' in columns: start_col = 'Id'
                        
                        if start_col:
                            cursor.execute(f"SELECT * FROM {table_name} WHERE {start_col}=?", (grp_id,))
                            row = cursor.fetchone()
                            if row:
                                print(f"✅ Found {grp_id} in table '{table_name}'")
                                found_in_any = True
                    except:
                        pass
                
                if not found_in_any:
                    print(f"❌ Completely missing {grp_id} in Raw DB")

            # Inspect data for one ability
            print("\nInspecting Ability 192762:")
            cursor.execute(f"PRAGMA table_info(Abilities)")
            columns = [c[1] for c in cursor.fetchall()]
            print("Columns:", columns)
            
            cursor.execute("SELECT * FROM Abilities WHERE GrpId=192762")
            row = cursor.fetchone()
            if row:
                print("Data:", dict(zip(columns, row)))

            conn.close()
        except Exception as e:
            print(f"Error checking raw DB: {e}")
    else:
        print("Raw DB not found at specified path")

if __name__ == "__main__":
    check_ids()
