#!/usr/bin/env python3
"""
Test our working MTGA data
"""

import pandas as pd
import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from replay_dtypes import get_dtypes
from pathlib import Path

def test_working_data():
    """Test the one file that worked"""
    print("🧪 Testing Working MTGA Data")
    print("===========================")
    
    data_dir = Path("data/17lands_data")
    
    # Find a working file (try smaller ones)
    test_files = [
        "replay_data_public.AFR.PremierDraft.csv.gz",  # Smaller file
        "replay_data_public.SNC.PremierDraft.csv.gz",  # Another test
    ]
    
    for filename in test_files:
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"❌ File not found: {filename}")
            continue
        
        try:
            print(f"\n🔄 Testing {filename}...")
            
            # Get official data types
            dtypes = get_dtypes(str(filepath), print_missing=False)
            print(f"   Found {len(dtypes)} official column types")
            
            # Use very permissive dtypes
            safe_dtypes = {}
            for col, dtype in dtypes.items():
                if dtype == 'float16':
                    safe_dtypes[col] = 'float32'
                elif dtype in ['int8', 'int16']:
                    safe_dtypes[col] = 'object'  # String for safety
                else:
                    safe_dtypes[col] = dtype
            
            # Load just first 1000 rows
            df = pd.read_csv(
                filepath,
                compression='gzip',
                nrows=1000,  # Only 1000 rows for testing
                dtype=safe_dtypes,
                low_memory=True,
                on_bad_lines='skip'
            )
            
            print(f"✅ Successfully loaded {len(df)} rows")
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {len(df.columns)}")
            print(f"   Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
            
            # Show sample data
            print(f"   Sample expansions: {df['expansion'].value_counts().head(3).to_dict()}")
            print(f"   Sample event types: {df['event_type'].value_counts().head(3).to_dict()}")
            
            # Save working sample
            df.to_parquet(f"working_sample_{filename.replace('.csv.gz', '.parquet')}", index=False)
            print(f"   ✅ Saved: working_sample_{filename.replace('.csv.gz', '.parquet')}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            continue
    
    return False

def main():
    success = test_working_data()
    
    if success:
        print(f"\n🎉 SUCCESS! Working MTGA data created!")
        print(f"Files: working_sample_*.parquet")
        print(f"Ready for: Task 1.3 (Decision Point Extraction)")
        print(f"Sample size: 1000 games for testing")
    else:
        print(f"\n❌ No working data could be created")

if __name__ == "__main__":
    main()
