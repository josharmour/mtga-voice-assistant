#!/usr/bin/env python3
"""
Compare EOE Game Formats for MTG AI Training
"""

import pandas as pd
from pathlib import Path
import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from replay_dtypes import get_dtypes

def analyze_format_safely(filepath, format_name, sample_size=100):
    """Safely analyze a format with robust error handling"""
    print(f"\n📊 {format_name}:")
    print(f"   File: {filepath.name}")
    print(f"   Size: {filepath.stat().st_size / 1024 / 1024:.1f} MB")

    try:
        # Get official dtypes
        dtypes = get_dtypes(str(filepath), print_missing=False)
        print(f"   Official dtypes: {len(dtypes)} columns")

        # Create extra-safe dtypes for analysis
        safe_dtypes = {}
        for col, dtype in dtypes.items():
            if dtype in ['float16', 'float32', 'float64']:
                safe_dtypes[col] = 'object'  # Load as string to avoid conversion issues
            elif dtype in ['int8', 'int16', 'int32', 'int64', 'bool']:
                safe_dtypes[col] = 'object'  # Load as string for safety
            else:
                safe_dtypes[col] = dtype

        # Load very small sample first
        df = pd.read_csv(
            filepath,
            compression='gzip',
            nrows=sample_size,
            dtype=safe_dtypes,
            on_bad_lines='skip',
            low_memory=True
        )

        print(f"   ✅ Loaded sample: {len(df)} games")
        print(f"   Columns: {len(df.columns)}")

        # Basic metrics that should always work
        if 'won' in df.columns:
            # Convert won to numeric safely
            wins = pd.to_numeric(df['won'], errors='coerce').dropna()
            if len(wins) > 0:
                win_rate = (wins == 1).mean() * 100
                print(f"   Win rate: {win_rate:.1f}% (from {len(wins)} valid games)")

        if 'num_turns' in df.columns:
            # Convert turns to numeric safely
            turns = pd.to_numeric(df['num_turns'], errors='coerce').dropna()
            if len(turns) > 0:
                avg_turns = turns.mean()
                print(f"   Avg turns: {avg_turns:.1f} (from {len(turns)} valid games)")

        if 'expansion' in df.columns:
            exp_counts = df['expansion'].value_counts()
            print(f"   Expansions: {dict(exp_counts)}")

        if 'event_type' in df.columns:
            event_counts = df['event_type'].value_counts()
            print(f"   Event types: {dict(event_counts)}")

        # Check for format-specific patterns
        if 'main_colors' in df.columns:
            color_counts = df['main_colors'].value_counts().head(3)
            print(f"   Top main colors: {dict(color_counts)}")

        # Look for columns unique to each format
        format_cols = [col for col in df.columns if format_name.lower().replace(' ', '') in col.lower()]
        if format_cols:
            print(f"   Format-specific columns: {len(format_cols)}")

        return {
            'format': format_name,
            'file_size_mb': filepath.stat().st_size / 1024 / 1024,
            'sample_games': len(df),
            'columns': len(df.columns),
            'win_rate': win_rate if 'win_rate' in locals() else None,
            'avg_turns': avg_turns if 'avg_turns' in locals() else None,
        }

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

def main():
    print("🔍 EOE Game Format Comparison")
    print("=============================")

    data_dir = Path("data/17lands_data")

    # All EOE formats
    formats = [
        ("PremierDraft", "replay_data_public.EOE.PremierDraft.csv.gz"),
        ("TradDraft", "replay_data_public.EOE.TradDraft.csv.gz"),
        ("Sealed", "replay_data_public.EOE.Sealed.csv.gz"),
        ("TradSealed", "replay_data_public.EOE.TradSealed.csv.gz")
    ]

    results = []

    for format_name, filename in formats:
        filepath = data_dir / filename
        if filepath.exists():
            result = analyze_format_safely(filepath, format_name)
            if result:
                results.append(result)
        else:
            print(f"\n❌ {format_name}: File not found")

    # Summary comparison
    print(f"\n📋 Summary Comparison")
    print(f"=====================")

    if results:
        for r in results:
            win_str = f"{r['win_rate']:.1f}%" if r['win_rate'] is not None else "N/A"
            turns_str = f"{r['avg_turns']:.1f}" if r['avg_turns'] is not None else "N/A"
            print(f"{r['format']:12} | {r['file_size_mb']:6.1f}MB | {r['sample_games']:4} games | {r['columns']:4} cols | Win: {win_str:6} | Turns: {turns_str:5}")

        print(f"\n🎯 Recommendations for MTG AI Training:")
        print(f"=====================================")

        # Sort by file size (data availability)
        largest = max(results, key=lambda x: x['file_size_mb'])
        print(f"• Most data: {largest['format']} ({largest['file_size_mb']:.1f}MB)")

        # Check variety
        if len(results) >= 3:
            print(f"• Good variety: {len(results)} different formats available")
        else:
            print(f"• Limited variety: Only {len(results)} formats found")

        # Data quality check
        valid_results = [r for r in results if r['win_rate'] is not None]
        if valid_results:
            avg_wr = sum(r['win_rate'] for r in valid_results) / len(valid_results)
            print(f"• Average win rate across formats: {avg_wr:.1f}%")

        print(f"\n💡 Suggested approach:")
        print(f"   • Use PremierDraft as primary dataset (largest)")
        print(f"   • Add TradDraft for draft variety (different decision patterns)")
        print(f"   • Include Sealed for constructed deck building diversity")
        print(f"   • TradSealed may be too small for reliable training")

if __name__ == "__main__":
    main()