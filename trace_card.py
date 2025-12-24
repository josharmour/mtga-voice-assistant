import os

filename = r"c:\Users\joshu\logparser\bug_reports\Bug Report #60\advisor.log"
search_term = "97381"

if not os.path.exists(filename):
    print(f"File not found: {filename}")
    exit(1)

print(f"File found. Size: {os.path.getsize(filename)} bytes")

import collections

with open(filename, 'r', encoding='utf-8', errors='replace') as f:
    last_lines = collections.deque(f, maxlen=10)
    print("--- Last 10 lines ---")
    for line in last_lines:
        print(line.strip()[:100] + "...")


