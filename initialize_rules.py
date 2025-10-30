#!/usr/bin/env python3
"""
Initialize the MTG rules vector database.
This only needs to be run once (or when rules are updated).
"""

from rag_advisor import RAGSystem
import logging

logging.basicConfig(level=logging.INFO)

print('Initializing rules database (this will take 2-5 minutes)...')
rag = RAGSystem()
rag.initialize_rules(force_recreate=False)
print('Done! Rules database is ready.')
