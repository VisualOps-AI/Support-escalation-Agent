#!/usr/bin/env python3
"""Seed the knowledge base with FAQ data."""

import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.knowledge import KnowledgeBase, KnowledgeIngester


async def main():
    print("Initializing knowledge base...")
    kb = KnowledgeBase(persist_directory="./chroma_data")
    ingester = KnowledgeIngester(kb)

    data_dir = Path(__file__).parent.parent / "data"

    faq_files = [
        ("billing_faq.json", "billing_knowledge"),
        ("technical_faq.json", "technical_knowledge"),
        ("account_faq.json", "account_knowledge"),
    ]

    total_docs = 0

    for filename, collection in faq_files:
        file_path = data_dir / filename
        if not file_path.exists():
            print(f"Warning: {filename} not found, skipping...")
            continue

        count = await ingester.ingest_faq_file(file_path, collection)
        print(f"Ingested {count} documents from {filename} into {collection}")
        total_docs += count

    print(f"\nTotal documents ingested: {total_docs}")
    print("\nCollection stats:")
    for collection in kb.list_collections():
        count = await kb.get_collection_count(collection)
        print(f"  {collection}: {count} documents")


if __name__ == "__main__":
    asyncio.run(main())
