import chromadb
from chromadb.config import Settings
from typing import Optional
import os


class KnowledgeBase:
    def __init__(self, persist_directory: Optional[str] = None):
        self.persist_directory = persist_directory or os.getenv(
            "CHROMA_PERSIST_DIR", "./chroma_data"
        )

        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )

        self._collections: dict[str, chromadb.Collection] = {}

    def get_or_create_collection(self, name: str) -> chromadb.Collection:
        if name not in self._collections:
            self._collections[name] = self.client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[name]

    async def add_documents(
        self,
        collection: str,
        documents: list[str],
        metadatas: list[dict],
        ids: list[str],
    ) -> None:
        coll = self.get_or_create_collection(collection)
        coll.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

    async def search(
        self,
        collection: str,
        query: str,
        top_k: int = 3,
        where: Optional[dict] = None,
    ) -> list[dict]:
        coll = self.get_or_create_collection(collection)

        results = coll.query(
            query_texts=[query],
            n_results=top_k,
            where=where,
        )

        if not results["documents"] or not results["documents"][0]:
            return []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(documents)
        distances = results["distances"][0] if results["distances"] else [0.0] * len(documents)

        return [
            {
                "content": doc,
                "source": meta.get("source", "unknown"),
                "score": 1 - dist,
                "metadata": meta,
            }
            for doc, meta, dist in zip(documents, metadatas, distances)
        ]

    async def delete_collection(self, name: str) -> None:
        self.client.delete_collection(name)
        if name in self._collections:
            del self._collections[name]

    def list_collections(self) -> list[str]:
        return [c.name for c in self.client.list_collections()]

    async def get_collection_count(self, name: str) -> int:
        coll = self.get_or_create_collection(name)
        return coll.count()
