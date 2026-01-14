import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .vector_store import KnowledgeBase


@dataclass
class Document:
    content: str
    source: str
    category: str
    title: Optional[str] = None
    metadata: Optional[dict] = None


class KnowledgeIngester:
    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base

    def _generate_id(self, content: str, source: str) -> str:
        hash_input = f"{source}:{content[:100]}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    async def ingest_documents(
        self,
        documents: list[Document],
        collection: str,
    ) -> int:
        if not documents:
            return 0

        texts = []
        metadatas = []
        ids = []

        for doc in documents:
            doc_id = self._generate_id(doc.content, doc.source)
            ids.append(doc_id)
            texts.append(doc.content)
            metadatas.append({
                "source": doc.source,
                "category": doc.category,
                "title": doc.title or "",
                **(doc.metadata or {}),
            })

        await self.kb.add_documents(
            collection=collection,
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )

        return len(documents)

    async def ingest_faq_file(
        self,
        file_path: Path,
        collection: str,
    ) -> int:
        with open(file_path) as f:
            faq_data = json.load(f)

        documents = []
        for item in faq_data:
            content = f"Q: {item['question']}\nA: {item['answer']}"
            documents.append(Document(
                content=content,
                source=f"faq:{file_path.name}",
                category=item.get("category", "general"),
                title=item.get("question", "")[:100],
            ))

        return await self.ingest_documents(documents, collection)

    async def ingest_markdown_file(
        self,
        file_path: Path,
        collection: str,
        category: str,
    ) -> int:
        with open(file_path) as f:
            content = f.read()

        sections = self._split_markdown_sections(content)

        documents = []
        for title, section_content in sections:
            if len(section_content.strip()) < 50:
                continue
            documents.append(Document(
                content=section_content,
                source=f"docs:{file_path.name}",
                category=category,
                title=title,
            ))

        return await self.ingest_documents(documents, collection)

    def _split_markdown_sections(self, content: str) -> list[tuple[str, str]]:
        sections = []
        current_title = "Introduction"
        current_content = []

        for line in content.split("\n"):
            if line.startswith("# "):
                if current_content:
                    sections.append((current_title, "\n".join(current_content)))
                current_title = line[2:].strip()
                current_content = []
            elif line.startswith("## "):
                if current_content:
                    sections.append((current_title, "\n".join(current_content)))
                current_title = line[3:].strip()
                current_content = []
            else:
                current_content.append(line)

        if current_content:
            sections.append((current_title, "\n".join(current_content)))

        return sections
