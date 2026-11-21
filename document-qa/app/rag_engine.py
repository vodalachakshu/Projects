"""Lightweight, self-contained RAG pipeline with retrieval and grounded generation."""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request

from app.config import Settings

logger = logging.getLogger("rag_service.engine")

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "get", "how", "i", "in", "is", "it", "of", "on", "or",
    "our", "the", "their", "them", "they", "this", "to", "what", "when",
    "where", "which", "who", "why", "with", "you", "your",
}


class RAGConfigurationError(Exception):
    """Raised when configuration or document directory is invalid."""


class RAGRetrievalError(Exception):
    """Raised when document loading, indexing, or retrieval fails."""


class RAGGenerationError(Exception):
    """Raised when answer generation fails."""


@dataclass
class DocumentChunk:
    """Represents a single indexed chunk from a source document."""

    chunk_id: str
    document: str
    text: str
    tokens: list[str]
    term_freq: dict[str, int]
    norm: float


def tokenize(text: str) -> list[str]:
    """Extract normalized alphanumeric tokens excluding common stop words."""
    return [t for t in TOKEN_PATTERN.findall(text.lower()) if t not in STOP_WORDS and len(t) > 1]


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping chunks while respecting paragraph boundaries when possible."""
    cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not cleaned:
        return []

    if len(cleaned) <= chunk_size:
        return [cleaned]

    step = max(1, chunk_size - chunk_overlap)
    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        if end < len(cleaned):
            split_pos = cleaned.rfind("\n", start + step // 2, end)
            if split_pos == -1:
                split_pos = cleaned.rfind(". ", start + step // 2, end)
                if split_pos != -1:
                    split_pos += 1
            if split_pos != -1 and split_pos > start:
                end = split_pos

        segment = cleaned[start:end].strip()
        if segment:
            chunks.append(segment)
        if end >= len(cleaned):
            break
        start = max(start + 1, end - chunk_overlap)

    return chunks


class DocumentQARAGEngine:
    """Loads 5-10 documents, builds a TF-IDF/cosine retriever, and generates grounded answers."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.chunks: list[DocumentChunk] = []
        self.idf: dict[str, float] = {}
        self.doc_stats: dict[str, dict[str, int]] = {}
        self.build_index()

    def _load_documents(self) -> list[tuple[str, str]]:
        """Load supported text/markdown documents from documents_dir."""
        docs_path = Path(self.settings.documents_dir)
        if not docs_path.exists() or not docs_path.is_dir():
            raise RAGConfigurationError(
                f"Documents directory not found: '{self.settings.documents_dir}'"
            )

        supported_exts = {".md", ".txt", ".rst"}
        files = sorted(
            [p for p in docs_path.iterdir() if p.is_file() and p.suffix.lower() in supported_exts]
        )
        if not files:
            raise RAGConfigurationError(
                f"No supported documents (.md, .txt, .rst) found in '{self.settings.documents_dir}'"
            )

        loaded: list[tuple[str, str]] = []
        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8")
            except OSError as exc:
                raise RAGRetrievalError(f"Failed reading file '{file_path.name}': {exc}") from exc
            if content.strip():
                loaded.append((file_path.name, content))

        if not loaded:
            raise RAGConfigurationError("All loaded documents were empty.")

        return loaded

    def build_index(self) -> None:
        """Chunk documents and precompute TF-IDF weights and norms."""
        raw_docs = self._load_documents()
        raw_chunks: list[tuple[str, str, str]] = []
        self.doc_stats = {}

        for doc_name, content in raw_docs:
            parts = chunk_text(
                content,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            )
            self.doc_stats[doc_name] = {
                "chunks": len(parts),
                "characters": len(content),
            }
            for idx, part in enumerate(parts, start=1):
                chunk_id = f"{Path(doc_name).stem}#c{idx}"
                raw_chunks.append((doc_name, chunk_id, part))

        num_chunks = len(raw_chunks)
        doc_freq: dict[str, int] = {}
        tokenized_parts: list[tuple[str, str, str, list[str], dict[str, int]]] = []

        for doc_name, chunk_id, text in raw_chunks:
            tokens = tokenize(text)
            tf: dict[str, int] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            for unique_tok in tf:
                doc_freq[unique_tok] = doc_freq.get(unique_tok, 0) + 1
            tokenized_parts.append((doc_name, chunk_id, text, tokens, tf))

        self.idf = {
            term: math.log((1.0 + num_chunks) / (1.0 + df)) + 1.0
            for term, df in doc_freq.items()
        }

        indexed_chunks: list[DocumentChunk] = []
        for doc_name, chunk_id, text, tokens, tf in tokenized_parts:
            weighted_sq_sum = 0.0
            for term, count in tf.items():
                weight = (1.0 + math.log(count)) * self.idf.get(term, 1.0)
                weighted_sq_sum += weight * weight
            norm = math.sqrt(weighted_sq_sum) if weighted_sq_sum > 0 else 1.0
            indexed_chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document=doc_name,
                    text=text,
                    tokens=tokens,
                    term_freq=tf,
                    norm=norm,
                )
            )

        self.chunks = indexed_chunks
        logger.info(
            "Indexed %d documents into %d chunks from '%s'",
            len(self.doc_stats),
            len(self.chunks),
            self.settings.documents_dir,
        )

    def retrieve(self, question: str, top_k: int | None = None) -> list[tuple[DocumentChunk, float]]:
        """Retrieve top_k most relevant chunks for a question using TF-IDF cosine similarity."""
        if not self.chunks:
            raise RAGRetrievalError("Vector index is empty. Cannot retrieve relevant context.")

        k = top_k or self.settings.top_k
        q_tokens = tokenize(question)
        if not q_tokens:
            return []

        q_tf: dict[str, int] = {}
        for tok in q_tokens:
            q_tf[tok] = q_tf.get(tok, 0) + 1

        q_weights: dict[str, float] = {}
        q_sq_sum = 0.0
        for term, count in q_tf.items():
            if term in self.idf:
                w = (1.0 + math.log(count)) * self.idf[term]
                q_weights[term] = w
                q_sq_sum += w * w

        if q_sq_sum == 0.0:
            return []

        q_norm = math.sqrt(q_sq_sum)
        scored: list[tuple[DocumentChunk, float]] = []

        for chunk in self.chunks:
            dot = 0.0
            for term, q_w in q_weights.items():
                c_count = chunk.term_freq.get(term, 0)
                if c_count > 0:
                    c_w = (1.0 + math.log(c_count)) * self.idf[term]
                    dot += q_w * c_w
            similarity = dot / (q_norm * chunk.norm) if (q_norm * chunk.norm) > 0 else 0.0
            if similarity > 0.0:
                scored.append((chunk, round(similarity, 4)))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]

    def _generate_extractive_grounded_answer(
        self,
        question: str,
        retrieved: list[tuple[DocumentChunk, float]],
    ) -> tuple[str, bool]:
        """Generate a strictly grounded answer from retrieved sentences without external fabrication."""
        if not retrieved:
            return (
                "I could not find relevant information in the provided documents to answer this question.",
                False,
            )

        q_terms = set(tokenize(question))
        candidate_sentences: list[tuple[float, str, str]] = []

        for chunk, chunk_score in retrieved:
            raw_units = re.split(r"(?<=[.!?])\s+|\n+-\s+", chunk.text)
            for unit in raw_units:
                sent = unit.strip().lstrip("-").strip()
                if len(sent) < 15:
                    continue
                s_tokens = set(tokenize(sent))
                overlap = len(q_terms & s_tokens)
                has_number = bool(re.search(r"\b\d+\b", sent))
                num_bonus = 0.5 if (has_number and any(w in q_terms for w in {"many", "days", "hours", "minutes", "limit", "cap"})) else 0.0
                score = (overlap * 1.5) + num_bonus + chunk_score
                if overlap > 0 or num_bonus > 0:
                    candidate_sentences.append((score, sent, chunk.document))

        if not candidate_sentences:
            top_chunk, _ = retrieved[0]
            return (
                f"Based on {top_chunk.document}, the closest relevant context is: {top_chunk.text[:350]}",
                True,
            )

        candidate_sentences.sort(key=lambda item: item[0], reverse=True)
        selected: list[str] = []
        seen_lower: set[str] = set()
        for _, sent, doc_name in candidate_sentences[:3]:
            key = sent.lower()
            if key not in seen_lower:
                seen_lower.add(key)
                selected.append(f"{sent} [{doc_name}]")

        return " ".join(selected), True

    def _generate_openai_answer(
        self,
        question: str,
        retrieved: list[tuple[DocumentChunk, float]],
    ) -> tuple[str, bool]:
        """Call OpenAI Chat Completions API constrained strictly to retrieved context."""
        if not self.settings.openai_api_key:
            raise RAGGenerationError("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")

        context_blocks = "\n\n".join(
            f"\n{chunk.text}"
            for chunk, _ in retrieved
        )
        system_prompt = (
            "You are a factual Document Q&A assistant. Answer the user's question using ONLY "
            "the provided context excerpts. If the context does not contain enough information, "
            "explicitly state that the documents do not cover it. Cite the source filename in brackets."
        )
        user_prompt = f"Context:\n{context_blocks}\n\nQuestion: {question}"

        payload: dict[str, Any] = {
            "model": self.settings.openai_model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        req = request.Request(
            url="https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.settings.openai_api_key}",
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=20) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            raise RAGGenerationError(f"OpenAI request failed: {exc}") from exc

        try:
            content = body["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RAGGenerationError("Unexpected response format from OpenAI API.") from exc

        return content, True

    def answer_question(self, question: str, top_k: int | None = None) -> dict[str, Any]:
        """Retrieve relevant chunks and generate a grounded answer with citations."""
        retrieved = self.retrieve(question=question, top_k=top_k)

        citations = [
            {
                "document": chunk.document,
                "chunk_id": chunk.chunk_id,
                "score": score,
                "excerpt": chunk.text[:240] + ("..." if len(chunk.text) > 240 else ""),
            }
            for chunk, score in retrieved
        ]

        provider = self.settings.llm_provider.lower().strip()
        if provider == "openai":
            answer, grounded = self._generate_openai_answer(question=question, retrieved=retrieved)
        elif provider == "extractive":
            answer, grounded = self._generate_extractive_grounded_answer(
                question=question, retrieved=retrieved
            )
        else:
            raise RAGConfigurationError(
                f"Unsupported LLM_PROVIDER '{self.settings.llm_provider}'. Use 'extractive' or 'openai'."
            )

        return {
            "question": question,
            "answer": answer,
            "grounded": grounded,
            "citations": citations,
            "provider": provider,
        }

    def get_corpus_status(self) -> dict[str, Any]:
        """Return metadata about indexed documents and chunks."""
        documents = [
            {
                "filename": doc_name,
                "chunks": stats["chunks"],
                "characters": stats["characters"],
            }
            for doc_name, stats in sorted(self.doc_stats.items())
        ]
        return {
            "document_count": len(documents),
            "chunk_count": len(self.chunks),
            "documents": documents,
        }
