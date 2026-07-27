from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass

import chromadb
import cohere
import numpy as np

from groundedops.config import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL, RERANK_K, RERANK_MODEL, RETRIEVE_K
from groundedops.documents import Chunk


@dataclass
class SearchResult:
    id: str
    text: str
    document: str
    source_path: str
    chunk_index: int
    retrieval_score: float
    rerank_score: float | None = None


class GroundedRetriever:
    def __init__(self) -> None:
        self.api_key = os.getenv("COHERE_API_KEY")
        self.co = cohere.ClientV2(api_key=self.api_key) if self.api_key else None
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    @property
    def using_cohere(self) -> bool:
        return self.co is not None

    def reset(self) -> None:
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    def index(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        existing = set(self.collection.get(include=[])["ids"])
        to_add = [chunk for chunk in chunks if chunk.id not in existing]
        if not to_add:
            return
        texts = [chunk.text for chunk in to_add]
        self.collection.add(
            ids=[chunk.id for chunk in to_add],
            documents=texts,
            embeddings=self.embed(texts, input_type="search_document"),
            metadatas=[
                {
                    "document": chunk.document,
                    "source_path": chunk.source_path,
                    "chunk_index": chunk.chunk_index,
                }
                for chunk in to_add
            ],
        )

    def retrieve(self, query: str, k: int = RETRIEVE_K) -> tuple[list[SearchResult], float]:
        started = time.perf_counter()
        response = self.collection.query(
            query_embeddings=self.embed([query], input_type="search_query"),
            n_results=min(k, max(1, self.collection.count())),
            include=["documents", "metadatas", "distances"],
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        results: list[SearchResult] = []
        for idx, doc_id in enumerate(response["ids"][0]):
            distance = float(response["distances"][0][idx])
            metadata = response["metadatas"][0][idx]
            results.append(
                SearchResult(
                    id=doc_id,
                    text=response["documents"][0][idx],
                    document=str(metadata["document"]),
                    source_path=str(metadata["source_path"]),
                    chunk_index=int(metadata["chunk_index"]),
                    retrieval_score=1 - distance,
                )
            )
        return results, elapsed_ms

    def rerank(self, query: str, results: list[SearchResult], k: int = RERANK_K) -> tuple[list[SearchResult], float]:
        if not results:
            return [], 0.0
        started = time.perf_counter()
        if self.co:
            docs = [{"text": f"{item.document}\n{item.text}"} for item in results]
            response = self.co.rerank(model=RERANK_MODEL, query=query, documents=docs, top_n=min(k, len(results)))
            ranked = []
            for item in response.results:
                original = results[item.index]
                ranked.append(
                    SearchResult(
                        **{**original.__dict__, "rerank_score": float(item.relevance_score)}
                    )
                )
        else:
            query_terms = set(query.lower().split())
            ranked = sorted(
                (
                    SearchResult(
                        **{
                            **item.__dict__,
                            "rerank_score": lexical_score(query_terms, item.text),
                        }
                    )
                    for item in results
                ),
                key=lambda item: item.rerank_score or 0,
                reverse=True,
            )[:k]
        elapsed_ms = (time.perf_counter() - started) * 1000
        return ranked, elapsed_ms

    def embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        if self.co:
            response = self.co.embed(
                model=EMBED_MODEL,
                texts=texts,
                input_type=input_type,
                embedding_types=["float"],
            )
            return response.embeddings.float
        return [local_embedding(text) for text in texts]


def local_embedding(text: str, dimensions: int = 384) -> list[float]:
    vector = np.zeros(dimensions, dtype=np.float32)
    for token in text.lower().split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % dimensions
        sign = 1 if digest[4] % 2 == 0 else -1
        vector[index] += sign
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector.tolist()


def lexical_score(query_terms: set[str], text: str) -> float:
    text_terms = set(text.lower().split())
    if not query_terms:
        return 0
    return len(query_terms & text_terms) / len(query_terms)
