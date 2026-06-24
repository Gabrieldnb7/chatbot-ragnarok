from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import chromadb
    from chromadb.config import Settings
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise RuntimeError(
        "ChromaDB is required. Install dependencies with: pip install -r requirements.txt"
    ) from exc

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
VECTOR_DB_DIR = DATA_DIR / "vector_db" / "chroma_data"
COLLECTION_NAME = "ragnarok_knowledge_base"
MODEL_NAME = "all-MiniLM-L6-v2"

VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_chroma_client():
    return chromadb.PersistentClient(
        path=str(VECTOR_DB_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


@lru_cache(maxsize=8)
def get_collection(collection_name: str = COLLECTION_NAME):
    return get_chroma_client().get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


@lru_cache(maxsize=1)
def _get_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError(
            "sentence-transformers is required. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    return SentenceTransformer(MODEL_NAME)


def _normalize_collection_name(collection_name: str) -> str:
    if not isinstance(collection_name, str) or not collection_name.strip():
        return COLLECTION_NAME
    return collection_name.strip()


def store_in_vector_db(embedded_chunks: list, collection_name: str = COLLECTION_NAME) -> bool:
    """Persist embedded chunks into ChromaDB."""
    if not isinstance(embedded_chunks, list):
        raise TypeError("embedded_chunks deve ser uma lista.")

    if not embedded_chunks:
        return False

    collection_name = _normalize_collection_name(collection_name)

    try:
        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for idx, chunk in enumerate(embedded_chunks):
            if not isinstance(chunk, dict):
                raise TypeError(f"Item na posicao {idx} nao e um dicionario.")
            if "embedding" not in chunk:
                raise KeyError(f"Chunk na posicao {idx} nao possui a chave 'embedding'.")
            if "texto" not in chunk:
                raise KeyError(f"Chunk na posicao {idx} nao possui a chave 'texto'.")

            ids.append(str(chunk.get("id", f"doc_chunk_{idx}")))
            embeddings.append(chunk["embedding"])
            documents.append(str(chunk["texto"]))
            metadatas.append(dict(chunk.get("metadata", {})))

        collection = get_collection(collection_name)
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        return True
    except Exception as exc:
        print(f"Erro ao persistir no banco vetorial: {exc}")
        return False


def search_similar_chunks(query: str, top_k: int = 4, collection_name: str = COLLECTION_NAME) -> list[dict[str, Any]]:
    """Search the vector store for chunks semantically similar to a query."""
    if not isinstance(query, str):
        raise TypeError("query deve ser uma string.")

    if top_k <= 0:
        raise ValueError("top_k deve ser maior que zero.")

    query = query.strip()
    if not query:
        return []

    collection_name = _normalize_collection_name(collection_name)
    model = _get_embedding_model()
    query_embedding = model.encode([query], show_progress_bar=False, normalize_embeddings=True)[0].tolist()

    result = get_collection(collection_name).query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    ids = (result.get("ids") or [[]])[0]

    retrieved: list[dict[str, Any]] = []
    for idx, texto in enumerate(documents):
        metadata = metadatas[idx] if idx < len(metadatas) and isinstance(metadatas[idx], dict) else {}
        distance = float(distances[idx]) if idx < len(distances) else 1.0
        retrieved.append(
            {
                "id": ids[idx] if idx < len(ids) else metadata.get("chunk_id", f"chunk_{idx + 1}"),
                "doc_id": metadata.get("doc_id", metadata.get("doc", "desconhecido")),
                "texto": texto,
                "metadata": metadata,
                "distance": distance,
                "score": max(0.0, 1.0 - distance),
            }
        )

    return retrieved


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    print(f"Vector DB path: {VECTOR_DB_DIR}")
    print(f"Collection: {COLLECTION_NAME}")
