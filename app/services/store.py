"""Vector store using LangChain FAISS and Pinecone integrations."""

from __future__ import annotations

import logging
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS, Pinecone
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import get_settings

logger = logging.getLogger(__name__)

_faiss_store: FAISS | None = None
_pinecone_store: Pinecone | None = None


def _faiss_paths():
    s = get_settings()
    return s.faiss_index_path(), s.faiss_metadata_path()


@lru_cache(maxsize=1)
def _get_embeddings() -> HuggingFaceEmbeddings:
    """Get LangChain embeddings instance."""
    settings = get_settings()
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False},
    )


def _get_faiss_store() -> FAISS:
    """Get or load FAISS vector store."""
    global _faiss_store
    if _faiss_store is None:
        idx_path, _ = _faiss_paths()
        embeddings = _get_embeddings()

        # `save_local(path)` stores files under a DIRECTORY `path/` (not a single file),
        # typically: `index.faiss` and `index.pkl`. Our config uses `data/faiss.index`
        # as that directory.
        index_file = idx_path / "index.faiss"
        meta_file = idx_path / "index.pkl"

        if idx_path.exists() and idx_path.is_dir() and index_file.exists() and meta_file.exists():
            _faiss_store = FAISS.load_local(str(idx_path), embeddings, allow_dangerous_deserialization=True)
            logger.info("Loaded existing FAISS store from %s", idx_path)
        else:
            # No on-disk index yet (or it's incomplete). Keep store uninitialized until first add.
            idx_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("FAISS store not found (or incomplete) at %s; will create on first ingest", idx_path)
    
    return _faiss_store


def _get_pinecone_store() -> Pinecone:
    """Get or create Pinecone vector store."""
    global _pinecone_store
    if _pinecone_store is None:
        settings = get_settings()
        if not settings.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY required when VECTOR_STORE=pinecone")
        
        embeddings = _get_embeddings()
        
        # Initialize Pinecone index if needed
        from pinecone import Pinecone as PineconeClient, ServerlessSpec
        pc = PineconeClient(api_key=settings.PINECONE_API_KEY)
        ix_name = settings.PINECONE_INDEX
        
        if ix_name not in [x.name for x in pc.list_indexes()]:
            pc.create_index(
                name=ix_name,
                dimension=len(embeddings.embed_query("test")),
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        
        _pinecone_store = Pinecone.from_existing_index(
            index_name=ix_name,
            embedding=embeddings,
        )
        logger.info("Connected to Pinecone index: %s", ix_name)
    
    return _pinecone_store


def add_vectors(vectors: list[list[float]], metadata: list[dict[str, Any]]) -> None:
    """Add vectors and metadata to the configured store (FAISS or Pinecone)."""
    store = get_settings().VECTOR_STORE
    if store == "faiss":
        _faiss_add(vectors, metadata)
    else:
        _pinecone_add(vectors, metadata)


def _faiss_add(vectors: list[list[float]], metadata: list[dict[str, Any]]) -> None:
    """Add vectors to FAISS using LangChain."""
    global _faiss_store
    
    idx_path, _ = _faiss_paths()
    embeddings = _get_embeddings()
    
    # Create texts from metadata
    texts = [m.get("text", "") for m in metadata]
    text_embeddings = list(zip(texts, vectors))
    
    index_file = idx_path / "index.faiss"
    meta_file = idx_path / "index.pkl"

    if not (idx_path.exists() and idx_path.is_dir() and index_file.exists() and meta_file.exists()):
        # Create new store with from_embeddings
        _faiss_store = FAISS.from_embeddings(
            text_embeddings=text_embeddings,
            embedding=embeddings,
            metadatas=metadata,
        )
    else:
        # Load existing store and add new embeddings
        if _faiss_store is None:
            _faiss_store = FAISS.load_local(str(idx_path), embeddings, allow_dangerous_deserialization=True)
        
        # Add new documents with their embeddings
        ids = [str(uuid.uuid4()) for _ in texts]
        _faiss_store.add_embeddings(
            text_embeddings=text_embeddings,
            ids=ids,
            metadatas=metadata,
        )
    
    # Save to disk
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    _faiss_store.save_local(str(idx_path))
    doc_count = len(_faiss_store.docstore._dict) if hasattr(_faiss_store, 'docstore') else len(texts)
    logger.info("FAISS: added %d vectors, total documents: %d", len(vectors), doc_count)


PINECONE_META_TEXT_MAX = 8_000  # stay under Pinecone metadata limits


def _pinecone_add(vectors: list[list[float]], metadata: list[dict[str, Any]]) -> None:
    """Add vectors to Pinecone using LangChain."""
    store = _get_pinecone_store()
    
    texts = [m.get("text", "") for m in metadata]
    ids = [str(uuid.uuid4()) for _ in texts]
    
    # Clean metadata for Pinecone (limit text size)
    cleaned_metadata = []
    for m in metadata:
        cleaned = {k: v for k, v in m.items() if k != "_id" and isinstance(v, (str, int, float, bool))}
        if "text" in cleaned and len(cleaned["text"]) > PINECONE_META_TEXT_MAX:
            cleaned["text"] = cleaned["text"][:PINECONE_META_TEXT_MAX]
        cleaned_metadata.append(cleaned)
    
    # Add documents with embeddings
    store.add_embeddings(
        text_embeddings=list(zip(texts, vectors)),
        ids=ids,
        metadatas=cleaned_metadata,
    )
    logger.info("Pinecone: added %d vectors", len(vectors))


def search_vectors(query_vector: list[float], top_k: int) -> list[dict[str, Any]]:
    """Search by vector. Returns list of dicts with 'text', 'source', 'score'."""
    store = get_settings().VECTOR_STORE
    if store == "faiss":
        return _faiss_search(query_vector, top_k)
    return _pinecone_search(query_vector, top_k)


def delete_document(document_id: str) -> None:
    """Best-effort delete of all vectors for a document_id from the configured store."""
    store = get_settings().VECTOR_STORE
    if store == "faiss":
        _faiss_delete_document(document_id)
    else:
        _pinecone_delete_document(document_id)


def _faiss_delete_document(document_id: str) -> None:
    """
    Delete all FAISS entries whose metadata 'source' equals document_id.

    LangChain's FAISS wrapper doesn't provide reliable per-doc deletes across versions,
    so we rebuild the index from remaining documents.
    """
    global _faiss_store
    idx_path, _ = _faiss_paths()
    if not idx_path.exists():
        return

    embeddings = _get_embeddings()
    store = _get_faiss_store()

    # Collect remaining docs
    remaining_texts: list[str] = []
    remaining_metas: list[dict[str, Any]] = []

    # store.docstore._dict contains Document objects
    doc_dict = getattr(getattr(store, "docstore", None), "_dict", {}) or {}
    for doc in doc_dict.values():
        try:
            meta = (doc.metadata or {}).copy()  # type: ignore[attr-defined]
            if meta.get("source") == document_id:
                continue
            text = getattr(doc, "page_content", "")  # type: ignore[attr-defined]
            if not text:
                continue
            meta["text"] = text
            remaining_texts.append(text)
            remaining_metas.append(meta)
        except Exception:
            continue

    # Rebuild (or clear if empty)
    if not remaining_texts:
        # Replace with a fresh empty-ish store
        _faiss_store = FAISS.from_texts([""], embeddings)
        _faiss_store.save_local(str(idx_path))
        logger.info("FAISS: cleared all vectors for document=%s (store rebuilt empty)", document_id)
        return

    _faiss_store = FAISS.from_texts(remaining_texts, embeddings, metadatas=remaining_metas)
    _faiss_store.save_local(str(idx_path))
    logger.info("FAISS: deleted vectors for document=%s (store rebuilt, remaining=%d)", document_id, len(remaining_texts))


def _pinecone_delete_document(document_id: str) -> None:
    """Delete all vectors for a document_id from Pinecone using metadata filter."""
    settings = get_settings()
    if not settings.PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY required when VECTOR_STORE=pinecone")

    from pinecone import Pinecone as PineconeClient

    pc = PineconeClient(api_key=settings.PINECONE_API_KEY)
    idx = pc.Index(settings.PINECONE_INDEX)
    # Filter delete by metadata source
    idx.delete(filter={"source": {"$eq": document_id}}, namespace="default")
    logger.info("Pinecone: deleted vectors for document=%s", document_id)

def _faiss_search(query_vector: list[float], top_k: int) -> list[dict[str, Any]]:
    """Search FAISS using LangChain."""
    idx_path, _ = _faiss_paths()
    if not idx_path.exists():
        return []
    
    store = _get_faiss_store()
    
    # Use similarity_search_with_score_by_vector if available, otherwise use the index directly
    try:
        # Try LangChain method first
        if hasattr(store, 'similarity_search_with_score_by_vector'):
            results = store.similarity_search_with_score_by_vector(query_vector, k=top_k)
        else:
            # Fallback: use the underlying FAISS index directly
            import numpy as np
            query_vec = np.array([query_vector], dtype=np.float32)
            scores, indices = store.index.search(query_vec, top_k)
            # Get documents from docstore
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < 0 or idx >= len(store.docstore._dict):
                    continue
                # Access docstore by index
                doc_id = str(idx)
                if doc_id in store.docstore._dict:
                    doc = store.docstore._dict[doc_id]
                    results.append((doc, float(scores[0][i])))
    except Exception as e:
        logger.error("FAISS search error: %s", e)
        return []
    
    out: list[dict[str, Any]] = []
    for doc, score in results:
        if hasattr(doc, 'metadata') and hasattr(doc, 'page_content'):
            meta = doc.metadata.copy()
            meta["text"] = doc.page_content
        else:
            # Fallback for direct docstore access
            meta = {"text": str(doc), "source": None}
        # FAISS uses L2 distance, lower is better; convert to similarity score
        meta["score"] = -float(score) if score > 0 else float(score)
        out.append(meta)
    
    return out


def _pinecone_search(query_vector: list[float], top_k: int) -> list[dict[str, Any]]:
    """Search Pinecone using LangChain."""
    store = _get_pinecone_store()
    
    # Use similarity_search_with_score_by_vector if available
    try:
        if hasattr(store, 'similarity_search_with_score_by_vector'):
            results = store.similarity_search_with_score_by_vector(query_vector, k=top_k)
        else:
            # Fallback: query Pinecone directly
            from langchain_core.documents import Document
            # Create a dummy query to use the retriever
            retriever = store.as_retriever(search_kwargs={"k": top_k})
            # We need to embed a query, but we have the vector already
            # For now, use the store's query method directly via the underlying index
            results = store.similarity_search_with_score_by_vector(query_vector, k=top_k)
    except Exception as e:
        logger.error("Pinecone search error: %s", e)
        # Fallback: query directly via Pinecone client
        try:
            settings = get_settings()
            from pinecone import Pinecone as PineconeClient
            pc = PineconeClient(api_key=settings.PINECONE_API_KEY)
            idx = pc.Index(settings.PINECONE_INDEX)
            r = idx.query(vector=query_vector, top_k=top_k, include_metadata=True, namespace="default")
            out: list[dict[str, Any]] = []
            for m in (r.matches or []):
                meta = dict(m.metadata or {})
                meta["text"] = meta.get("text", "")
                meta["score"] = float(m.score or 0.0)
                out.append(meta)
            return out
        except Exception as e2:
            logger.error("Pinecone direct query also failed: %s", e2)
            return []
    
    out: list[dict[str, Any]] = []
    for doc, score in results:
        if hasattr(doc, 'metadata') and hasattr(doc, 'page_content'):
            meta = doc.metadata.copy()
            meta["text"] = doc.page_content
        else:
            meta = {"text": str(doc), "source": None}
        meta["score"] = float(score)
        out.append(meta)
    
    return out


def ensure_faiss_loaded() -> None:
    """Load FAISS index from disk if not already in memory."""
    if get_settings().VECTOR_STORE != "faiss":
        return
    _get_faiss_store()  # This will load if needed
