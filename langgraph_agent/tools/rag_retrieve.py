# tools/rag_retrieve.py
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_core.tools import StructuredTool
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import os, time

# -------------------------
# Pydantic Schemas
# -------------------------
class RagRetrieveInput(BaseModel):
    q: str = Field(..., description="Natural language query.")
    k: int = Field(5, ge=1, le=20, description="Number of chunks to return.")
    persist_dir: str = Field("chroma", description="Path to Chroma persist directory.")
    collection: str = Field("research_corpus", description="Chroma collection name.")
    min_score: Optional[float] = Field(0.2, ge=0.0, le=1.0, description="Filter out weak matches.")

class RagChunk(BaseModel):
    doc_id: Optional[str] = None
    chunk_id: Optional[str] = None
    text: str
    score: float
    source: Optional[str] = None  # e.g., file path or URL

class RagRetrieveOutput(BaseModel):
    chunks: List[RagChunk]
    latency_ms: int

# -------------------------
# Embeddings (Ollama, local)
# -------------------------
def _get_embedder():
    # Configure via env if you want; sensible defaults below.
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("RAG_EMBED_MODEL", "nomic-embed-text")
    return OllamaEmbeddings(model=model, base_url=base_url)

# -------------------------
# Core retrieval
# -------------------------
@retry(stop=stop_after_attempt(2), wait=wait_exponential(0.5, 0.5, 2.0))
def _retrieve(args: RagRetrieveInput) -> RagRetrieveOutput:
    t0 = time.time()
    embedder = _get_embedder()

    # Load existing persistent collection
    vs = Chroma(
        collection_name=args.collection,
        persist_directory=args.persist_dir,
        embedding_function=embedder,
    )

    # similarity_search_with_relevance_scores returns (Document, score in [0..1])
    results = vs.similarity_search_with_relevance_scores(args.q, k=args.k)

    out: List[RagChunk] = []
    for doc, score in results:
        if args.min_score is not None and score < args.min_score:
            continue
        md = doc.metadata or {}
        out.append(
            RagChunk(
                doc_id=md.get("doc_id") or md.get("source") or md.get("path"),
                chunk_id=md.get("chunk_id") or md.get("id"),
                text=doc.page_content.strip(),
                score=float(score),
                source=md.get("source") or md.get("path") or md.get("url"),
            )
        )

    return RagRetrieveOutput(chunks=out, latency_ms=int((time.time() - t0) * 1000))

def _rag_retrieve_tool(**kwargs) -> RagRetrieveOutput:
    args = RagRetrieveInput(**kwargs)
    return _retrieve(args)


rag_retrieve_tool = StructuredTool.from_function(
    name="rag_retrieve",
    description=(
        "Retrieve top-k relevant chunks from a persistent Chroma vectorstore. "
        "Use this to answer questions from your local notes/PDFs. "
        "Make sure the same embedding model was used at indexing time."
    ),
    func=_rag_retrieve_tool,
    args_schema=RagRetrieveInput,
    return_schema=RagRetrieveOutput,
)
