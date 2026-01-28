# Architecture

High-level flow: **upload → enqueue ingest → parse/chunk/embed → store (FAISS or Pinecone)**; **query → rate limit → embed → retrieve → HF Inference → answer**.

## Diagram (Mermaid)

Use this diagram in docs, or recreate it in [draw.io](https://app.diagrams.net/) to produce `architecture.drawio`.

```mermaid
flowchart TB
    subgraph Client
        User[User]
    end

    subgraph API[FastAPI API]
        Upload[POST /documents/upload]
        Query[POST /query]
        Status[GET /jobs/id]
        RateLimit[Rate Limiter]
    end

    subgraph Workers[Background Workers]
        ARQ[ARQ + Redis]
        Ingest[Ingest Job: parse, chunk, embed, store]
    end

    subgraph Stores
        FS[(File Store)]
        VS[FAISS or Pinecone]
    end

    subgraph HF_Inference
        LLM[HF Inference API]
    end

    User -->|upload PDF/TXT| Upload
    Upload --> FS
    Upload -->|enqueue| ARQ
    ARQ --> Ingest
    Ingest --> VS
    User -->|question| Query
    Query --> RateLimit
    RateLimit -->|embed query| VS
    VS -->|retrieve chunks| Query
    Query -->|chunks + question| LLM
    LLM -->|answer| User
    User -->|check status| Status
    Status --> ARQ
```

## Components

- **API:** FastAPI; `/documents/upload`, `/documents/jobs/{id}`, `/query`, `/health`; slowapi rate limiting.
- **Worker:** ARQ + Redis; ingest task parses PDF/TXT, chunks, embeds via HF API, stores in FAISS or Pinecone.
- **Stores:** File store (uploads), vector store (FAISS or Pinecone).
- **Inference:** Hugging Face Inference API for embeddings and text-generation only; no local models.
