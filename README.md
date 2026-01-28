# RAG Assistant

A production-ready Retrieval-Augmented Generation (RAG) application with a modern UI, streaming responses, and support for multiple LLM providers.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![React](https://img.shields.io/badge/React-18+-61DAFB)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Features

- **Document Upload**: Support for PDF and TXT files
- **Intelligent Chunking**: Automatic text splitting with overlap for context preservation
- **Vector Search**: FAISS-powered similarity search with sentence-transformer embeddings
- **Streaming Responses**: Real-time answer generation with Server-Sent Events
- **Multiple LLM Providers**: Support for Groq, HuggingFace, and more via the Inference Providers API
- **Modern UI**: Clean, responsive interface with dark theme
- **Production Ready**: Rate limiting, error handling, and health checks

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Frontend  │────▶│   FastAPI   │────▶│    FAISS    │
│   (React)   │◀────│   Backend   │◀────│  (Vectors)  │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  LLM API    │
                    │ (Groq/HF)   │
                    └─────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- HuggingFace API token

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/rag-assistant.git
   cd rag-assistant
   ```

2. **Set up Python environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env and add your HF_TOKEN
   ```

4. **Build frontend**
   ```bash
   cd frontend
   npm install
   npm run build
   cd ..
   ```

5. **Run the application**
   ```bash
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

6. **Open** http://localhost:8000 in your browser

## Configuration

Key environment variables in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_TOKEN` | HuggingFace API token | Required |
| `HF_INFERENCE_URL` | Model and provider | `meta-llama/Llama-3.3-70B-Instruct:groq` |
| `EMBEDDING_MODEL` | Sentence transformer model | `all-MiniLM-L6-v2` |
| `CHUNK_SIZE` | Document chunk size | `512` |
| `TOP_K` | Number of chunks to retrieve | `10` |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/documents/upload` | POST | Upload a document |
| `/documents` | GET | List all documents |
| `/documents/{id}` | DELETE | Delete a document |
| `/query` | POST | Query with non-streaming response |
| `/query/stream` | POST | Query with streaming response |
| `/health` | GET | Health check |
| `/docs` | GET | OpenAPI documentation |

## Project Structure

```
.
├── app/
│   ├── api/              # API endpoints
│   ├── services/         # Business logic
│   │   ├── llm.py        # LLM integration
│   │   ├── embeddings.py # Vector embeddings
│   │   ├── store.py      # FAISS vector store
│   │   └── retrieval.py  # RAG retrieval
│   ├── schemas/          # Pydantic models
│   └── utils/            # Utilities
├── frontend/
│   └── src/              # React application
├── data/                 # Vector store data
├── uploads/              # Uploaded documents
└── requirements.txt
```

## LLM Providers

The application supports multiple LLM providers through HuggingFace's Inference Providers:

| Provider | Model Example | Free Tier |
|----------|---------------|-----------|
| **Groq** | `meta-llama/Llama-3.3-70B-Instruct:groq` | Yes |
| **HF Inference** | `HuggingFaceTB/SmolLM3-3B:hf-inference` | Yes |
| **Together** | `meta-llama/Llama-3-70B:together` | Limited |

## Deployment

### Docker

```bash
docker build -t rag-assistant .
docker run -p 8000:8000 -e HF_TOKEN=your_token rag-assistant
```

### Railway / Render

1. Push to GitHub
2. Connect your repository
3. Set environment variables
4. Deploy!

## Development

```bash
# Backend (with hot reload)
uvicorn app.main:app --reload --port 8000

# Frontend (development server)
cd frontend && npm run dev
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
