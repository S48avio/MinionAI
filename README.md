# MinionAI

<p align="center">
  <img src="frontend/public/statics/logo.png" alt="MinionAI logo" width="160" />
</p>

MinionAI is an Aonz.ai chat and PDF retrieval application with a React/Vite frontend and a FastAPI backend. It routes Aonz.ai questions through a Chroma knowledge base and sends unrelated questions directly to an OpenAI-compatible LiteLLM proxy. Requests are traced with Langfuse and responses can be cached in Valkey using exact and semantic matching.

## Request flow

```text
Question
  -> exact Valkey cache
  -> semantic Valkey cache
  -> Aonz.ai routing agent
       -> RAG: Chroma retrieval -> LiteLLM answer formatting
       -> General: LiteLLM answer
  -> store final response in Valkey
```

Valkey is treated as an optimization. If it is unavailable, chat requests continue through the normal LLM/RAG flow.

## Project structure

```text
minion/
├── backend/
│   ├── cache.py         # Exact and semantic Valkey cache
│   ├── config/
│   │   └── logger.py    # Rotating application/error log configuration
│   ├── logs/            # Generated runtime logs (Git-ignored)
│   ├── schemas/
│   │   └── models.py    # Pydantic request and response schemas
│   ├── main.py          # Chat routing, RAG orchestration, and health API
│   ├── rag.py           # Mounted PDF ingestion and retrieval service
│   └── chroma_data/     # Generated Chroma vectors (Git-ignored)
├── frontend/
│   ├── src/             # React components and styles
│   └── package.json     # Frontend dependencies and scripts
├── pyproject.toml       # Python dependencies
└── uv.lock              # Locked Python dependencies
```

## Requirements

- Python 3.13 or newer
- [uv](https://docs.astral.sh/uv/)
- Node.js and npm
- A running LiteLLM proxy
- A Langfuse project
- Valkey with the Valkey Search module for semantic caching

The application can run without Valkey, but caching will be disabled. Exact caching needs standard Valkey commands; semantic caching additionally requires `FT.CREATE` and `FT.SEARCH` support.

## Environment variables

Copy the backend environment template:

```bash
cp backend/.env.example backend/.env
```

Then replace the placeholder values in `backend/.env`:

```env
LITELLM_API_KEY=your-litellm-api-key
LITELLM_BASE_URL=http://your-litellm-host
LITELLM_MODEL=gpt4

LANGFUSE_PUBLIC_KEY=your-langfuse-public-key
LANGFUSE_SECRET_KEY=your-langfuse-secret-key
LANGFUSE_HOST=https://cloud.langfuse.com

VALKEY_HOST=localhost
VALKEY_PORT=6379
VALKEY_DB=0
CACHE_TTL=3600
SEMANTIC_THRESHOLD=0.92
```

Do not commit real API keys to version control.

The frontend calls `http://127.0.0.1:8000` by default. To use another backend URL, create `frontend/.env`:

```env
VITE_API_URL=http://your-backend-host
```

## Install dependencies

Install the backend dependencies from the project root:

```bash
uv sync
```

Install the frontend dependencies:

```bash
cd frontend
npm install
cd ..
```

## Run locally

Start the FastAPI backend from the backend directory:

```bash
cd backend
uv run uvicorn main:app --reload
```

The Sentence Transformer embedding model is loaded during startup. Its first run may take longer while the model is downloaded.

The backend runs at `http://127.0.0.1:8000`. The APIs have separate interactive documentation while running in the same FastAPI/Uvicorn process:

- Chat API: `http://127.0.0.1:8000/docs`
- RAG API: `http://127.0.0.1:8000/rag/docs`

The RAG routes are available under `/rag`, including `/rag/upload`, `/rag/search`, `/rag/documents`, and `/rag/health`.

In a second terminal, start the frontend:

```bash
cd frontend
npm run dev
```

Open the URL printed by Vite, normally `http://localhost:5173`.

## Chat API

Send a request to `POST /chat`:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What services does Aonz.ai provide?","top_k":5}'
```

Example response:

```json
{
  "answer": "Aonz.ai provides...",
  "route": "rag",
  "sources": [
    {
      "filename": "aonz.pdf",
      "document_id": "document-uuid",
      "page_number": 2,
      "chunk_number": 1,
      "distance": 0.18
    }
  ]
}
```

The routing agent returns `rag` for Aonz.ai questions and `general` for unrelated questions. RAG responses include the retrieved sources unless the final answer came from the cache. Each uncached request creates Langfuse observations for routing and answer generation.

## Valkey cache

Cache lookup occurs before the routing agent:

1. Exact caching normalizes the question and looks up its SHA-256 key.
2. Semantic caching embeds the question and uses an HNSW cosine-distance search to find a similarly worded cached question.
3. A semantic result is used only when its similarity meets `SEMANTIC_THRESHOLD`.
4. On a miss, the final LLM response and route are stored with `CACHE_TTL`.

For example, after caching `What services does Aonz.ai provide?`, a question such as `Tell me what Aonz offers` may produce a semantic cache hit even though the text differs. Cached RAG responses currently return an empty `sources` list because only the final answer and route are cached.

## RAG API

The RAG application is mounted at `/rag` and has its own Swagger UI at `http://127.0.0.1:8000/rag/docs`.

Upload and embed a PDF:

```bash
curl -X POST http://127.0.0.1:8000/rag/upload \
  -F "file=@document.pdf;type=application/pdf"
```

Search the embedded content:

```bash
curl -X POST http://127.0.0.1:8000/rag/search \
  -H "Content-Type: application/json" \
  -d '{"query":"What are the main findings?","top_k":5}'
```

Other RAG endpoints:

- `GET /rag/documents` lists uploaded documents.
- `DELETE /rag/documents/{document_id}` removes a document and its vectors.
- `GET /rag/health` reports the collection status and vector count.

Chroma persists generated vectors in `backend/chroma_data/`. This directory is local runtime data and is excluded from Git. Deleting it resets the local document collection.

## Health and logs

Check the complete backend and Valkey connection:

```bash
curl http://127.0.0.1:8000/health
```

Check the RAG collection separately:

```bash
curl http://127.0.0.1:8000/rag/health
```

Runtime logs use rotation and are written to:

- `backend/logs/application.log` for application activity.
- `backend/logs/error.log` for errors and tracebacks.

If `/chat` reports an OpenAI `502 Bad Gateway`, Valkey and Chroma may still be healthy. A 502 in this path means the configured LiteLLM proxy or its upstream model failed; check the LiteLLM service, model mapping, and credentials.

## Build the frontend

```bash
cd frontend
npm run build
```

The production files are generated in `frontend/dist`.
