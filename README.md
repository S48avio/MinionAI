# MinionAI

<p align="center">
  <img src="frontend/public/statics/logo.png" alt="MinionAI logo" width="160" />
</p>

MinionAI is a chat application with a React and Vite frontend and a FastAPI backend. The backend sends chat requests through an OpenAI-compatible LiteLLM proxy and records model traces with Langfuse.

## Project structure

```text
minion/
├── backend/
│   └── main.py          # FastAPI chat endpoint
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

## Environment variables

Copy the backend environment template:

```bash
cp backend/.env.example backend/.env
```

Then replace the placeholder values in `backend/.env`:

```env
LITELLM_API_KEY=your-litellm-api-key
LITELLM_BASE_URL=http://your-litellm-host

LANGFUSE_PUBLIC_KEY=your-langfuse-public-key
LANGFUSE_SECRET_KEY=your-langfuse-secret-key
LANGFUSE_HOST=https://cloud.langfuse.com
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

Start the FastAPI backend from the project root:

```bash
uv run uvicorn backend.main:app --reload
```

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
  -d '{"message":"What is a lion?"}'
```

Example response:

```json
{
  "answer": "A lion is a large cat..."
}
```

Each request creates a Langfuse trace for the complete API operation and a generation observation for the model call, including its response and token usage.

## Build the frontend

```bash
cd frontend
npm run build
```

The production files are generated in `frontend/dist`.
