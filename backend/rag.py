from pathlib import Path
from uuid import uuid4

import chromadb
import pymupdf
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


app = FastAPI(
    title="Persistent Chroma RAG Service",
    description="Upload, search, list, and delete PDF documents.",
)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

CHROMA_PATH = Path(__file__).resolve().parent / "chroma_data"
COLLECTION_NAME = "pdf_documents"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


# ---------------------------------------------------------
# Embedding model
# ---------------------------------------------------------

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


# ---------------------------------------------------------
# Persistent Chroma database
# ---------------------------------------------------------

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={
        "hnsw:space": "cosine"
    }
)


# ---------------------------------------------------------
# Request models
# ---------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


# ---------------------------------------------------------
# PDF processing
# ---------------------------------------------------------

def extract_pdf_pages(file_bytes: bytes) -> list[dict]:
    try:
        document = pymupdf.open(
            stream=file_bytes,
            filetype="pdf"
        )
    except Exception as exc:
        raise ValueError("The uploaded file is not a valid PDF") from exc

    pages = []

    try:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            if text:
                pages.append(
                    {
                        "page_number": page_number,
                        "text": text
                    }
                )
    finally:
        document.close()

    return pages


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100
) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError(
            "Chunk overlap must be smaller than chunk size"
        )

    words = text.split()
    chunks = []

    step = chunk_size - overlap

    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

    return chunks


def create_document_chunks(
    pages: list[dict],
    document_id: str,
    filename: str
) -> tuple[list[str], list[str], list[dict]]:
    ids = []
    documents = []
    metadatas = []

    for page in pages:
        page_number = page["page_number"]

        page_chunks = chunk_text(
            text=page["text"],
            chunk_size=500,
            overlap=100
        )

        for chunk_number, chunk in enumerate(
            page_chunks,
            start=1
        ):
            chunk_id = (
                f"{document_id}-"
                f"page-{page_number}-"
                f"chunk-{chunk_number}"
            )

            ids.append(chunk_id)
            documents.append(chunk)

            metadatas.append(
                {
                    "document_id": document_id,
                    "filename": filename,
                    "page_number": page_number,
                    "chunk_number": chunk_number,
                    "embedding_model": EMBEDDING_MODEL_NAME
                }
            )

    return ids, documents, metadatas


# ---------------------------------------------------------
# Upload and embed PDF
# ---------------------------------------------------------

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...)
):
    filename = file.filename or "uploaded.pdf"

    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded PDF is empty"
        )

    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="PDF exceeds the 20 MB upload limit"
        )

    if not file_bytes.startswith(b"%PDF"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file does not appear to be a valid PDF"
        )

    try:
        pages = extract_pdf_pages(file_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc

    if not pages:
        raise HTTPException(
            status_code=400,
            detail=(
                "No extractable text was found. "
                "The PDF may be scanned and require OCR."
            )
        )

    document_id = str(uuid4())

    ids, documents, metadatas = create_document_chunks(
        pages=pages,
        document_id=document_id,
        filename=filename
    )

    if not documents:
        raise HTTPException(
            status_code=400,
            detail="No usable text chunks were created"
        )

    embeddings = embedding_model.encode(
        documents,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False
    ).tolist()

    try:
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save embeddings: {exc}"
        ) from exc

    return {
        "message": "PDF embedded successfully",
        "document_id": document_id,
        "filename": filename,
        "pages_processed": len(pages),
        "chunks_created": len(documents),
        "vectors_in_database": collection.count()
    }


# ---------------------------------------------------------
# Search relevant chunks
# ---------------------------------------------------------

@app.post("/search")
def search_documents(request: SearchRequest):
    query_embedding = embedding_model.encode(
        [request.query],
        normalize_embeddings=True,
        show_progress_bar=False
    ).tolist()

    available_chunks = collection.count()

    if available_chunks == 0:
        raise HTTPException(
            status_code=404,
            detail="No documents have been uploaded"
        )

    result = collection.query(
        query_embeddings=query_embedding,
        n_results=min(request.top_k, available_chunks),
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    matches = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):
        matches.append(
            {
                "text": document,
                "filename": metadata["filename"],
                "document_id": metadata["document_id"],
                "page_number": metadata["page_number"],
                "chunk_number": metadata["chunk_number"],
                "distance": distance
            }
        )

    return {
        "query": request.query,
        "matches": matches
    }


# ---------------------------------------------------------
# List uploaded documents
# ---------------------------------------------------------

@app.get("/documents")
def list_documents():
    result = collection.get(
        include=["metadatas"]
    )

    documents = {}

    for metadata in result["metadatas"]:
        document_id = metadata["document_id"]

        if document_id not in documents:
            documents[document_id] = {
                "document_id": document_id,
                "filename": metadata["filename"],
                "chunks": 0
            }

        documents[document_id]["chunks"] += 1

    return {
        "documents": list(documents.values()),
        "total_vectors": collection.count()
    }


# ---------------------------------------------------------
# Delete one uploaded document
# ---------------------------------------------------------

@app.delete("/documents/{document_id}")
def delete_document(document_id: str):
    existing = collection.get(
        where={
            "document_id": document_id
        },
        include=["metadatas"]
    )

    if not existing["ids"]:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )

    collection.delete(
        where={
            "document_id": document_id
        }
    )

    return {
        "message": "Document deleted successfully",
        "document_id": document_id,
        "deleted_chunks": len(existing["ids"]),
        "vectors_remaining": collection.count()
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "collection": COLLECTION_NAME,
        "vectors": collection.count()
    }
