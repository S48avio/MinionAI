from typing import Literal

from pydantic import BaseModel, Field

# Pydantic schemas for chat
class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class Source(BaseModel):
    filename: str
    document_id: str
    page_number: int
    chunk_number: int
    distance: float


class ChatResponse(BaseModel):
    answer: str
    route: Literal["rag", "general"]
    sources: list[Source] = Field(default_factory=list)


# Pydantic schemas for RAG
class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
