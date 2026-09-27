from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field

class Citation(BaseModel):
    document_id: str
    document_name: str
    page: int
    section: str | None = None
    chunk_id: str
    excerpt: str

class Finding(BaseModel):
    id: str
    title: str
    severity: Literal["critical", "high", "medium", "low", "info"]
    category: str
    summary: str
    evidence: list[Citation]
    recommended_actions: list[str]

class AnalysisResponse(BaseModel):
    documents_analyzed: int
    findings: list[Finding]
    summary: str
    generated_by: str

class QueryRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int = Field(default=5, ge=1, le=10)

class QueryResponse(BaseModel):
    answer: str
    provider: str
    citations: list[Citation]

class IngestResponse(BaseModel):
    ingested_documents: int
    skipped_files: list[str]
    chunks: int
    entities: int
    relationships: int

class GraphResponse(BaseModel):
    nodes: list[dict[str, Any]]
    edges: list[dict[str, str]]
