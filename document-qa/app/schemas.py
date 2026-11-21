"""Pydantic request and response models for the Document Q&A API.""[span_26](start_span)"[span_26](end_span)

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Payload for asking a question against indexed documents.""[span_27](start_span)"[span_27](end_span)

    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="User question to answer using the indexed document corpus",
        examples=["How many annual paid leave days do full-time employees get?"],
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=10,
        description="Optional override for number of retrieved chunks",
    )


class Citation(BaseModel):
    """Metadata for a retrieved source chunk used to ground the answer.""[span_28](start_span)"[span_28](end_span)

    document: str = Field(..., description="Source document filename")
    chunk_id: str = Field(..., description="Unique identifier of the retrieved chunk")
    score: float = Field(..., description="Relevance similarity score")
    excerpt: str = Field(..., description="Relevant snippet from the source document")


class AskResponse(BaseModel):
    """Response containing the grounded answer and supporting citations.""[span_29](start_span)"[span_29](end_span)

    question: str
    answer: str
    grounded: bool = Field(
        ...,
        description="True when answer is supported by retrieved document chunks",
    )
    citations: list[Citation] = Field(default_factory=list)
    provider: str = Field(..., description="Answer generation provider used")
    request_id: str = Field(..., description="Correlation ID for tracing and logs")


class DocumentSummary(BaseModel):
    """Summary of an indexed document.""[span_30](start_span)"[span_30](end_span)

    filename: str
    chunks: int
    characters: int


class CorpusStatusResponse(BaseModel):
    """Status of the loaded document corpus and vector index.""[span_31](start_span)"[span_31](end_span)

    document_count: int
    chunk_count: int
    documents: list[DocumentSummary]


class HealthResponse(BaseModel):
    """Service health status response.""[span_32](start_span)"[span_32](end_span)

    status: str
    service: str
    version: str
    indexed_documents: int
    indexed_chunks: int
