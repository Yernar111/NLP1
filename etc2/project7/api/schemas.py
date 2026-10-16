from pydantic import BaseModel, Field
from typing import List

class QueryRequest(BaseModel):
    query: str = Field(..., description="Вопрос для модели", example="When was Abraham Lincoln born?")
    top_k: int = Field(default=2, ge=1, le=5, description="Количество релевантных пассажей для контекста")

class SourceDocument(BaseModel):
    passage_id: str | int
    score: float
    text: str

class QueryResponse(BaseModel):
    query: str
    response: str = Field(..., description="Сгенерированный моделью ответ")
    context_used: List[SourceDocument]
