from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException

from api.schemas import QueryRequest, QueryResponse
from api.services import RAGService
from api import dependencies

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Данный блок выполняется ПРИ СТАРТЕ сервера
    print("Запуск FastAPI... Загружаем модели в память.")
    dependencies.rag_service_instance = RAGService()
    print("Модели успешно загружены!")
    
    yield  # Сервер работает и принимает запросы
    
    # Данный блок выполняется ПРИ ОСТАНОВКЕ сервера
    print("Завершение работы FastAPI... Освобождаем ресурсы.")
    dependencies.rag_service_instance = None

app = FastAPI(
    title="Qwen2.5 RAG API",
    description="Pet-проект для проверки эффективности RAG на базе Qwen2.5-0.5B-Instruct и Qdrant",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok", "service": "RAG API"}

@app.post("/api/v1/query", response_model=QueryResponse, tags=["RAG"])
def query_rag(
    request: QueryRequest,
    rag_service: RAGService = Depends(dependencies.get_rag_service)
):
    try:
        response, sources = rag_service.answer_query(
            query=request.query,
            top_k=request.top_k
        )
        return QueryResponse(
            query=request.query,
            response=response,
            context_used=sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки запроса: {str(e)}")