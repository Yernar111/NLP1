from fastapi import FastAPI, HTTPException, status, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from config import settings
from schemas import BiologyQueryRequest, BiologyQueryResponse
from prompt import prepare_messages_with_context_window, create_updated_history
from model import LLMService 


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Инициализация модели при старте веб-сервера
    LLMService()
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="API ИИ-репетитора по школьной биологии на базе Qwen2.5-0.5B",
    lifespan=lifespan
)


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "ok", "model": settings.MODEL_NAME, "device": settings.DEVICE}


@app.post("/api/v1/explain", response_model=BiologyQueryResponse)
async def explain_biology_topic(request: BiologyQueryRequest):
    try:
        llm_service = LLMService()
        
        # user_prompt = build_biology_user_prompt(request.topic, request.grade_level)

        # 1. Применение алгоритма скользящего окна к истории сообщений
        messages, context_stats = prepare_messages_with_context_window(
            topic=request.topic,
            grade_level=request.grade_level,
            history=request.history
        )

        # raw_result = await llm_service.generate_async(user_prompt)
        raw_result = await llm_service.generate_async(messages, context_stats)

        new_history = create_updated_history(
            existing_history=request.history,
            topic=request.topic,
            grade_level=request.grade_level,
            raw_result=raw_result
        )
        
        return BiologyQueryResponse(
            topic=request.topic,
            explanation=raw_result["explanation"],
            analogy=raw_result["analogy"],
            key_points=raw_result["key_points"],
            context_stats=raw_result["context_stats"],
            updated_history=new_history
        ) 
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при генерации ответа: {str(e)}"
        )



@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Кастомный обработчик 422 ошибок для удобной отладки JSON."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Ошибка формата входящего JSON-запроса",
            "errors": exc.errors()
        },
    ) 