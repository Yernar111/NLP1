from fastapi import FastAPI, HTTPException, status, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from api.schemas import GenerateRequest, GenerateResponse
from api.model import LLMService 

from src.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    LLMService()
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="Qwen2.5-0.5B with Fine-Tuning",
    lifespan=lifespan
)


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "ok", "model": settings.BASE_MODEL_NAME, "device": settings.DEVICE}


@app.post("/api/v1/generate", response_model=GenerateResponse)
async def generate_response(request: GenerateRequest):
    try:
        llm_service = LLMService()

        result = await llm_service.generate_async(request.prompt)

        return GenerateResponse(
            prompt=request.prompt,
            response=result
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