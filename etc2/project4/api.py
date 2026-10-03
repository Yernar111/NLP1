from fastapi import FastAPI, HTTPException, status
from contextlib import asynccontextmanager

from config import settings
from schemas import CodeReviewRequest, ExplainConceptRequest, MentorResponse
from prompt import build_code_review_user_prompt, build_explain_user_prompt
from model import LLMService


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Инициализация модели при старте приложения
    LLMService()
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "ok", "model": settings.MODEL_NAME, "device": settings.DEVICE}


@app.post("/api/v1/review", response_model=MentorResponse)
async def code_review(request: CodeReviewRequest):
    try:
        llm_service = LLMService()
        user_prompt = build_code_review_user_prompt(request.code, request.language, request.context)
        result = await llm_service.generate_async(user_prompt)
        return MentorResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке запроса: {str(e)}")


@app.post("/api/v1/explain", response_model=MentorResponse)
async def explain_concept(request: ExplainConceptRequest):
    try:
        llm_service = LLMService()
        user_prompt = build_explain_user_prompt(request.topic, request.experience_level)
        result = await llm_service.generate_async(user_prompt)
        return MentorResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке запроса: {str(e)}")