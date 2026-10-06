from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Вопрос или инструкция для модели", example="Объясни принцип работы рекурсии простыми словами.")

class GenerateResponse(BaseModel):
    prompt: str
    response: str = Field(..., description="Сгенерированный моделью ответ")