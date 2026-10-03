from pydantic import BaseModel, Field
from typing import Optional

# BaseModel нужен для валидации данных и генерации схемы OpenAPI. Field используется для добавления метаданных к полям модели, таких как описание и пример значения. Optional позволяет указать, что поле может быть пустым (None).


class CodeReviewRequest(BaseModel): # Модель запроса для ревью кода
    code: str = Field(..., description="Исходный код для ревью", example="def add(a, b): return a+b") 
    language: str = Field(default="python", description="Язык программирования")
    context: Optional[str] = Field(None, description="Дополнительный контекст или задача")
 

class ExplainConceptRequest(BaseModel): # Модель запроса для объяснения концепции
    topic: str = Field(..., description="Тема для объяснения", example="Декораторы в Python")
    experience_level: str = Field(
        default="Junior", 
        description="Уровень разработчика (Junior, Middle, Student)"
    )


class MentorResponse(BaseModel): # Модель ответа наставника с разбором кода и рекомендациями
    thinking_process: str = Field(..., description="Шаги рассуждения модели (Chain-of-Thought)")
    structured_feedback: str = Field(..., description="Основной ответ/ревью")
    action_items: list[str] = Field(..., description="Чёткие рекомендательные шаги")