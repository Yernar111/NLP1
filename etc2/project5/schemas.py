from pydantic import BaseModel, Field
from typing import Optional, Literal

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"] = Field(..., description="Роль отправителя (user или assistant)")
    content: str = Field(..., description="Текст сообщения")

class ContextWindowStats(BaseModel):
    total_history_messages: int = Field(..., description="Всего сообщений в переданной истории")
    active_context_messages: int = Field(..., description="Сообщений, поместившихся в активное окно")
    dropped_messages: int = Field(..., description="Отброшено устаревших сообщений из-за лимита окна")
    input_tokens_count: int = Field(..., description="Итоговый объём входного контекста в токенах")


class BiologyQueryRequest(BaseModel):
    topic: str = Field(
        ..., 
        description="Биологический термин, орган, процесс или понятие", 
        example="Митохондрия"
    )
    grade_level: Optional[str] = Field(
        default="5-9 класс", 
        description="Уровень сложности (например, '5-6 класс', '10-11 класс')"
    )
    history: Optional[list[ChatMessage]] = Field(
        default=[],
        description="История прошлых сообщений для сохранения контекста беседы"
    )

 
class BiologyQueryResponse(BaseModel):
    topic: str = Field(..., description="Запрошенная тема")
    explanation: str = Field(..., description="Простое и понятное объяснение сути")
    analogy: str = Field(..., description="Наглядная аналогия из жизни")
    key_points: list[str] = Field(..., description="Главные выводы и запоминающиеся факты")
    context_stats: ContextWindowStats = Field(..., description="Статистика использования контекстного окна")
    updated_history: list[ChatMessage] = Field(
        ..., 
        description="Готовая обновлённая история диалога. Скопируйте её в поле 'history' для следующего запроса."
    )