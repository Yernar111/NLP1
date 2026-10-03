from config import settings
from schemas import ChatMessage
SYSTEM_PROMPT = """Ты — доброжелательный и увлеченный учитель школьной биологии. Твоя цель — объяснять биологические понятия, термины и процессы доступным языком для школьников.

### ПРАВИЛА И ОГРАНИЧЕНИЯ:
1. Отвечай СТРОГО на русском языке.
2. Не используй слишком сложные академические термины без их моментального пояснения.
3. Твой ответ ДОЛЖЕН содержать ровно три блока с XML-тегами:
   - <explanation>: Понятное и интересное объяснение термина или процесса.
   - <analogy>: Наглядная и простая аналогия из повседневной жизни.
   - <points>: Список из 2-3 главных фактов (каждый пункт с новой строки, начинающийся с '- ').
4. Не используй приветствия ("Привет!", "Здравствуйте"). Сразу начинай ответ с тега <explanation>.

### ПРИМЕР ИДЕАЛЬНОГО ОТВЕТА (FEW-SHOT):
 
<explanation>
Микроскоп — это оптический прибор, который позволяет рассмотреть мелкие объекты, не видимые невооруженным глазом. Он увеличивает изображение с помощью системы линз, благодаря чему мы можем увидеть клетки растений, животных и бактерии.
</explanation>

<analogy>
Микроскоп можно сравнить с супер-лупой или волшебным окном, которое открывает перед нами целый невидимый микромир.
</analogy>

<points>
- Главные части микроскопа — это окуляр, объектив и предметный столик.
- Световой микроскоп увеличивает объекты за счет света и линз.
- Позволил ученым узнать, что все живые организмы состоят из клеток.
</points>
"""


# def build_biology_user_prompt(topic: str, grade_level: str = "5-9 класс") -> str: # Функция формирует пользовательский промпт для объяснения темы по биологии.
#     return (
#         f"Объясни тему или термин по биологии: '{topic}'.\n"
#         f"Уровень сложности: для учеников ({grade_level})."
#     )

def prepare_messages_with_context_window(
    topic: str,
    grade_level: str = "5-9 класс",
    history: list[ChatMessage] | None = None
) -> tuple[list[dict], dict]:
    """
    Формирует список сообщений с применением алгоритма Sliding Context Window (Скользящего окна).
    Отбрасывает устаревшие сообщения, гарантируя, что размер контекста не превысит возможности модели.
    """
    if history is None:
        history = []

    total_messages = len(history)
    
    # Расчёт вместимости: N пар "user-assistant" -> N * 2 сообщений
    max_messages = settings.MAX_CONTEXT_TURNS * 2
    trimmed_history = history[-max_messages:] if total_messages > max_messages else history # Обрезка истории до максимального количества сообщений
    dropped_count = total_messages - len(trimmed_history)

    # 1. Системный промпт — всегда в начале окна
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # 2. Активная часть истории, вошедшая в скользящее окно
    for msg in trimmed_history:
        messages.append({"role": msg.role, "content": msg.content})

    # 3. Новое сообщение пользователя — в конце окна
    current_user_content = f"Тема/Вопрос: '{topic}'. Уровень: ({grade_level})."
    messages.append({"role": "user", "content": current_user_content})
    print(messages)

    stats = {
        "total_history_messages": total_messages,
        "active_context_messages": len(trimmed_history) + 1,  # Включая текущий запрос
        "dropped_messages": dropped_count,
    }

    return messages, stats

def create_updated_history(
    existing_history: list[ChatMessage] | None,
    topic: str,
    grade_level: str,
    raw_result: dict
) -> list[ChatMessage]:
    """Формирует обновлённую историю с добавлением текущего шага диалога."""
    new_history = list(existing_history) if existing_history else []

    # 1. Формируем сообщение пользователя
    user_content = f"Тема/Вопрос: '{topic}'. Уровень: ({grade_level})."
    new_history.append(ChatMessage(role="user", content=user_content))

    # 2. Формируем ответ ассистента в формате XML-тегов, с которыми работает модель
    points_text = "\n".join([f"- {pt}" for pt in raw_result["key_points"]])
    assistant_content = (
        f"<explanation>{raw_result['explanation']}</explanation>\n"
        f"<analogy>{raw_result['analogy']}</analogy>\n"
        f"<points>\n{points_text}\n</points>"
    )
    new_history.append(ChatMessage(role="assistant", content=assistant_content))

    return new_history