from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

from peft import PeftModel

# Если кто-то пришлет запрос без поля "user_message", API автоматически вернет ошибку.
class ChatRequest(BaseModel):
    user_message: str
    max_tokens: int = 100 # Значение по умолчанию

class ChatResponse(BaseModel):
    bot_reply: str

# Глобальная переменная для хранения загруженной модели
ai_pipeline = None

@asynccontextmanager # asynccontextmanager позволяет выполнять асинхронные операции при старте и завершении приложения FastAPI
async def lifespan(app: FastAPI): # Lifespan для управление жизненным циклом. А именно для загрузки весов модели с жесткого диска в оперативную память (RAM). Мы делаем это ОДИН РАЗ при старте сервера, чтобы при самих запросах отвечать мгновенно.
    global ai_pipeline
    print("Загрузка модели в память. Пожалуйста, подождите...")
    
    # Если папки нет, для теста можно вписать "Qwen/Qwen2.5-0.5B-Instruct"
    adapter_path = "./my-support-model"
    base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    # Скачиваем/загружаем базовую модель (1 ГБ) с серверов Hugging Face
    print("Загрузка базовой модели (может занять время при первом запуске)...")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="cpu",
        torch_dtype=torch.float32 # Для CPU используем стандартный тип данных
    )

    # НАКАТЫВАЕМ АДАПТЕР: объединяем базовую модель с нашими весами LoRA
    print("Применение весов LoRA...")
    model = PeftModel.from_pretrained(base_model, adapter_path) # adapter_path/adapter_model.safetensors
    
    # Используем высокоуровневый pipeline из Hugging Face
    ai_pipeline = pipeline(
        "text-generation", 
        model=model,
        tokenizer=tokenizer,
        device="cpu" # Запускаем на процессоре
    )
    print("Модель готова к работе!")
    
    yield # Сервер начинает принимать запросы
    
    print("Выключение сервера. Очистка памяти...")
    ai_pipeline = None

# Инициализация приложения FastAPI
app = FastAPI(title="AI Support API", lifespan=lifespan)

@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    if not ai_pipeline:
        raise HTTPException(status_code=503, detail="Модель еще не загрузилась")
    
    # Обертываем сообщение пользователя в шаблон
    prompt = f"<|im_start|>user\n{request.user_message}<|im_end|>\n<|im_start|>assistant\n"
    
    # Передаем текст в модель
    outputs = ai_pipeline(
        prompt, 
        max_new_tokens=request.max_tokens, 
        return_full_text=False # Вернуть только сгенерированный ответ, без самого вопроса
    )
    
    reply = outputs[0]["generated_text"].strip()
    
    # Возвращаем ответ в виде JSON
    return ChatResponse(bot_reply=reply)