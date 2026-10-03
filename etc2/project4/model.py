# Модуль инференса модели
# Загрузка LLM — ресурсоёмкая операция. Применяется паттерн Singleton для загрузки модели в память один раз при старте сервера. Генерация выносится в асинхронный поток `asyncio.to_thread`, чтобы не блокировать Event Loop FastAPI. Дополнительно встроен парсинг XML-тегов, заданных в промпте.

import re
import asyncio
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import settings
from prompt import SYSTEM_PROMPT


class LLMService:
    _instance = None # Статическое поле класса

    def __new__(cls): # Метод чтобы во всем приложении использовалась одна и та же модель 
        if cls._instance is None:
            cls._instance = super(LLMService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        print(f"Загрузка модели {settings.MODEL_NAME} на устройство {settings.DEVICE}...")
        self.tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_NAME, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.MODEL_NAME,
            torch_dtype=settings.TORCH_DTYPE,
            device_map=settings.DEVICE,
            trust_remote_code=True
        )
        print("Модель успешно загружена в память.")

    def _generate(self, user_prompt: str) -> dict:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        # Применение официального шаблона Qwen ChatML
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(settings.DEVICE)

        with torch.no_grad(): # Генерация ответа без вычисления градиентов
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=settings.MAX_NEW_TOKENS,
                temperature=settings.TEMPERATURE,
                top_p=settings.TOP_P,
                do_sample=True, # Сэмплирование лучше для генерации объяснений, чем жадный поиск
                pad_token_id=self.tokenizer.eos_token_id # eos_token_id — это специальный токен, который обозначает конец последовательности. Он используется для правильного завершения генерации текста и предотвращения выхода за пределы допустимой длины.
            )

        # Отсекаем входной промпт из результатов
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0] # Декодируем сгенерированные токены обратно в текст, пропуская специальные токены (например, <pad>, <eos>), чтобы получить чистый текст ответа модели.

        return self._parse_structured_output(response_text)

    def _parse_structured_output(self, text: str) -> dict:
        """Парсинг структурированных тегов из вывода модели."""
        thinking = re.search(r'<thinking>(.*?)</thinking>', text, re.DOTALL) # re.DOTALL позволяет точке (.) в регулярном выражении соответствовать любому символу, включая символ новой строки. Это важно, когда мы хотим захватить многострочный текст между тегами <thinking> и </thinking>.
        feedback = re.search(r'<feedback>(.*?)</feedback>', text, re.DOTALL)
        actions = re.search(r'<actions>(.*?)</actions>', text, re.DOTALL)

        raw_actions = actions.group(1).strip() if actions else "" # Если тег <actions> найден, извлекаем его содержимое и удаляем лишние пробелы в начале и конце строки. Если тег не найден, устанавливаем пустую строку.
        actions_list = [ # Создаем список действий внутри тега <actions>, удаляя лишние пробелы и символы '-' в начале каждой строки
            line.strip('- ').strip()
            for line in raw_actions.split('\n') # Разделяем текст на строки по символу новой строки
            if line.strip().startswith('-') or line.strip() # Фильтруем строки, оставляя только те, которые начинаются с '-' или не пустые
        ]

        return {
            "thinking_process": thinking.group(1).strip() if thinking else "Анализ выполнен по умолчанию.",
            "structured_feedback": feedback.group(1).strip() if feedback else text,
            "action_items": actions_list if actions_list else ["Изучить предложенный рефакторинг."]
        }

    async def generate_async(self, user_prompt: str) -> dict: # Асинхронный метод для генерации ответа
        """Асинхронная обертка для предотвращения блокировки Event Loop."""
        return await asyncio.to_thread(self._generate, user_prompt) # Используем asyncio.to_thread для выполнения синхронного метода _generate в отдельном потоке, чтобы не блокировать основной Event Loop FastAPI.