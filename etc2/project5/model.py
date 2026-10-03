import re
import asyncio
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import settings
# from prompt import SYSTEM_PROMPT


class LLMService:
    _instance = None
 
    def __new__(cls):
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
        print("Модель биологического ИИ-репетитора успешно загружена.")

    # def _generate(self, user_prompt: str) -> dict:
    #     messages = [
    #         {"role": "system", "content": SYSTEM_PROMPT},
    #         {"role": "user", "content": user_prompt}
    #     ] 
        
    #     text = self.tokenizer.apply_chat_template(
    #         messages,
    #         tokenize=False,
    #         add_generation_prompt=True
    #     )
        
    #     model_inputs = self.tokenizer([text], return_tensors="pt").to(settings.DEVICE)

    #     with torch.no_grad():
    #         generated_ids = self.model.generate(
    #             **model_inputs,
    #             max_new_tokens=settings.MAX_NEW_TOKENS,
    #             temperature=settings.TEMPERATURE,
    #             top_p=settings.TOP_P,
    #             do_sample=True,
    #             pad_token_id=self.tokenizer.eos_token_id
    #         )

    #     generated_ids = [
    #         output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    #     ]
    #     response_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    #     return self._parse_structured_output(response_text)

    def _generate_with_context(self, messages: list[dict], stats_meta: dict) -> dict:
        # Формирование единого текстового контекста из всей истории через ChatML
        text = self.tokenizer.apply_chat_template( 
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(settings.DEVICE)
        input_tokens_count = model_inputs.input_ids.shape[1] # Подсчёт количества токенов во входном контексте для статистики

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=settings.MAX_NEW_TOKENS,
                temperature=settings.TEMPERATURE,
                top_p=settings.TOP_P,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        parsed_response = self._parse_structured_output(response_text) # Парсинг ответа модели. Возвращает словарь с объяснением, аналогией и ключевыми пунктами
        
        # Запись точной длины контекста в метаданные. Переписываем stats_meta(из аргумента метода), чтобы включить количество токенов во входном контексте.
        stats_meta["input_tokens_count"] = input_tokens_count
        parsed_response["context_stats"] = stats_meta # Добавляем к parsed_response статистику использования контекстного окна
        
        return parsed_response # Возвращаем полный словарь с объяснением, аналогией, ключевыми пунктами и статистикой контекста.

    def _parse_structured_output(self, text: str) -> dict:
        """Парсинг тегов с защитой от незакрытых тегов малой модели."""
        exp_match = re.search(r'<explanation>(.*?)(?:</explanation>|<analogy>|$)', text, re.DOTALL | re.IGNORECASE)
        ana_match = re.search(r'<analogy>(.*?)(?:</analogy>|<points>|$)', text, re.DOTALL | re.IGNORECASE)
        pts_match = re.search(r'<points>(.*?)(?:</points>|$)', text, re.DOTALL | re.IGNORECASE)

        explanation = exp_match.group(1).strip() if exp_match else text.strip() # Если тег <explanation> не найден, возвращаем весь текст как объяснение.
        analogy = ana_match.group(1).strip() if ana_match else "Аналогия: представьте этот процесс как работу слаженного механизма."
        
        points_list = []
        if pts_match: # Если тег найден, извлекаем его содержимое и удаляем лишние пробелы по краям.
            raw_pts = pts_match.group(1).strip()
            points_list = [
                line.strip('- *').strip() 
                for line in raw_pts.split('\n') 
                if line.strip()
            ]

        return {
            "explanation": explanation,
            "analogy": analogy,
            "key_points": points_list if points_list else ["Тема требует внимательного изучения по учебнику."]
        }

    async def generate_async(self, messages: list[dict], stats_meta: dict) -> dict:
        # return await asyncio.to_thread(self._generate, user_prompt)
        return await asyncio.to_thread(self._generate_with_context, messages, stats_meta)