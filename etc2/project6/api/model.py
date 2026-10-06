import asyncio
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
from peft import PeftModel

from src.config import settings
from src.prompt import SYSTEM_PROMPT

class LLMService:
    _instance = None
 
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        print(f"Загрузка модели {settings.BASE_MODEL_NAME} на устройство {settings.DEVICE}...")
        self.tokenizer = AutoTokenizer.from_pretrained(settings.BASE_MODEL_NAME, trust_remote_code=True)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=settings.TORCH_DTYPE
        ) if torch.cuda.is_available() else None

        model = AutoModelForCausalLM.from_pretrained(
            settings.BASE_MODEL_NAME,
            quantization_config=bnb_config if torch.cuda.is_available() else None,
            # torch_dtype=settings.TORCH_DTYPE,
            device_map=settings.DEVICE,
            trust_remote_code=True
        )
        if os.path.exists(settings.ADAPTER_DIR):
            print(f"--> Обнаружен LoRA-адаптер в {settings.ADAPTER_DIR}. Накладываем веса...")
            self.model = PeftModel.from_pretrained(model, settings.ADAPTER_DIR)
            self.is_adapter_loaded = True
        else:
            print("--> ВНИМАНИЕ: LoRA-адаптер не найден! Запуск в режиме базовой модели.")
            self.model = model
            self.is_adapter_loaded = False

        self.model.eval()
        print("Модель успешно загружена.")

    def _generate(self, user_prompt: str) -> dict:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ] 
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(settings.DEVICE)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=settings.MAX_NEW_TOKENS,
                temperature=settings.TEMPERATURE,
                top_p=settings.TOP_P,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        generated_ids = generated_ids[0][model_inputs.input_ids.shape[1]:]
        response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return response_text.strip() 

    async def generate_async(self, user_prompt: str) -> dict:
        return await asyncio.to_thread(self._generate, user_prompt)