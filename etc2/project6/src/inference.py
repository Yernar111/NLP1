import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from config import settings

from prompt import SYSTEM_PROMPT


def load_model_and_tokenizer():
    """Загрузка базовой модели Qwen2.5-0.5B и наложение LoRA-адаптера."""
    print(f"--> 1/3 Загрузка токенизатора для {settings.BASE_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(settings.BASE_MODEL_NAME, trust_remote_code=True)

    # Настройка 4-битного квантования NF4 при наличии GPU
    bnb_config = BitsAndBytesConfig( 
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=settings.TORCH_DTYPE
    ) if torch.cuda.is_available() else None

    print(f"--> 2/3 Загрузка базовой модели на устройство [{settings.DEVICE}]...")
    base_model = AutoModelForCausalLM.from_pretrained(
        settings.BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )

    # Проверка и подключение обученного адаптера LoRA
    if os.path.exists(settings.ADAPTER_DIR):
        print(f"--> 3/3 Наложение LoRA-адаптера из '{settings.ADAPTER_DIR}'...")
        model = PeftModel.from_pretrained(base_model, settings.ADAPTER_DIR)
        print("--> [УСПЕХ] LoRA-адаптер успешно подключен!")
    else:
        print("--> [ВНИМАНИЕ] Директория с адаптером не найдена! Использована БАЗОВАЯ модель.")
        model = base_model

    # Отключение dropout и переподготовка слоев к генерации
    model.eval()
    return model, tokenizer


def generate_response(
    model, 
    tokenizer, 
    prompt: str
) -> tuple[str, float]:
    """Генерация ответа модели на один запрос и замер времени выполнения."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    # Формирование текста в формате ChatML
    formatted_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([formatted_input], return_tensors="pt").to(settings.DEVICE)

    # start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=settings.MAX_NEW_TOKENS,
            temperature=settings.TEMPERATURE,
            top_p=settings.TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Извлекаем только сгенерированные токены (без текста входящего промпта)
    generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
    response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return response_text.strip()


def main():
    print("=" * 60)
    print(" ЛОКАЛЬНЫЙ ТЕСТ ИНФЕРЕНСА МОДЕЛИ (без запуска API-сервера)")
    print("=" * 60)

    model, tokenizer = load_model_and_tokenizer()

    # Запуск интерактивного консольного диалога
    print("Введите запрос (или 'exit' для выхода):")
    
    while True:
        try:
            user_input = input("\nВы > ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ["exit", "quit", "выход"]:
                print("Завершение сессии инференса.")
                break

            response_text = generate_response(model, tokenizer, user_input)
            print(response_text)
            
        except KeyboardInterrupt:
            print("\nПринудительное завершение работы.")
            sys.exit(0)


if __name__ == "__main__":
    main()