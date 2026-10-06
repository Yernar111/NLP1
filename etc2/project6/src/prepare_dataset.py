from config import settings
from datasets import load_dataset

def prepare_dataset(tokenizer):
    """Загрузка и форматирование датасета LDJnr/Capybara."""
    print(f"--> Загрузка датасета {settings.DATASET_NAME}...")
    dataset = load_dataset(settings.DATASET_NAME, split="train")
    
    # Ограничиваем количество примеров согласно инструкции (<= 10 000)
    if len(dataset) > settings.MAX_TRAIN_SAMPLES:
        dataset = dataset.select(range(settings.MAX_TRAIN_SAMPLES))
    
    def format_prompts(example):
        """Преобразование структуры диалога Capybara в ChatML формат Qwen."""
        messages = []
        # Структура Capybara содержит массив диалогов в поле 'conversation'
        for turn in example["conversation"]:
            user_text = turn.get("input") or ""
            assistant_text = turn.get("output") or ""

            if not user_text or not assistant_text:
                continue

            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": assistant_text})
        
        # Применяем официальный ChatTemplate модели
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        return {"text": text}

    print("--> Форматирование диалогов под ChatML...")
    formatted_dataset = dataset.map(format_prompts, remove_columns=dataset.column_names)
    return formatted_dataset