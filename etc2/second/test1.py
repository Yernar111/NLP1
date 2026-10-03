# 1. Установка необходимых библиотек
# !pip install -q -U transformers accelerate bitsandbytes

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 2. Настройка 4-битного квантования NF4 (Normal Float 4). Квантование позволяет уменьшить размер модели и ускорить инференс, сохраняя при этом точность.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",             # Тип квантования, оптимизированный для LLM
    bnb_4bit_compute_dtype=torch.bfloat16, # Вычисления в bfloat16/float16 для ускорения
    bnb_4bit_use_double_quant=True         # Двойное квантование (экономы дополнительных ~0.37 бит на параметр)
)

# 3. Выбор модели (Qwen 2.5 3B Instruct — доступная и открытая модель)
model_id = "Qwen/Qwen2.5-3B-Instruct"

# 4. Загрузка токенизатора и модели в 4-битном формате
tokenizer = AutoTokenizer.from_pretrained(model_id) # Загрузка токенизатора для преобразования текста в токены и обратно. Этот токенизатор совместим с моделью Qwen 2.5 3B Instruct.
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto" # Автоматическое распределение слоёв на GPU
)

# 5. Формирование диалогового контекста. 
messages = [
    {"role": "system", "content": "Ты опытный инженер по машинному обучению."},
    {"role": "user", "content": "Объясни в двух предложениях, чем 4-битное квантование помогает инференсу LLM."}
]

# 6. Применение Chat Template и токенизация
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) # Создание подходящего промпта для модели на основе диалогового контекста. Функция apply_chat_template формирует текстовый ввод основываясь на messages, который учитывает роли участников диалога (system, user) и добавляет специальные токены для генерации ответа.
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# 7. Генерация ответа
outputs = model.generate(
    **inputs,
    max_new_tokens=150, # Максимальное количество новых токенов, которые модель может сгенерировать в ответ на запрос. Ограничение предотвращает слишком длинные ответы и контролирует время инференса.
    temperature=0.7, # Параметр температуры управляет случайностью генерации. Более низкие значения (ближе к 0) делают ответы более детерминированными, в то время как более высокие значения (ближе к 1) увеличивают разнообразие и креативность ответов.
    do_sample=True # Флаг do_sample=True указывает модели использовать стохастический метод выборки токенов при генерации, что позволяет получать более разнообразные и креативные ответы, в отличие от детерминированного выбора наиболее вероятного токена.
)

# 8. Декодирование и вывод результата (отсекая исходный промпт)
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True) # Декодирование сгенерированных токенов обратно в текст. Функция decode преобразует числовые токены в читаемый текст, а параметр skip_special_tokens=True удаляет специальные токены, которые не нужны в финальном ответе.
print(response)