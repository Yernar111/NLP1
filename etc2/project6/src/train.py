import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from config import settings

from prepare_dataset import prepare_dataset

def build_quantization_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=settings.TORCH_DTYPE,
        bnb_4bit_use_double_quant=True
    )

def build_lora_config():
    return LoraConfig(
        r=settings.LORA_R,
        lora_alpha=settings.LORA_ALPHA,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=settings.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )

def build_sft_config():
    return SFTConfig(
        output_dir="./results",
        per_device_train_batch_size=settings.BATCH_SIZE,
        gradient_accumulation_steps=settings.GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=False,
        learning_rate=settings.LEARNING_RATE,
        num_train_epochs=settings.NUM_TRAIN_EPOCHS,
        logging_steps=10,
        save_strategy="no",                                    # Сохраняем только финальный адаптер вручную
        fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        optim="paged_adamw_8bit",
        dataset_text_field="text",                             # Имя поля с форматированным текстом
        max_length=settings.MAX_SEQ_LENGTH, # Максимальная длина последовательности для токенизации
        packing=False                                          # Отключаем упаковку для сохранения границ диалогов
    )


def train():
    print(f"--> Инициализация обучения на устройстве: {settings.DEVICE}")
    
    tokenizer = AutoTokenizer.from_pretrained(settings.BASE_MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = build_quantization_config()

    model = AutoModelForCausalLM.from_pretrained(
        settings.BASE_MODEL_NAME,
        attn_implementation="sdpa",
        quantization_config=bnb_config if torch.cuda.is_available() else None,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )

    # model.config.use_cache = False

    model = prepare_model_for_kbit_training(model)

    peft_config = build_lora_config()

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_dataset = prepare_dataset(tokenizer)

    sft_config = build_sft_config()

    # Инициализация и запуск SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        # peft_config=peft_config,
        args=sft_config,
        processing_class=tokenizer
    )

    print("--> Запуск SFT-обучения...")
    trainer.train()

    print(f"--> Сохранение LoRA-адаптеров в {settings.ADAPTER_DIR}...")
    trainer.model.save_pretrained(settings.ADAPTER_DIR)
    tokenizer.save_pretrained(settings.ADAPTER_DIR)
    print("--> Обучение и сохранение завершено успешно!")


if __name__ == "__main__":
    train()