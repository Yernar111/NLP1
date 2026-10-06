import torch
from pydantic_settings import BaseSettings

 
class Settings(BaseSettings):
    PROJECT_NAME: str = "Fine-tuning"
    BASE_MODEL_NAME: str = "Qwen/Qwen2.5-0.5B-Instruct"
    DATASET_NAME: str = "LDJnr/Capybara"
    ADAPTER_DIR: str = "./qwen-capybara-lora"
    

    MAX_TRAIN_SAMPLES: int = 4000
    MAX_SEQ_LENGTH: int = 256

    LORA_R: int = 16
    LORA_ALPHA: int = 32
    LORA_DROPOUT: float = 0.05

    BATCH_SIZE: int = 4
    GRADIENT_ACCUMULATION_STEPS: int = 2  # Эффективный батч = 2 * 4 = 8
    LEARNING_RATE: float = 2e-4
    NUM_TRAIN_EPOCHS: int = 1
    
    MAX_NEW_TOKENS: int = 512
    TEMPERATURE: float = 0.2
    TOP_P: float = 0.85

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    TORCH_DTYPE: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # MAX_CONTEXT_TURNS: int = 3
    # MAX_CONTEXT_TOKENS: int = 1500

    class Config:
        env_file = ".env"


settings = Settings()