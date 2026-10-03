import torch
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Code Reviewer & Mentor API"
    MODEL_NAME: str = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # Выбор устройства: CUDA if available, else CPU
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    TORCH_DTYPE: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    # Гиперпараметры генерации для малых LLM
    MAX_NEW_TOKENS: int = 512
    TEMPERATURE: float = 0.3  # Низкая температура для строгого соблюдения формата
    TOP_P: float = 0.85 # top_p это порог вероятности для сэмплирования токенов, ограничивающий генерацию наиболее вероятными токенами. 0.85 означает, что модель будет выбирать токены из топ-85% вероятных вариантов, что помогает контролировать разнообразие и креативность генерации.

    class Config:
        env_file = ".env"


settings = Settings()