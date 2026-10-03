import torch
from pydantic_settings import BaseSettings

 
class Settings(BaseSettings):
    PROJECT_NAME: str = "Школьная Биология — ИИ-Репетитор API"
    MODEL_NAME: str = "Qwen/Qwen2.5-0.5B-Instruct"
    
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    TORCH_DTYPE: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    MAX_NEW_TOKENS: int = 512
    TEMPERATURE: float = 0.4
    TOP_P: float = 0.85

    MAX_CONTEXT_TURNS: int = 3  # Хранить в окне не более 3 последних пар сообщений
    MAX_CONTEXT_TOKENS: int = 1500

    class Config:
        env_file = ".env"


settings = Settings()