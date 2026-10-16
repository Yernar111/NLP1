import torch
import os
from pydantic import BaseModel, ConfigDict

 
class Settings(BaseModel):
    # model_config = ConfigDict(arbitrary_types_allowed=True)
    QDRANT_PATH: str = "../qdrant_storage"
    BASE_MODEL_NAME: str = "Qwen/Qwen2.5-0.5B-Instruct"
    EMBEDDER_MODEL_NAME: str = "all-MiniLM-L6-v2"
    COLLECTION_NAME: str = "documents"
    
    
    MAX_NEW_TOKENS: int = 256
    TEMPERATURE: float = 0.2
    TOP_P: float = 0.85

    # TORCH_DTYPE: torch.dtype = (torch.float16 if torch.cuda.is_available() else torch.float32)

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


settings = Settings()

# if __name__ == "__main__":
#     print("Настройки проекта:")
#     print(settings.dict())