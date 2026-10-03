# Файл для создания API для инференса модели
from fastapi import FastAPI
from pathlib import Path
from typing import List

import pickle

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field

import config as config
from model import LSTMClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VOCAB_PATH = PROJECT_ROOT / config.VOCAB_SAVE_PATH
MODEL_PATH = PROJECT_ROOT / config.MODEL_SAVE_PATH

app = FastAPI(
    title="Text Emotion Classification API",
    version="1.0.0",
    description="API для предсказания эмоций по тексту на основе LSTM-модели.",
)


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Текст для классификации")


class TextsRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, description="Список текстов для batch-предсказания")


class PredictionResponse(BaseModel):
    text: str
    label: str
    confidence: float


class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]


@app.on_event("startup")
def load_model_and_vocab():
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)

    dummy_embedding_matrix = torch.zeros(
        (len(vocab), config.EMBEDDING_DIM),
        dtype=torch.float32,
    )

    model = LSTMClassifier(
        embedding_matrix=dummy_embedding_matrix,
        hidden_dim=config.HIDDEN_DIM,
        num_classes=config.NUM_CLASSES,
        num_layers=config.NUM_LAYERS,
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=config.DEVICE))
    model.to(config.DEVICE)
    model.eval()

    app.state.vocab = vocab
    app.state.model = model
    app.state.emotions = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]


def ensure_loaded():
    if not hasattr(app.state, "model") or not hasattr(app.state, "vocab"):
        load_model_and_vocab()


def predict_single_text(model: torch.nn.Module, vocab: dict, text: str):
    tokens = str(text).split()
    indices = [vocab.get(token, vocab["<UNK>"]) for token in tokens][: config.MAX_SEQ_LEN]

    padding_length = config.MAX_SEQ_LEN - len(indices)
    indices = indices + [vocab["<PAD>"]] * padding_length

    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1)
        predicted_class_id = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class_id].item()

    label = app.state.emotions[predicted_class_id]
    return {
        "text": text,
        "label": label,
        "confidence": round(float(confidence), 4),
    }


@app.get("/health")
def healthcheck():
    ensure_loaded()
    return {"status": "ok", "model": "LSTMClassifier"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: TextRequest):
    ensure_loaded()
    model = app.state.model
    vocab = app.state.vocab
    return predict_single_text(model=model, vocab=vocab, text=request.text)


@app.post("/predict_batch", response_model=BatchPredictionResponse)
def predict_batch(request: TextsRequest):
    ensure_loaded()
    model = app.state.model
    vocab = app.state.vocab

    results = [
        predict_single_text(model=model, vocab=vocab, text=text)
        for text in request.texts
    ]

    return {"results": results}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
