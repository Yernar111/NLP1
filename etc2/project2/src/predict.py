import torch
import pickle
import numpy as np
import config
from model import LSTMClassifier

import torch.nn.functional as F

# 1. Загрузка сохраненного словаря
with open(config.VOCAB_SAVE_PATH, "rb") as f:
    vocab = pickle.load(f)

vocab_size = len(vocab)

dummy_embedding_matrix = torch.zeros((vocab_size, config.EMBEDDING_DIM))

model = LSTMClassifier(
    embedding_matrix=dummy_embedding_matrix,
    hidden_dim=config.HIDDEN_DIM,
    num_classes=config.NUM_CLASSES,
    num_layers=config.NUM_LAYERS
)

model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
model.to(config.DEVICE)
model.eval()

emotions_list = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]

def predict_sentiment(texts):

    # Предобработка/токенизация входящих текстов
    tokens = str(texts).split()

    indices = [vocab.get(token, vocab["<UNK>"]) for token in tokens][:config.MAX_SEQ_LEN]

    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(config.DEVICE)

    padding_length = config.MAX_SEQ_LEN - len(indices)
    indices = indices + [vocab["<PAD>"]] * padding_length

    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1)
        predicted_class_id = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class_id].item()
        
    # predicted_label = CLASS_MAPPING.get(predicted_class_id, f"Класс {predicted_class_id}")
    predicted_label = emotions_list[predicted_class_id]
    return predicted_label, confidence


if __name__ == "__main__":
    sample_text = "im scared"
    
    label, confidence = predict_sentiment(sample_text)
    
    print(f"Text: {sample_text}")
    print(f"Label: {label}")
    print(f"Confidence: {confidence:.2%}")