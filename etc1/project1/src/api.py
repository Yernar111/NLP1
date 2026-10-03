import pickle
import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel

# 1. Загрузка сохраненного словаря
with open("../models/text_vectorizer_vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

# 2. Восстановление слоя TextVectorization
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=10000,
    output_sequence_length=50
)

vectorizer.set_vocabulary(vocab)

# 1. Глобальная загрузка артефактов один раз при старте скрипта
model = tf.keras.models.load_model("../models/lstm_wiki_model.keras")

emotions_list = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]

# 2. Инициализация приложения
app = FastAPI()

# 3. Схема входного JSON-запроса
class TextRequest(BaseModel):
    texts: list[str]

@app.get("/")
def main():
    return {"message": "ok"}

# 4. Эндпоинт для инференса
@app.post("/predict")
def predict(request: TextRequest):
    # Преобразуем входящий массив строк в векторы
    vectorized_data = vectorizer(request.texts)
    
    # Прогоняем через модель
    probabilities = model.predict(vectorized_data, verbose=0)
    
    # Собираем результат
    results = []
    for text, prob in zip(request.texts, probabilities):
        class_id = int(np.argmax(prob))
        confidence = float(prob[class_id])
        label_name = emotions_list[class_id]
        results.append({
            "text": text,
            "label": label_name,
            "confidence": round(confidence, 4)
        })
        
    return {"predictions": results}

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)