import pickle
import numpy as np
import tensorflow as tf

# 1. Загрузка сохраненного словаря
with open("models/text_vectorizer_vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

# 2. Восстановление слоя TextVectorization
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=10000,
    output_sequence_length=50
)

vectorizer.set_vocabulary(vocab)

# Загрузка обученной модели Keras
model = tf.keras.models.load_model("models/rnn_wiki_model.keras")

emotions_list = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]

def predict_sentiment(texts):

    # Предобработка/токенизация входящих текстов
    vectorized_texts = vectorizer(texts)

    # Предсказание (возвращает матрицу размерностью [len(texts), 6])
    predictions = model.predict(vectorized_texts, verbose=0)

    # Формирование читаемого ответа
    results = []
    for text, prob in zip(texts, predictions):
        class_id = int(np.argmax(prob))
        confidence = float(prob[class_id])

        label_name = emotions_list[class_id]
        results.append({
            "text": text,
            "label": label_name,
            "confidence": round(confidence, 4)
        })
    return results


if __name__ == "__main__":

    # Примеры новых сообщений для инференса
    new_messages = [
        "Great service, everything was fast and smooth!",
        "The package arrived broken and customer support was useless.",
        "I am very angry",
        "Send money real quick",
        "I have a lot of deadlines tomorrow"
    ]

    # Запуск предсказания
    output = predict_sentiment(new_messages)

    for item in output:
        print(f"Text: '{item['text']}'")
        print(f"Emotion: {item['label']} (Confidence: {item['confidence']})\n")