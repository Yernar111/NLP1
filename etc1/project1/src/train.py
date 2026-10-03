import pandas as pd
from data_preprocessing import create_vectorizer_and_matrix
from model import build_rnn_model

def train_pipeline():
    # 1. Загрузка данных (предполагаем столбцы 'text' и 'label')
    df = pd.read_csv("data/text.csv")
    texts = df['text'].values
    labels = df['label'].values

    max_seq_len = 50 # Фиксированная длина входящей последовательности токенов
    
    # 2. Подготовка векторайзера и матрицы wiki-words
    vectorizer, emb_matrix, vocab_size = create_vectorizer_and_matrix(texts, max_seq_len)

    # 3. Преобразование текста в индексы
    X = vectorizer(texts).numpy() # X.shape будет (N, 50), где N - количество текстов. Каждый текст превращается в последовательность индексов слов
    y = labels
    
    # 4. Сборка и обучение модели
    model = build_rnn_model(vocab_size, emb_matrix, max_seq_len)
    
    print("Начало обучения RNN...")
    model.fit(X, y, epochs=5, batch_size=128, validation_split=0.2)
    
    # 5. Сохранение (сохраняем и саму модель, и логику токенизации)
    model.save("models/lstm_wiki_model.keras")

    print("Model saved")
    

if __name__ == "__main__":
    train_pipeline()