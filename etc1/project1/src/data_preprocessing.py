import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

import pickle # pickle нужен для сохранения словаря в файл, чтобы потом его можно было загрузить и использовать при инференсе модели

def create_vectorizer_and_matrix(texts, max_seq_len=50, max_tokens=10000):
    # 1. Токенизируем текст (разбиваем на индексы)
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=max_tokens, # Ограничиваем словарь 10000 самыми частыми словами
        output_sequence_length=max_seq_len # Фиксированная длина входящей последовательности токенов. Сообщения с более 50 токенами будут обрезаны, а если текст короче - дополнены нулями 
    )
    vectorizer.adapt(texts) # Обучаем векторайзер на нашем датасете. Набор текстов разбивается на токены, подсчитываются частоты слов, создается словарь в котором каждое слово имеет свой индекс.
    
    # 2. Получаем словарь датасета
    vocab = vectorizer.get_vocabulary()

    with open("models/text_vectorizer_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    print("Vectorizer saved")

    vocab_size = len(vocab)
    
    # 3. Загружаем wiki-words и генерируем векторы для нашего словаря
    print("Загрузка wiki-words-250...")
    embed_model = hub.load("../wiki_words")
    
    # tfhub принимает список слов и возвращает тензор векторов
    # Передаем словарь батчами (или целиком, если позволяет ОЗУ)
    word_vectors = embed_model(vocab).numpy()
    
    # Создаем матрицу весов для слоя Embedding
    embedding_matrix = np.zeros((vocab_size, 250)) # Матрица эмбеддингов в которой каждая строка это эмбеддинг для одного слова из словаря
    for i, word in enumerate(vocab): # Даем каждому слову индекс и проходимся по словарю
        # Индексы 0 (padding) и 1 (unknown) оставляем нулевыми
        if i > 1:
            embedding_matrix[i] = word_vectors[i]
            
    return vectorizer, embedding_matrix, vocab_size