import numpy as np
import torch
from src.config import (MIN_WORD_FREQ, VOCAB_SAVE_PATH)
import pickle

def build_vocab_and_embeddings(texts, embedding_path, embed_dim, min_freq=MIN_WORD_FREQ):
    """
    Создает словарь из корпуса и матрицу весов на основе GloVe200.
    """
    # 1. Подсчет частот каждого слова
    word_counts = {}
    for text in texts:
        for word in str(text).split():
            word_counts[word] = word_counts.get(word, 0) + 1
            
    # 2. Формирование словаря (0: PAD, 1: UNK)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = len(vocab) # Каждое новое слово получает следующий индекс

    with open(VOCAB_SAVE_PATH, "wb") as f:
        pickle.dump(vocab, f)
    
    print("Vocabulary saved")
            
    vocab_size = len(vocab)
    embedding_matrix = np.zeros((vocab_size, embed_dim))
    
    # Инициализация <UNK> случайным нормальным распределением. Это позволит модели обучаться на неизвестных словах, а не игнорировать их полностью. Потому что если мы оставим вектор <UNK> нулевым, то модель будет игнорировать неизвестные слова, что может ухудшить качество предсказаний.
    embedding_matrix[1] = np.random.normal(scale=0.6, size=(embed_dim,))
    
    # 3. Загрузка предобученных весов wiki-words
    loaded_words = 0
    with open(embedding_path, "r", encoding="utf-8") as f: # Открываем текстовый файл эмбеддингов. Каждая строка имеет формат: "word 0.123 0.456 ... 0.789" 
        for line in f:
            # values = line.strip().split()

            # Разделяем строку СПРАВА ровно embed_dim раз. Так как в одной строке embed_dim чисел и 1 слово. Это нужно, чтобы корректно обрабатывать слова, которые могут содержать пробелы (например, "New York"). Мы хотим получить 1 слово + embed_dim чисел.
            values = line.rstrip().rsplit(' ', embed_dim)

            # Проверяем, что строка корректно разбилась на (1 слово + embed_dim чисел)
            if len(values) != embed_dim + 1:
                continue

            word = values[0] # Первый элемент - это слово, остальные embed_dim элементов - это вектор эмбеддинга.
            if word in vocab:
                vector = np.asarray(values[1:], dtype="float32") # Преобразуем оставшиеся embed_dim элементов в массив numpy типа float32. В итоге получаем вектор эмбеддинга для текущего слова.
                idx = vocab[word]
                embedding_matrix[idx] = vector
                loaded_words += 1
                
    print(f"Успешно найдено {loaded_words}/{vocab_size} слов из словаря в файле GloVe200.")
    return vocab, torch.tensor(embedding_matrix, dtype=torch.float32)