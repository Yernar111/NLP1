import tensorflow as tf
import numpy as np

def build_rnn_model(vocab_size: int, embedding_matrix: np.ndarray, max_seq_len: int):
    model = tf.keras.Sequential([
        # Входной слой (принимает последовательность индексов/токенов)
        tf.keras.layers.Input(shape=(max_seq_len,)), # max_seq_len - это длина входящей последовательности токенов. Каждый текст будет представлен как последовательность индексов слов (чисел), длиной max_seq_len.
        
        # Слой эмбеддингов, инициализированный векторами wiki-words
        # trainable=False блокирует их изменение при обучении (Fine-tuning отключен)
        tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=250,
            weights=[embedding_matrix], # Передаем матрицу эмбеддингов, чтобы слой использовал предобученные векторы
            trainable=False,
            mask_zero=True # Игнорируем нули от padding
        ),
        
        # Рекуррентный слой LSTM
        tf.keras.layers.LSTM(64, return_sequences=False, dropout=0.2), # return_sequences=False означает, что мы хотим получить только последний скрытый вектор, а не всю последовательность скрытых состояний. dropout=0.2 добавляет регуляризацию для предотвращения переобучения.
        
        # Полносвязный слой
        tf.keras.layers.Dense(32, activation='relu'),

        # Выходной слой для 6 классов
        tf.keras.layers.Dense(6, activation='softmax')
    ])
    
    model.compile( # Компилируем модель, указываем оптимизатор, функцию потерь и метрики для оценки качества модели
        optimizer='adam',
        loss='sparse_categorical_crossentropy', # Используем sparse_categorical_crossentropy, так как у нас есть целые числа (индексы классов) в качестве меток, а не one-hot векторы.
        metrics=['accuracy']
    )
    
    return model