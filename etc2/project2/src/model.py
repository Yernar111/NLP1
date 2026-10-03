import torch
import torch.nn as nn

# Архитектура модели LSTM.
# Сначала мы инициализируем слой эмбеддингов с предобученной матрицей эмбеддингов, затем передаем эти эмбеддинги в LSTM слой, а на выходе получаем логиты для каждого класса через полносвязный слой. В конце мы применяем dropout для регуляризации и предотвращения переобучения.

class LSTMClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, num_classes, num_layers=2, bidirectional=True, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        
        # Инициализация слоя эмбеддингов
        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False, padding_idx=0) # nn.Embedding.from_pretrained позволяет использовать предобученные эмбеддинги (например, GloVe или wiki-words) в качестве начальных весов для слоя эмбеддингов. freeze=False означает, что эти веса будут обновляться во время обучения модели, что может улучшить качество классификации на специфичных данных. padding_idx=0 указывает, что индекс 0 используется для паддинга и не должен обновляться во время обучения.
        
        # LSTM слой
        self.lstm = nn.LSTM(
            input_size=embed_dim, # В данном случае 200
            hidden_size=hidden_dim,
            num_layers=num_layers, # Количество рекуррентных слоев, чаще всего ставят 2
            batch_first=True,
            bidirectional=bidirectional, # Двунаправленная LSTM (bidirectional=True), что позволяет модели учитывать контекст как слева, так и справа от текущего слова. Это особенно полезно для задач классификации текста, где контекст может быть важен для правильного понимания смысла предложения.
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Классификатор (умножаем на 2, если LSTM двунаправленная)
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * direction_factor, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)
        # embedded shape: (batch_size, seq_len, embed_dim)
        
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Берем последнее скрытое состояние
        if self.lstm.bidirectional:
            # Конкатенируем последние скрытые состояния прямого и обратного проходов
            hidden_last = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden_last = hidden[-1,:,:]
            
        hidden_last = self.dropout(hidden_last)
        logits = self.fc(hidden_last)
        
        return logits # shape: (batch_size, num_classes)