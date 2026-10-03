import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Линейные слои для получения Q, K, V из входного вектора
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, embed_dim = x.shape
        
        # 1. Проектируем в Q, K, V
        Q = self.q_proj(x)  # (batch_size, seq_len, embed_dim)
        K = self.k_proj(x)  # (batch_size, seq_len, embed_dim)
        V = self.v_proj(x)  # (batch_size, seq_len, embed_dim)
        
        # 2. Вычисляем скалярное произведение Q и K^T
        # K.transpose(-2, -1) меняет местами последние две размерности: (batch_size, embed_dim, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch_size, seq_len, seq_len)
        
        # 3. Масштабируем на sqrt(d_k)
        d_k = embed_dim
        scores = scores / (d_k ** 0.5)
        
        # 4. Применяем Softmax вдоль последней оси (по всем ключам для каждого запроса)
        attention_weights = F.softmax(scores, dim=-1)  # (batch_size, seq_len, seq_len)
        
        # 5. Перемножаем веса на V
        output = torch.matmul(attention_weights, V)  # (batch_size, seq_len, embed_dim)
        
        return output, attention_weights

# --- Проверка работы модуля ---
if __name__ == "__main__":
    batch_size = 2
    seq_len = 4     # Предложение из 4 токенов
    embed_dim = 16  # Размерность векторного представления слова

    # Создаём случайный входной тензор (например, эмбеддинги слов)
    x = torch.randn(batch_size, seq_len, embed_dim)

    attention_layer = SelfAttention(embed_dim)
    output, weights = attention_layer(x)

    print("Размерность входа: ", x.shape)         # torch.Size([2, 4, 16])
    print("Размерность выхода:", output.shape)    # torch.Size([2, 4, 16])
    print("Размерность матрицы внимания:", weights.shape) # torch.Size([2, 4, 4])