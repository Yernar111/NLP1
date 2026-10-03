import torch
import torch.nn.functional as F

def create_causal_mask(seq_len: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """Создает нижнетреугольную маску причинности (1 - разрешено, 0 - заблокировано)."""
    # torch.tril оставляет элементы на и ниже главной диагонали, остальное зануляет
    mask = torch.tril(torch.ones((seq_len, seq_len), device=device))
    return mask

# --- Имитация работы в слое Attention ---

seq_len = 3
batch_size = 1

# 1. Симулируем сырые оценки внимания QK^T / sqrt(d_k)
raw_scores = torch.tensor([
    [[2.0, 1.0, 5.0],   # Токен 1 "видит" сырые оценки к токенам 1, 2, 3
     [1.0, 4.0, 3.0],   # Токен 2
     [3.0, 2.0, 6.0]]   # Токен 3
])

# 2. Генерируем маску (размер 3x3)
mask = create_causal_mask(seq_len)

# 3. Заменяем нули в маске на -inf
# masked_fill заменяет элементы, где mask == 0, на -inf
masked_scores = raw_scores.masked_fill(mask == 0, float('-inf'))

# 4. Вычисляем вероятности через Softmax по последней оси
attention_weights = F.softmax(masked_scores, dim=-1)

print("--- Маска (1 = доступно, 0 = заблокировано) ---")
print(mask.int())

print("\n--- Скоры после маскирования (-inf) ---")
print(masked_scores[0])

print("\n--- Финальные веса внимания (Softmax) ---")
print(attention_weights[0])