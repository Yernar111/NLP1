import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Данные и пути
DATA_PATH = "data/text.csv"
EMBEDDING_PATH = "../embedding/wiki_giga_2024_200_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt"
MODEL_SAVE_PATH = "models/best_lstm.pth"
VOCAB_SAVE_PATH = "models/vocab.pkl"
LOSS_HISTORY_PATH = "models/loss_history.json"
METRICS_HISTORY_PATH = "models/metrics_history.json"

# Гиперпараметры текста
MAX_SEQ_LEN = 50
MIN_WORD_FREQ = 2
EMBEDDING_DIM = 200

# Гиперпараметры модели и обучения
HIDDEN_DIM = 128
NUM_CLASSES = 6       # Зависит от вашего датасета
NUM_LAYERS = 2
BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 1e-3
DROPOUT = 0.3

# etc
PATIENCE = 3
TEST_SIZE=0.2
RANDOM_STATE=42