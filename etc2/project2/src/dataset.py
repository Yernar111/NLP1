import torch
from torch.utils.data import Dataset

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        tokens = str(text).split()
        
        # Проходимся по словарю проверяя что токен существует в словаре, иначе используем индекс <UNK>. Перевод в индексы с обрезкой до max_len. 
        indices = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens][:self.max_len]
        
        # Паддинг (дополнение нулями до max_len)
        padding_length = self.max_len - len(indices)
        indices = indices + [self.vocab["<PAD>"]] * padding_length
        
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long) # Возвращаем кортеж (текст в виде индексов, метка)