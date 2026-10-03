import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import src.config as config
from src.utils import build_vocab_and_embeddings
from src.dataset import TextClassificationDataset
from src.model import LSTMClassifier
from src.train import run_epoch

from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

import json

import matplotlib.pyplot as plt

def main():
    print(f"Device: {config.DEVICE}")
    
    # 1. Загрузка данных
    # Ожидается CSV с колонками 'text' и 'label' (числа от 0 до NUM_CLASSES-1)
    df = pd.read_csv(config.DATA_PATH)
    texts, labels = df['text'].values, df['label'].values
    
    # 2. Построение словаря и матрицы эмбеддингов
    vocab, embedding_matrix = build_vocab_and_embeddings(
        texts=texts, 
        embedding_path=config.EMBEDDING_PATH, 
        embed_dim=config.EMBEDDING_DIM
    )
    
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
    
    train_dataset = TextClassificationDataset(X_train, y_train, vocab, config.MAX_SEQ_LEN)
    val_dataset = TextClassificationDataset(X_val, y_val, vocab, config.MAX_SEQ_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # 3. Инициализация модели, функции потерь и оптимизатора
    model = LSTMClassifier(
        embedding_matrix=embedding_matrix,
        hidden_dim=config.HIDDEN_DIM,
        num_classes=config.NUM_CLASSES,
        num_layers=config.NUM_LAYERS
    ).to(config.DEVICE)
    
    # Для мультиклассовой классификации (Softmax встроен внутрь)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    acc_metric = MulticlassAccuracy(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    f1_metric = MulticlassF1Score(num_classes=config.NUM_CLASSES, average='macro').to(config.DEVICE)

    loss_train = [] 
    loss_val = []

    acc_val = []
    f1_val = []
    
    # 6. Цикл обучения
    best_val_acc = 0.0
    patience = config.PATIENCE
    patience_counter = 0
    for epoch in range(config.EPOCHS):
        print(f"\nЭпоха {epoch+1}/{config.EPOCHS}")
        
        train_loss = run_epoch(model, train_loader, criterion, config.DEVICE, optimizer)
        val_loss, val_acc, val_f1 = run_epoch(model, val_loader, criterion, config.DEVICE, optimizer=None, acc_metric=acc_metric, f1_metric=f1_metric)
        
        print(f"Mean Train Loss: {train_loss:.4f}")
        print(f"Mean Validation Loss: {val_loss:.4f}")

        loss_train.append(train_loss)
        loss_val.append(val_loss) 

        acc_val.append(val_acc)
        f1_val.append(val_f1)

        print(f"Эпоха завершена | Accuracy: {val_acc:.4f} | F1-Score: {val_f1:.4f}")

        acc_metric.reset()
        f1_metric.reset()
        
        # Сохранение лучшей модели
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print("New best model saved")
        else:
            patience_counter += 1
            print(f"-> Validation Accuracy не улучшился. Ожидание: {patience_counter}/{patience}")
    
            if patience_counter >= patience:
                print(f" Early Stopping! Обучение остановлено на эпохе {epoch+1}.")
                break
    loss_history = {"train_loss": loss_train, "val_loss": loss_val}
    with open(config.LOSS_HISTORY_PATH, "w") as f:
        json.dump(loss_history, f, indent=2)

    metrics_history = {"val_accuracy": acc_val, "val_f1": f1_val}
    with open(config.METRICS_HISTORY_PATH, "w") as f:
        json.dump(metrics_history, f, indent=2)

    print("Metrics saved")

    plt.plot(loss_train, label="Training Loss")
    plt.plot(loss_val, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()
    plt.savefig("loss_plot1.png")

if __name__ == "__main__":
    main()