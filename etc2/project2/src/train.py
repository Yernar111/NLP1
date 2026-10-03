import torch

def run_epoch(model, dataloader, criterion, device, optimizer=None, acc_metric=None, f1_metric=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        
        # Клиппинг градиентов (защита от взрыва градиентов в RNN)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            acc_metric.update(outputs, labels)
            f1_metric.update(outputs, labels)
    
        total_loss += loss.item()

    if is_train:
        return total_loss / len(dataloader)
    else:
        val_acc = acc_metric.compute().item()
        val_f1 = f1_metric.compute().item()
        return total_loss / len(dataloader), val_acc, val_f1