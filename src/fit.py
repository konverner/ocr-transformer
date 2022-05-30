from time import time
import numpy as np
import torch
from config import DEVICE
from utils import labels_to_text, char_error_rate, evaluate, log_metrics

def train(model, optimizer, criterion, train_loader):
    """
    params
    ---
    model : nn.Module
    optimizer : nn.Object
    criterion : nn.Object
    train_loader : torch.utils.data.DataLoader

    returns
    ---
    epoch_loss / len(train_loader) : float
        overall loss
    """
    model.train()
    epoch_loss = 0
    for src, trg in train_loader:
        src, trg = src.to(DEVICE), trg.to(DEVICE)
        optimizer.zero_grad()
        output = model(src, trg[:-1, :])

        loss = criterion(output.view(-1, output.shape[-1]), torch.reshape(trg[1:, :], (-1,)))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(train_loader)


# GENERAL FUNCTION FROM TRAINING AND VALIDATION
def fit(model, optimizer, criterion, train_loader, val_loader, epoch_limit):
    metrics = []
    for epoch in range(0, epoch_limit):
      epoch_metrics = {}
      start_time = time()
      train_loss = train(model, optimizer, criterion, train_loader)
      end_time = time()
      epoch_metrics = evaluate(model, criterion, val_loader)
      epoch_metrics['train_loss'] = train_loss
      epoch_metrics['epoch'] = epoch
      epoch_metrics['time'] = end_time - start_time
      metrics.append(epoch_metrics)
      log_metrics(epoch_metrics)
