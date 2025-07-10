import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import numpy as np
import wandb

from load_data import init_data

def get_transformC(mean, std):
    transformC = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
    ])
    return transformC

def init_dataC(hyp):
  _, _, mean, std = init_data(hyp)
  transformC = get_transformC(mean, std)
  full_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transformC)
  train_size = int(0.75*(len(full_dataset)))
  valid_size = len(full_dataset) - train_size
  train_dataset, valid_dataset = random_split(full_dataset,[train_size, valid_size])

  train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
  valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

  return train_loader, valid_loader

class Classifier(nn.Module):
  def __init__(self, encoder, num_classes=10, device='cuda'): #чистый энкодер с пустым encoder.fc
    super().__init__()
    self.device = device
    self.encoder = encoder
    for p in self.encoder.parameters():
      p.requires_grad = False
    with torch.no_grad():
      dummy = torch.randn(2, 3, 32, 32)
      feat_dim = self.encoder(dummy).shape[1]
      # print('размерность выхода энкодера:', self.encoder(dummy).shape)
    self.flatten = nn.Flatten()
    self.classifier = nn.Linear(feat_dim, num_classes)
    model_params = list(self.classifier.parameters())
    optimizer = torch.optim.AdamW(model_params)
    self.criterion = nn.CrossEntropyLoss().to(self.device)
    self.optimizer = optimizer
    self.train_losses = []
    self.valid_losses = []
    self.train_accs = []
    self.valis_accs = []

  def forward(self, x):
    x = x.to(self.device)
    x = self.encoder(x)
    x = self.flatten(x)
    return self.classifier(x)

  def train_step(self, train_loader):
    total_loss, correct, total = (0, 0, 0)

    train_iter = tqdm(train_loader, desc='Training', leave=False, position=0)
    for x, y in train_iter:
      x, y = x.to(self.device), y.to(self.device)
      self.optimizer.zero_grad()

      logits = self(x)
      loss = self.criterion(logits, y)
      loss.backward()
      self.optimizer.step()

      predict = torch.argmax(logits, dim=1)
      correct += (predict == y).sum().item()
      total_loss += loss.item()
      total += len(y)

      train_iter.set_postfix({
          'loss': f"{total_loss/(total/len(y)):.4f}",
          'acc': f"{correct/total:.2%}"
      })

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total

    return avg_loss, accuracy

  def valid_step(self, valid_loader):
      self.eval()
      total_loss, correct, total = (0, 0, 0)
      valid_iter = tqdm(valid_loader, desc="Validation", leave=False, position=0)
      for x, y in valid_iter:
        x, y = x.to(self.device), y.to(self.device)
        logits = self(x)
        loss = self.criterion(logits, y)

        predict = torch.argmax(logits, dim=1)
        correct += (predict == y).sum().item()
        total_loss += loss.item()
        total += len(y)

        valid_iter.set_postfix({
            'val_loss': f"{total_loss/(total/len(y)):.4f}",
            'val_acc': f"{correct/total:.2%}"
        })

      avg_loss = total_loss / len(valid_loader)
      accuracy = correct / total

      return avg_loss, accuracy

  def run(self, train_loader, valid_loader, num_epochs):
      self.to(self.device)
      epoch_iter = tqdm(range(num_epochs), desc="Epochs")
      for epoch in epoch_iter:
        train_loss, train_accuracy = self.train_step(train_loader)
        valid_loss, valid_accuracy = self.valid_step(valid_loader)
        self.train_losses.append(train_loss)
        self.train_accs.append(train_accuracy)
        self.valid_losses.append(valid_loss)
        self.valis_accs.append(valid_accuracy)

        epoch_iter.set_postfix({
            'train_loss': f"{train_loss:.4f}",
            'train_acc': f"{train_accuracy:.2%}",
            'val_loss': f"{valid_loss:.4f}",
            'val_acc': f"{valid_accuracy:.2%}"
        })


      # self.lr_sheduler.step(train_loss)

      torch.cuda.empty_cache()