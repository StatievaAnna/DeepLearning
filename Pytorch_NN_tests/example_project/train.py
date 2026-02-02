import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm, trange
import wandb

from hparams import config

def compute_accuracy(preds, targets):
    return (targets == preds).float().mean()

def get_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

def load_datasets(root="CIFAR10", download=False):
    tfm = get_transforms()
    train_ds = CIFAR10(root + '/train', train=True, transform=tfm, download=download)
    test_ds = CIFAR10(root + '/test', train=False, transform=tfm, download=download)
    return train_ds, test_ds

def create_data_loaders(train_ds, test_ds, batch_size, train_size=None, test_size=None):
    if train_size is not None:
        train_ds = Subset(train_ds, list(range(train_size)))
    if test_size is not None:
        test_ds = Subset(test_ds, list(range(test_size)))
        
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def initialize_model(device, zero_init_residual):
    model = models.resnet18(
        pretrained=False,
        num_classes=10,
        zero_init_residual=zero_init_residual
    ).to(device)
    wandb.watch(model)
    return model

def create_optimizer(model, lr, weight_decay):
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

def train_single_batch(model, batch, criterion, optimizer, device):
    images, labels = batch
    images, labels = images.to(device), labels.to(device)
    
    outputs = model(images)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            all_preds.append(preds)
            all_labels.append(y)
    
    accuracy = compute_accuracy(torch.cat(all_preds), torch.cat(all_labels))
    model.train()
    return accuracy

def log_metrics(metrics, epoch, batch_idx, batch_size, dataset_len):
    step = epoch * dataset_len + (batch_idx + 1) * batch_size
    wandb.log(metrics, step=step)

def save_artifacts(model):
    torch.save(model.state_dict(), "model.pt")
    with open("run_id.txt", "w+") as f:
        print(wandb.run.id, file=f)

def train_model(config):
    """Основной цикл обучения."""
    # Инициализация WandB
    wandb.init(config=config, project="effdl_example", name="test", dir="wandb_logs")
    
    # Настройка устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Загрузка данных
    train_dataset, test_dataset = load_datasets(root="CIFAR10", download=True)
    train_loader, test_loader = create_data_loaders(
        train_dataset, test_dataset,
        batch_size=config["batch_size"],
        train_size=config["train_size"], # для тестов
        test_size=config["test_size"]
    )
    
    # Инициализация модели
    model = initialize_model(
        device,
        zero_init_residual=config["zero_init_residual"]
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(
        model,
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    
    last_loss = 0

    for epoch in trange(config["epochs"], desc="Epochs"):
        model.train()
        for i, batch in enumerate(tqdm(train_loader, desc="Batches", leave=False)):
            loss = train_single_batch(model, batch, criterion, optimizer, device)
            
            # Логирование каждые 100 батчей
            if i % 100 == 0:
                accuracy = evaluate_model(model, test_loader, device)
                log_metrics(
                    {"test_acc": accuracy, "train_loss": loss.item()},
                    epoch, i, config["batch_size"], len(train_dataset)
                )

            last_loss = loss    
    accuracy = evaluate_model(model, test_loader, device)
    wandb.log({"final_train_loss": last_loss.item(), "final_test_acc": accuracy.item()})

    save_artifacts(model)

if __name__ == '__main__':
    train_model(config) 