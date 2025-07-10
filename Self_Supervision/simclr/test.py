from loss import *
from simclr import *
from cifar import load_cifar10
from transforms import *

import pytest

def get_data():
    X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10("cifar_data", channels_last=True)

    mean = np.mean(X_train, axis=(0, 1, 2), keepdims=True).squeeze()
    std = np.std(X_train, axis=(0, 1, 2), keepdims=True).squeeze()

    train_transform, valid_transform = get_transforms(mean, std)
    train_dataset = CLDataset(X_train, y_train, transform_augment=train_transform)
    valid_dataset = CLDataset(X_val, y_val, transform_augment=valid_transform)

    return train_dataset, valid_dataset

def test_load_cifar():
    X_train, y_train, X_val, y_val, x_test, y_test = load_cifar10("cifar_data", channels_last=True)

    assert (X_train.shape, y_train.shape) == ((40000, 32, 32, 3), (40000, ))
    assert (X_val.shape, y_val.shape) == ((10000, 32, 32, 3), (10000,))

def test_loader():
    batch_size = 32
    n_workers = 0

    train_dataset, valid_dataset = get_data()

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=n_workers)
    
    val_loader = torch.utils.data.DataLoader(valid_dataset,
                                             batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=n_workers)
    x1, x2, _, _ = next(iter(train_loader))
    x3, x4, _, _ = next(iter(val_loader))
    assert (x1.shape, x2.shape) == (torch.Size([32, 3, 32, 32]), torch.Size([32, 3, 32, 32]))
    assert (x3.shape, x4.shape) == (torch.Size([32, 3, 32, 32]), torch.Size([32, 3, 32, 32]))

def test_out_shape():
    device = "cuda"
    model = PreModel()
    model = model.to(device)

    x = np.random.random((32,3,224,224)).astype(np.float32)
    out = model(torch.tensor(x, device=device, dtype=torch.float32))
    assert out.shape == torch.Size([32, 128])

def test_loss():
    x1 = torch.tensor(np.random.random((4, 128)))
    x2 = torch.tensor(np.random.random((4, 128)))

    criterion = SimCLR_Loss(batch_size=4,
                                     temperature=0.2)
    loss = criterion(x1, x2)
    assert loss.item() > 0 
