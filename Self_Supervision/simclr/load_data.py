from transforms import get_transforms, CLDataset
import numpy as np
from torch.utils.data import DataLoader

from cifar import load_cifar10

def get_cropped_data_idxs(data, crop_coef: float = 1.0):
    crop_coef = np.clip(crop_coef, 0, 1)

    init_data_size = len(data)
    final_data_size = int(init_data_size * crop_coef)

    random_idxs = np.random.choice(tuple(range(init_data_size)), final_data_size, replace=False)
    return random_idxs

def load_datasets(crop_coef=0.2):
    X_train, y_train, X_val, y_val, _, _ = load_cifar10("cifar_data", channels_last=True)
    train_idxs = get_cropped_data_idxs(X_train, crop_coef=crop_coef)
    train_data = X_train[train_idxs]
    train_labels = y_train[train_idxs]

    valid_idxs = get_cropped_data_idxs(X_val, crop_coef=crop_coef)
    valid_data = X_val[valid_idxs]
    valid_labels = y_val[valid_idxs]

    mean = np.mean(X_train, axis=(0, 1, 2), keepdims=True).squeeze()
    std = np.std(X_train, axis=(0, 1, 2), keepdims=True).squeeze()

    train_transform, valid_transform = get_transforms(mean, std)

    train_dataset = CLDataset(train_data, train_labels, transform_augment=train_transform)
    valid_dataset = CLDataset(valid_data, valid_labels, transform_augment=valid_transform)

    return train_dataset, valid_dataset, mean, std

def init_data(hyp):
    train_dataset, valid_dataset, mean, std = load_datasets(crop_coef=1.4)
    print('Train size:', len(train_dataset), 'Valid size:', len(valid_dataset))

    train_loader = DataLoader(train_dataset,
                                    batch_size=hyp['batch_size'],
                                    shuffle=True,
                                    num_workers=hyp['n_workers'],
                                    pin_memory=True,
                                    drop_last=True
                                    )

    valid_loader = DataLoader(valid_dataset,
                                    batch_size=hyp['batch_size'],
                                    shuffle=True,
                                    num_workers=hyp['n_workers'],
                                    pin_memory=True,
                                    drop_last=True
                                    )
    
    return train_loader, valid_loader, mean, std