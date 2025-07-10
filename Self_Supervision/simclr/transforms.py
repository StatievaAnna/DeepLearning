import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import numpy as np

# CHANGE TRANSFORMS
def get_transforms(mean, std):
    train_transform = A.Compose([
    A.OneOf([
        A.ColorJitter(),
        A.ToGray(),
#         A.GaussNoise(),
    ]),
    A.OneOf([
        A.CoarseDropout(max_holes=1, max_height=10, max_width=10, min_holes=1, min_height=5, min_width=5),
        A.Compose([
            A.RandomCrop(height=28, width=28),  # Сначала кадрируем
            A.Resize(height=32, width=32)
        ]),
        A.GaussianBlur(blur_limit=(1,3)),
    ]),
    A.HorizontalFlip(),
    A.RandomRotate90(),
    A.Normalize(mean=mean, std=std),
    ToTensorV2()
    ])

    valid_transform = A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    return train_transform, valid_transform

class CLDataset(Dataset):
    def __init__(self, x_data, y_data, transform_augment=None):
        self.x_data = x_data
        self.y_data = y_data

        assert transform_augment is not None, 'set transform_augment'
        # TODO: pass your code
        self.transform_augment = transform_augment

    def __len__(self):
        # TODO: pass your code
        return len(self.x_data)

    def __getitem__(self, item):
        image = self.x_data[item]
        image = (image * 255).astype(np.uint8)
        label = self.y_data[item]

        # TODO: pass your code
        x1 = self.transform_augment(image=image)['image']
        x2 = self.transform_augment(image=image)['image']

        image = torch.tensor(image).permute(2, 0, 1)

        return x1, x2, label, image