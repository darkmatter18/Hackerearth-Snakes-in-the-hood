import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize, \
    Compose, RandomResizedCrop


class SnakeDataset(Dataset):
    def __init__(self, dataroot: str, phase: str, data: np.array, is_train: bool = False, is_y: bool = False,
                 load_size: int = 156, crop_size: int = 128):
        self.data = data
        self.load_size = load_size
        self.crop_size = crop_size
        self.is_train = is_train
        self.is_y = is_y

        self.transforms = self.get_transform(self.is_train)
        self.image_dir = os.path.join(dataroot, phase)
        print(self.transforms)

    def __getitem__(self, index):
        if self.is_y:
            d = self.data[index, :]
            f_name = d[0]
            label = torch.from_numpy(np.array(d[-1], dtype=np.int64))
        else:
            d = self.data[index]
            f_name = d[0]
            label = np.array([])

        file = os.path.join(self.image_dir, f'{f_name}.jpg')
        img = Image.open(file).convert('RGB')
        img_t = self.transforms(img)

        return {'image': img_t, 'label': label, 'image_id': f_name}

    def __len__(self):
        return len(self.data)

    def get_transform(self, is_train: bool = False):
        if is_train:
            return Compose([
                RandomResizedCrop(self.crop_size),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            return Compose([
                Resize(self.load_size),
                CenterCrop(self.crop_size),
                ToTensor(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
