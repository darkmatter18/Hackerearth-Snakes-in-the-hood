import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, RandomCrop, RandomRotation, RandomHorizontalFlip, ToTensor, Normalize, \
    Compose


class SnakeDataset(Dataset):
    def __init__(self, dataroot: str, phase: str, data: np.array, preprocess: str = 'resize,crop,rotate,flip',
                 load_size: int = 156, crop_size: int = 128):
        self.data = data
        self.preprocess = preprocess
        self.load_size = load_size
        self.crop_size = crop_size

        self.transforms = self.get_transform()
        self.image_dir = os.path.join(dataroot, phase)

    def __getitem__(self, index):
        d = self.data[index, :]
        file = os.path.join(self.image_dir, f'{d[0]}.jpg')
        img = Image.open(file).convert('RGB')
        img_t = self.transforms(img)
        label = torch.from_numpy(np.array(d[-1], dtype=np.int64))
        return {'image': img_t, 'label': label}

    def __len__(self):
        return len(self.data)

    def get_transform(self, convert=True):
        transform_list = []

        if 'resize' in self.preprocess:
            transform_list.append(Resize([self.load_size, self.load_size]))

        if 'rotate' in self.preprocess:
            transform_list.append(RandomRotation(30))

        if 'crop' in self.preprocess:
            transform_list.append(RandomCrop(self.crop_size))

        if 'flip' in self.preprocess:
            transform_list.append(RandomHorizontalFlip())

        if convert:
            transform_list.append(ToTensor())
            transform_list.append(Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        return Compose(transform_list)
