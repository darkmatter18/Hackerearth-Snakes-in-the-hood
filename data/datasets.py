import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, CenterCrop, RandomRotation, RandomHorizontalFlip, ToTensor, Normalize, \
    Compose


class SnakeDataset(Dataset):
    def __init__(self, dataroot: str, phase: str, data: np.array, preprocess: str = 'resize,crop,rotate,flip',
                 apply_augmentation: bool = False, test_mode: bool = False, load_size: int = 156, crop_size: int = 128):
        self.data = data
        self.preprocess = preprocess
        self.load_size = load_size
        self.crop_size = crop_size
        self.test_mode = test_mode

        self.transforms = self.get_transform(self.preprocess, augmentation=apply_augmentation)
        self.image_dir = os.path.join(dataroot, phase)

    def __getitem__(self, index):
        if not self.test_mode:
            d = self.data[index, :]
            f_name = d[0]
        else:
            d = self.data[index]
            f_name = d
        file = os.path.join(self.image_dir, f'{f_name}.jpg')
        img = Image.open(file).convert('RGB')
        img_t = self.transforms(img)

        if not self.test_mode:
            label = torch.from_numpy(np.array(d[-1], dtype=np.int64))
        else:
            label = np.array([])

        return {'image': img_t, 'label': label, 'image_id': f_name}

    def __len__(self):
        return len(self.data)

    def get_transform(self, preprocess: str, augmentation: bool = False, convert=True):
        transform_list = []

        if 'resize' in preprocess:
            transform_list.append(Resize([self.load_size, self.load_size]))

        if 'rotate' in preprocess and augmentation:
            transform_list.append(RandomRotation(20))

        if 'flip' in preprocess and augmentation:
            transform_list.append(RandomHorizontalFlip())

        if 'crop' in preprocess:
            transform_list.append(CenterCrop(self.crop_size))

        if convert:
            transform_list.append(ToTensor())
            transform_list.append(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        return Compose(transform_list)
