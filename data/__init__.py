import math
import os
import pickle

import numpy as np
import pandas as pd

from .dataloaders import SnakeDataLoader
from .datasets import SnakeDataset


def create_dataset(opt):
    data = pd.read_csv(f'{opt.dataroot}/{opt.phase}.csv')

    breeds = sorted(set(data['breed'].values))
    breeds_to_idx = {v: i for i, v in enumerate(breeds)}
    idx_to_breeds = {i: v for i, v in enumerate(breeds)}

    with open(os.path.join(opt.checkpoints_dir, opt.name, f'breads.pkl'), 'wb') as f:
        pickle.dump({'breeds_to_idx': breeds_to_idx, 'idx_to_breeds': idx_to_breeds}, f)

    data = data.replace({"breed": breeds_to_idx})

    dataset = data.values
    train_idx = np.random.choice(dataset.shape[0],
                                 math.floor(opt.train_ratio * len(dataset)), replace=False)

    train_data = dataset[train_idx]  # Train data
    test_data = np.delete(dataset, train_idx, axis=0)  # Test data

    print(f"No of train data is {len(train_data)}",
          f"No of test data is {len(test_data)}", sep="\n")

    train_dataset = SnakeDataset(dataroot=opt.dataroot, phase=opt.phase, data=train_data, preprocess=opt.preprocess,
                                 apply_augmentation=True, test_mode=not opt.isTrain, load_size=opt.load_size,
                                 crop_size=opt.crop_size)
    test_dataset = SnakeDataset(dataroot=opt.dataroot, phase=opt.phase, data=train_data, preprocess=opt.preprocess,
                                apply_augmentation=False, test_mode=not opt.isTrain, load_size=opt.load_size,
                                crop_size=opt.crop_size)

    trainloader = SnakeDataLoader(train_dataset, opt.batch_size, opt.num_threads, not opt.no_train_shuffle)
    testloader = SnakeDataLoader(test_dataset, opt.batch_size, opt.num_threads, shuffle=False)

    return trainloader, testloader


def create_test_dataset(opt):
    data = pd.read_csv(f'{opt.dataroot}/{opt.phase}.csv')

    if opt.phase == "train":
        breeds = sorted(set(data['breed'].values))
        breeds_to_idx = {v: i for i, v in enumerate(breeds)}
        data = data.replace({"breed": breeds_to_idx})

    test_dataset = SnakeDataset(dataroot=opt.dataroot, phase=opt.phase, data=data.values, preprocess=opt.preprocess,
                                apply_augmentation=False, test_mode=opt.phase == "test",
                                load_size=opt.load_size, crop_size=opt.crop_size)

    testloader = SnakeDataLoader(test_dataset, opt.batch_size, opt.num_threads, shuffle=False)

    return testloader
