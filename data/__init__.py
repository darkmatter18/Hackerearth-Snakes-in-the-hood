import os
import math
import pickle
import pandas as pd
from .datasets import SnakeDataset, SnakeTestDataset
from .dataloaders import SnakeDataLoader


def create_dataset(opt):
    data = pd.read_csv(f'{opt.dataroot}/{opt.phase}.csv')

    breeds = sorted(set(data['breed'].values))
    breeds_to_idx = {v: i for i, v in enumerate(breeds)}
    idx_to_breeds = {i: v for i, v in enumerate(breeds)}

    with open(os.path.join(opt.checkpoints_dir, opt.name, f'breads.pkl'), 'wb') as f:
        pickle.dump({'breeds_to_idx': breeds_to_idx, 'idx_to_breeds': idx_to_breeds}, f)

    data = data.replace({"breed": breeds_to_idx})

    dataset = data.values
    train_idx = math.floor(opt.train_ratio * len(dataset))

    train_data = dataset[0:train_idx]
    test_data = dataset[train_idx:-1]

    print(f"No of train data is {len(train_data)}",
          f"No of test data is {len(test_data)}", sep="\n")

    train_dataset = SnakeDataset(opt.dataroot, opt.phase, train_data, opt.preprocess, True, opt.load_size,
                                 opt.crop_size)
    test_dataset = SnakeDataset(opt.dataroot, opt.phase, test_data, opt.preprocess, False, opt.load_size, opt.crop_size)

    trainloader = SnakeDataLoader(train_dataset, opt.batch_size, opt.num_threads, not opt.no_train_shuffle)
    testloader = SnakeDataLoader(test_dataset, opt.batch_size, opt.num_threads, shuffle=False)

    return trainloader, testloader


def create_test_dataset(opt):
    data = pd.read_csv(f'{opt.dataroot}/{opt.phase}.csv')['image_id'].values

    test_dataset = SnakeTestDataset(opt.dataroot, opt.phase, data, opt.preprocess, opt.load_size, opt.crop_size)
    testloader = SnakeDataLoader(test_dataset, opt.batch_size, opt.num_threads, shuffle=False)

    return testloader