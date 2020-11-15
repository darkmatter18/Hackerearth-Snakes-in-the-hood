import os
import math
import pickle
import pandas as pd
from .datasets import SnakeDataset
from .dataloaders import SnakeDataLoader


def create_dataset(opt):
    data = pd.read_csv(f'{opt.dataroot}/{opt.phase}.csv')
    ds = pd.get_dummies(data, columns=['breed'])

    idx = {i: v[6:] for i, v in enumerate(ds.columns.values[1:])}
    with open(os.path.join(opt.checkpoints_dir, opt.name, f'breads.pkl'), 'wb') as f:
        pickle.dump(idx, f)

    data = ds.values
    train_idx = math.floor(opt.train_ratio * len(data))

    train_data = data[0:train_idx]
    test_data = data[train_idx:-1]

    print(f"No of train data is {len(train_data)}",
          f"No of test data is {len(test_data)}", sep="\n")

    train_dataset = SnakeDataset(opt.dataroot, opt.phase, train_data, opt.preprocess, opt.load_size, opt.crop_size)
    test_dataset = SnakeDataset(opt.dataroot, opt.phase, test_data, opt.preprocess, opt.load_size, opt.crop_size)

    trainloader = SnakeDataLoader(train_dataset, opt.batch_size, opt.num_threads, not opt.no_train_shuffle)
    testloader = SnakeDataLoader(test_dataset, opt.batch_size, opt.num_threads, shuffle=False)

    return trainloader, testloader
