import math
import pandas as pd


def create_dataset(opt):
    data = pd.read_csv(f'{opt.dataroot}/{opt.phase}.csv').values
    train_idx = math.floor(opt.train_ratio * len(data))

    train_data = data[0:train_idx]
    test_data = data[train_idx:-1]

    print(f"No of train data is {len(train_data)}",
          f"No of test data is {len(test_data)}", sep="\n")
