import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


class TestStore:
    def __init__(self, opt):
        self.opt = opt
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.test_file = os.path.join(self.save_dir, opt.test_file_name)

        with open(os.path.join(opt.checkpoints_dir, opt.name, f'breads.pkl'), 'rb') as f:
            self.idx_to_breeds = pickle.load(f)['idx_to_breeds']

        self.test_data = pd.DataFrame(columns=['image_id', 'breed'])

    def load_test_data(self, image_id: np.ndarray, output: np.ndarray, label_orig: np.ndarray) -> None:
        breed = pd.Series(output).replace(self.idx_to_breeds).values
        print("Output:", output, "Breed:", breed,  sep="\n")

        if label_orig:
            print("Label Original", label_orig, sep="\n")
            print("F1 Score: ", f1_score(label_orig, output, average='weighted'))

        df = pd.DataFrame(np.vstack((image_id, breed)).T, columns=['image_id', 'breed'])
        # print(df)
        self.test_data = pd.concat([self.test_data, df], ignore_index=True)
        # print(self.test_data)

    def write(self):
        print(f"Writing to {self.test_file}")
        self.test_data.to_csv(self.test_file, index=False)
