from torch.utils.data import DataLoader


class SITHDataLoader:
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, torch_dataset, batch_size=1, num_threads=0, shuffle=True):
        """
        SITHDataLoader Class

        :param torch_dataset: Torch dataset
        :param batch_size: no of examples per batch. Default: 1
        :param num_threads: number of workers for multiprocessing. Default: 0
        :param shuffle: Whether the dataset has shuffle or not. Default: True
        """

        self.dataset = torch_dataset
        self.batch_size = batch_size
        print("dataset [%s] was created" % type(self.dataset).__name__)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=int(num_threads))

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return len(self.dataset)

    def __iter__(self):
        """Return a batch of data"""
        for data in self.dataloader:
            yield data
