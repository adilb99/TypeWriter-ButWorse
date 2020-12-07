import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TWDataset(Dataset):
    def __init__(self, data_dir):
        with open(data_dir) as f:
            self.data_list = [line.rstrip() for line in f]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        '''
            sample should contain input info and labels
        '''
        fname = self.data_list[index]
        sample = np.load(fname)
        return sample
