import numpy as np
import torch
from torch.utils.data import Dataset

class TWDataset(Dataset):
    def __init__(self, data_dir, fnames):
        with open(data_dir) as f:
            self.data_list = [line.split(',').rstrip() for line in f]
        self.data_dir = data_dir

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        '''
            sample should contain input info and labels
        '''
        fnames = self.data_list[index]
        sample = [np.load(self.data_dir + fname) for fname in fnames]
        sample[0] = sample[0].reshape(-1, 113)
        for i in range(len(sample)):
            sample[i] = torch.tensor(sample[i])
        return sample
