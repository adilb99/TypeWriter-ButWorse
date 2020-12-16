import numpy as np
import torch
from torch.utils.data import Dataset

class TWDataset(Dataset):
    def __init__(self, data_dir, fnames):
        with open(fnames) as f:
            self.data_list = [line.split(',') for line in f]
        self.data_dir = data_dir

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        '''
            sample should contain input info and labels
        '''
        fnames = self.data_list[index]
        # sample = [np.load(self.data_dir + fname) for fname in fnames]
        sample = []
        for fname in fnames:
            if fname.endswith("\n"):
                fname = fname[:-1]
            sample.append(np.load(self.data_dir + fname))
        sample[0] = sample[0].reshape(-1, 113)
        if sample[0].shape[0]==5080:
            sample[0] = np.concatenate((sample[0], np.zeros((120,113), dtype=np.bool)))
        sample[2] = sample[2].reshape(1, -1)
        for i in range(len(sample)):
            sample[i] = torch.tensor(sample[i])
        return sample


def collate_fn(batch):
    func_body = [data[0].unsqueeze(0) for data in batch]
    docstr = [data[1].unsqueeze(0) for data in batch]
    occur = [data[2].unsqueeze(0) for data in batch]
    labels = [data[3].unsqueeze(0) for data in batch]

    fb = func_body[0]
    doc = docstr[0]
    occ = occur[0]
    gt = labels[0]
    for fb_, doc_, occ_, gt_ in zip(func_body[1:], docstr[1:], occur[1:], labels[1:]):
        fb = torch.cat((fb, fb_), dim=0)
        doc = torch.cat((doc, doc_), dim=0)
        occ = torch.cat((occ, occ_), dim=0)
        gt = torch.cat((gt, gt_), dim=0)

    return fb, doc, occ, gt
