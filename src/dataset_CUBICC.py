import os
import numpy as np
import torch
class CUBICCDataset(torch.utils.data.Dataset):

    def __init__(self, datadir):
        self.images = torch.load(os.path.join(datadir,'images.pt'), weights_only=True)
        self.captions = torch.load(os.path.join(datadir,'captions.pt'), weights_only=True) 
        self.labels = torch.load(os.path.join(datadir,'labels.pt'), weights_only=True)
        self.labels_traintest = torch.load(os.path.join(datadir,'train_test_labelling.pt'), weights_only=True) # NOT USED NOR EXPOSED
        self.labels_original = torch.load(os.path.join(datadir,'original_labels.pt'), weights_only=True) # NOT USED NOR EXPOSED
        self.train_split = np.load(os.path.join(datadir, 'train_split.npy'))
        self.validation_split = np.load(os.path.join(datadir, 'validation_split.npy'))
        self.test_split = np.load(os.path.join(datadir, 'test_split.npy'))

    def __getitem__(self, idx):
        return (self.images[idx], self.captions[idx]), self.labels[idx]

    def __len__(self):
        return len(self.labels)