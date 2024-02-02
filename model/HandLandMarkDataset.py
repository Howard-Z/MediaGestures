import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

class HandLandmarkDataset(Dataset):
    def crawler(self, path):
        self.file_paths = []
        self.labels = []
        for root,d_names,f_names in os.walk(path):
            for f in f_names:
                #STUPID MAC SHIT
                if ".DS_Store" not in f:
                    #The big jank here:
                    if "pinch" in root:
                        self.labels.append(1)
                    self.file_paths.append(os.path.join(root, f))
        return len(self.file_paths)

    def __init__(self, root, transform = None):
        self.root = root
        self.transform = transform
        self.len = self.crawler(root)

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = np.load(self.file_paths[idx], allow_pickle=True)
        label = self.labels[idx]

        return sample, label
        



tester = HandLandmarkDataset("/Users/howardzhu/Documents/git_repos/MediaGestures/parsed_data")
sample, label = tester.__getitem__(5)
print(sample)
print(label)