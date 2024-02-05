import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import sys

base_path = os.curdir
sys.path.insert(0, base_path)
from utils.get_root_dir import *



class HandLandmarkDataset(Dataset):
    def crawler(self, path):
        for root, d_names, f_names in os.walk(path):
            for f in f_names:
                # STUPID MAC SHIT
                if ".DS_Store" not in f:
                    # The big jank here:
                    if "pinch" in root:
                        self.labels.append(1)
                    self.file_paths.append(os.path.join(root, f))
        return len(self.file_paths)

    def __init__(self, root, transform = None):
        self.root = root
        self.transform = transform
        self.labels = []
        self.file_paths = []
        self.len = self.crawler(root)
        
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = np.load(self.file_paths[idx], allow_pickle=True)
        label = self.labels[idx]

        return sample, label
        

# tester = HandLandmarkDataset(get_root_dir() + "/parsed_data")
# temp = os.path.join(get_root_dir(), "parsed_data")
tester = HandLandmarkDataset(os.path.join(get_root_dir(), "parsed_data"))
sample, label = tester[0]
print(sample)
print(label)