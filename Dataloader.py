from torch.utils.data import Dataset
from torch import optim
import torch.nn as nn
import torch
import numpy as np
import pandas as pd

class CustomDataset(Dataset):
    '''
    Dataloader for custom dataset:
    DataPath: Path for the dataset
    DatasetClass: class of training, validation, or test.
    file: ends with 'T' -> positive data sample || ends with 'F' -> negative data sample
    '''
    def __init__(self, DataPath, DatasetClass):
        self.csv_path = []
        self.label = []
        for set in os.listdir(DataPath):
            set_path = os.path.join(DataPath, set)
            if os.path.isdir(set_path):
                if set == status:
                    for file in os.listdir(set_path):
                        if file.endswith('.txt'):
                            file_path = os.path.join(set_path, file)
                            self.csv_path.append(file_path)
                            if file.startswith('T'):
                                self.label.append(1)
                            elif file.startswith('F'):
                                self.label.append(0)

    def __getitem__(self, index):
        csv_path = self.csv_path[index]
        label = self.label[index]
        data = torch.tensor(np.loadtxt(csv_path))
        data = data.reshape((1, data.shape[0]))
        label = torch.tensor(label)
        return data, label

    def __len__(self):
        return len(self.label)