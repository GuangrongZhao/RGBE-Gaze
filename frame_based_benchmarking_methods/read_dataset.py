import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import h5py
from torch.utils.data import Dataset, DataLoader


class gaze_Dataset(Dataset):
    def __init__(self, image_path_list, label_path_list):
        super().__init__()

        self.keys = []
        self.values = []
        self.total_data_num = 0
        for i in range(len(image_path_list)):
            with h5py.File(image_path_list[i], 'r') as image_file:
                self.keys.append(self.total_data_num)
                self.values.append([image_path_list[i], label_path_list[i]])
                self.total_data_num += image_file['data'].shape[-1]

    def __getitem__(self, index):
        def load_data(file_index, adjusted_index):
            with h5py.File(self.values[file_index][0], 'r') as image_file, \
                    h5py.File(self.values[file_index][1], 'r') as label_file:
                image = image_file['data'][:, :, adjusted_index].T
                label = label_file['data'][:, adjusted_index].T
                image = torch.from_numpy(image).type(torch.FloatTensor) / 255
                image = image.unsqueeze(0)
                label = torch.from_numpy(label).type(torch.FloatTensor)
                return image, label

        for i in range(len(self.keys) - 1):
            if self.keys[i] <= index < self.keys[i + 1]:
                return load_data(i, index - self.keys[i])

        return load_data(-1, index - self.keys[-1])

    def __len__(self):
        return self.total_data_num

