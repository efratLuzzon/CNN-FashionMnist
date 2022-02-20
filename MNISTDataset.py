import numpy as np
import torch
from torch.utils.data import Dataset


class MNISTDataset(Dataset):
    def __init__(self, path_images, path_labels, transform=None):
        self.transform = transform
        self.train_x = np.loadtxt(path_images)
        self.train_x /= 255
        self.train_y = np.loadtxt(path_labels)

    def __len__(self):
        return len(self.train_x)

    def __getitem__(self, index):
        image = torch.FloatTensor(self.train_x[index])
        if self.transform:
            image = self.transform(image)
        return image, self.train_y[index]