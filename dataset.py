import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
import skimage
import numpy as np
from PIL import Image
import torch
class CustomDataset(Dataset):
    def __init__(self):
        self.data = pd.read_csv("./Data/english.csv")
        map = "01234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.labels = self.data.iloc[:, 1]
        for i in range(len(self.labels)):
            curr = [0] * 63
            index = map.index(self.labels[i])
            curr[index] = 1
            self.labels[i] = map.index(self.labels[i])
        self.files = self.data.iloc[:, 0]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img = np.array(Image.open("./Data/" + self.files[idx]))[:, :, 0]
        small_img = skimage.measure.block_reduce(img, (50, 50), np.min)
        trans = transforms.Compose([transforms.ToTensor()])
        return trans(small_img), torch.tensor(self.labels[idx])
