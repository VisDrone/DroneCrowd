import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from image_test import *
import torchvision.transforms.functional as F

class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None,  train=False, batch_size=1, num_workers=4):
        if shuffle:
            random.shuffle(root)
        
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __len__(self):
        return self.nSamples
    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_paths = self.lines[index]
        imgs1, imgs2, imgs3, imgs4, gtnum, gts = load_data(img_paths, self.train)
        if self.transform is not None:
            for i in range(len(imgs1)):
                imgs1[i] = self.transform(imgs1[i])
                imgs2[i] = self.transform(imgs2[i])
                imgs3[i] = self.transform(imgs3[i])
                imgs4[i] = self.transform(imgs4[i])

        return imgs1, imgs2, imgs3, imgs4, gtnum, gts
