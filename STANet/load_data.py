import os
import random
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import scipy.io as io
from torchvision import transforms
import h5py

class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None, train=False, batch_size=1, num_workers=4):
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
        if self.train:
            img1,img2,dengt11,dengt12,dengt13,dengt21,dengt22,dengt23,locgt11,locgt12,locgt13,locgt21,locgt22,locgt23,trkgt1,trkgt2 = load_train_data(img_paths)

            if self.transform is not None:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
        
            return img1,img2,dengt11,dengt12,dengt13,dengt21,dengt22,dengt23,locgt11,locgt12,locgt13,locgt21,locgt22,locgt23,trkgt1,trkgt2

        else:
            img11,img12,img13,img14,img21,img22,img23,img24,loc11,loc12,loc13,loc14,loc21,loc22,loc23,loc24,gt_count1,gt1,gt_count2,gt2 = load_test_data(img_paths)

            if self.transform is not None:
                img11 = self.transform(img11)
                img12 = self.transform(img12)
                img13 = self.transform(img13)
                img14 = self.transform(img14)
                img21 = self.transform(img21)
                img22 = self.transform(img22)
                img23 = self.transform(img23)
                img24 = self.transform(img24)
            return img11,img12,img13,img14,img21,img22,img23,img24,loc11,loc12,loc13,loc14,loc21,loc22,loc23,loc24,gt_count1,gt1,gt_count2,gt2

def load_train_data(img_paths):
    # DroneCrowd
    pt_path1 = img_paths[0].replace('.jpg', '_max.h5').replace('images', 'ground_truth')
    pt_file1 = h5py.File(pt_path1)
    locgt1 = np.asarray(pt_file1['location'])
    trkgt1 = np.asarray(pt_file1['identity'])
    pt_path2 = img_paths[1].replace('.jpg', '_max.h5').replace('images', 'ground_truth')
    pt_file2 = h5py.File(pt_path2)
    locgt2 = np.asarray(pt_file2['location'])
    trkgt2 = np.asarray(pt_file2['identity'])

    gt_path1 = img_paths[0].replace('.jpg', '.h5').replace('images', 'ground_truth')
    img1 = Image.open(img_paths[0]).convert('RGB')
    gt_file1 = h5py.File(gt_path1)
    dengt1 = np.asarray(gt_file1['density'])
    gt_path2 = img_paths[1].replace('.jpg', '.h5').replace('images', 'ground_truth')
    img2 = Image.open(img_paths[1]).convert('RGB')
    gt_file2 = h5py.File(gt_path2)
    dengt2 = np.asarray(gt_file2['density'])

    # crop the images
    crop_factor = 0.5
    crop_size = (int(img1.size[0] * crop_factor), int(img1.size[1] * crop_factor))
    dx = int(random.randint(0, 1) * img2.size[0] * 1. / 2)
    dy = int(random.randint(0, 1) * img2.size[1] * 1. / 2)
    img1 = img1.crop((dx, dy, crop_size[0] + dx, crop_size[1] + dy))
    img2 = img2.crop((dx, dy, crop_size[0] + dx, crop_size[1] + dy))
    dengt1 = dengt1[dy:crop_size[1] + dy, dx:crop_size[0] + dx]
    dengt2 = dengt2[dy:crop_size[1] + dy, dx:crop_size[0] + dx]

    locgt1 = locgt1[int(dy / 2):int(crop_size[1] / 2 + dy / 2), int(dx / 2):int(crop_size[0] / 2 + dx / 2)]
    locgt2 = locgt2[int(dy / 2):int(crop_size[1] / 2 + dy / 2), int(dx / 2):int(crop_size[0] / 2 + dx / 2)]
    trkgt1 = trkgt1[int(dy / 2):int(crop_size[1] / 2 + dy / 2), int(dx / 2):int(crop_size[0] / 2 + dx / 2)]
    trkgt2 = trkgt2[int(dy / 2):int(crop_size[1] / 2 + dy / 2), int(dx / 2):int(crop_size[0] / 2 + dx / 2)]

    if np.random.random() < 0.5:
        img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
        img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
        dengt1 = np.fliplr(dengt1)
        dengt2 = np.fliplr(dengt2)
        locgt1 = np.fliplr(locgt1)
        locgt2 = np.fliplr(locgt2)
        trkgt1 = np.fliplr(trkgt1)
        trkgt2 = np.fliplr(trkgt2)

    # multi-scale density gt
    dengt11 = cv2.resize(dengt1, (dengt1.shape[1] / 2, dengt1.shape[0] / 2), interpolation=cv2.INTER_CUBIC) * 4.
    dengt12 = cv2.resize(dengt1, (dengt1.shape[1] / 4, dengt1.shape[0] / 4), interpolation=cv2.INTER_CUBIC) * 16.
    dengt13 = cv2.resize(dengt1, (dengt1.shape[1] / 8, dengt1.shape[0] / 8), interpolation=cv2.INTER_CUBIC) * 64.
    dengt21 = cv2.resize(dengt2, (dengt2.shape[1] / 2, dengt2.shape[0] / 2), interpolation=cv2.INTER_CUBIC) * 4.
    dengt22 = cv2.resize(dengt2, (dengt2.shape[1] / 4, dengt2.shape[0] / 4), interpolation=cv2.INTER_CUBIC) * 16.
    dengt23 = cv2.resize(dengt2, (dengt2.shape[1] / 8, dengt2.shape[0] / 8), interpolation=cv2.INTER_CUBIC) * 64.
    # multi-scale localization gt
    locgt11 = locgt1.copy()
    locgt12 = cv2.resize(locgt1, (locgt1.shape[1] / 2, locgt1.shape[0] / 2), interpolation=cv2.INTER_CUBIC) * 4.
    locgt13 = cv2.resize(locgt1, (locgt1.shape[1] / 4, locgt1.shape[0] / 4), interpolation=cv2.INTER_CUBIC) * 16.
    locgt21 = locgt2.copy()
    locgt22 = cv2.resize(locgt2, (locgt2.shape[1] / 2, locgt2.shape[0] / 2), interpolation=cv2.INTER_CUBIC) * 4.
    locgt23 = cv2.resize(locgt2, (locgt2.shape[1] / 4, locgt2.shape[0] / 4), interpolation=cv2.INTER_CUBIC) * 16.
    # target id gt
    trkgt1 = trkgt1.copy()
    trkgt2 = trkgt2.copy()

    return img1, img2, dengt11, dengt12, dengt13, dengt21, dengt22, dengt23, locgt11, locgt12, locgt13, locgt21, locgt22, locgt23, trkgt1, trkgt2


def load_test_data(img_paths):
    img1 = Image.open(img_paths[0]).convert('RGB')
    img2 = Image.open(img_paths[1]).convert('RGB')
    #DroneCrowd
    pt_path1 = img_paths[0].replace('.jpg', '_max.h5').replace('images', 'ground_truth')

    if os.path.exists(pt_path1):
    	pt_file1 = h5py.File(pt_path1)
    	location1 = np.asarray(pt_file1['location'])
    	pt_path2 = img_paths[1].replace('.jpg', '_max.h5').replace('images', 'ground_truth')
    	pt_file2 = h5py.File(pt_path2)
    	location2 = np.asarray(pt_file2['location'])
    else:
    	location1,location2=[],[]


    mat_path = img_paths[0].replace('.jpg', '.mat').replace('images', 'ground_truth').replace('img', 'GT_img')
    mat = io.loadmat(mat_path)
    gt1 = mat["image_info"][0, 0][0, 0][0]
    gt_count1 = np.sum(gt1.shape[0])
    gt1 = np.array(gt1, dtype=np.float32)
    mat_path = img_paths[1].replace('.jpg', '.mat').replace('images', 'ground_truth').replace('img', 'GT_img')
    mat = io.loadmat(mat_path)
    gt2 = mat["image_info"][0, 0][0, 0][0]
    gt_count2 = np.sum(gt2.shape[0])
    gt2 = np.array(gt2, dtype=np.float32)

    # crop the images
    crop_factor = 0.5
    crop_size = (int(img1.size[1] * crop_factor), int(img1.size[0] * crop_factor))
    imgs1 = transforms.FiveCrop(crop_size)(img1)
    img11, img12, img13, img14 = imgs1[0:4]
    crop_size = (int(img2.size[1] * crop_factor), int(img2.size[0] * crop_factor))
    imgs2 = transforms.FiveCrop(crop_size)(img2)
    img21, img22, img23, img24 = imgs2[0:4]

    # crop the localization gt map
    if len(location1):
    	crop_size = (int(img1.size[0] * crop_factor / 2), int(img1.size[1] * crop_factor / 2))
    	dy, dx = 0, 0
    	loc11 = location1[dy:crop_size[1] + dy, dx:crop_size[0] + dx]
    	loc21 = location2[dy:crop_size[1] + dy, dx:crop_size[0] + dx]
    	dy, dx = 0, crop_size[0]
    	loc12 = location1[dy:crop_size[1] + dy, dx:crop_size[0] + dx]
    	loc22 = location2[dy:crop_size[1] + dy, dx:crop_size[0] + dx]
    	dy, dx = crop_size[1], 0
    	loc13 = location1[dy:crop_size[1] + dy, dx:crop_size[0] + dx]
    	loc23 = location2[dy:crop_size[1] + dy, dx:crop_size[0] + dx]
    	dy, dx = crop_size[1], crop_size[0]
    	loc14 = location1[dy:crop_size[1] + dy, dx:crop_size[0] + dx]
    	loc24 = location2[dy:crop_size[1] + dy, dx:crop_size[0] + dx]
    else:
    	loc11, loc12, loc13, loc14, loc21, loc22, loc23, loc24 = [], [], [], [], [], [], [], []

    return img11, img12, img13, img14, img21, img22, img23, img24, loc11, loc12, loc13, loc14, loc21, loc22, loc23, loc24, gt_count1, gt1, gt_count2, gt2
