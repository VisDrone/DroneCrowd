import random
import os
from PIL import Image,ImageFilter,ImageDraw
import numpy as np
import h5py
from PIL import ImageStat
import cv2
import pdb
import scipy.io as io
from torchvision import transforms


def load_data(img_paths, train = False):
    #DroneCrowd
    imgset = []
    gts = []
    for i in range(len(img_paths)):
        img = Image.open(img_paths[i]).convert('RGB')
        imgset.append(img)
        mat_path = img_paths[i].replace('.jpg', '.mat').replace('images', 'ground_truth').replace('img', 'GT_img')
        mat = io.loadmat(mat_path)
        gt = mat["image_info"][0, 0][0, 0][0]
        gt = np.array(gt, dtype=np.float32)
        gt_ = np.zeros(shape=[512 - gt.shape[0], 2])
        all_gt = np.vstack((gt, gt_))
        if i == 0:
            gtnum = np.sum(gt.shape[0])
        else:
            gtnum = np.vstack((gtnum, np.sum(gt.shape[0])))
        gts.append(all_gt)

    # crop the images
    crop_factor = 0.5
    imgs1, imgs2, imgs3, imgs4 = [], [], [], []
    for i in range(len(imgset)):
        # crop the images
        crop_size = (int(imgset[0].size[1] * crop_factor), int(imgset[0].size[0] * crop_factor))
        imgs = transforms.FiveCrop(crop_size)(imgset[i])
        img1, img2, img3, img4 = imgs[0:4]
        imgs1.append(img1)
        imgs2.append(img2)
        imgs3.append(img3)
        imgs4.append(img4)

    return imgs1, imgs2, imgs3, imgs4, gtnum, gts
