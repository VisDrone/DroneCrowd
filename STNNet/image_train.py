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


def load_data(img_paths, train = True):
    #parse annotation of DroneCrowd
    imgset = []
    dengt, trkgt = [], []
    reggts1, reggts2, reggts3 = [], [], []
    for i in range(len(img_paths)):
        img = Image.open(img_paths[i]).convert('RGB')
        imgset.append(img)

        gt_path = img_paths[i].replace('.jpg', '.h5').replace('images', 'ground_truth')
        gt_file = h5py.File(gt_path, "r")
        den = np.asarray(gt_file['density'])
        gt_path = img_paths[i].replace('.jpg','_max.h5').replace('images','ground_truth')
        gt_file = h5py.File(gt_path, "r")
        trk = np.asarray(gt_file['identity'])
        reg1 = np.asarray(gt_file['offset1'])
        reg2 = np.asarray(gt_file['offset2'])
        reg3 = np.asarray(gt_file['offset3'])
        dengt.append(den)
        trkgt.append(trk)
        reggts1.append(reg1)
        reggts2.append(reg2)
        reggts3.append(reg3)
        if not train:
            mat_path = img_paths[i].replace('.jpg', '.mat').replace('images', 'ground_truth').replace('img', 'GT_img')
            mat = io.loadmat(mat_path)
            gt = mat["image_info"][0, 0][0, 0][0]
            if i == 0:
                gtnum = np.sum(gt.shape[0])
            else:
                gtnum = np.vstack((gtnum, np.sum(gt.shape[0])))

    # crop the images
    if train:  
        crop_factor = 0.5
        crop_size = (int(imgset[0].size[0]*crop_factor),int(imgset[0].size[1]*crop_factor))
        dx = int(random.randint(0,1)*imgset[0].size[0]*1./2)
        dy = int(random.randint(0,1)*imgset[0].size[1]*1./2)
        flag = False
        for i in range(len(img_paths)):
            imgset[i] = imgset[i].crop((dx,dy,crop_size[0]+dx,crop_size[1]+dy))
            dengt[i] = dengt[i][int(dy/2):int(crop_size[1]/2+dy/2),int(dx/2):int(crop_size[0]/2+dx/2)]
            trkgt[i] = trkgt[i][int(dy/2):int(crop_size[1]/2+dy/2),int(dx/2):int(crop_size[0]/2+dx/2)]
            reggts1[i] = reggts1[i][int(dy/2):int(crop_size[1]/2+dy/2),int(dx/2):int(crop_size[0]/2+dx/2),:]
            reggts2[i] = reggts2[i][int(dy/4):int(crop_size[1]/4+dy/4),int(dx/4):int(crop_size[0]/4+dx/4),:]
            reggts3[i] = reggts3[i][int(dy/8):int(crop_size[1]/8)+int(dy/8),int(dx/8):int(crop_size[0]/8)+int(dx/8),:]

            if np.random.random() < 0.5 and i == 0:
                flag = True
            if flag:
                imgset[i] = imgset[i].transpose(Image.FLIP_LEFT_RIGHT)
                dengt[i] = np.fliplr(dengt[i])
                trkgt[i] = np.fliplr(trkgt[i])
                reggts1[i] = np.fliplr(reggts1[i])
                reggts2[i] = np.fliplr(reggts2[i])
                reggts3[i] = np.fliplr(reggts3[i])

        for i in range(len(imgset)):
            # multi-scale density gt
            den1 = dengt[i].copy()
            den2 = cv2.resize(dengt[i],(int(dengt[i].shape[1]/2),int(dengt[i].shape[0]/2)),interpolation = cv2.INTER_CUBIC)*4
            den3 = cv2.resize(dengt[i],(int(dengt[i].shape[1]/4),int(dengt[i].shape[0]/4)),interpolation = cv2.INTER_CUBIC)*16
            # multi-scale regression gt
            reg1 = reggts1[i].copy()
            reg2 = reggts2[i].copy()
            reg3 = reggts3[i].copy()
            if i == 0:
                dengt1 = np.expand_dims(den1,0)
                dengt2 = np.expand_dims(den2,0)
                dengt3 = np.expand_dims(den3,0)
                reggt1 = np.expand_dims(reg1,0)
                reggt2 = np.expand_dims(reg2,0)
                reggt3 = np.expand_dims(reg3,0)
                trkgt4 = np.expand_dims(trkgt[i],0)
            else:
                dengt1 = np.vstack((dengt1,np.expand_dims(den1,0)))
                dengt2 = np.vstack((dengt2,np.expand_dims(den2,0)))
                dengt3 = np.vstack((dengt3,np.expand_dims(den3,0)))
                reggt1 = np.vstack((reggt1,np.expand_dims(reg1,0)))
                reggt2 = np.vstack((reggt2,np.expand_dims(reg2,0)))
                reggt3 = np.vstack((reggt3,np.expand_dims(reg3,0)))
                trkgt4 = np.vstack((trkgt4,np.expand_dims(trkgt[i],0)))

        return imgset, dengt1, dengt2, dengt3, reggt1, reggt2, reggt3, trkgt4

    else:
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

            # crop the localization map
            crop_size = (int(imgset[0].size[0] * crop_factor / 2), int(imgset[0].size[1] * crop_factor / 2))

            dy, dx = 0, 0
            reg1 = reggts1[i][dy:crop_size[1] + dy, dx:crop_size[0] + dx, :]
            dy, dx = 0, crop_size[0]
            reg2 = reggts1[i][dy:crop_size[1] + dy, dx:crop_size[0] + dx, :]
            dy, dx = crop_size[1], 0
            reg3 = reggts1[i][dy:crop_size[1] + dy, dx:crop_size[0] + dx, :]
            dy, dx = crop_size[1], crop_size[0]
            reg4 = reggts1[i][dy:crop_size[1] + dy, dx:crop_size[0] + dx, :]

            # crop the id map
            dy, dx = 0, 0
            trk1 = trkgt[i][dy:crop_size[1] + dy, dx:crop_size[0] + dx]
            dy, dx = 0, crop_size[0]
            trk2 = trkgt[i][dy:crop_size[1] + dy, dx:crop_size[0] + dx]
            dy, dx = crop_size[1], 0
            trk3 = trkgt[i][dy:crop_size[1] + dy, dx:crop_size[0] + dx]
            dy, dx = crop_size[1], crop_size[0]
            trk4 = trkgt[i][dy:crop_size[1] + dy, dx:crop_size[0] + dx]

            if i == 0:
                regs1 = np.expand_dims(reg1,0)
                regs2 = np.expand_dims(reg2,0)
                regs3 = np.expand_dims(reg3,0)
                regs4 = np.expand_dims(reg4,0)
                trks1 = np.expand_dims(trk1,0)
                trks2 = np.expand_dims(trk2,0)
                trks3 = np.expand_dims(trk3,0)
                trks4 = np.expand_dims(trk4,0)
            else:
                regs1 = np.vstack((regs1,np.expand_dims(reg1,0)))
                regs2 = np.vstack((regs2,np.expand_dims(reg2,0)))
                regs3 = np.vstack((regs3,np.expand_dims(reg3,0)))
                regs4 = np.vstack((regs4,np.expand_dims(reg4,0)))
                trks1 = np.vstack((trks1,np.expand_dims(trk1,0)))
                trks2 = np.vstack((trks2,np.expand_dims(trk2,0)))
                trks3 = np.vstack((trks3,np.expand_dims(trk3,0)))
                trks4 = np.vstack((trks4,np.expand_dims(trk4,0)))

        return imgs1, imgs2, imgs3, imgs4, gtnum, regs1, regs2, regs3, regs4, trks1, trks2, trks3, trks4
