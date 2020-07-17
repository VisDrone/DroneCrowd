import sys
import os
import warnings

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import math
import argparse
import cv2
import time
import scipy
import glob

import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters

import load_data
from stanet import STANet
from utils import load_test_data, load_pretrained_model, load_txt

parser = argparse.ArgumentParser(description='Test PyTorch STANet')

parser.add_argument('--loc', dest='use_loc', action='store_true',
                      help='whether using the location head or not')

parser.add_argument('--trk', dest='use_trk', action='store_true',
                      help='whether using the location head or not')

def main():
    
    global args, best_prec1
        
    args = parser.parse_args()
    args.workers = 16
    args.seed = long(time.time())

    # prepare the test dataset
    root = '../dataset/'
    test_step = 5
    test_pair = load_test_data(root, test_step)

    if args.use_loc:
        model_name = 'locmodel_best.pth.tar'
        use_loc = True
        use_trk = False
        if args.use_trk:
            model_name = 'trkmodel_best.pth.tar'
            use_loc = True
            use_trk = True
    else:
        model_name = 'denmodel_best.pth.tar'
        use_loc = False
        use_trk = False

    model = STANet(use_loc, use_trk).cuda()
    model = load_pretrained_model(model_name, model)

    criterion = nn.MSELoss(size_average=False).cuda()
    with torch.no_grad():
       validate(test_pair, model, criterion)


def validate(test_pair, model, criterion):
    # attributes index of sequences in DroneCrowd dataset
    high = [11, 15, 16, 22, 23, 24, 34, 35, 42, 43, 44, 61, 62, 63, 65, 69, 70, 74]
    low = [17, 18, 75, 82, 88, 95, 101, 103, 105, 108, 111, 112]
    cloudy = [11, 22, 23, 24, 43, 61, 62, 63, 65, 75, 82, 88, 95, 101, 108]
    sunny = [15, 16, 34, 35, 42, 44, 103, 105, 111, 112]
    night = [17, 18, 69, 70, 74]
    crowd = [11, 15, 22, 23, 24, 88, 101, 105, 108, 111, 112]
    sparse = [16, 17, 18, 34, 35, 42, 43, 44, 61, 62, 63, 65, 69, 70, 74, 75, 82, 95, 103]

    test_loader = torch.utils.data.DataLoader(
        load_data.listDataset(test_pair,
                              shuffle=False,
                              transform=transforms.Compose([
                                  transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                              std=[0.229, 0.224, 0.225]),
                              ]), train=False), batch_size=1)
    model.eval()
    mae = 0
    mse = 0
    mae_high = 0
    mse_high = 0
    mae_low = 0
    mse_low = 0
    mae_cloudy = 0
    mse_cloudy = 0
    mae_sunny = 0
    mse_sunny = 0
    mae_night = 0
    mse_night = 0
    mae_crowd = 0
    mse_crowd = 0
    mae_sparse = 0
    mse_sparse = 0

    count_high = 0
    count_low = 0
    count_cloudy = 0
    count_sunny = 0
    count_night = 0
    count_crowd = 0
    count_sparse = 0

    mae1, mae2, mae3, mae4 = 0., 0., 0., 0.
    mse1, mse2, mse3, mse4 = 0., 0., 0., 0.

    allposnum, allnegnum, allgtnum = 0, 0, 0
    for i, (
    img11, img12, img13, img14, img21, img22, img23, img24, loc11, loc12, loc13, loc14, loc21, loc22, loc23, loc24,
    gt_count1, gt1, gt_count2, gt2) in enumerate(test_loader):
        seqID = int(test_pair[i][1][-10:-7])
        img11 = Variable(img11.cuda())
        img21 = Variable(img21.cuda())
        den_g11, den_g12, den_g13, den_g14, loc_g11, loc_g12, loc_g13, loc_g14, fea_g14 = model(img11, img21)
        img12 = Variable(img12.cuda())
        img22 = Variable(img22.cuda())
        den_g21, den_g22, den_g23, den_g24, loc_g21, loc_g22, loc_g23, loc_g24, fea_g24 = model(img12, img22)
        img13 = Variable(img13.cuda())
        img23 = Variable(img23.cuda())
        den_g31, den_g32, den_g33, den_g34, loc_g31, loc_g32, loc_g33, loc_g34, fea_g34 = model(img13, img23)
        img14 = Variable(img14.cuda())
        img24 = Variable(img24.cuda())
        den_g41, den_g42, den_g43, den_g44, loc_g41, loc_g42, loc_g43, loc_g44, fea_g44 = model(img14, img24)

        # calculate density result
        denout21 = max(0,den_g11[:,1,:,:].data.sum()) + max(0,den_g21[:,1,:,:].data.sum()) + \
                   max(0,den_g31[:,1,:,:].data.sum()) + max(0,den_g41[:,1,:,:].data.sum())
        denout22 = max(0,den_g12[:,1,:,:].data.sum()) + max(0,den_g22[:,1,:,:].data.sum()) + \
                   max(0,den_g32[:,1,:,:].data.sum()) + max(0,den_g42[:,1,:,:].data.sum())
        denout23 = max(0,den_g13[:,1,:,:].data.sum()) + max(0,den_g23[:,1,:,:].data.sum()) + \
                   max(0,den_g33[:,1,:,:].data.sum()) + max(0,den_g43[:,1,:,:].data.sum())
        denout24 = max(0,den_g14[:,1,:,:].data.sum()) + max(0,den_g24[:,1,:,:].data.sum()) + \
                   max(0,den_g34[:,1,:,:].data.sum()) + max(0,den_g44[:,1,:,:].data.sum())

        cmae1 = abs(denout21 - float(gt_count2))
        cmae2 = abs(denout22 - float(gt_count2))
        cmae3 = abs(denout23 - float(gt_count2))
        cmae4 = abs(denout24 - float(gt_count2))
        val = cmae1
        if seqID in high:
            mae_high += val
            mse_high += val ** 2
            count_high += 1
        else:
            mae_low += val
            mse_low += val ** 2
            count_low += 1
        if seqID in crowd:
            mae_crowd += val
            mse_crowd += val ** 2
            count_crowd += 1
        else:
            mae_sparse += val
            mse_sparse += val ** 2
            count_sparse += 1

        if seqID in cloudy:
            mae_cloudy += val
            mse_cloudy += val ** 2
            count_cloudy += 1
        elif seqID in sunny:
            mae_sunny += val
            mse_sunny += val ** 2
            count_sunny += 1
        else:
            mae_night += val
            mse_night += val ** 2
            count_night += 1
        mae += val
        mse += val ** 2

        mae1 += cmae1
        mae2 += cmae2
        mae3 += cmae3
        mae4 += cmae4
        mse1 += cmae1 ** 2
        mse2 += cmae2 ** 2
        mse3 += cmae3 ** 2
        mse4 += cmae4 ** 2

        # calculate localization result
        bina_thre = 0.005
        data12 = np.hstack((den_g11[:,1,:,:].data.cpu().numpy().squeeze(),den_g21[:,1,:,:].data.cpu().numpy().squeeze()))
        data34 = np.hstack((den_g31[:,1,:,:].data.cpu().numpy().squeeze(),den_g41[:,1,:,:].data.cpu().numpy().squeeze()))
        out_data = np.vstack((data12, data34))

        # output final results
        gtnum, posnum, negnum, pts = calc_pt_match(gt2, out_data, bina_thre, 0.001, 8, 2.0)
        np.savetxt('results/' + test_pair[i][1][-13:-4] + '_pt.txt', pts)
        allgtnum += gtnum
        allposnum += posnum
        allnegnum += negnum
        #if i % 300 == 0:
        #    print(i, allgtnum, allposnum, allnegnum)

        if (i % 300 == 0 or i == len(test_loader) - 1):
            print('img{i:d}*MAE1 {mae1:.1f} MAE2 {mae2:.1f} MAE3 {mae3:.1f} MAE4 {mae4:.1f}'.format(i=i + 1,
                                                                                                    mae1=mae1 / (i + 1),
                                                                                                    mae2=mae2 / (i + 1),
                                                                                                    mae3=mae3 / (i + 1),
                                                                                                    mae4=mae4 / (
                                                                                                    i + 1)))

    mae1 = math.sqrt((mae1 / len(test_loader)) ** 2)
    mae2 = math.sqrt((mae2 / len(test_loader)) ** 2)
    mae3 = math.sqrt((mae3 / len(test_loader)) ** 2)
    mae4 = math.sqrt((mae4 / len(test_loader)) ** 2)
    mse1 = math.sqrt(mse1 / len(test_loader))
    mse2 = math.sqrt(mse2 / len(test_loader))
    mse3 = math.sqrt(mse3 / len(test_loader))
    mse4 = math.sqrt(mse4 / len(test_loader))

    print(
    '*MAE1 {mae1:.1f} MAE2 {mae2:.1f} MAE3 {mae3:.1f} MAE4 {mae4:.1f} MSE1 {mse1:.1f} MSE2 {mse2:.1f} MSE3 {mse3:.1f} MSE4 {mse4:.1f}'.format(
        mae1=mae1, mae2=mae2, mae3=mae3, mae4=mae4, mse1=mse1, mse2=mse2, mse3=mse3, mse4=mse4))

    mae_overall = mae.data.cpu().numpy() / len(test_loader)
    mse_overall = math.sqrt(mse / len(test_loader))

    mae_high = mae_high.data.cpu().numpy() / count_high
    mse_high = math.sqrt(mse_high / count_high)
    mae_low = mae_low.data.cpu().numpy() / max(1.0, count_low)
    mse_low = math.sqrt(mse_low / max(1.0, count_low))

    mae_cloudy = mae_cloudy.data.cpu().numpy() / count_cloudy
    mse_cloudy = math.sqrt(mse_cloudy / count_cloudy)
    mae_sunny = mae_sunny.data.cpu().numpy() / max(1.0, count_sunny)
    mse_sunny = math.sqrt(mse_sunny / max(1.0, count_sunny))
    mae_night = mae_night.data.cpu().numpy() / max(1.0, count_night)
    mse_night = math.sqrt(mse_night / max(1.0, count_night))

    mae_crowd = mae_crowd.data.cpu().numpy() / max(1.0, count_crowd)
    mse_crowd = math.sqrt(mse_crowd / max(1.0, count_crowd))
    mae_sparse = mae_sparse.data.cpu().numpy() / count_sparse
    mse_sparse = math.sqrt(mse_sparse / count_sparse)

    print('*overall-MAE {mae_overall:.1f} MSE {mse_overall:.1f}\n'
          '*high-MAE {mae_high:.1f} MSE {mse_high:.1f}\n'
          '*low-MAE {mae_low:.1f} MSE {mse_low:.1f}\n'
          '*cloudy-MAE {mae_cloudy:.1f} MSE {mse_cloudy:.1f}\n'
          '*sunny-MAE {mae_sunny:.1f} MSE {mse_sunny:.1f}\n'
          '*night-MAE {mae_night:.1f} MSE {mse_night:.1f}\n'
          '*crowd-MAE {mae_crowd:.1f} MSE {mse_crowd:.1f}\n'
          '*sparse-MAE {mae_sparse:.1f} MSE {mse_sparse:.1f}\n'
          .format(mae_overall=mae_overall, mse_overall=mse_overall, mae_high=mae_high, mse_high=mse_high,
                  mae_low=mae_low, mse_low=mse_low,
                  mae_cloudy=mae_cloudy, mse_cloudy=mse_cloudy, mae_sunny=mae_sunny, mse_sunny=mse_sunny,
                  mae_night=mae_night, mse_night=mse_night,
                  mae_crowd=mae_crowd, mse_crowd=mse_crowd, mae_sparse=mae_sparse, mse_sparse=mse_sparse))

def calc_pt_match(gt, ourmap, bina_thre, diff_thre, neighbor_thre, ratio):
    binamap = ourmap > bina_thre
    ourmap[binamap == 0] = 0
    data_max = filters.maximum_filter(ourmap, neighbor_thre)
    data_min = filters.minimum_filter(ourmap, neighbor_thre)
    maxima = ourmap == data_max
    diffmap = ((data_max - data_min) > diff_thre)
    maxima[diffmap == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)

    x, y, sc = [], [], []
    for dy, dx in slices:
        x_center = (dx.start + dx.stop - 1) / 2 * ratio
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1) / 2 * ratio
        y.append(y_center)
        sc.append(ourmap[int(y_center/ratio), int(x_center/ratio)])

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    sc = np.asarray(sc, dtype=np.float32)

    gx, gy = [], []
    for k in range(len(gt[0])):
        gx.append(gt[0][k][0])
        gy.append(gt[0][k][1])

    sorted_ind = np.argsort(-sc)
    outx = x[sorted_ind]
    outy = y[sorted_ind]

    cnt = 0
    posnum, negnum, gtnum = 0, 0, 0
    for ii in range(len(x)):
        curx = outx[ii]
        cury = outy[ii]
        min_dist = np.inf
        for jj in range(len(gx)):
            curgx = gx[jj]
            curgy = gy[jj]
            dist = np.sqrt((curx - curgx) ** 2 + (cury - curgy) ** 2)
            if dist < min_dist:
                min_dist = dist
        if min_dist <= 10:
            cnt += 1
    posnum += cnt
    negnum += len(x) - cnt
    gtnum += len(gx)
    pts = np.hstack((np.expand_dims(x.T,axis=1), np.expand_dims(y.T,axis=1), np.expand_dims(sc.T,axis=1)))

    return gtnum, posnum, negnum, pts


if __name__ == '__main__':
    main()
