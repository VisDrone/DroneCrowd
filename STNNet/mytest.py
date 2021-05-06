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
import pdb

from stnmodel import STNNet
from loss_func import calc_loc_loss, calc_trk_loss
from utils import save_checkpoint
import dataset_test

import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters

parser = argparse.ArgumentParser(description='PyTorch STNNet for Crowd Counting, Localization and Tracking')
parser.add_argument('--loc', dest='use_loc', action='store_true',
                      help='whether using the localization subnet or not')
parser.add_argument('--trk', dest='use_trk', action='store_true',
                      help='whether using the association subnet or not')
parser.add_argument('--tem', dest='use_tem', action='store_true',
                    help='whether only using the temporal prediction constraint in the loss')
parser.add_argument('--cyc', dest='use_cyc', action='store_true',
                    help='whether using the cycle loss')
parser.add_argument('--bs', dest='batch_size', default=1, type=int,
                      help='batch_size')

def main():
    
    global args        
    args = parser.parse_args()
    args.seed = time.time()
    args.workers = 10
    args.factor = 100.
    args.frame_length = 2

    # prepare the test dataset
    root = '../dataset/'
    test_path = os.path.join(root,'test_data','images')
    test_list = []
    for img_path in glob.glob(os.path.join(test_path, '*.jpg')):
        test_list.append(img_path)
    test_list.sort()
    test_clip = []
    test_step = 0
    for idx in range(len(test_list)):
        if (test_step) % 1 == 0:
            clip, imgs, seqids = [], [], []
            for i in range(args.frame_length):
                cur_img = test_list[max(0, idx - i)]
                seqid = cur_img[-10:-7]
                imgs.append(cur_img)
                seqids.append(seqid)

            for i in range(args.frame_length - 1):
                cur_img = imgs[i]
                pre_img = imgs[i + 1]
                seq_id1 = seqids[i]
                seq_id2 = seqids[i + 1]
                if i == 0:
                    if seq_id1 == seq_id2:
                        clip.append(cur_img)
                        clip.append(pre_img)
                    else:
                        clip.append(cur_img)
                        clip.append(cur_img)
                else:
                    if seq_id1 == seq_id2:
                        clip.append(pre_img)
                    else:
                        clip.append(cur_img)
            test_clip.append(clip[::-1])
        test_step += 1

    if args.use_loc:
        model_name = 'models/loc_model_best.pth.tar'
        use_loc = True
        use_trk = False
        save_path_prefix = 'loc_pts'
        if args.use_trk:
            model_name = 'models/trk_model_best.pth.tar'
            save_path_prefix = 'trk_pts'
            if args.use_tem:
                model_name = 'models/tem_model_best.pth.tar' 
                save_path_prefix = 'tem_pts'
            elif args.use_cyc:
                model_name = 'models/cyc_model_best.pth.tar'
                save_path_prefix = 'cyc_pts'
            use_loc = True
            use_trk = True
    else:
        model_name = 'models/den_model_best.pth.tar'
        use_loc = False
        use_trk = False
        save_path_prefix = 'den_pts'
    if not os.path.isdir('den_pts'): os.mkdir('den_pts')
    if not os.path.isdir('loc_pts'): os.mkdir('loc_pts')
    if not os.path.isdir(save_path_prefix): os.mkdir(save_path_prefix)

    model = STNNet(use_loc, use_trk).cuda()

    print("=> loading checkpoint...")
    checkpoint = torch.load(model_name)
    my_models = model.state_dict()
    pre_models = list(checkpoint['state_dict'].items())
    count = 0
    for layer_name, value in my_models.items():
        prelayer_name, pre_weights = pre_models[count]
        my_models[layer_name] = pre_weights
        count += 1  
    model.load_state_dict(my_models)
    test(test_clip, model, save_path_prefix)

#########################################Test##########################################################################
@torch.no_grad()
def test(val_pair, model, save_path_prefix):
    # video-level attributes
    small = [11, 15, 16, 22, 23, 24, 34, 35, 42, 43, 44, 61, 62, 63, 65, 69, 70, 74]
    large = [17, 18, 75, 82, 88, 95, 101, 103, 105, 108, 111, 112]
    cloudy = [11, 22, 23, 24, 43, 61, 62, 63, 65, 75, 82, 88, 95, 101, 108]
    sunny = [15, 16, 34, 35, 42, 44, 103, 105, 111, 112]  
    night = [17, 18, 69, 70, 74]
    crowd = [11, 15, 22, 23, 24, 88, 101, 105, 108, 111, 112]
    sparse = [16, 17, 18, 34, 35, 42, 43, 44, 61, 62, 63, 65, 69, 70, 74, 75, 82, 95, 103]
		
    test_loader = torch.utils.data.DataLoader(
    dataset_test.listDataset(val_pair,
                   shuffle=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]),  train=False), batch_size=1)    

    model.eval()    
    mae = 0
    mse = 0
    mae_small = 0
    mse_small = 0
    mae_large = 0
    mse_large = 0
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

    count_small = 0
    count_large = 0
    count_cloudy = 0
    count_sunny = 0
    count_night = 0
    count_crowd = 0
    count_sparse = 0

    mae1 = 0
    mae2 = 0
    mae3 = 0
    mse1 = 0
    mse2 = 0
    mse3 = 0

    allposnum,allnegnum,allgtnum = 0,0,0
    for i, (imgs1, imgs2, imgs3, imgs4, gtnum, gt) in enumerate(test_loader):
        seqID = int(val_pair[i][0][-10:-7])
        for j in range(len(imgs1)):
            imgs1[j] = Variable(imgs1[j].cuda())
            imgs2[j] = Variable(imgs2[j].cuda())
            imgs3[j] = Variable(imgs3[j].cuda())
            imgs4[j] = Variable(imgs4[j].cuda())

        den_g11, den_g12, den_g13, loc_g11, loc_g12, loc_g13, loc_g14, reg_g11, reg_g12, reg_g13, reg_g14, trk_s14, trk_o14, trk_p14 = model(imgs1)
        den_g21, den_g22, den_g23, loc_g21, loc_g22, loc_g23, loc_g24, reg_g21, reg_g22, reg_g23, reg_g24, trk_s24, trk_o24, trk_p24 = model(imgs2)
        den_g31, den_g32, den_g33, loc_g31, loc_g32, loc_g33, loc_g34, reg_g31, reg_g32, reg_g33, reg_g34, trk_s34, trk_o34, trk_p34 = model(imgs3)
        den_g41, den_g42, den_g43, loc_g41, loc_g42, loc_g43, loc_g44, reg_g41, reg_g42, reg_g43, reg_g44, trk_s44, trk_o44, trk_p44 = model(imgs4)

        # calculate density result
        denout1 = max(0,den_g11[1].data.sum())+max(0,den_g21[1].data.sum())+max(0,den_g31[1].data.sum())+max(0,den_g41[1].data.sum())
        denout2 = max(0,den_g12[1].data.sum())+max(0,den_g22[1].data.sum())+max(0,den_g32[1].data.sum())+max(0,den_g42[1].data.sum())
        denout3 = max(0,den_g13[1].data.sum())+max(0,den_g23[1].data.sum())+max(0,den_g33[1].data.sum())+max(0,den_g43[1].data.sum())

        cmae1 = abs(denout1/args.factor - float(gtnum[0][1][0]))
        cmae2 = abs(denout2/args.factor - float(gtnum[0][1][0]))
        cmae3 = abs(denout3/args.factor - float(gtnum[0][1][0]))
        val = cmae1

        if seqID in small:
	        mae_small += val
	        mse_small += val**2
	        count_small += 1
        else:
	        mae_large += val
	        mse_large += val**2
	        count_large += 1

        if seqID in crowd:
	        mae_crowd += val
	        mse_crowd += val**2
	        count_crowd += 1
        else:
	        mae_sparse += val
	        mse_sparse += val**2
	        count_sparse += 1
	
        if seqID in cloudy:
	        mae_cloudy += val
	        mse_cloudy += val**2
	        count_cloudy += 1
        elif seqID in sunny:
	        mae_sunny += val
	        mse_sunny += val**2
	        count_sunny += 1
        else:
	        mae_night += val
	        mse_night += val**2
	        count_night += 1
        mae += val
        mse += val**2
   
        mae1 += cmae1
        mae2 += cmae2
        mae3 += cmae3
        mse1 += cmae1**2
        mse2 += cmae2**2
        mse3 += cmae3**2

        # calculate localization result
        if args.use_loc and not args.use_trk:
            loc1 = torch.exp(loc_g14[1][:,0,:,:]) / (torch.exp(loc_g14[1][:,1,:,:]) + torch.exp(loc_g14[1][:,0,:,:]))
            loc2 = torch.exp(loc_g24[1][:,0,:,:]) / (torch.exp(loc_g24[1][:,1,:,:]) + torch.exp(loc_g24[1][:,0,:,:]))
            loc3 = torch.exp(loc_g34[1][:,0,:,:]) / (torch.exp(loc_g34[1][:,1,:,:]) + torch.exp(loc_g34[1][:,0,:,:]))
            loc4 = torch.exp(loc_g44[1][:,0,:,:]) / (torch.exp(loc_g44[1][:,1,:,:]) + torch.exp(loc_g44[1][:,0,:,:]))
            data12 = np.hstack((loc1.data.cpu().numpy().squeeze(),loc2.data.cpu().numpy().squeeze()))
            data34 = np.hstack((loc3.data.cpu().numpy().squeeze(),loc4.data.cpu().numpy().squeeze()))
            loc_data = np.vstack((data12,data34))
            data12 = torch.cat((reg_g14[1],reg_g24[1]), 3)
            data34 = torch.cat((reg_g34[1],reg_g44[1]), 3)
            reg_data = torch.cat((data12,data34), 2)
            reg_data = reg_data.data.cpu().numpy().squeeze()
            save_path = 'loc_pts/' + val_pair[i][1][-13:-4] + '_loc.txt'
        elif args.use_loc and args.use_trk:
            p1 = (trk_p14[0]-trk_o14[0]*5.).data.cpu().numpy().squeeze()
            p2 = (trk_p24[0]-trk_o24[0]*5.).data.cpu().numpy().squeeze()
            p2[:,1] = p2[:,1]+480.
            p3 = (trk_p34[0]-trk_o34[0]*5.).data.cpu().numpy().squeeze()
            p3[:,0] = p3[:,0]+270.
            p4 = (trk_p44[0]-trk_o44[0]*5.).data.cpu().numpy().squeeze()
            p4[:,0], p4[:,1] = p4[:,0]+270., p4[:,1]+480.
            outpts = np.vstack((p1,p2,p3,p4))
            s1 = trk_s14[0].data.cpu().numpy().squeeze()
            s2 = trk_s24[0].data.cpu().numpy().squeeze()
            s3 = trk_s34[0].data.cpu().numpy().squeeze()
            s4 = trk_s44[0].data.cpu().numpy().squeeze()
            outscs = np.hstack((s1,s2,s3,s4))
            save_path = save_path_prefix + '/' + val_pair[i][1][-13:-4] + '_loc.txt'
        else:
            save_path = 'den_pts/' + val_pair[i][1][-13:-4] + '_loc.txt'
        # output density maps
        data12 = np.hstack((den_g11[1].data.cpu().numpy().squeeze(),den_g21[1].data.cpu().numpy().squeeze()))
        data34 = np.hstack((den_g31[1].data.cpu().numpy().squeeze(),den_g41[1].data.cpu().numpy().squeeze()))
        out_data = np.vstack((data12,data34))

        # output final results
        if not args.use_loc and not args.use_trk: # with density heads
            xp,yp,sc = calc_denpt(out_data/100.,8,2.0)
        elif args.use_loc and not args.use_trk: # with localization subnet
            xp,yp,sc = calc_locpt(loc_data,reg_data,0.5,out_data/100.,8,2.0)
        elif args.use_loc and args.use_trk: # with association subnet
            xp,yp,sc = calc_trkpt(outpts,outscs,0.5,out_data/100.,8,2.0)
        gtnum,posnum,negnum,pts = pt_eval(gt[1],xp,yp,sc)

        np.savetxt(save_path, pts)
        allgtnum += gtnum
        allposnum += posnum
        allnegnum += negnum
        if (i % 5 == 0 or i == len(test_loader)-1):
            print('img{i:d}*GT {gtnum:d} Pos {posnum:d} Neg {negnum:d} MAE1 {mae1:.1f} MAE2 {mae2:.1f} MAE3 {mae3:.1f}'
                 .format(i=i+1,gtnum=allgtnum, posnum=allposnum, negnum=allnegnum, mae1=mae1/(i+1),mae2=mae2/(i+1),mae3=mae3/(i+1)))

    mae1 = math.sqrt((mae1/len(test_loader))**2)
    mae2 = math.sqrt((mae2/len(test_loader))**2)    
    mae3 = math.sqrt((mae3/len(test_loader))**2)   
    mse1 = math.sqrt(mse1/len(test_loader)) 
    mse2 = math.sqrt(mse2/len(test_loader))    
    mse3 = math.sqrt(mse3/len(test_loader))

    print('*MAE1 {mae1:.1f} MAE2 {mae2:.1f} MAE3 {mae3:.1f} MSE1 {mse1:.1f} MSE2 {mse2:.1f} MSE3 {mse3:.1f}'
          .format(mae1=mae1,mae2=mae2,mae3=mae3,mse1=mse1,mse2=mse2,mse3=mse3))

    # counting results in different video-level attributes
    print('*Overall MAE {mae:.1f} MSE {mse:.1f}'.format( mae=mae.data.cpu().numpy()/len(test_loader), mse=math.sqrt(mse/len(test_loader)) ))

    print('*Small MAE {mae:.1f} MSE {mse:.1f}'.format( mae=mae_small.data.cpu().numpy()/max(1.,count_small), mse=math.sqrt(mse_small/max(1.,count_small)) ))	
    print('*Large MAE {mae:.1f} MSE {mse:.1f}'.format( mae=mae_large.data.cpu().numpy()/max(1.,count_large), mse=math.sqrt(mse_large/max(1.,count_large)) ))

    print('*Cloudy MAE {mae:.1f} MSE {mse:.1f}'.format( mae=mae_cloudy.data.cpu().numpy()/max(1.,count_cloudy), mse=math.sqrt(mse_cloudy/max(1.,count_cloudy)) ))	
    print('*Sunny MAE {mae:.1f} MSE {mse:.1f}'.format( mae=mae_sunny.data.cpu().numpy()/max(1.,count_sunny), mse=math.sqrt(mse_sunny/max(1.,count_sunny)) ))
    print('*Night MAE {mae:.1f} MSE {mse:.1f}'.format( mae=mae_night.data.cpu().numpy()/max(1.,count_night), mse=math.sqrt(mse_night/max(1.,count_night)) ))
	
    print('*Crowded MAE {mae:.1f} MSE {mse:.1f}'.format( mae=mae_crowd.data.cpu().numpy()/max(1.,count_crowd), mse=math.sqrt(mse_crowd/max(1.,count_crowd)) ))
    print('*Sparse MAE {mae:.1f} MSE {mse:.1f}'.format( mae=mae_sparse.data.cpu().numpy()/max(1.,count_sparse), mse=math.sqrt(mse_sparse/max(1.,count_sparse)) ))

def pt_eval(gt,x,y,sc):
    gx, gy = [], []
    for k in range(len(gt[0])):
        if gt[0][k][0] > 0:
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


def calc_trkpt(outpts, outscs, thre, ourmap, neighbor_thre, ratio):
    binamap = ourmap > 0
    ourmap[binamap == 0] = 0
    data_max = filters.maximum_filter(ourmap, neighbor_thre)
    data_min = filters.minimum_filter(ourmap, neighbor_thre)
    maxima = ourmap == data_max
    diffmap = ((data_max - data_min) > 0.001)
    maxima[diffmap == 0] = 0
    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    x, y, sc = [], [], []
    # points from density
    for dy, dx in slices:
        x_center = (dx.start + dx.stop - 1) / 2 * ratio
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1) / 2 * ratio
        y.append(y_center)
        sc.append(ourmap[int(y_center/ratio), int(x_center/ratio)])

    # points from tracking
    for k in range(outpts.shape[0]):
        if outscs[k]>=thre:
            x_center = outpts[k,1] * ratio
            y_center = outpts[k,0] * ratio
            x.append(x_center)
            y.append(y_center)
            sc.append(outscs[k])

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    sc = np.asarray(sc, dtype=np.float32)
    return x, y, sc

def calc_locpt(locmap, regmap, thre, ourmap, neighbor_thre, ratio):
    binamap = ourmap > 0
    ourmap[binamap == 0] = 0
    data_max = filters.maximum_filter(ourmap, neighbor_thre)
    data_min = filters.minimum_filter(ourmap, neighbor_thre)
    maxima = ourmap == data_max
    diffmap = ((data_max - data_min) > 0.001)
    maxima[diffmap == 0] = 0
    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    x, y, sc = [], [], []
    # points from density
    for dy, dx in slices:
        x_center = (dx.start + dx.stop - 1) / 2 * ratio
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1) / 2 * ratio
        y.append(y_center)
        sc.append(ourmap[int(y_center/ratio), int(x_center/ratio)])

    binamap = locmap > thre
    locmap[binamap == 0] = 0
    data_max = filters.maximum_filter(locmap, neighbor_thre)
    data_min = filters.minimum_filter(locmap, neighbor_thre)
    maxima = locmap == data_max
    diffmap = ((data_max - data_min) > 0.001)
    maxima[diffmap == 0] = 0
    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    # points from localization
    for dy, dx in slices:
        x_center = (dx.start + dx.stop - 1) / 2 * ratio
        y_center = (dy.start + dy.stop - 1) / 2 * ratio
        offset = regmap[:, int(y_center / ratio), int(x_center / ratio)]
        x.append(x_center+offset[1]*5*ratio)
        y.append(y_center+offset[0]*5*ratio)
        sc.append(locmap[int(y_center/ratio), int(x_center/ratio)])

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    sc = np.asarray(sc, dtype=np.float32)
    return x, y, sc
 
def calc_denpt(ourmap, neighbor_thre, ratio):
    binamap = ourmap > 0
    ourmap[binamap == 0] = 0
    data_max = filters.maximum_filter(ourmap, neighbor_thre)
    data_min = filters.minimum_filter(ourmap, neighbor_thre)
    maxima = ourmap == data_max
    diffmap = ((data_max - data_min) > 0.001)
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
    return x, y, sc

if __name__ == '__main__':
    main()        
