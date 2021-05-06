import sys
import os
import warnings

from stnmodel import STNNet
from loss_func import calc_loc_loss, calc_trk_loss
from utils import AverageMeter, save_checkpoint
import dataset_train

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import math
import argparse
import cv2
import time
import glob
import pdb
parser = argparse.ArgumentParser(description='PyTorch STNNet for Crowd Counting, Localization and Tracking')
parser.add_argument('--mGPUs', dest='use_mGPUs', action='store_true',
                    help='whether using multiple GPUs')
parser.add_argument('--loc', dest='use_loc', action='store_true',
                    help='whether using the localization subnet or not')
parser.add_argument('--trk', dest='use_trk', action='store_true',
                    help='whether fine-tuning the association subnet or not')
parser.add_argument('--tem', dest='use_tem', action='store_true',
                    help='whether only using the temporal prediction constraint in the loss')
parser.add_argument('--cyc', dest='use_cyc', action='store_true',
                    help='whether using the cycle loss')
parser.add_argument('--bs', dest='batch_size', default=1, type=int,
                    help='batch_size')
parser.add_argument('--fr', dest='frame_length', default=2, type=int,
                    help='the length of frames in the clip')
parser.add_argument('--pre', dest='pre_train', default=None, type=str,
                    help='path to the pre-trained model')
parser.add_argument('task', metavar='TASK', type=str,
                    help='model name')

def main():
    global args, best_prec1, best_mte1
    best_prec1 = 1e6
    best_mte1 = 1e6
    args = parser.parse_args()
    args.lr = 1e-6
    args.start_epoch = 0
    args.epochs = 100
    args.workers = 10
    args.factor = 100.
    args.seed = time.time()
    args.print_freq = 10
    nframe = args.frame_length
    # prepare the train/val dataset
    root = '../dataset/'
    train_path = os.path.join(root, 'train_data', 'images')
    train_list = []
    for img_path in glob.glob(os.path.join(train_path, '*.jpg')):
        train_list.append(img_path)
    train_list.sort()
    train_clip = []
    train_step = 0
    for idx in range(len(train_list)):
        if train_step % 1 == 0:
            clip, imgs, seqids = [], [], []
            for i in range(nframe):
                cur_img = train_list[max(0, idx - i)]
                seqid = cur_img[-10:-7]
                imgs.append(cur_img)
                seqids.append(seqid)

            for i in range(nframe - 1):
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
            train_clip.append(clip[::-1])
        train_step += 1

    test_path = os.path.join(root, 'val_data', 'images')
    test_list = []
    for img_path in glob.glob(os.path.join(test_path, '*.jpg')):
        test_list.append(img_path)
    test_list.sort()
    test_clip = []
    test_step = 0
    for idx in range(len(test_list)):
        if (test_step) % 1 == 0:
            clip, imgs, seqids = [], [], []
            for i in range(nframe):
                cur_img = test_list[max(0, idx - i)]
                seqid = cur_img[-10:-7]
                imgs.append(cur_img)
                seqids.append(seqid)

            for i in range(nframe - 1):
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

    torch.cuda.manual_seed(args.seed)
    use_loc = True if args.use_loc else False
    use_trk = True if args.use_trk else False
    model = STNNet(use_loc, use_trk).cuda()
    if args.use_mGPUs: model = nn.DataParallel(model)
    criterion = nn.MSELoss(reduction='sum').cuda()
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(model_parameters, lr=args.lr, betas=(0.5, 0.999))
    if args.pre_train:
        if os.path.isfile(args.pre_train):
            print("=> loading checkpoint '{}'".format(args.pre_train))
            checkpoint = torch.load(args.pre_train)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            my_models = model.state_dict()
            pre_models = list(checkpoint['state_dict'].items())
            preweight_dict = {}
            for i in range(len(pre_models)):
                prelayer_name, pre_weights = pre_models[i]
                preweight_dict[prelayer_name] = pre_weights
            for layer_name, value in my_models.items():
                if 'module.'+layer_name in preweight_dict.keys() or layer_name in preweight_dict.keys():
                    try:
                        my_models[layer_name] = preweight_dict['module.'+layer_name]
                    except:
                        my_models[layer_name] = preweight_dict[layer_name]
            model.load_state_dict(my_models)
            print("=> loaded checkpoint '{}' (epoch {})".format(args.pre_train, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre_train))

    for epoch in range(args.start_epoch, args.epochs):
        train(train_clip, model, criterion, optimizer, epoch)
        with torch.no_grad():
            prec1, mte1 = validate(test_clip, model, criterion)
            is_best = prec1 <= best_prec1 and mte1 <= best_mte1
            best_prec1 = min(prec1, best_prec1)
            best_mte1 = min(mte1, best_mte1)
            print(' * best MAE {mae:.3f} best MTE {mte:.3f} '.format(mae=best_prec1,mte=best_mte1))

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.pre_train,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best, args.task)

###########################################Training########################################################################
def train(train_clip, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loc_ratio = 1.0
    trk_ratio = 1.0
    use_tem = True if args.use_tem else False
    train_loader = torch.utils.data.DataLoader(
        dataset_train.listDataset(train_clip,
                            shuffle=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),
                            ]),
                            train=True,
                            batch_size=args.batch_size,
                            num_workers=args.workers), pin_memory=True,
        batch_size=args.batch_size)
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))

    model.train()
    end = time.time()

    for i, (imgset, dengt1, dengt2, dengt3, reggt1, reggt2, reggt3, trkgt) in enumerate(train_loader):
        data_time.update(time.time() - end)
        for j in range(len(imgset)):
            imgset[j] = Variable(imgset[j].cuda())
        den_g1, den_g2, den_g3, loc_g1, loc_g2, loc_g3, loc_g4, reg_g1, reg_g2, reg_g3, reg_g4, trk_g4, trk_o4, trk_p4 = model(imgset)

        # loss for density
        den_loss1, den_loss2, den_loss3 = 0., 0., 0.
        for j in range(len(imgset)):
            den1 = Variable(dengt1[:,j,:,:].type(torch.FloatTensor).unsqueeze(1).cuda())
            den2 = Variable(dengt2[:,j,:,:].type(torch.FloatTensor).unsqueeze(1).cuda())
            den3 = Variable(dengt3[:,j,:,:].type(torch.FloatTensor).unsqueeze(1).cuda())
            den_loss1 += criterion(den_g1[j], den1*args.factor) * 2.0 / float(args.batch_size) / 3. /args.factor/10.
            den_loss2 += criterion(den_g2[j], den2*args.factor) * 0.5 / float(args.batch_size) / 3. /args.factor/10.
            den_loss3 += criterion(den_g3[j], den3*args.factor) * 0.05 / float(args.batch_size) / 3. /args.factor/10.

        den_loss1 = den_loss1/float(len(imgset))
        den_loss2 = den_loss2/float(len(imgset))
        den_loss3 = den_loss3/float(len(imgset))
        den_loss = (den_loss1.mean() + den_loss2.mean() + den_loss3.mean())

        # loss for localization
        loc_loss, loc_loss1, loc_loss2, loc_loss3, loc_loss4 = 0., 0., 0., 0., 0.
        conf_loss, conf_loss1, conf_loss2, conf_loss3, conf_loss4 = 0., 0., 0., 0., 0.
        reg_loss, reg_loss1, reg_loss2, reg_loss3, reg_loss4 = 0., 0., 0., 0., 0.
        if args.use_loc:          
            for j in range(len(imgset)):
                reg1 = Variable(reggt1[:,j,:,:].type(torch.FloatTensor).unsqueeze(1).cuda())
                reg2 = Variable(reggt2[:,j,:,:].type(torch.FloatTensor).unsqueeze(1).cuda())
                reg3 = Variable(reggt3[:,j,:,:].type(torch.FloatTensor).unsqueeze(1).cuda())
                loc_loss1_,conf_loss1_,reg_loss1_=calc_loc_loss(criterion,loc_g1[j],reg_g1[j],reg1,loc_ratio,1.0,trkgt[:,j,:,:])
                loc_loss2_,conf_loss2_,reg_loss2_=calc_loc_loss(criterion,loc_g2[j],reg_g2[j],reg2,loc_ratio,2.0,trkgt[:,j,:,:])
                loc_loss3_,conf_loss3_,reg_loss3_=calc_loc_loss(criterion,loc_g3[j],reg_g3[j],reg3,loc_ratio,4.0,trkgt[:,j,:,:])
                loc_loss4_,conf_loss4_,reg_loss4_=calc_loc_loss(criterion,loc_g4[j],reg_g4[j],reg1,loc_ratio,1.0,trkgt[:,j,:,:])
                loc_loss1+=loc_loss1_
                loc_loss2+=loc_loss2_
                loc_loss3+=loc_loss3_
                loc_loss4+=loc_loss4_
                conf_loss1+=conf_loss1_
                conf_loss2+=conf_loss2_
                conf_loss3+=conf_loss3_
                conf_loss4+=conf_loss4_
                reg_loss1+=reg_loss1_
                reg_loss2+=reg_loss2_
                reg_loss3+=reg_loss3_
                reg_loss4+=reg_loss4_
            loc_loss1,conf_loss1,reg_loss1 = loc_loss1/float(args.batch_size*len(imgset)),conf_loss1/float(args.batch_size*len(imgset)),reg_loss1/float(args.batch_size*len(imgset))
            loc_loss2,conf_loss2,reg_loss2 = loc_loss2/float(args.batch_size*len(imgset)),conf_loss2/float(args.batch_size*len(imgset)),reg_loss2/float(args.batch_size*len(imgset))
            loc_loss3,conf_loss3,reg_loss3 = loc_loss3/float(args.batch_size*len(imgset)),conf_loss3/float(args.batch_size*len(imgset)),reg_loss3/float(args.batch_size*len(imgset))
            loc_loss4,conf_loss4,reg_loss4 = loc_loss4/float(args.batch_size*len(imgset)),conf_loss4/float(args.batch_size*len(imgset)),reg_loss4/float(args.batch_size*len(imgset))

            loc_loss = (loc_loss1.mean() + loc_loss2.mean() + loc_loss3.mean() + loc_loss4.mean())
            conf_loss = (conf_loss1.mean() + conf_loss2.mean() + conf_loss3.mean() + conf_loss4.mean())
            reg_loss = (reg_loss1.mean() + reg_loss2.mean() + reg_loss3.mean() + reg_loss4.mean())
           
        # loss for tracking
        trk_loss, pt_recall, pt_precision = 0., 0., 0.
        if args.use_trk:
            trk_loss_1, pt_recall_1, pt_precision_1 = calc_trk_loss(loc_g1[0],trkgt[:,0,:,:],trkgt[:,1,:,:],trk_o4[0],trk_p4[0], use_tem)  # forward tracking
            if args.use_cyc:
                trk_loss_2, pt_recall_2, pt_precision_2 = calc_trk_loss(loc_g1[1],trkgt[:,1,:,:],trkgt[:,0,:,:],-trk_o4[1],trk_p4[1], use_tem) # backward tracking
            else:
                trk_loss_2, pt_recall_2, pt_precision_2 = trk_loss_1, pt_recall_1, pt_precision_1
            trk_loss += (trk_loss_1+trk_loss_2)*trk_ratio/float(len(imgset))
            pt_recall += (pt_recall_1+pt_recall_2)/float(len(imgset))
            pt_precision += (pt_precision_1+pt_precision_2)/float(len(imgset))
        if trk_loss == 0:
            loss = den_loss + loc_loss
        else:
            loss = den_loss + loc_loss + trk_loss

        if loss > den_loss:
            losses.update(loss.item(), args.batch_size)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if args.use_mGPUs:
            den_loss1 = den_loss1.mean().item()
            den_loss2 = den_loss2.mean().item()
            den_loss3 = den_loss3.mean().item()

            if args.use_loc:
                loc_loss1,conf_loss1,reg_loss1 = loc_loss1.mean().item(),conf_loss1.mean().item(),reg_loss1.mean().item()
                loc_loss2,conf_loss2,reg_loss2 = loc_loss2.mean().item(),conf_loss2.mean().item(),reg_loss2.mean().item()
                loc_loss3,conf_loss3,reg_loss3 = loc_loss3.mean().item(),conf_loss3.mean().item(),reg_loss3.mean().item()
                loc_loss4,conf_loss4,reg_loss4 = loc_loss4.mean().item(),conf_loss4.mean().item(),reg_loss4.mean().item()
        else:
            den_loss1 = den_loss1.item()
            den_loss2 = den_loss2.item()
            den_loss3 = den_loss3.item()
            if args.use_loc:
                loc_loss1,conf_loss1,reg_loss1 = loc_loss1.item(),conf_loss1.item(),reg_loss1.item()
                loc_loss2,conf_loss2,reg_loss2 = loc_loss2.item(),conf_loss2.item(),reg_loss2.item()
                loc_loss3,conf_loss3,reg_loss3 = loc_loss3.item(),conf_loss3.item(),reg_loss3.item()
                loc_loss4,conf_loss4,reg_loss4 = loc_loss4.item(),conf_loss4.item(),reg_loss4.item()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.2f} ({loss.avg:.2f})\t'
                  'denLoss {den_loss1:.2f}/{den_loss2:.2f}/{den_loss3:.2f}\t'
                  'locLoss {loc_loss1:.2f}/{loc_loss2:.2f}/{loc_loss3:.2f}/{loc_loss4:.2f}\t'
                  'locRes {conf_loss:.4f}/{reg_loss:.4f}/{pt_recall:.2f}/{pt_precision:.2f}\t'
                  'allLoss {den_loss:.2f}/{loc_loss:.2f}/{trk_loss:.2f}\t'
                  .format(epoch, i, len(train_loader), loss=losses,
                          den_loss1=den_loss1, den_loss2=den_loss2, den_loss3=den_loss3,
                          loc_loss1=loc_loss1, loc_loss2=loc_loss2, loc_loss3=loc_loss3, loc_loss4=loc_loss4,
                          conf_loss=conf_loss, reg_loss=reg_loss, pt_recall=pt_recall, pt_precision=pt_precision,
                          den_loss=den_loss, loc_loss=loc_loss, trk_loss=trk_loss))

###########################################Validation########################################################################
def validate(val_clip, model, criterion):
    print('begin validation...')
    test_loader = torch.utils.data.DataLoader(
        dataset_reg.listDataset(val_clip,
                                 shuffle=False,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                 std=[0.229, 0.224, 0.225]),
                                 ]), train=False),
        batch_size=1)

    model.eval()
    nfr = len(val_clip[0])
    maes, mses, mles, mtes = np.zeros(shape=[nfr,3]), np.zeros(shape=[nfr,3]), np.zeros(shape=[nfr,3]), np.zeros(shape=[nfr,3])
    for i, (imgs1, imgs2, imgs3, imgs4, gtnum, regs1, regs2, regs3, regs4, trks1, trks2, trks3, trks4) in enumerate(test_loader):
        for j in range(len(imgs1)):
            imgs1[j] = Variable(imgs1[j].cuda())
            imgs2[j] = Variable(imgs2[j].cuda())
            imgs3[j] = Variable(imgs3[j].cuda())
            imgs4[j] = Variable(imgs4[j].cuda())

        den_g11, den_g12, den_g13, loc_g11, loc_g12, loc_g13, loc_g14, reg_g11, reg_g12, reg_g13, reg_g14, fea_g14, trk_o14, trk_p14 = model(imgs1)
        den_g21, den_g22, den_g23, loc_g21, loc_g22, loc_g23, loc_g24, reg_g21, reg_g22, reg_g23, reg_g24, fea_g24, trk_o24, trk_p24 = model(imgs2)
        den_g31, den_g32, den_g33, loc_g31, loc_g32, loc_g33, loc_g34, reg_g31, reg_g32, reg_g33, reg_g34, fea_g34, trk_o34, trk_p34 = model(imgs3)
        den_g41, den_g42, den_g43, loc_g41, loc_g42, loc_g43, loc_g44, reg_g41, reg_g42, reg_g43, reg_g44, fea_g44, trk_o44, trk_p44 = model(imgs4)

        # save density result
        for j in range(len(imgs1)):
            denout1=max(0,den_g11[j].data.sum())+max(0,den_g21[j].data.sum())+max(0,den_g31[j].data.sum())+max(0,den_g41[j].data.sum())
            denout2=max(0,den_g12[j].data.sum())+max(0,den_g22[j].data.sum())+max(0,den_g32[j].data.sum())+max(0,den_g42[j].data.sum())
            denout3=max(0,den_g13[j].data.sum())+max(0,den_g23[j].data.sum())+max(0,den_g33[j].data.sum())+max(0,den_g43[j].data.sum())

            cmae1 = abs(denout1/args.factor - float(gtnum[0][j][0]))
            cmae2 = abs(denout2/args.factor - float(gtnum[0][j][0]))
            cmae3 = abs(denout3/args.factor - float(gtnum[0][j][0]))
            maes[j,0] += cmae1
            maes[j,1] += cmae2
            maes[j,2] += cmae3
            mses[j,0] += cmae1 ** 2
            mses[j,1] += cmae2 ** 2
            mses[j,2] += cmae3 ** 2

        # save localization result
        if args.use_loc:
            for j in range(len(imgs1)):
                reg1 = Variable(regs1[:,j,:,:].type(torch.FloatTensor).unsqueeze(1).cuda())
                reg2 = Variable(regs2[:,j,:,:].type(torch.FloatTensor).unsqueeze(1).cuda())
                reg3 = Variable(regs3[:,j,:,:].type(torch.FloatTensor).unsqueeze(1).cuda())
                reg4 = Variable(regs4[:,j,:,:].type(torch.FloatTensor).unsqueeze(1).cuda())
                loc_loss1,conf_loss1,reg_loss1=calc_loc_loss(criterion,loc_g14[j],reg_g14[j],reg1,1.0,1.0,trks1[:,j,:,:])
                loc_loss2,conf_loss2,reg_loss2=calc_loc_loss(criterion,loc_g24[j],reg_g24[j],reg2,1.0,1.0,trks2[:,j,:,:])
                loc_loss3,conf_loss3,reg_loss3=calc_loc_loss(criterion,loc_g34[j],reg_g34[j],reg3,1.0,1.0,trks3[:,j,:,:])
                loc_loss4,conf_loss4,reg_loss4=calc_loc_loss(criterion,loc_g44[j],reg_g14[j],reg4,1.0,1.0,trks4[:,j,:,:])
                mles[j,0] += loc_loss1.item()+loc_loss2.item()+loc_loss3.item()+loc_loss4.item()
                mles[j,1] += conf_loss1.item()+conf_loss2.item()+conf_loss3.item()+conf_loss4.item()
                mles[j,2] += reg_loss1.item()+reg_loss2.item()+reg_loss3.item()+reg_loss4.item()
            mles = mles/float(len(imgs4))
        # save tracking result
        if args.use_trk:
            cmte11, ptrec11, ptpre11 = calc_trk_loss(loc_g14[0], trks1[:,0,:,:], trks1[:,1,:,:], trk_o14[0], trk_p14[0])
            cmte21, ptrec21, ptpre21 = calc_trk_loss(loc_g24[0], trks2[:,0,:,:], trks2[:,1,:,:], trk_o24[0], trk_p24[0])
            cmte31, ptrec31, ptpre31 = calc_trk_loss(loc_g34[0], trks3[:,0,:,:], trks3[:,1,:,:], trk_o34[0], trk_p34[0])
            cmte41, ptrec41, ptpre41 = calc_trk_loss(loc_g44[0], trks4[:,0,:,:], trks4[:,1,:,:], trk_o44[0], trk_p44[0])
            cmte12, ptrec12, ptpre12 = calc_trk_loss(loc_g14[1], trks1[:,1,:,:], trks1[:,0,:,:], -trk_o14[1], trk_p14[1])
            cmte22, ptrec22, ptpre22 = calc_trk_loss(loc_g24[1], trks2[:,1,:,:], trks2[:,0,:,:], -trk_o24[1], trk_p24[1])
            cmte32, ptrec32, ptpre32 = calc_trk_loss(loc_g34[1], trks3[:,1,:,:], trks3[:,0,:,:], -trk_o34[1], trk_p34[1])
            cmte42, ptrec42, ptpre42 = calc_trk_loss(loc_g44[1], trks4[:,1,:,:], trks4[:,0,:,:], -trk_o44[1], trk_p44[1])
            mtes[0,0] += cmte11+cmte21+cmte31+cmte41
            mtes[0,1] += (ptrec11+ptrec21+ptrec31+ptrec41)/4.
            mtes[0,2] += (ptpre11+ptpre21+ptpre31+ptpre41)/4.
            mtes[1,0] += cmte12+cmte22+cmte32+cmte42
            mtes[1,1] += (ptrec12+ptrec22+ptrec32+ptrec42)/4.
            mtes[1,2] += (ptpre12+ptpre22+ptpre32+ptpre42)/4.

    for i in range(len(imgs4)):
        for j in range(3):
            maes[i,j] = maes[i,j]/float(len(test_loader))
            mses[i,j] = math.sqrt(mses[i,j]/float(len(test_loader)))
        print('*MAE1 {mae1:.1f} MAE2 {mae2:.1f} MAE3 {mae3:.1f} MSE1 {mse1:.1f} MSE2 {mse2:.1f} MSE3 {mse3:.1f}'.format(
            mae1=maes[i,0], mae2=maes[i,1], mae3=maes[i,2], mse1=mses[i,0], mse2=mses[i,1], mse3=mses[i,2]))
        if args.use_loc:
            mles[i] = mles[i]/float(len(test_loader))
            print('*MLE1 {mle1:.4f} conf {mle2:.4f} reg {mle3:.4f}'.format(mle1=mles[i,0],mle2=mles[i,1],mle3=mles[i,2]))
        if args.use_trk:
            mtes[i] = mtes[i]/float(len(test_loader))
            print('*MTE1 {mte1:.2f} RECALL {mte2:.4f} PRECISION {mte3:.4f}'.format(mte1=mtes[i,0], mte2=mtes[i,1], mte3=mtes[i,2]))

    mae = np.min(maes[1,:])
    mte = mtes[0,0]
    return mae, mte

if __name__ == '__main__':
    main()
