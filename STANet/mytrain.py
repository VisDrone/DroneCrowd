import sys
import warnings

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import scipy

import numpy as np
import math
import argparse
import cv2
import time

import load_data
from stanet import STANet
from utils import load_train_data, load_val_data, load_txt, save_checkpoint

parser = argparse.ArgumentParser(description='Train PyTorch STANet')

parser.add_argument('--mGPUs', dest='use_mGPUs', action='store_true',
                    help='whether using multiple GPUs')

parser.add_argument('--loc', dest='use_loc', action='store_true',
                    help='whether using the localization branch')

parser.add_argument('--trk', dest='use_trk', action='store_true',
                    help='whether using the association branch')

parser.add_argument('--bs', dest='batch_size', default=1, type=int,
                    help='batch_size')

parser.add_argument('--pre', dest='pre_train', default=None, type=str,
                    help='path to the pretrained model')

parser.add_argument('task', metavar='TASK', type=str,
                    help='task id to use.')


def main():
    global args, best_prec1
    best_prec1 = 1e6
    args = parser.parse_args()
    args.start_epoch = 0
    args.epochs = 50
    args.workers = 16
    args.seed = time.time()
    args.print_freq = 100

    # load the dronecrowd dataset
    root = '../dataset/'
    train_step = 1
    train_pair = load_train_data(root, train_step)
    val_step = 1
    val_pair = load_val_data(root, val_step)

    torch.cuda.manual_seed(args.seed)
    # load model
    use_loc = True if args.use_loc else False
    use_trk = True if args.use_trk else False
    model = STANet(use_loc, use_trk).cuda()

    if args.use_mGPUs:
        model = nn.DataParallel(model)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer1 = torch.optim.Adam(model_parameters, lr=0.000001, betas=(0.5, 0.999))
    optimizer2 = torch.optim.Adam(model_parameters, lr=0.00001, betas=(0.5, 0.999))
    optimizer3 = torch.optim.Adam(model_parameters, lr=0.000005, betas=(0.5, 0.999))
    if args.pre_train:
        if os.path.isfile(args.pre_train):
            print("=> loading checkpoint '{}'".format(args.pre_train))
            checkpoint = torch.load(args.pre_train)
            if checkpoint['epoch'] < 10:
                optimizer = optimizer1
            elif checkpoint['epoch'] >= 10 and checkpoint['epoch'] < 30:
                optimizer = optimizer2
            else:
                optimizer = optimizer3
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pre_train, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre_train))

    criterion = nn.MSELoss(size_average=False).cuda()

    for epoch in range(args.start_epoch, args.epochs):
        if epoch < 10:
            optimizer = optimizer1
        elif epoch >= 10 and epoch < 30:
            optimizer = optimizer2
        else:
            optimizer = optimizer3

        train(train_pair, model, criterion, optimizer, epoch)
        with torch.no_grad():
            prec1 = validate(val_pair, model, criterion)
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print(' * best MAE {mae:.3f} '.format(mae=best_prec1))
        save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.pre_train,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
        }, is_best, args.task)


def calc_trk_loss(feat, gt1, gt2):
    feat_dim = feat.shape[1] / 2
    feat1 = feat[:, :feat_dim, :, :]
    feat2 = feat[:, feat_dim:, :, :]
    loss = 0
    for k in range(feat.shape[0]):
        cur_loss = 0
        pos_inds1 = gt1[k] > 0
        pos_inds2 = gt2[k] > 0
        pos_ids1 = gt1[k][pos_inds1]
        pos_ids2 = gt2[k][pos_inds2]
        num1 = len(pos_ids1)
        num2 = len(pos_ids2)
        if num1 > 0 and num2 > 0:
            pfeat1 = feat1[k, :, pos_inds1.squeeze()].permute(1, 0)
            pfeat2 = feat2[k, :, pos_inds2.squeeze()].permute(1, 0)
            X1 = torch.unsqueeze(pfeat1, dim=1).expand(num1, num2, feat_dim)
            X2 = torch.unsqueeze(pfeat2, dim=0).expand(num1, num2, feat_dim)
            feat_dist = torch.sqrt(torch.sum((X1 - X2) ** 2, dim=2))
            loss_same = 0
            loss_diff = 0
            min_diff = np.inf
            max_same = -np.inf
            for i in range(num1):
                for j in range(num2):
                    if pos_ids1[i] == pos_ids2[j]:
                        loss_same = max(max_same, feat_dist[i][j])
                        max_same = loss_same
                    else:
                        loss_diff = min(min_diff, feat_dist[i][j])
                        min_diff = loss_diff
            cur_loss += max(loss_same - loss_diff + 0.2, 0)
        cur_loss = cur_loss / max(1., float(num1))
    loss += cur_loss
    return loss


def train(train_pair, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    den_ratio = float(args.batch_size) / 4.
    loc_ratio = 0.0001 * den_ratio
    if epoch < 0:
        trk_ratio = 0
    else:
        trk_ratio = 10

    if epoch < 30:
        merge_weight = 2.0
    else:
        merge_weight = 4.0

    train_loader = torch.utils.data.DataLoader(
        load_data.listDataset(train_pair,
                                  shuffle=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                  std=[0.229, 0.224, 0.225]),
                                  ]),
                                  train=True,
                                  batch_size=args.batch_size,
                                  num_workers=args.workers), pin_memory=True,
        batch_size=args.batch_size)
    print('epoch %d, processed %d samples' % (epoch, epoch * len(train_loader.dataset)))

    model.train()

    for i, (img1, img2, dengt11, dengt12, dengt13, dengt21, dengt22, dengt23, locgt11, locgt12, locgt13, locgt21, locgt22,
    locgt23, trkgt1, trkgt2) in enumerate(train_loader):
        img1 = Variable(img1.cuda())
        img2 = Variable(img2.cuda())

        den_g1, den_g2, den_g3, den_g4, loc_g1, loc_g2, loc_g3, loc_g4, fea_g4 = model(img1, img2)
        # loss for density
        dengt11 = Variable(dengt11.type(torch.FloatTensor).unsqueeze(1).cuda())
        dengt12 = Variable(dengt12.type(torch.FloatTensor).unsqueeze(1).cuda())
        dengt13 = Variable(dengt13.type(torch.FloatTensor).unsqueeze(1).cuda())
        dengt21 = Variable(dengt21.type(torch.FloatTensor).unsqueeze(1).cuda())
        dengt22 = Variable(dengt22.type(torch.FloatTensor).unsqueeze(1).cuda())
        dengt23 = Variable(dengt23.type(torch.FloatTensor).unsqueeze(1).cuda())

        den_loss11 = criterion(den_g1[:, 0, :, :].unsqueeze(1), dengt11) * 2.0 * den_ratio
        den_loss12 = criterion(den_g2[:, 0, :, :].unsqueeze(1), dengt12) * 0.5 * den_ratio
        den_loss13 = criterion(den_g3[:, 0, :, :].unsqueeze(1), dengt13) * 0.05 * den_ratio
        den_loss14 = criterion(den_g4[:, 0, :, :].unsqueeze(1), dengt11) * merge_weight * den_ratio
        den_loss21 = criterion(den_g1[:, 1, :, :].unsqueeze(1), dengt21) * 2.0 * den_ratio
        den_loss22 = criterion(den_g2[:, 1, :, :].unsqueeze(1), dengt22) * 0.5 * den_ratio
        den_loss23 = criterion(den_g3[:, 1, :, :].unsqueeze(1), dengt23) * 0.05 * den_ratio
        den_loss24 = criterion(den_g4[:, 1, :, :].unsqueeze(1), dengt21) * merge_weight * den_ratio

        den_loss1 = (den_loss11 + den_loss21) / 2.0
        den_loss2 = (den_loss12 + den_loss22) / 2.0
        den_loss3 = (den_loss13 + den_loss23) / 2.0
        den_loss4 = (den_loss14 + den_loss24) / 2.0
        den_loss = (den_loss1.mean() + den_loss2.mean() + den_loss3.mean() + den_loss4.mean())

        # loss for localization
        if args.use_loc and loc_ratio > 0:

            locgt11 = Variable(locgt11.type(torch.FloatTensor).unsqueeze(1).cuda())
            locgt12 = Variable(locgt12.type(torch.FloatTensor).unsqueeze(1).cuda())
            locgt13 = Variable(locgt13.type(torch.FloatTensor).unsqueeze(1).cuda())
            locgt21 = Variable(locgt21.type(torch.FloatTensor).unsqueeze(1).cuda())
            locgt22 = Variable(locgt22.type(torch.FloatTensor).unsqueeze(1).cuda())
            locgt23 = Variable(locgt23.type(torch.FloatTensor).unsqueeze(1).cuda())

            loc_loss11 = criterion(loc_g1[:, 0, :, :].unsqueeze(1), locgt11) * 2.0 * loc_ratio
            loc_loss12 = criterion(loc_g2[:, 0, :, :].unsqueeze(1), locgt12) * 0.5 * loc_ratio
            loc_loss13 = criterion(loc_g3[:, 0, :, :].unsqueeze(1), locgt13) * 0.05 * loc_ratio
            loc_loss14 = criterion(loc_g4[:, 0, :, :].unsqueeze(1), locgt11) * merge_weight * loc_ratio
            loc_loss21 = criterion(loc_g1[:, 1, :, :].unsqueeze(1), locgt21) * 2.0 * loc_ratio
            loc_loss22 = criterion(loc_g2[:, 1, :, :].unsqueeze(1), locgt22) * 0.5 * loc_ratio
            loc_loss23 = criterion(loc_g3[:, 1, :, :].unsqueeze(1), locgt23) * 0.05 * loc_ratio
            loc_loss24 = criterion(loc_g4[:, 1, :, :].unsqueeze(1), locgt21) * merge_weight * loc_ratio

            loc_loss1 = (loc_loss11 + loc_loss21) / 2.0
            loc_loss2 = (loc_loss12 + loc_loss22) / 2.0
            loc_loss3 = (loc_loss13 + loc_loss23) / 2.0
            loc_loss4 = (loc_loss14 + loc_loss24) / 2.0
            loc_loss = (loc_loss1.mean() + loc_loss2.mean() + loc_loss3.mean() + loc_loss4.mean())
        else:
            loc_loss = 0.
            loc_loss1, loc_loss2, loc_loss3, loc_loss4 = 0., 0., 0., 0.

        # loss for tracking
        #pdb.set_trace()
        if args.use_trk and trk_ratio > 0:
            trk_loss = calc_trk_loss(fea_g4, trkgt1, trkgt2) / float(args.batch_size) * trk_ratio
        else:
            trk_loss = 0.

        loss = den_loss + loc_loss + trk_loss
        losses.update(loss.item(), args.batch_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.use_mGPUs:
            den_loss1 = den_loss1.mean().item()
            den_loss2 = den_loss2.mean().item()
            den_loss3 = den_loss3.mean().item()
            den_loss4 = den_loss4.mean().item()
            if args.use_loc:
                loc_loss1 = loc_loss1.mean().item()
                loc_loss2 = loc_loss2.mean().item()
                loc_loss3 = loc_loss3.mean().item()
                loc_loss4 = loc_loss4.mean().item()
        else:
            den_loss1 = den_loss1.item()
            den_loss2 = den_loss2.item()
            den_loss3 = den_loss3.item()
            den_loss4 = den_loss4.item()
            if args.use_loc:
                loc_loss1 = loc_loss1.item()
                loc_loss2 = loc_loss2.item()
                loc_loss3 = loc_loss3.item()
                loc_loss4 = loc_loss4.item()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.2f} ({loss.avg:.2f})\t'
                  'denLoss {den_loss1:.2f}/{den_loss2:.2f}/{den_loss3:.2f}/{den_loss4:.2f}\t'
                  'locLoss {loc_loss1:.2f}/{loc_loss2:.2f}/{loc_loss3:.2f}/{loc_loss4:.2f}\t'
                  'allLoss {den_loss:.2f}/{loc_loss:.2f}/{trk_loss:.2f}\t'
                  .format(epoch, i, len(train_loader), loss=losses,
                          den_loss1=den_loss1, den_loss2=den_loss2, den_loss3=den_loss3, den_loss4=den_loss4,
                          loc_loss1=loc_loss1, loc_loss2=loc_loss2, loc_loss3=loc_loss3, loc_loss4=loc_loss4,
                          den_loss=den_loss, loc_loss=loc_loss, trk_loss=trk_loss))

def validate(val_pair, model, criterion):
    val_loader = torch.utils.data.DataLoader(
        load_data.listDataset(val_pair,
                                  shuffle=False,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                  std=[0.229, 0.224, 0.225]),
                                  ]), train=False),
        batch_size=1)

    num_sample = len(val_loader)

    model.eval()
    maes, mses, mle = np.zeros(shape=[4,]), np.zeros(shape=[4,]), 0

    for i, (img11,img12,img13,img14,img21,img22,img23,img24,loc11,loc12,loc13,loc14,loc21,loc22,loc23,loc24,gt_count1,gt1,gt_count2,gt2) in enumerate(val_loader):
        img13 = Variable(img13.cuda())
        img23 = Variable(img23.cuda())
        den_g11, den_g12, den_g13, den_g14, loc_g11, loc_g12, loc_g13, loc_g14, fea_g14 = model(img11, img21)
        img14 = Variable(img14.cuda())
        img24 = Variable(img24.cuda())
        den_g21, den_g22, den_g23, den_g24, loc_g21, loc_g22, loc_g23, loc_g24, fea_g24 = model(img12, img22)
        img13 = Variable(img13.cuda())
        img23 = Variable(img23.cuda())
        den_g31, den_g32, den_g33, den_g34, loc_g31, loc_g32, loc_g33, loc_g34, fea_g34 = model(img13, img23)
        img14 = Variable(img14.cuda())
        img24 = Variable(img24.cuda())
        den_g41, den_g42, den_g43, den_g44, loc_g41, loc_g42, loc_g43, loc_g44, fea_g44 = model(img14, img24)

        # calculate density result
        denout21 = max(0,den_g11[:,1,:,:].data.sum())+max(0,den_g21[:,1,:,:].data.sum())+max(0,den_g31[:,1,:,:].data.sum())+max(0,den_g41[:,1,:,:].data.sum())
        denout22 = max(0,den_g12[:,1,:,:].data.sum())+max(0,den_g22[:,1,:,:].data.sum())+max(0,den_g32[:,1,:,:].data.sum())+max(0,den_g42[:,1,:,:].data.sum())
        denout23 = max(0,den_g13[:,1,:,:].data.sum())+max(0,den_g23[:,1,:,:].data.sum())+max(0,den_g33[:,1,:,:].data.sum())+max(0,den_g43[:,1,:,:].data.sum())
        denout24 = max(0,den_g14[:,1,:,:].data.sum())+max(0,den_g24[:,1,:,:].data.sum())+max(0,den_g34[:,1,:,:].data.sum())+max(0,den_g44[:,1,:,:].data.sum())

        cmae1 = abs(denout21 - float(gt_count2))
        cmae2 = abs(denout22 - float(gt_count2))
        cmae3 = abs(denout23 - float(gt_count2))
        cmae4 = abs(denout24 - float(gt_count2))

        maes[0] += cmae1
        maes[1] += cmae2
        maes[2] += cmae3
        maes[3] += cmae4
        mses[0] += cmae1 ** 2
        mses[1] += cmae2 ** 2
        mses[2] += cmae3 ** 2
        mses[3] += cmae4 ** 2

        if args.use_loc:
            loc21 = Variable(loc21.type(torch.FloatTensor).unsqueeze(1).cuda())
            loc22 = Variable(loc22.type(torch.FloatTensor).unsqueeze(1).cuda())
            loc23 = Variable(loc23.type(torch.FloatTensor).unsqueeze(1).cuda())
            loc24 = Variable(loc24.type(torch.FloatTensor).unsqueeze(1).cuda())
            cmle24 = criterion(loc_g11[:,1,:,:].unsqueeze(1), loc21).item() + criterion(loc_g21[:,1,:,:].unsqueeze(1), loc22).item()+\
                     criterion(loc_g31[:,1,:,:].unsqueeze(1), loc23).item() + criterion(loc_g41[:,1,:,:].unsqueeze(1), loc24).item()
            mle += cmle24

    for i in range(4):
        maes[i] = maes[i]/num_sample
        mses[i] = math.sqrt(mses[i]/num_sample)

    print('*MAE21 {mae21:.1f} MAE22 {mae22:.1f} MAE23 {mae23:.1f} MAE24 {mae24:.1f} MSE21 {mse21:.1f} MSE22 {mse22:.1f} MSE23 {mse23:.1f} MSE24 {mse24:.1f}'.format(
        mae21=maes[0], mae22=maes[1], mae23=maes[2], mae24=maes[3], mse21=mses[0], mse22=mses[1], mse23=mses[2], mse24=mses[3]))

    if args.use_loc:
        mle = mle / num_sample
        print('*MLE {mle:.1f}'.format(mle=mle))

    mae = np.min(maes)

    return mae

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
