import torch.nn as nn
import torch
from torchvision import models
from spatial_correlation_sampler import SpatialCorrelationSampler
from pointconv import PointConvDensitySetAbstraction
import numpy as np
import random
import torch.nn.functional as F
import torch.nn.init as init
import math

##############################VGG16 Backbone#######################################
class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.backbone1 = make_layers([64, 64, 'M', 128, 128])
        self.backbone2 = make_layers(['M', 256, 256, 256], in_channels=128)
        self.backbone3 = make_layers(['M', 512, 512, 512], in_channels=256)

    def forward(self, x):
        x1 = self.backbone1(x)
        x2 = self.backbone2(x1)
        x3 = self.backbone3(x2)

        return x1, x2, x3

def make_layers(cfg, in_channels=3, batch_norm=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

##############################Space-Time Neighbor-Aware Network####################
class STNNet(nn.Module):
    def __init__(self, use_loc=False, use_trk=False):
        super(STNNet, self).__init__()
        self.loc = use_loc
        self.trk = use_trk
        # backbone
        self.backbone = BaseNet()
        self.relu = nn.ReLU()

        # multi-scale combination
        self.deconv1 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1, dilation=2)
        self.deconv2 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1, dilation=2)

        self.conv1 = nn.Conv2d(512, 256, 1)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1, dilation=1)
        self.conv3 = nn.Conv2d(256, 128, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1, dilation=1)

        # attention layers
        self.s_weight1 = SpatialWeightLayer()
        self.s_weight2 = SpatialWeightLayer()
        self.s_weight3 = SpatialWeightLayer()

        # density output layers
        self.den_output_layer1 = nn.Conv2d(128, 1, kernel_size=1)
        self.den_output_layer2 = nn.Conv2d(256, 1, kernel_size=1)
        self.den_output_layer3 = nn.Conv2d(512, 1, kernel_size=1)

        # fix above parameters when training the association heads
        if self.trk:
            for para in self.parameters():
                para.requires_grad = False

        if self.loc:
            # localization output layers
            self.loc_output_layer1 = nn.Conv2d(128, 2, 3, padding=1, dilation=1)
            self.loc_output_layer2 = nn.Conv2d(256, 2, 3, padding=1, dilation=1)
            self.loc_output_layer3 = nn.Conv2d(512, 2, 3, padding=1, dilation=1)
            self.reg_output_layer1 = nn.Conv2d(128, 2, 3, padding=1, dilation=1)
            self.reg_output_layer2 = nn.Conv2d(256, 2, 3, padding=1, dilation=1)
            self.reg_output_layer3 = nn.Conv2d(512, 2, 3, padding=1, dilation=1)

            self.s_weight_loc = nn.Conv2d(6, 6, 3, padding=1)
            self.merge_loc = nn.Conv2d(6, 2, 1, padding=0)
            self.s_weight_reg = nn.Conv2d(6, 6, 3, padding=1)
            self.merge_reg = nn.Conv2d(6, 2, 1, padding=0)

        if self.trk:
            # tracking output layers
            self.corr_layer1 = SpatialCorrelationSampler(1, 11, 1, 0, 1)
            self.corr_layer2 = SpatialCorrelationSampler(1, 11, 1, 0, 1)
            self.corr_layer3 = SpatialCorrelationSampler(1, 11, 1, 0, 1)
            self.trk_output_layer1 = nn.Conv2d(363, 128, 3, padding=1, dilation=1)
            self.trk_output_layer2 = nn.Conv2d(128, 64, 3, padding=1, dilation=1)
            # graph layers
            self.gcn_layers = PointConvDensityClsSsg(num_classes=2, num_pt=128)

        # load weights of the backbone network
        mod = models.vgg16(pretrained = True)
        self._initialize_weights()
        my_models = self.backbone.state_dict()
        pre_models = list(mod.state_dict().items())
        count = 0
        for layer_name, value in my_models.items():
            prelayer_name, pre_weights = pre_models[count]
            my_models[layer_name] = pre_weights
            count += 1
        self.backbone.load_state_dict(my_models)

    def forward(self, imgset):
        base_g1, base_g2, base_g3 = [], [], []
        den_g1, den_g2, den_g3, den_g4 = [], [], [], []
        loc_g1, loc_g2, loc_g3, loc_g4 = [], [], [], []
        reg_g1, reg_g2, reg_g3, reg_g4 = [], [], [], []
        trk_g1, trk_g2, trk_g3 = [], [], []
        trk_g, trk_o, trk_p = [], [], []
        # combine features
        for i in range(len(imgset)):
            f1, f2, f3 = self.backbone(imgset[i])
            if i == 0:
                g1_size = (f1.shape[2], f1.shape[3])
                g2_size = (f2.shape[2], f2.shape[3])

            g2 = self.deconv1(f3)
            g2 = self.relu(g2)
            g2 = nn.Upsample(size=g2_size, mode='bilinear', align_corners=True)(g2)
            g2 = torch.cat((g2, f2), 1)
            g2 = self.conv1(g2)
            g2 = self.relu(g2)
            g2 = self.conv2(g2)
            g2 = self.relu(g2)

            g1 = self.deconv2(g2)
            g1 = self.relu(g1)
            g1 = nn.Upsample(size=g1_size, mode='bilinear', align_corners=True)(g1)
            g1 = torch.cat((g1, f1), 1)
            g1 = self.conv3(g1)
            g1 = self.relu(g1)
            g1 = self.conv4(g1)
            g1 = self.relu(g1)

            g1 = self.s_weight1(g1)
            g2 = self.s_weight2(g2)
            f3 = self.s_weight3(f3)
            base_g1.append(g1)
            base_g2.append(g2)
            base_g3.append(f3)

            # generate density map
            den1 = self.den_output_layer1(g1)  # x2 resolution
            den2 = self.den_output_layer2(g2)  # x4 resolution
            den3 = self.den_output_layer3(f3)  # x8 resolution
            den_g1.append(den1)
            den_g2.append(den2)
            den_g3.append(den3)

            # generate localization map
            if self.loc:
                loc1 = self.loc_output_layer1(g1)  # x2 resolution
                loc2 = self.loc_output_layer2(g2)  # x4 resolution
                loc3 = self.loc_output_layer3(f3)  # x8 resolution
                loc4 = torch.cat((loc1, nn.Upsample(size=g1_size, mode='bilinear', align_corners=True)(loc2),
                              nn.Upsample(size=g1_size, mode='bilinear', align_corners=True)(loc3)), 1)
                loc4 = self.s_weight_loc(loc4)
                loc4 = self.merge_loc(loc4)

                reg1 = self.reg_output_layer1(g1)  # x2 resolution
                reg2 = self.reg_output_layer2(g2)  # x4 resolution
                reg3 = self.reg_output_layer3(f3)  # x8 resolution
                reg4 = torch.cat((reg1, nn.Upsample(size=g1_size, mode='bilinear', align_corners=True)(reg2),
                              nn.Upsample(size=g1_size, mode='bilinear', align_corners=True)(reg3)), 1)
                reg4 = self.s_weight_reg(reg4)
                reg4 = self.merge_reg(reg4)

                loc_g1.append(loc1)
                loc_g2.append(loc2)
                loc_g3.append(loc3)
                loc_g4.append(loc4)

                reg_g1.append(reg1)
                reg_g2.append(reg2)
                reg_g3.append(reg3)
                reg_g4.append(reg4)

        if not self.trk:
            return den_g1, den_g2, den_g3, loc_g1, loc_g2, loc_g3, loc_g4, reg_g1, reg_g2, reg_g3, reg_g4, trk_g, trk_o, trk_p

        # generate tracking map
        trk1 = self.corr_layer1(base_g1[0], base_g1[1]).view(g1.shape[0], 121, g1.shape[2], g1.shape[3])
        trk2 = self.corr_layer2(base_g2[0], base_g2[1]).view(g2.shape[0], 121, g2.shape[2], g2.shape[3])
        trk3 = self.corr_layer3(base_g3[0], base_g3[1]).view(f2.shape[0], 121, f3.shape[2], f3.shape[3])
        trk2 = nn.Upsample(size=g1_size, mode='bilinear', align_corners=True)(trk2)
        trk3 = nn.Upsample(size=g1_size, mode='bilinear', align_corners=True)(trk3)
        trk_g1.append(trk1)
        trk_g2.append(trk2)
        trk_g3.append(trk3)

        trk_123 = torch.cat((trk1, trk2, trk3), 1)
        trk_123 = self.trk_output_layer1(trk_123)
        trk_123 = self.relu(trk_123)
        trk_123 = self.trk_output_layer2(trk_123)
        topk = 128 # top 128 proposal points
        for i in range(len(imgset)):
            # extract maximal points
            loc = torch.exp(loc_g4[i][:,0,:,:])/(torch.exp(loc_g4[i][:,1,:,:])+torch.exp(loc_g4[i][:,0,:,:])) # confidence of points
            reg = reg_g4[i] # offset of points

            for k in range(loc.shape[0]):
                cur_loc = loc[k]
                max_loc, keep_idx = nms(cur_loc.unsqueeze(0))
                # extract points based on location maps
                points = np.where(max_loc[0].cpu() > 0)
                points = torch.cat((torch.from_numpy(points[0]).unsqueeze(1), torch.from_numpy(points[1]).unsqueeze(1)), 1)
                points = points.type(torch.FloatTensor).cuda()
                num_points = points.shape[0]
                points = points.type(torch.long)
                reg = reg.type(torch.long)
                for j in range(num_points):
                    score = max_loc[0, points[j, 0], points[j, 1]].unsqueeze(0)
                    offset = reg[k, :, points[j, 0], points[j, 1]].unsqueeze(0)
                    points[j,:] = points[j,:] - offset*5. # add the offset at the current frame
                    if j == 0:
                        scs = score
                    else:
                        scs = torch.cat((scs, score), 0)
                scs, idx = torch.sort(scs, descending=True)
                points = points.type(torch.float)
                outscs = scs[:topk]
                pts = points[idx[:topk], :]

                # extract features based on correlation maps
                real_pts = pts.type(torch.long)
                h, w = trk_123.shape[2]-1, trk_123.shape[3]-1
                for j in range(topk):
                    feat = trk_123[k, :, int(max(0,min(h,real_pts[j,0]))), int(max(0,min(w,real_pts[j,1])))].unsqueeze(0)
                    if j == 0:
                        fts = feat
                    else:
                        fts = torch.cat((fts, feat), 0)

                maxpoint, minpoint = torch.max(pts), torch.min(pts)
                pts = (pts - minpoint) / float(maxpoint - minpoint)
                pts = pts.unsqueeze(0).transpose(2, 1)
                fts = fts.unsqueeze(0).transpose(2, 1)

                # extract locations and offsets based on gcns
                random_offset = self.gcn_layers(pts, fts)
                random_offset = random_offset.permute(0, 2, 1).contiguous()
                if k == 0:
                    random_scores = outscs.unsqueeze(0)
                    random_offsets = random_offset
                    random_points = real_pts.unsqueeze(0)
                else:
                    random_scores = torch.cat((random_scores, outscs.unsqueeze(0)), 0)
                    random_offsets = torch.cat((random_offsets, random_offset), 0)
                    random_points = torch.cat((random_points, real_pts.unsqueeze(0)), 0)
            trk_g.append(random_scores)  # scores of selected points
            trk_o.append(random_offsets)  # offsets of selected points
            trk_p.append(random_points)  # locations of selected points

        return den_g1, den_g2, den_g3, loc_g1, loc_g2, loc_g3, loc_g4, reg_g1, reg_g2, reg_g3, reg_g4, trk_g, trk_o, trk_p

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_((m.weight), std=0.01)
                if m.bias is not None:
                    nn.init.constant_((m.bias), 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_((m.weight), std=0.01)
                if m.bias is not None:
                    nn.init.constant_((m.bias), 0)   

def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def nms(featmap, kernel=7):
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(featmap, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == featmap)
    return featmap * keep.float(), keep

##############################Graph Convolutional Network#######################################
class PointConvDensityClsSsg(nn.Module):
    def __init__(self, num_classes=2, num_pt=256):
        super(PointConvDensityClsSsg, self).__init__()
        self.sa1 = PointConvDensitySetAbstraction(npoint=num_pt, nsample=8, in_channel=64 + 2, mlp=[64, 128],
                                                  bandwidth=0.1, group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(npoint=num_pt, nsample=8, in_channel=128 + 2, mlp=[128, 64],
                                                  bandwidth=0.2, group_all=False)
        self.sa3 = PointConvDensitySetAbstraction(npoint=num_pt, nsample=8, in_channel=64 + 2, mlp=[64, num_classes],
                                                  bandwidth=0.4, group_all=False)

    def forward(self, xyz, feat):
        B, _, _ = xyz.shape
        l1_xyz, l1_points = self.sa1(xyz, feat)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        return l3_points


##############################Attention Network#######################################
class ChannelWeightLayer(nn.Module):
    def __init__(self):
        super(ChannelWeightLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(3, 3, bias=False),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ChannelWeightLayer1(nn.Module):
    def __init__(self):
        super(ChannelWeightLayer1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(6, 6, bias=False),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SpatialWeightLayer(nn.Module):
    def __init__(self):
        super(SpatialWeightLayer, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
