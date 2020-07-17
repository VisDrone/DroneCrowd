import torch.nn as nn
import torch
from torchvision import models
from dcn.dcn_v2 import DCN

##############################VGG16 Backbone Network#######################################
class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.backbone1 = make_layers([64, 64, 'M', 128, 128])
        self.backbone2 = make_layers(['M', 256, 256, 256], in_channels = 128)
        self.backbone3 = make_layers(['M', 512, 512, 512], in_channels = 256)

    def forward(self, x):
        x1 = self.backbone1(x)
        x2 = self.backbone2(x1)
        x3 = self.backbone3(x2)

        return x1, x2, x3

def make_layers(cfg, in_channels = 3, batch_norm=False):
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

##############################Space-Time Multi-Scale Attention Network#######################################
class STANet(nn.Module):
    def __init__(self, use_loc=False, use_trk=False):
        super(STANet, self).__init__()
        self.seen = 0
        self.loc = use_loc
        self.trk = use_trk        
        # backbone
        self.backbone = BaseNet()
        self.refine = DCN(512,512,kernel_size=(3,3),stride=1,padding=1,dilation=1,deformable_groups=2)
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
        self.c_weight_den = ChannelWeightLayer()
        self.s_weight_den = SpatialWeightLayer()
        self.merge_den = nn.Conv2d(6, 2, 3, padding=1) 
        self.den_output_layer1 = nn.Conv2d(128, 2, kernel_size=1)
        self.den_output_layer2 = nn.Conv2d(256, 2, kernel_size=1)
        self.den_output_layer3 = nn.Conv2d(512, 2, kernel_size=1)
        # localization output layers
        if self.loc:
            self.c_weight_loc = ChannelWeightLayer()
            self.s_weight_loc = SpatialWeightLayer()
            self.merge_loc = nn.Conv2d(6, 2, 3, padding=1)
            self.loc_output_layer1 = nn.Conv2d(128, 2, 3, padding=1, dilation=1)
            self.loc_output_layer2 = nn.Conv2d(256, 2, 3, padding=1, dilation=1)
            self.loc_output_layer3 = nn.Conv2d(512, 2, 3, padding=1, dilation=1)
        # association output layers
        if self.trk:
            self.trk_output_layer1 = nn.Conv2d(128, 16, 3, padding=1, dilation=1)
            self.trk_norm = nn.BatchNorm2d(16)
        # load weights of the backbone network
        vgg16 = models.vgg16(pretrained = True)       
        self._initialize_weights()    
        my_models = self.backbone.state_dict()
        pre_models = vgg16.state_dict().items()
        count = 0
        for layer_name, value in my_models.items():
            prelayer_name, pre_weights = pre_models[count]
            my_models[layer_name] = pre_weights
            count += 1  
        self.backbone.load_state_dict(my_models)

    def forward(self, img1, img2):  
        # combine features    
        feat1_1, feat2_1, feat3_1 = self.backbone(img1)
        feat1_2, feat2_2, feat3_2 = self.backbone(img2)
        g1_size = (feat1_1.shape[2], feat1_1.shape[3])
        g2_size = (feat2_1.shape[2], feat2_1.shape[3])

        f1 = feat1_1 + feat1_2
        f2 = feat2_1 + feat2_2
        f3 = feat3_1 + feat3_2

        f3 = self.refine(f3)
        f3 = self.relu(f3)

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
        
        base_g1 = self.s_weight1(g1) 
        base_g2 = self.s_weight2(g2)
        base_g3 = self.s_weight3(f3) 

        # generate density map
        den_g1 = self.den_output_layer1(base_g1) #x2 resolution
        den_g2 = self.den_output_layer2(base_g2) #x4 resolution
        den_g3 = self.den_output_layer3(base_g3) #x8 resolution
        # attention for final density map
        den_g4 = torch.cat((den_g1, nn.Upsample(size=g1_size, mode='bilinear', align_corners=True)(den_g2/4.), nn.Upsample(size=g1_size, mode='bilinear', align_corners=True)(den_g3/16.)), 1)
        den_g4 = self.c_weight_den(den_g4) 
        den_g4 = self.s_weight_den(den_g4)
        den_g4 = self.merge_den(den_g4)

        # generate localization map
        if self.loc:
            loc_g1 = self.loc_output_layer1(base_g1) #x2 resolution
            loc_g2 = self.loc_output_layer2(base_g2) #x4 resolution
            loc_g3 = self.loc_output_layer3(base_g3) #x8 resolution
            loc_g4 = torch.cat((loc_g1, nn.Upsample(size=g1_size, mode='bilinear', align_corners=True)(loc_g2/4.), nn.Upsample(size=g1_size, mode='bilinear', align_corners=True)(loc_g3/16.)), 1)
            loc_g4 = self.c_weight_loc(loc_g4) 
            loc_g4 = self.s_weight_loc(loc_g4)
            loc_g4 = self.merge_loc(loc_g4)
        else:
            loc_g1,loc_g2,loc_g3,loc_g4 = [],[],[],[]
        # generate tracking map
        if self.trk:
            trk_g41 = self.trk_output_layer1(feat1_1)
            trk_g41 = self.trk_norm(trk_g41)
            trk_g42 = self.trk_output_layer1(feat1_2)
            trk_g42 = self.trk_norm(trk_g42)
            trk_g4 = torch.cat((trk_g41, trk_g42), 1)
        else:
            trk_g4 = []

        return den_g1, den_g2, den_g3, den_g4, loc_g1, loc_g2, loc_g3, loc_g4, trk_g4

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

##############################Attention Network#######################################
class ChannelWeightLayer(nn.Module):
    def __init__(self):
        super(ChannelWeightLayer, self).__init__()
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
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = nn.functional.sigmoid(x_out) # broadcasting
        return x * scale

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
