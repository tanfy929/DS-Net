# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from My_module import MyModule
from resnet_factory import get_resnet_backbone
from functools import partial

nonlinearity = partial(F.relu, inplace=True)


torch.backends.cudnn.benchmark=True
torch.backends.cudnn.deterministic=True


class preEMnet(MyModule):
    def __init__(self, HSInetL, subnetL, num_M, inimu, inisig):
        super(preEMnet,self).__init__()
        self.HSInetL = HSInetL
        self.subnetL = subnetL
        self.num_M = num_M
        self.mu = nn.Parameter(torch.tensor(inimu*100), requires_grad=True)
        self.sig = nn.Parameter(torch.tensor(inisig), requires_grad=True)
        self.resCNNet = resCNNnet(num_M,subnetL)
        self.resCNNetPlus = resCNNnetPlus(num_M,subnetL)

        self.CENet = CE_Net()
        self.CENet1 = CE_Net()
        self.CENet2 = CE_Net()
        self.etaList = self.eta_stage(2,torch.Tensor([0.001]))
        iniC = GetIniC(num_M)
        self.C_0 = nn.Parameter(torch.FloatTensor(iniC), requires_grad=True)
        iniC_1 = GetIniC(num_M)
        self.C = nn.Parameter(torch.FloatTensor(iniC_1), requires_grad=True)
    def eta_stage(self, iters, value):
        eta_t = value.unsqueeze(dim=0)
        eta = eta_t.expand(iters, -1)
        eta_out = nn.Parameter(data=eta, requires_grad = True)
        return eta_out

    def forward(self, X, Z0, ifpre=1):
        ListZ = []
        ListCM = []
        ListM = []

        Z = self.CENet(X)
        ListZ.append(Z)
        if ifpre < 0.5:
            Z1 = Z0
        else:
            Z1 = Z
        W = torch.sum(Z1 / (2 * self.sig + 0.0001), dim = 1)
        W = torch.unsqueeze(W, dim = 1)
        mu = self.mu / 100
        B = torch.sum(torch.unsqueeze(Z1, dim=2) * mu / torch.unsqueeze((2 * self.sig + 0.0001), dim=2), dim=1)
        B = B / (W + 0.0000001)
        E = (B - X)
        G = F.conv_transpose2d(E,self.C_0,stride=1,padding=9//2,output_padding = 0)
        M = self.resCNNet(-self.etaList[0,:]/10.0 * G)

        CM = F.conv2d(M, self.C, stride=1, padding=9 // 2)
        ListCM.append(CM)
        ListM.append(M)
        Z = self.CENet1(X - CM)
        ListZ.append(Z)
        if ifpre < 0.5:
            Z1 = Z0
        else:
            Z1 = Z
        W = torch.sum(Z1 / (2 * self.sig + 0.0001), dim=1)
        W = torch.unsqueeze(W, dim=1)
        B = torch.sum(torch.unsqueeze(Z1, dim=2) * mu / torch.unsqueeze((2 * self.sig + 0.0001), dim=2), dim=1)
        B = B / (W + 0.0000001)

        E = W * (CM + B - X)
        G = F.conv_transpose2d(E, self.C, stride=1, padding=9 // 2, output_padding=0)
        M = self.resCNNetPlus(M - self.etaList[1, :] / 10.0 * G)
        CM = F.conv2d(M, self.C, stride=1, padding=9 // 2)
        ListCM.append(CM)
        ListM.append(M)
        Z = self.CENet2(X - CM)

        return Z, ListZ, ListCM, ListM, self.sig, mu, W, B, self.C



def GetIniC(numM, Csize = 9):
    C = (np.random.rand(3,numM,Csize,Csize)-0.5)*2*2.4495/np.sqrt(numM*Csize*Csize)
    return torch.FloatTensor(C)

class CE_Net(nn.Module):
    def __init__(self, num_classes=5, num_channels=3):
        super(CE_Net, self).__init__()
        filters = [64, 128, 256, 512]
        # resnet = models.resnet34(pretrained=True)
        resnet = get_resnet_backbone('resnet34')(pretrain=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = DACblock(512)
        self.spp = SPPblock(512)

        self.decoder4 = DecoderBlock(516, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        # Center
        e4 = self.dblock(e4)
        e4 = self.spp(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        out = nn.Softmax2d()(out)
        
        return out

class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # nonlinearity:F.relu
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out

class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
        self.layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
        self.layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
        self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w), mode='bilinear')

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

        return out

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


def lossFuncionCE(Zout,Z):
    # beta = [1, 1, 1, 1, 1]  # 背景1，MA2，HE1，EX2，SE1
    # beta_n = [1, 1, 1, 1, 1]
    beta = [2, 4, 1, 0.5, 4]
    beta_n = [5, 5, 6, 6.5, 5]
    eps1 = [0.0000001, 0.0000001, 0.0000001, 0.0000001, 0.0000001]
    loss = 0
    for i in range(5):  # 五类病灶
        loss = loss - torch.mean(beta[i] * Z[:, i, :, :] * torch.log(Zout[:, i, :, :] + eps1[i])
                                                   + beta_n[i] * (1 - Z[:, i, :, :]) * torch.log(1 - Zout[:, i, :, :] + eps1[i]))
    return loss


class MyBN(MyModule):
    def __init__(self, Channel, eps=1.0e-5):
        super(MyBN, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, Channel, 1, 1), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(1, Channel, 1, 1), requires_grad=True)
        self.orlBN = nn.BatchNorm2d(Channel, affine=True)
        self.feature_mean = 0
        self.feature_var = 1
        self.eps = eps

    def forward(self, feature):
        if not self.ft_BN:
            out = self.orlBN(feature)

            self.feature_mean = self.orlBN.running_mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            self.feature_var = self.orlBN.running_var.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        else:
            feature_normalized = (feature - self.feature_mean) / torch.sqrt(self.feature_var + self.eps)
            out = self.alpha * feature_normalized + self.beta
        return out

class resCNNnet(nn.Module):
    def __init__(self,channel,levelN):
        super(resCNNnet,self).__init__()
        self.channel = channel
        self.levelN = levelN
        self.resLevelList_temp = [resLevel(9,channel) for _ in range(self.levelN - 1)]
        self.resLevelList = nn.Sequential(*self.resLevelList_temp)
        self.reduceM = reduceM(channel)

    def forward(self,X):
        for i in range(self.levelN - 1):
            X = self.resLevelList[i](X)
            X = self.reduceM(X)
        return X

class resCNNnetPlus(nn.Module):
    def __init__(self,channel,levelN):
        super(resCNNnetPlus,self).__init__()
        self.channel = channel
        self.levelN = levelN
        self.resLevelList_temp = [resLevel(9, channel) for _ in range(self.levelN - 1)]
        self.resLevelList = nn.Sequential(*self.resLevelList_temp)
        self.ThroM = ThroM(channel)

    def forward(self,X):
        for i in range(self.levelN - 1):
            X = self.resLevelList[i](X)
            X = self.ThroM(X)
        return X

class resLevel(nn.Module):
    def __init__(self, Fsize, Channel):
        super(resLevel, self).__init__()
        self.conv_bn_1 = nn.Sequential(
                                        nn.Conv2d(Channel, Channel+3, kernel_size=Fsize, stride=1, padding=(Fsize-1)//2),
                                        MyBN(Channel+3),
                                        nn.ReLU(inplace=True),
                                        )
        self.conv_bn_2 = nn.Sequential(
                                        nn.Conv2d(Channel+3, Channel, kernel_size=Fsize, stride=1, padding=(Fsize-1)//2),
                                        MyBN(Channel),
                                        nn.ReLU(inplace=True),
                                        )

    def forward(self, X):
        X_1 = self.conv_bn_1(X)
        X_2 = self.conv_bn_2(X_1)
        X_out = X + X_2
        return X_out

class reduceM(nn.Module):
    def __init__(self, num_M):
        super(reduceM,self).__init__()
        self.num_M = num_M
        self.blurC = nn.Parameter(torch.ones([2 * self.num_M, 1, 9, 9]) / 81, requires_grad=True) # 高斯核
        self.Thro = nn.Parameter(torch.full(size=[1, self.num_M, 1, 1], fill_value=0.2),requires_grad=True)
    def forward(self,M):
        B, C, H, W = M.size()
        blurM = F.conv2d(M.view(1, B*self.num_M, H, W), self.blurC, stride=1, padding=9//2, groups=B*self.num_M).view(B, self.num_M, H, W)
        M = M - blurM
        M = M - self.Thro
        M = nn.ReLU(True)(M)
        return M

class ThroM(nn.Module):
    def __init__(self,num_M):
        super(ThroM, self).__init__()
        self.num_M = num_M
        self.Thro = nn.Parameter(torch.full(size=[1, num_M, 1, 1],fill_value=0.5),requires_grad = True)
    def forward(self,M):
        M = M - self.Thro
        M = nn.ReLU(True)(M)
        return M

