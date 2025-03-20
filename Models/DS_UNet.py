# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from My_module import MyModule

torch.backends.cudnn.benchmark=True
torch.backends.cudnn.deterministic=True

class preEMnet(MyModule):
    def __init__(self, subnetL, num_M, inimu, inisig):
        super(preEMnet,self).__init__()
        self.subnetL = subnetL
        self.num_M = num_M
        self.mu = nn.Parameter(torch.tensor(inimu*100), requires_grad=True)
        self.sig = nn.Parameter(torch.tensor(inisig), requires_grad=True)
        self.resCNNet = resCNNnet(num_M,subnetL)
        self.resCNNetPlus = resCNNnetPlus(num_M,subnetL)
        self.UNet = UNet(3)
        self.UNet1 = UNet(3)
        self.UNet2 = UNet(3)
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
        Z = self.UNet(X)
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
        Z = self.UNet1(X - CM)
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
        Z = self.UNet2(X - CM)
        return Z, ListZ, ListCM, ListM, self.sig, mu, W, B, self.C

def GetIniC(numM, Csize = 9):
    C = (np.random.rand(3,numM,Csize,Csize)-0.5)*2*2.4495/np.sqrt(numM*Csize*Csize)
    return torch.FloatTensor(C)

class UNet(nn.Module):
    def __init__(self,inDim):
        super(UNet,self).__init__()
        self.inDim = inDim
        self.ReLU = nn.ReLU()
        self.ConLevel11 = ConLevel(3, inDim, 32)
        self.ConLevel12 = ConLevel(3, 32, 32)
        self.ConLevel21 = ConLevel(3, 32, 64)
        self.ConLevel22 = ConLevel(3, 64, 64)
        self.ConLevel31 = ConLevel(3, 64, 128)
        self.ConLevel32 = ConLevel(3, 128, 128)
        self.ConLevel41 = ConLevel(3, 128, 256)
        self.ConLevel42 = ConLevel(3, 256, 256)
        self.ConLevel51 = ConLevel(3, 256, 512)
        self.ConLevel52 = ConLevel(3, 512, 512)

        self.up1 = U_up(2,512)
        self.C61 = ConLevel(3,512,256)
        self.C62 = ConLevel(3,256,256)
        self.up2 = U_up(2, 256)
        self.C71 = ConLevel(3, 256, 128)
        self.C72 = ConLevel(3, 128, 128)
        self.up3 = U_up(2, 128)
        self.C81 = ConLevel(3, 128, 64)
        self.C82 = ConLevel(3, 64, 64)
        self.up4 = U_up(2, 64)
        self.C91 = ConLevel(3, 64, 32)
        self.C92 = ConLevel(3, 32, 32)

        self.pred = ConLevel(1,32,5)
    def forward(self, X):
        conv1_1 = self.ConLevel11(X)
        conv1_2 = self.ConLevel12(conv1_1)
        pool1 = nn.MaxPool2d(2,2,padding=(2-1)//2)(conv1_2)

        conv2_1 = self.ConLevel21(pool1)
        conv2_2 = self.ConLevel22(conv2_1)
        pool2 = nn.MaxPool2d(2,2,padding=(2-1)//2)(conv2_2)

        conv3_1 = self.ConLevel31(pool2)
        conv3_2 = self.ConLevel32(conv3_1)
        pool3 = nn.MaxPool2d(2,2,padding=(2-1)//2)(conv3_2)

        conv4_1 = self.ConLevel41(pool3)
        conv4_2 = self.ConLevel42(conv4_1)
        pool4 = nn.MaxPool2d(2,2,padding=(2-1)//2)(conv4_2)

        conv5_1 = self.ConLevel51(pool4)
        conv5_2 = self.ConLevel52(conv5_1)

        # Upsample
        u1 = self.up1(conv5_2,conv4_2)
        u11 = self.C61(u1)
        u12 = self.C62(u11)

        u2 = self.up2(u12,conv3_2)
        u21 = self.C71(u2)
        u22 = self.C72(u21)

        u3 = self.up3(u22,conv2_2)
        u31 = self.C81(u3)
        u32 = self.C82(u31)

        u4 = self.up4(u32,conv1_2)
        u41 = self.C91(u4)
        u42 = self.C92(u41)

        Z = self.pred(u42)
        Z = nn.Softmax2d()(Z)
        return Z

class ConLevel(nn.Module):
    def __init__(self,Fsize,inC,outC):
        super(ConLevel,self).__init__()
        self.Fsize = Fsize
        self.inC = inC
        self.outC = outC
        self.conv = nn.Sequential(nn.Conv2d(self.inC, self.outC, kernel_size=self.Fsize, stride=1, padding=self.Fsize//2, bias=True),
                                  nn.BatchNorm2d(outC, affine=True),
                                  nn.ReLU(inplace=True)
                                 )
    def forward(self,X):
        conv = self.conv(X)
        return conv


class U_up(nn.Module):
    def __init__(self, scale, indim):
        super(U_up, self).__init__()
        self.scale = scale
        self.indim = indim
        self.up = nn.Sequential(torch.nn.ConvTranspose2d(indim,indim//2,kernel_size=scale*2,stride=scale,padding=scale//2),
                                nn.ReLU())
    def forward(self, x, y):
        x = self.up(x)
        z = torch.cat([x,y],1)
        return z

def lossFuncionCE(Zout,Z):
    beta = [2, 4, 2, 1.5, 4]
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
        self.blurC = nn.Parameter(torch.ones([2 * self.num_M, 1, 9, 9]) / 81, requires_grad=True)
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

