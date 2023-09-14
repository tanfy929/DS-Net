# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as  F
from My_module import MyModule

torch.backends.cudnn.benchmark=False
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
        self.Lsegnet = Lsegnet(3)
        self.LsegnetPlusList_temp = [LsegnetPlus(inDim=3, stageNum=i) for i in range(HSInetL+1)]
        self.LsegnetPlusList = nn.Sequential(*self.LsegnetPlusList_temp)
        self.etaList = self.eta_stage(HSInetL+1, torch.Tensor([0.001]))
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
        sizeX = X.shape
        sizeM = [sizeX[0], self.num_M, sizeX[2], sizeX[3]]

        ListZ = []
        ListTemp = []
        ListCM = []
        ListM = []

        Z, Ztemp = self.Lsegnet(X)

        ListZ.append(Z)
        ListTemp.append(Ztemp)
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
        for i in range(self.HSInetL):
            CM = F.conv2d(M,self.C,stride=1,padding=9//2)
            ListCM.append(CM)
            ListM.append(M)
            Z,Ztemp = self.LsegnetPlusList[i](X-CM,ListZ)
            ListZ.append(Z)
            ListTemp.append(Ztemp)
            if ifpre < 0.5:
                Z1 = Z0
            else:
                Z1 = Z
            W = torch.sum(Z1 / (2 * self.sig + 0.0001),dim=1)
            W = torch.unsqueeze(W, dim=1)
            B = torch.sum(torch.unsqueeze(Z1, dim=2) * mu / torch.unsqueeze((2 * self.sig + 0.0001),dim=2), dim=1)
            B = B / (W + 0.0000001)

            E = W * (CM + B - X)
            G = F.conv_transpose2d(E, self.C, stride=1, padding=9//2, output_padding=0)
            M = self.resCNNetPlus(M - self.etaList[i+1, :]/10.0 * G)
        CM = F.conv2d(M, self.C,stride=1, padding=9//2)
        ListCM.append(CM)
        ListM.append(M)
        Z,Ztemp = self.LsegnetPlusList[self.HSInetL](X-CM,ListZ)

        W = torch.sum(Z / (2 * self.sig + 0.0001), dim=1)
        W = torch.unsqueeze(W, dim=1)
        B = torch.sum(torch.unsqueeze(Z, dim=2) * mu / torch.unsqueeze((2 * self.sig + 0.0001), dim=2), dim=1)
        B = B / (W + 0.0000001)

        return Z, ListZ, Ztemp, ListTemp, ListCM, ListM, self.sig, mu, W, B, self.C


def GetIniC(numM, Csize = 9):
    C = (np.random.rand(3,numM,Csize,Csize)-0.5)*2*2.4495/np.sqrt(numM*Csize*Csize)
    return torch.FloatTensor(C)


def lossFuncionCE(Zout,Ztemp,Z):
    loss1 = 0
    loss2 = 0
    loss3 = 0
    loss4 = 0
    loss5 = 0
    loss6 = 0
    weit = [10, 1, 1, 1, 1, 1]
    losslist = [loss1, loss2, loss3, loss4, loss5, loss6]
    # 背景，MA，HE，EX，SE
    beta = [2, 4, 1, 1, 4]
    beta_n = [5, 5, 6, 6, 5.5]
    eps1 = [0.0000001, 0.0000001, 0.0000001, 0.0000001, 0.0000001]
    ds = [Zout, Ztemp[0],Ztemp[1],Ztemp[2],Ztemp[3],Ztemp[4]]

    for j in range(6):
        for i in range(5):  # 五类病灶
            losslist[j] = losslist[j] - torch.mean(beta[i] * Z[:, i, :, :] * torch.log(ds[j][:, i, :, :] + eps1[i])
                                                   + beta_n[i] * (1 - Z[:, i, :, :]) * torch.log(1 - ds[j][:, i, :, :] + eps1[i]))
    loss = 0
    # losslist[]分别存了Z，五个Ztemp的loss
    for i in range(6):
        loss = weit[i] * torch.mean(losslist[i]) + loss
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


class ConLevel(nn.Module):
    def __init__(self,Fsize,inC,outC):
        super(ConLevel,self).__init__()
        self.Fsize = Fsize
        self.inC = inC
        self.outC = outC
        self.conv = nn.Conv2d(self.inC, self.outC, kernel_size=self.Fsize, stride=1, padding=self.Fsize//2, bias=True)
        self.MyBN = MyBN(outC)
        self.ReLU = nn.ReLU()
    def forward(self,X):
        conv = self.conv(X)
        X = self.MyBN(conv)
        X = self.ReLU(X)
        return X


class UpsampleNet(nn.Module):
    def __init__(self, scale, indim, outdim):
        super(UpsampleNet, self).__init__()
        self.scale = scale
        self.indim = indim
        self.outdim = outdim
        self.conv = nn.Conv2d(indim, outdim, kernel_size=1, stride=1)
        self.up = torch.nn.ConvTranspose2d(outdim,outdim,kernel_size=scale*2,stride=scale,padding=scale//2)

    def forward(self, x):
        x = self.conv(x)
        x = self.up(x)
        return x

class UpsampleNet_add(nn.Module):
    def __init__(self, scale, indim, outdim):
        super(UpsampleNet_add, self).__init__()
        self.scale = scale
        self.indim = indim
        self.outdim = outdim
        self.conv = nn.Sequential(
                                  nn.Conv2d(indim, outdim, kernel_size=1, stride=1),
                                  nn.BatchNorm2d(outdim,affine=True),
                                  nn.ReLU(inplace=True)
                                 )
        self.up = torch.nn.ConvTranspose2d(outdim,outdim,kernel_size=scale*2,stride=scale,padding=scale//2)

    def forward(self, x):
        x = self.conv(x)
        x = self.up(x)
        return x


class Lsegnet(nn.Module):
    def __init__(self,inDim):
        super(Lsegnet,self).__init__()
        self.inDim = inDim
        self.ConLevel11 = ConLevel(3,inDim,64)
        self.ConLevel12 = ConLevel(3,64,64)
        self.ConLevel21 = ConLevel(3,64,128)
        self.ConLevel22 = ConLevel(3,128,128)
        self.ConLevel31 = ConLevel(3,128,256)
        self.ConLevel32 = ConLevel(3,256,256)
        self.ConLevel33 = ConLevel(3,256,256)
        self.ConLevel41 = ConLevel(3,256,512)
        self.ConLevel42 = ConLevel(3,512,512)
        self.ConLevel43 = ConLevel(3,512,512)
        self.ConLevel51 = ConLevel(3,512,512)
        self.ConLevel52 = ConLevel(3,512,512)
        self.ConLevel53 = ConLevel(3,512,512)

        self.up1 = nn.Sequential(
                                 nn.Conv2d(64, 5, 1, stride=1, bias=True),
                                 nn.BatchNorm2d(5,affine=True),
                                 nn.Softmax2d()
                                 )
        self.up2 = nn.Sequential(
                                 UpsampleNet_add(2, 128, 5),
                                 nn.Softmax2d()
                                 )
        self.up3 = nn.Sequential(
                                 UpsampleNet_add(4, 256, 5),
                                 nn.Softmax2d()
                                 )
        self.up4 = nn.Sequential(
                                 UpsampleNet_add(8, 512, 5),
                                 nn.Softmax2d()
                                 )
        self.up5 = nn.Sequential(
                                 UpsampleNet_add(16, 512, 5),
                                 nn.Softmax2d()
                                 )
        self.ConLevel111 = ConLevel(1, 5, 5)
        self.ConLevel112 = ConLevel(1, 5, 5)
        self.ConLevel113 = ConLevel(1, 5, 5)
        self.ConLevel114 = ConLevel(1, 5, 5)
        self.ConLevel115 = ConLevel(1, 5, 5)

        self.ConLevel1 = ConLevel(1, 5, 1)
        self.ConLevel2 = ConLevel(1, 5, 1)
        self.ConLevel3 = ConLevel(1, 5, 1)
        self.ConLevel4 = ConLevel(1, 5, 1)
        self.ConLevel5 = ConLevel(1, 5, 1)


    def forward(self,X):
        conv1_1 = self.ConLevel11(X)
        conv1_2 = self.ConLevel12(conv1_1)
        pool1 = nn.MaxPool2d(2, 2, padding = (2-1)//2)(conv1_2)
        conv2_1 = self.ConLevel21(pool1)
        conv2_2 = self.ConLevel22(conv2_1)
        pool2 = nn.MaxPool2d(2, 2,padding=(2-1)//2)(conv2_2)
        conv3_1 = self.ConLevel31(pool2)
        conv3_2 = self.ConLevel32(conv3_1)
        conv3_3 = self.ConLevel33(conv3_2)
        pool3 = nn.MaxPool2d(2, 2,padding=(2-1)//2)(conv3_3)
        conv4_1 = self.ConLevel41(pool3)
        conv4_2 = self.ConLevel42(conv4_1)
        conv4_3 = self.ConLevel43(conv4_2)
        pool4 = nn.MaxPool2d(2, 2,padding=(2-1)//2)(conv4_3)
        conv5_1 = self.ConLevel51(pool4)
        conv5_2 = self.ConLevel52(conv5_1)
        conv5_3 = self.ConLevel53(conv5_2)
        Ztemp = []
        Ztemp.append(self.up1(conv1_2))
        Ztemp.append(self.up2(conv2_2))
        Ztemp.append(self.up3(conv3_3))
        Ztemp.append(self.up4(conv4_3))
        Ztemp.append(self.up5(conv5_3))

        dsnfin = []
        for i in range(5): # 五个通道
            dsn = torch.unsqueeze(Ztemp[0][:, i, :, :], 1)
            for j in range(4):
                dsn = torch.cat([dsn, torch.unsqueeze(Ztemp[j + 1][:, i, :, :], 1)], 1)  # 融合五个边缘特征的病灶通道维
            dsnfin.append(dsn)
        dsnfin1 = self.ConLevel111(dsnfin[0])
        dsnfin2 = self.ConLevel112(dsnfin[1])
        dsnfin3 = self.ConLevel113(dsnfin[2])
        dsnfin4 = self.ConLevel114(dsnfin[3])
        dsnfin5 = self.ConLevel115(dsnfin[4])

        dsnfin1 = self.ConLevel1(dsnfin1)
        dsnfin2 = self.ConLevel2(dsnfin2)
        dsnfin3 = self.ConLevel3(dsnfin3)
        dsnfin4 = self.ConLevel4(dsnfin4)
        dsnfin5 = self.ConLevel5(dsnfin5)

        dsnfin = torch.cat([dsnfin1, dsnfin2, dsnfin3, dsnfin4, dsnfin5], 1)
        dsnfin = nn.Softmax2d()(dsnfin)

        return dsnfin, Ztemp


class LsegnetPlus(MyModule):
    def __init__(self,inDim,stageNum):
        super(LsegnetPlus, self).__init__()
        self.inDim = inDim
        self.stageNum = stageNum
        self.ConLevel11 = ConLevel(3, inDim, 64)
        self.ConLevel12 = ConLevel(3, 64, 64)
        self.ConLevel21 = ConLevel(3, 64, 128)
        self.ConLevel22 = ConLevel(3, 128, 128)
        self.ConLevel31 = ConLevel(3, 128, 256)
        self.ConLevel32 = ConLevel(3, 256, 256)
        self.ConLevel33 = ConLevel(3, 256, 256)
        self.ConLevel41 = ConLevel(3, 256, 512)
        self.ConLevel42 = ConLevel(3, 512, 512)
        self.ConLevel43 = ConLevel(3, 512, 512)
        self.ConLevel51 = ConLevel(3, 512, 512)
        self.ConLevel52 = ConLevel(3, 512, 512)
        self.ConLevel53 = ConLevel(3, 512, 512)


        self.up1 = nn.Sequential(
                                 nn.Conv2d(64, 5, 1, stride=1),
                                 nn.BatchNorm2d(5,affine=True),
                                 nn.Softmax2d()
                                 )
        self.up2 = nn.Sequential(
                                 UpsampleNet_add(2, 128, 5),
                                 nn.Softmax2d()
                                 )
        self.up3 = nn.Sequential(
                                 UpsampleNet_add(4, 256, 5),
                                 nn.Softmax2d()
                                 )
        self.up4 = nn.Sequential(
                                 UpsampleNet_add(8, 512, 5),
                                 nn.Softmax2d()
                                 )
        self.up5 = nn.Sequential(
                                 UpsampleNet_add(16, 512, 5),
                                 nn.Softmax2d()
                                 )
        self.ConLevel111 = ConLevel(1, stageNum + 5 + 1, 10)
        self.ConLevel112 = ConLevel(1, stageNum + 5 + 1, 10)
        self.ConLevel113 = ConLevel(1, stageNum + 5 + 1, 10)
        self.ConLevel114 = ConLevel(1, stageNum + 5 + 1, 10)
        self.ConLevel115 = ConLevel(1, stageNum + 5 + 1, 10)

        self.ConLevel1 = ConLevel(1, 10, 1)
        self.ConLevel2 = ConLevel(1, 10, 1)
        self.ConLevel3 = ConLevel(1, 10, 1)
        self.ConLevel4 = ConLevel(1, 10, 1)
        self.ConLevel5 = ConLevel(1, 10, 1)

    def forward(self, X, ZList):
        conv1_1 = self.ConLevel11(X)
        conv1_2 = self.ConLevel12(conv1_1)
        pool1 = nn.MaxPool2d(3, 2, padding=(3-1)//2)(conv1_2)
        conv2_1 = self.ConLevel21(pool1)
        conv2_2 = self.ConLevel22(conv2_1)
        pool2 = nn.MaxPool2d(3, 2, padding=(3-1)//2)(conv2_2)
        conv3_1 = self.ConLevel31(pool2)
        conv3_2 = self.ConLevel32(conv3_1)
        conv3_3 = self.ConLevel33(conv3_2)
        pool3 = nn.MaxPool2d(3, 2, padding=(3-1)//2)(conv3_3)
        conv4_1 = self.ConLevel41(pool3)
        conv4_2 = self.ConLevel42(conv4_1)
        conv4_3 = self.ConLevel43(conv4_2)
        pool4 = nn.MaxPool2d(3, 2, padding=(3-1)//2)(conv4_3)
        conv5_1 = self.ConLevel51(pool4)
        conv5_2 = self.ConLevel52(conv5_1)
        conv5_3 = self.ConLevel53(conv5_2)

        Ztemp = []
        Ztemp.append(self.up1(conv1_2))
        Ztemp.append(self.up2(conv2_2))
        Ztemp.append(self.up3(conv3_3))
        Ztemp.append(self.up4(conv4_3))
        Ztemp.append(self.up5(conv5_3))

        dsnfin = []
        for i in range(5): # 五个通道
            dsn = torch.unsqueeze(ZList[0][:, i, :, :], 1)
            for k in range(self.stageNum):
                dsn = torch.cat([dsn, torch.unsqueeze(ZList[k + 1][:, i, :, :], 1)], 1)
            for j in range(5):
                dsn = torch.cat([dsn,torch.unsqueeze(Ztemp[j][:, i, :, :], 1)], 1)
            dsnfin.append(dsn)

        dsnfin1 = self.ConLevel111(dsnfin[0])
        dsnfin2 = self.ConLevel112(dsnfin[1])
        dsnfin3 = self.ConLevel113(dsnfin[2])
        dsnfin4 = self.ConLevel114(dsnfin[3])
        dsnfin5 = self.ConLevel115(dsnfin[4])

        dsnfin1 = self.ConLevel1(dsnfin1)
        dsnfin2 = self.ConLevel2(dsnfin2)
        dsnfin3 = self.ConLevel3(dsnfin3)
        dsnfin4 = self.ConLevel4(dsnfin4)
        dsnfin5 = self.ConLevel5(dsnfin5)

        dsnfin = torch.cat([dsnfin1, dsnfin2, dsnfin3, dsnfin4, dsnfin5], 1)
        dsnfin = nn.Softmax2d()(dsnfin)

        return dsnfin, Ztemp


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


class resCNNnet(nn.Module):
    def __init__(self,channel,levelN):
        super(resCNNnet,self).__init__()
        self.channel = channel
        self.levelN = levelN
        self.resLevelList_temp = [resLevel(9,channel) for _ in range(self.levelN - 1)]
        self.resLevelList = nn.Sequential(*self.resLevelList_temp)
        # self.resLevel = resLevel(9,channel)
        self.reduceM = reduceM(channel)

    def forward(self,X):
        for i in range(self.levelN - 1):
            X = self.resLevelList[i](X)
            X =self.reduceM(X)
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
            X =self.ThroM(X)
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

