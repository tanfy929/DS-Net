import numpy as np
import torch
import torch.nn as nn
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
        self.sig = nn.Parameter(torch.tensor(inisig), requires_grad=True)  # sig初始大小为[1,5,1,1]

        self.resCNNet = resCNNnet(num_M,subnetL)
        self.resCNNetPlus = resCNNnetPlus(num_M,subnetL)

        self.UNetplus = UNetplus(3)
        self.UNetplus1 = UNetplus(3)
        self.UNetplus2 = UNetplus(3)
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

        Z = self.UNetplus(X)
        ListZ.append(Z)
        if ifpre < 0.5:
            Z1 = Z0
        else:
            Z1 = Z[3]
        W = torch.sum(Z1 / (2 * self.sig + 0.0001), dim = 1)
        W = torch.unsqueeze(W, dim = 1)
        mu = self.mu / 100
        B = torch.sum(torch.unsqueeze(Z1, dim=2) * mu / torch.unsqueeze((2 * self.sig + 0.0001), dim=2), dim=1)
        B = B / (W + 0.0000001)
        E = (B - X) # M初始化为0
        # 实现卷积的转置
        G = F.conv_transpose2d(E,self.C_0,stride=1,padding=9//2,output_padding = 0)
        M = self.resCNNet(-self.etaList[0,:]/10.0 * G)
        CM = F.conv2d(M, self.C, stride=1, padding=9 // 2)
        ListCM.append(CM)
        ListM.append(M)
        Z = self.UNetplus1(X - CM)
        ListZ.append(Z)
        if ifpre < 0.5:
            Z1 = Z0
        else:
            Z1 = Z[3]
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
        Z = self.UNetplus2(X - CM)

        return Z, ListZ, ListCM, ListM, self.sig, mu, W, B, self.C

def GetIniC(numM, Csize = 9):
    C = (np.random.rand(3,numM,Csize,Csize)-0.5)*2*2.4495/np.sqrt(numM*Csize*Csize)
    return torch.FloatTensor(C)

class UNetplus(nn.Module):
    def __init__(self,inDim):
        super(UNetplus,self).__init__()
        self.inDim = inDim
        self.ReLU = nn.ReLU()
        # ConLevel(Fsize,inC,outC)
        self.ConLevel00 = ConLevel(3, inDim, 32)
        self.ConLevel001 = ConLevel(3,32,32)
        self.ConLevel10 = ConLevel(3,32,64)
        self.ConLevel20 = ConLevel(3,64,128)
        self.ConLevel201 = ConLevel(3,128,128)
        self.ConLevel30 = ConLevel(3,128,256)
        self.ConLevel301 = ConLevel(3,256,256)
        self.ConLevel40 = ConLevel(3,256,512)
        self.ConLevel401 = ConLevel(3,512,512)
        self.upsample10 = Upsample(2,64)
        self.ConLevel01 = ConLevel(3,32*2,32)
        self.ConLevelout1 = ConLevel(3,32,5)
        self.upsample20 = Upsample(2,128)
        self.ConLevel11 = ConLevel(3,64*2,64)
        self.upsample11 = Upsample(2,64)
        self.ConLevel02 = ConLevel(3,32*3,32)
        self.ConLevelout2 = ConLevel(3,32,5)
        self.upsample30 = Upsample(2,256)
        self.ConLevel21 = ConLevel(3,128*2,128)
        self.upsample21 = Upsample(2,128)
        self.ConLevel12 = ConLevel(3,64*3,64)
        self.upsample12 = Upsample(2,64)
        self.ConLevel03 = ConLevel(3,32*4,32)
        self.ConLevelout3 = ConLevel(3,32,5)
        self.upsample40 = Upsample(2,512)
        self.ConLevel31 = ConLevel(3,256*2,256)
        self.upsample31 = Upsample(2,256)
        self.ConLevel22 = ConLevel(3,128*3,128)
        self.upsample22 = Upsample(2,128)
        self.ConLevel13 = ConLevel(3,64*4,64)
        self.upsample13 = Upsample(2,64)
        self.ConLevel04 = ConLevel(3,32*5,32)
        self.ConLevelout4 = ConLevel(3,32,5)
    def forward(self,X):
        output = []
        x_00 = self.ConLevel00(X)
        x_00 = self.ConLevel001(x_00)
        x_10 = nn.MaxPool2d(2, 2, padding=(2 - 1) // 2)(x_00)
        x_10 = self.ConLevel10(x_10)
        x_20 = nn.MaxPool2d(2, 2, padding=(2 - 1) // 2)(x_10)
        x_20 = self.ConLevel20(x_20)
        x_20 = self.ConLevel201(x_20)

        x_30 = nn.MaxPool2d(2, 2, padding=(2 - 1) // 2)(x_20)
        x_30 = self.ConLevel30(x_30)
        x_30 = self.ConLevel301(x_30)

        x_40 = nn.MaxPool2d(2, 2, padding=(2 - 1) // 2)(x_30)
        x_40 = self.ConLevel40(x_40)
        x_40 = self.ConLevel401(x_40)

        up10 = self.upsample10(x_10)
        x_01 = self.ConLevel01(torch.cat([x_00,up10],1))
        output1 = self.ConLevelout1(x_01)
        output1 = nn.Softmax2d()(output1)

        up20 = self.upsample20(x_20)
        x_11 = self.ConLevel11(torch.cat([x_10,up20],1))
        up11 = self.upsample11(x_11)
        x_02 = self.ConLevel02(torch.cat([x_00,x_01,up11],1))
        output2 = self.ConLevelout2(x_02)
        output2 = nn.Softmax2d()(output2)

        up30 = self.upsample30(x_30)
        x_21 = self.ConLevel21(torch.cat([x_20,up30],1))
        up21 = self.upsample21(x_21)
        x_12 = self.ConLevel12(torch.cat([x_10,x_11,up21],1))
        up12 = self.upsample12(x_12)
        x_03 = self.ConLevel03(torch.cat([x_00,x_01,x_02,up12],1))
        output3 = self.ConLevelout3(x_03)
        output3 = nn.Softmax2d()(output3)

        up40 = self.upsample40(x_40)
        x_31 = self.ConLevel31(torch.cat([x_30,up40],1))
        up31 = self.upsample31(x_31)
        x_22 = self.ConLevel22(torch.cat([x_20,x_21,up31],1))
        up22 = self.upsample22(x_22)
        x_13 = self.ConLevel13(torch.cat((x_10,x_11,x_12,up22),1))
        up13 = self.upsample13(x_13)
        x_04 = self.ConLevel04(torch.cat([x_00,x_01,x_02,x_03,up13],1))
        output4 = self.ConLevelout4(x_04)
        output4 = nn.Softmax2d()(output4)

        output.append(output1)
        output.append(output2)
        output.append(output3)
        output.append(output4)
        return output


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

class Upsample(nn.Module):
    def __init__(self, scale, indim):
        super(Upsample, self).__init__()
        self.scale = scale
        self.indim = indim
        self.up = nn.Sequential(torch.nn.ConvTranspose2d(indim,indim//2,kernel_size=scale*2,stride=scale,padding=scale//2),
                                nn.ReLU())
    def forward(self, x):
        x = self.up(x)
        return x

def lossFuncionCE(Zout,Z):
    loss1 = 0
    loss2 = 0
    loss3 = 0
    loss4 = 0
    weit = [1, 1, 1, 10]
    losslist = [loss1, loss2, loss3, loss4]
    beta = [1,1,1,1,1]
    beta_n = [1,1,1,1,1]
    eps1 = [0.0000001, 0.0000001, 0.0000001, 0.0000001, 0.0000001]
    ds = [Zout[0], Zout[1], Zout[2], Zout[3]]

    for j in range(4):
        for i in range(5):  # 五类病灶
            losslist[j] = losslist[j] - torch.mean(beta[i] * Z[:, i, :, :] * torch.log(ds[j][:, i, :, :] + eps1[i])
                                                   + beta_n[i] * (1 - Z[:, i, :, :]) * torch.log(1 - ds[j][:, i, :, :] + eps1[i]))
    loss = 0
    # losslist[]分别存了Z，五个Ztemp的loss
    for i in range(4):
        loss = weit[i] * torch.mean(losslist[i]) + loss
    return loss

class MyBN(MyModule):
    def __init__(self, Channel, eps=1.0e-5):
        super(MyBN, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, Channel, 1, 1), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(1, Channel, 1, 1), requires_grad=True)
        self.orlBN = nn.BatchNorm2d(Channel, affine=True) # affine代表gamma(*)，beta(+)是否可学
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

