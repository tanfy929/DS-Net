import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from My_module import MyModule

from networks.vision_transformer import SwinUnet as ViT_seg
from config import get_config

torch.backends.cudnn.benchmark=True
torch.backends.cudnn.deterministic=True

# torch.manual_seed(3407)
# torch.cuda.manual_seed_all(3407)
# np.random.seed(3407)


class preEMnet(MyModule):
    def __init__(self, subnetL, num_M, inimu, inisig, opt):
        super(preEMnet,self).__init__()
        self.subnetL = subnetL
        self.num_M = num_M
        # self.mu = Variable(inimu * 100, requires_grad=True) # / 100
        self.mu = nn.Parameter(torch.tensor(inimu*100), requires_grad=True)
        # self.mu = nn.Parameter(mu, requires_grad=True) / 100 # inimu初始大小为[1,5,3,1,1]
        # self.sig = nn.Parameter(torch.tensor(inisig,dtype=torch.float32),requires_grad = True).cuda() # sig初始大小为[1,5,1,1]
        self.sig = nn.Parameter(torch.tensor(inisig), requires_grad=True)  # sig初始大小为[1,5,1,1]
        # self.sig = Variable(inisig, requires_grad=True)

        self.resCNNet = resCNNnet(num_M,subnetL,opt)
        self.resCNNetPlus = resCNNnetPlus(num_M,subnetL)

        config = get_config(opt)
        network = ViT_seg(config, img_size=opt.image_size, num_classes=5)
        network.load_from(config)

        self.SwinUNet = network
        self.SwinUNet1 = network
        self.SwinUNet2 = network
        # self.UNetPlusList_temp = [UNet(3) for _ in range(HSInetL+1)]
        # self.UNetPlusList = nn.Sequential(*self.UNetPlusList_temp)
        self.etaList = self.eta_stage(2,torch.Tensor([0.001])) # 这样eta才能更新
      
        iniC = GetIniC(num_M)
        self.C_0 = nn.Parameter(torch.FloatTensor(iniC), requires_grad=True)
        iniC_1 = GetIniC(num_M)
        self.C = nn.Parameter(torch.FloatTensor(iniC_1), requires_grad=True)
        # self.C = nn.Parameter(create_kernel(shape=[3, self.num_M, 9, 9]),requires_grad = True).cuda()
    def eta_stage(self, iters, value):
        eta_t = value.unsqueeze(dim=0)
        eta = eta_t.expand(iters, -1)
        eta_out = nn.Parameter(data=eta, requires_grad = True)
        return eta_out

    def forward(self, X, Z0, ifpre=1):
        ListZ = []
        ListCM = []
        ListM = []

        Z = self.SwinUNet(X) 
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
        E = (B - X) # M初始化为0
        # 实现卷积的转置
        G = F.conv_transpose2d(E,self.C_0,stride=1,padding=9//2,output_padding = 0) # padding参数不确定
        M = self.resCNNet(-self.etaList[0,:]/10.0 * G)

        CM = F.conv2d(M, self.C, stride=1, padding=9 // 2)  # (out_channels，in_channe/groups，H，W)
        ListCM.append(CM)
        ListM.append(M)
        Z = self.SwinUNet1(X - CM)
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
        G = F.conv_transpose2d(E, self.C, stride=1, padding=9 // 2, output_padding=0)  # padding参数不确定
        M = self.resCNNetPlus(M - self.etaList[1, :] / 10.0 * G)
        CM = F.conv2d(M, self.C, stride=1, padding=9 // 2)  # 改变了C的size
        ListCM.append(CM)
        ListM.append(M)
        Z = self.SwinUNet2(X - CM)

        return Z, ListZ, ListCM, ListM, self.sig, mu, W, B, self.C

def GetIniC(numM, Csize = 9):
    C = (np.random.rand(3,numM,Csize,Csize)-0.5)*2*2.4495/np.sqrt(numM*Csize*Csize)
    return torch.FloatTensor(C)


def lossFuncionCE(Zout,Z):
    beta = [1, 2, 1, 1.5, 1]  # 背景1，MA2，HE1，EX2，SE1
    beta_n = [1,1,1,1,1]
    # beta = [2, 4, 1, 0.5, 4]
    # beta_n = [5, 5, 6, 6.5, 5]
    eps1 = [0.0000001, 0.0000001, 0.0000001, 0.0000001, 0.0000001]
    loss = 0
    for i in range(5):  # 五类病灶
        loss = loss - torch.mean(beta[i] * Z[:, i, :, :] * torch.log(Zout[:, i, :, :] + eps1[i])
                                                   + beta_n[i] * (1 - Z[:, i, :, :]) * torch.log(1 - Zout[:, i, :, :] + eps1[i]))
    
    return loss


class MyBN(MyModule):
    def __init__(self, Channel, eps=1.0e-5): # 相比原版删除了参数pretrain
        super(MyBN, self).__init__()
        #        self.Pre_BN = pretrain
        self.alpha = nn.Parameter(torch.ones(1, Channel, 1, 1), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(1, Channel, 1, 1), requires_grad=True)
        # self.orlBN = nn.BatchNorm2d(Channel,affine=False)
        self.orlBN = nn.BatchNorm2d(Channel, affine=True) # affine代表gamma(*)，beta(+)是否可学
        self.feature_mean = 0
        self.feature_var = 1
        self.eps = eps

    def forward(self, feature):
        if not self.ft_BN:  # 普通BN
            # print("feature device",feature.device)
            # self.orlBN.running_mean = self.orlBN.running_mean
            # self.orlBN.running_var = self.orlBN.running_var
            out = self.orlBN(feature)

            self.feature_mean = self.orlBN.running_mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            self.feature_var = self.orlBN.running_var.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        #            feature_mean            = torch.mean
        #            feature_var             = self.orlBN.running_var.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        #            print(self.feature_mean)
        else:  # 不更新mean和var
            #            self.orlBN(feature)
            #            feature_mean       = self.orlBN.running_mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            #            feature_var        = self.orlBN.running_var.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            feature_normalized = (feature - self.feature_mean) / torch.sqrt(self.feature_var + self.eps)
            out = self.alpha * feature_normalized + self.beta
        #            print('training')
        return out

class resCNNnet(nn.Module):
    def __init__(self,channel,levelN,opt):
        super(resCNNnet,self).__init__()
        self.channel = channel
        self.levelN = levelN
        self.resLevelList_temp = [resLevel(9,channel) for _ in range(self.levelN - 1)]
        self.resLevelList = nn.Sequential(*self.resLevelList_temp)
        # self.resLevel = resLevel(9,channel)
        # self.reduceM = reduceM(channel,opt)

    def forward(self,X):
        for i in range(self.levelN - 1):
            X = self.resLevelList[i](X)
            # X = self.resLevel(X)
            # print('X',X.shape) # (2,10,304,304)
            # X = self.reduceM(X)
        return X

class resCNNnetPlus(nn.Module):
    def __init__(self,channel,levelN):
        super(resCNNnetPlus,self).__init__()
        self.channel = channel
        self.levelN = levelN
        # self.resLevelList = nn.ModuleList(resLevel(9, channel) for _ in range(self.levelN - 1)])    # 这样应该没问题，我以前一般照下面写
        self.resLevelList_temp = [resLevel(9, channel) for _ in range(self.levelN - 1)]  # 这样子好像网络是不更新的，不会注册到网络中
        self.resLevelList = nn.Sequential(*self.resLevelList_temp)
        # self.resLevel = resLevel(9,channel)
        self.ThroM = ThroM(channel)

    def forward(self,X):
        for i in range(self.levelN - 1):
            X = self.resLevelList[i](X)
            # X = self.resLevel(X)
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
    def __init__(self, num_M, opt):
        super(reduceM,self).__init__()
        self.num_M = num_M
        self.blurC = nn.Parameter(torch.ones([opt.batch_size * self.num_M, 1, 9, 9]) / 81, requires_grad=True) # 高斯核
        self.Thro = nn.Parameter(torch.full(size=[1, self.num_M, 1, 1], fill_value=0.2),requires_grad=True)
    def forward(self,M):
        B, C, H, W = M.size() # (2,10,304,304)
        # print('blurC', self.blurC.shape) # ([20, 1, 9, 9])
        blurM = F.conv2d(M.view(1, B*self.num_M, H, W), self.blurC, stride=1, padding=9//2, groups=B*self.num_M).view(B, self.num_M, H, W)  # tf里是深度卷积
        # print('blurM',blurM.shape) # (2,10,304,304)
        M = M - blurM
        # print('M-blurM',M.shape) # (2,10,304,304)
        # print('Thro',Thro.shape) # (1,10,1,1)
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

