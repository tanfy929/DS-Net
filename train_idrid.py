
import torch
import torch.nn.functional as  F
import numpy as np
import scipy.io as sio
import re
import os
import eyeDataReader_idrid as Crd
import MyLib as ML
import random
import DS_UNetplusplus as EM # 根据需要导入不同网络
import warnings
import argparse
import torch.optim as optim
warnings.filterwarnings('ignore')

torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(1)

# 参数设置
parser = argparse.ArgumentParser()
# 模式：训练、测试
parser.add_argument('--mode',type=str,default='train',help='train or test or iniTest or testAll.')
# 是否进行预训练
parser.add_argument('--ifpretrain',type=str,default='Yes',help='Yes or No')
# 类数
parser.add_argument('--theK',type=int,default=5,help='the number of classes')
# M通道数
parser.add_argument('--theKM', type=int,default=10,help='the number of classes')
# ResNet网络层数
parser.add_argument('--subnetL', type=int,default=4,help='layer number of subnet')
# 学习率
parser.add_argument('--learning_rate',type=float,default= 0.001,help='learning_rate')
# epoch 存多少次
parser.add_argument('--epoch', type=int,default=70,help='epoch')
# ini model
parser.add_argument('--ini_dir', type=str,default='temp/finaltrainEMlseg_threestage',help='ini model')
# 训练过程数据的存放路径
parser.add_argument('--train_dir', type=str,default='temp/finaltrainEMlseg_threestage_two',
                    help='Directory to keep training outputs.')
# 测试过程数据的存放路径
parser.add_argument('--test_dir', type=str,default='TestResult/finaltestEMlseg_threestage_two/',
                    help='Directory to keep eval outputs.')
# 数据参数 Patch 大小
parser.add_argument('--image_size', type=int,default=304,help='Image side length.')
# 每个epoch的迭代次数
parser.add_argument('--BatchIter', type=int,default=2000,help="""number of training h5 files.""")
# batch的大小
parser.add_argument('--batch_size', type=int,default=2,help="""Batch size.""")
# GPU设备数量（0代表CPU）
parser.add_argument('--num_gpus', type=int,default=1,help='Number of gpus used for training. (0 or 1)')

opt = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# train
def train(network,optimizer,lr_scheduler):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ML.mkdir('tempIm_train')
    Tname = ['MA', 'HE', 'EX', 'SE']
    colorMatrix = np.array([[0, 0, 0], [1, 0.3, 0.3], [0.3, 1, 0.3], [0.3, 0.3, 1], [0.3, 1, 1]])

    if os.path.exists(opt.ini_dir):
        ckpt = torch.load(opt.ini_dir)
        network.load_state_dict(ckpt)
        ckpt_num = re.findall(r"\d", ckpt)
        if len(ckpt_num) == 3:
            start_point = 100 * int(ckpt_num[0]) + 10 * int(ckpt_num[1]) + int(ckpt_num[2])
        elif len(ckpt_num) == 2:
            start_point = 10 * int(ckpt_num[0]) + int(ckpt_num[1])
        else:
            start_point = int(ckpt_num[0])
        print("Load success")
    else:
        print("re-training")
        start_point = 0

    random.seed(start_point + 1)

    allX, allZ = Crd.all_train_data_in()

    for j in range(start_point,opt.epoch):
        network.train()
        if j < 5:
            if opt.ifpretrain == 'Yes':
                ifpre_ = 0
                print('pretraining epoch')
                save_path = opt.train_dir + 'pre/'
                ML.mkdir(save_path)
            else:
                ifpre_ = 1
                print('training epoch')
                save_path = opt.train_dir + '/'
                ML.mkdir(save_path)
        else:
            ifpre_ = 1
            print('training epoch')
            save_path = opt.train_dir + '/'
            ML.mkdir(save_path)

        lr = optimizer.param_groups[0]['lr']
        Training_Loss = 0

        for num in range(opt.BatchIter):
            optimizer.zero_grad()
            batch_X, batch_Z = Crd.train_data_in(allX, allZ, opt.image_size, opt.batch_size)
            # 转换为tensor并交换维度
            batch_X = torch.FloatTensor(batch_X).cuda().to(device)
            batch_Z = torch.FloatTensor(batch_Z).cuda().to(device)
            batch_X = batch_X.permute(0,3,1,2)
            batch_Z = batch_Z.permute(0,3,1,2)
            pre_Z, pre_ListZ, pre_CM, pre_M, pre_sigma, pre_mu, pre_W, pre_B, _ = network(X=batch_X,Z0=batch_Z,ifpre=ifpre_)
            usedBG, smoothmask = getBackground2(batch_X, batch_Z)
            # loss function
            loss = 30 * EM.lossFuncionCE(pre_Z,batch_Z)
            loss = loss + 15 * EM.lossFuncionCE(pre_ListZ[0], batch_Z)
            for i in range(2):
                loss = loss + 5 * EM.lossFuncionCE(pre_ListZ[i], batch_Z) + .5 * torch.mean(torch.pow(usedBG - pre_CM[i], 2))
                loss = loss + 1 * torch.mean(torch.abs(pre_M[i]))
            loss = loss + 2 * torch.mean(torch.pow(usedBG - pre_CM[1], 2))
            loss = loss + 0.1 * torch.mean(batch_Z * torch.sum((torch.unsqueeze(torch.log(pre_sigma), 0) +
                                                                torch.pow(torch.unsqueeze(batch_X - pre_CM[1], 2)
                                                                    - torch.transpose(pre_mu, dim0=1, dim1=2), 2) /
                                                                (2 * torch.pow(torch.unsqueeze(pre_sigma, 0), 2) + 0.01)), 1))

            loss.backward()
            optimizer.step()
            Training_Loss += loss   # training loss

            _, ifshow = divmod(num + 1, 200)

            if ifshow == 1:
                CurLoss = Training_Loss / (num + 1)

                print('...Training with the %d-th banch ....' % (num + 1))
                print('.. %d epoch training, learning rate = %.8f, Training_Loss = %.4f..'
                      % (j + 1, lr, CurLoss))

                for a in range(4):
                    pre_Z[a] = pre_Z[a].cpu().detach().numpy()
                batch_Z = batch_Z.cpu().detach().numpy()
                batch_X = batch_X.permute(0, 2, 3, 1).cpu().detach().numpy()
                pre_W = pre_W.permute(0, 2, 3, 1).cpu().detach().numpy()
                pre_B = pre_B.permute(0, 2, 3, 1).cpu().detach().numpy()
                usedBG = usedBG.permute(0, 2, 3, 1).cpu().detach().numpy()
                pre_ListZ[0][3] = pre_ListZ[0][3].cpu().detach().numpy()
                pre_ListZ[1][3] = pre_ListZ[1][3].cpu().detach().numpy()
                for i in range(2):
                    pre_CM[i] = pre_CM[i].permute(0, 2, 3, 1)
                    pre_CM[i] = pre_CM[i].cpu().detach().numpy()

                toshow = np.hstack((np.tensordot(pre_Z[3][0, :, :, :], colorMatrix, [0, 0]),
                                    np.tensordot(batch_Z[0, :, :, :], colorMatrix, [0, 0])))
                Xshow = batch_X[0, :, :, :]

                toshow = np.hstack((np.tensordot(pre_ListZ[0][3][0, :, :, :], colorMatrix, [0, 0]),
                                    np.tensordot(pre_ListZ[1][3][0, :, :, :], colorMatrix, [0, 0]),
                                    toshow))
                toshow2 = np.hstack((np.tensordot(pre_ListZ[0][3][0, :, :, :], colorMatrix, [0, 0]),
                                     ML.normalized(np.tile(pre_W[0, :, :], (1, 1, 3))),
                                     ML.normalized(pre_B[0, :, :, :]),
                                     ML.setRange(-pre_CM[1][0, :, :, :] + batch_X[0, :, :, :], 20,
                                                 -10)))
                toshow3 = np.hstack((np.abs(usedBG[0, :, :, :]) / 200,
                                     (np.abs(pre_CM[1][0, :, :, :])) / 200,
                                     (np.abs(pre_CM[1][0, :, :, :] + pre_B[0, :, :, :])) / 200,
                                     Xshow / 200))
                ML.imwrite(np.vstack((toshow2, toshow, toshow3)),('tempIm_train/epoch%d_num%d.png' % (j + 1, num + 1)))

        # adjust the learning rate
        lr_scheduler.step()
        # save mu and sigma
        pre_sigma = pre_sigma.cpu().detach().numpy()
        pre_mu = pre_mu.cpu().detach().numpy()
        np.save('tempIm_train/sigma_epoch%d.npy'  %  (j + 1), pre_sigma)
        np.save('tempIm_train/mu_epoch%d.npy'  %  (j + 1), pre_mu)
        # sio.savemat('tempIm_train/MuSigmaofepoch%d' % (j + 1), {'mu': pre_mu, 'sigma': pre_sigma})
        model_name = 'model-epoch'  # save model

        save_path_full = save_path + model_name
        torch.save(network.state_dict(), save_path + 'latest.pth')

        if (j % 10 == 0):
            torch.save(network.state_dict(), save_path_full + str(j) + '.pth')  # 保存模型参数

        print('... mu and sigma of the  %d-th epoch ....' % (j + 1))
        print(np.squeeze(pre_mu))
        print(np.squeeze(pre_sigma))
        print('=========================================')
        print('*****************************************')

def getBackground2(X, Z):
    smoothmask = F.conv2d(1-torch.unsqueeze(Z[:, 0, :, :], 1),torch.ones([1,1,3,3]).cuda(),stride=1,padding=3//2)
    smoothmask = torch.min(smoothmask,1)
    largemask = F.conv2d(1-torch.unsqueeze(Z[:, 0, :, :], 1),torch.ones([1,1,11,11]).cuda(),stride=1,padding=11//2)
    b = torch.ones(largemask.shape).cuda()
    largemask = torch.min(largemask,b)
    Gauss = torch.FloatTensor(gauss(59, 24)).cuda()
    Gauss = torch.reshape(Gauss, [1, 1, 59, 59])

    smallmask = F.conv2d(1 - torch.unsqueeze(Z[:, 0, :, :], 1), torch.ones([1, 1, 5, 5]).cuda(), stride=1, padding=5//2)
    smallmask = torch.min(smallmask, b)

    maskeye = torch.ge(X, 0.1 * 255).float()
    maskeye = torch.unsqueeze(maskeye[:, 0, :, :],1)
    mask = maskeye * (largemask - smallmask)
    maskX = mask * X
    meanX = torch.sum(maskX, [2, 3], keepdims=True) / (torch.sum(mask, [2, 3], keepdims=True) + 0.0001)

    maskX0 = maskeye * F.conv2d(torch.unsqueeze(maskX[:, 0, :, :], 1), Gauss, stride=1, padding=59//2)
    maskX1 = maskeye * F.conv2d(torch.unsqueeze(maskX[:, 1, :, :], 1), Gauss, stride=1, padding=59//2)
    maskX2 = maskeye * F.conv2d(torch.unsqueeze(maskX[:, 2, :, :], 1), Gauss, stride=1, padding=59//2)
    maskX = torch.cat([maskX0, maskX1, maskX2], 1)

    blurM = maskeye * F.conv2d(mask, Gauss, stride=1, padding=59//2)
    maskX = maskX + 0.0001 * meanX
    showX = (1 - smallmask) * X + smallmask * maskX / (blurM + 0.0001)

    return showX, smoothmask

def gauss(kernel_size, sigma):
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2 - 0.5
    if sigma <= 0:
        sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8

    s = sigma ** 2
    sum_val = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - center, j - center

            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / 2 / s)
            sum_val += kernel[i, j]

    kernel = kernel / sum_val
    return kernel

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data = sio.loadmat('iniMuSig_idrid')
    inimu = torch.FloatTensor(data['mu']).cuda().to(device)
    inisig = torch.FloatTensor(np.sqrt(data['sigma'])).cuda().to(device)
    inimu = inimu.permute(0,3,4,1,2)
    inisig = inisig.permute(0,3,1,2)
    network = EM.preEMnet(opt.subnetL,opt.theKM,inimu,inisig).to(device)
    optimizer = optim.Adam(network.parameters(), lr=opt.learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[23, 47], gamma=0.1)
    train(network,optimizer,scheduler)

