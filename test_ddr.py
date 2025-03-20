
import torch
import torch.nn.functional as  F
import numpy as np
import scipy.io as sio
import re
import os
import eyeDataReader_ddr as Crd
import MyLib as ML
import random
import DS_UNetplusplus as EM # 根据需要导入不同的网络
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve
import Roc_ddr as Roc
import warnings
import argparse
import torch.optim as optim
warnings.filterwarnings('ignore')


torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

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
# ResNet网络的层数
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
# batch大小
parser.add_argument('--batch_size', type=int,default=2,help="""Batch size.""")
# GPU设备数量（0代表CPU）
parser.add_argument('--num_gpus', type=int,default=1,help='Number of gpus used for training. (0 or 1)')

opt = parser.parse_args()
# ==============================================================================#

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# ==============================================================================#
def testAll(network):
    allX, allZ = Crd.all_test_data_in()
    colorMatrix = np.array([[0, 0, 0], [0.3, 0.3, 1], [0.3, 1, 0.3], [1, 0.3, 0.3], [0.3, 1, 1]])
    ML.mkdir('tempIm_test')
    ML.mkdir(opt.test_dir)
    with torch.no_grad():
        for i in range(113):
            if i == 112:
                testband = [2*i-1, 2*i]
                inX = allX[:, :, :, testband]
                inX = hpadding(inX)
                inX = torch.FloatTensor(inX).cuda().to(device)
                inX = inX.permute(3, 2, 0, 1)
                pred_Z, _, ListCM, ListM, _, _, _, _, pred_C = network(X=inX,Z0=0,ifpre=1)
                pred_Z = pred_Z[3][1,:,:,:]
                pred_Z = dehpadding(pred_Z)
                pred_Z = pred_Z.permute(1, 2, 0)
                pred_Z = pred_Z.cpu().detach().numpy()
                sio.savemat(opt.test_dir + ('_%s' % (2 * i)), {'outZ': pred_Z, 'orlZ': allZ[2 * i]})
            else:
                testband = [2*i,2*i+1]
                inX = allX[:, :, :, testband]
                inX = hpadding(inX)
                inX = torch.FloatTensor(inX).cuda().to(device)
                inX = inX.permute(3,2,0,1)
                pred_Z, _, ListCM, ListM, _, _, _, _, pred_C = network(X=inX,Z0=0,ifpre=1)
                pred_CM = ListCM[1][0,:,:,:]
                pred_M = ListM[1][0,:,:,:]
                pred_Z0 = pred_Z[3][0,:,:,:]
                pred_Z1 = pred_Z[3][1,:,:,:]
                pred_Z0 = dehpadding(pred_Z0)
                pred_Z1 = dehpadding(pred_Z1)
                pred_Z0 = pred_Z0.permute(1,2,0)
                pred_Z1 = pred_Z1.permute(1,2,0)
                pred_Z0 = pred_Z0.cpu().detach().numpy()
                pred_Z1 = pred_Z1.cpu().detach().numpy()
                sio.savemat(opt.test_dir + ('_%s' % (2*i)),{'outZ': pred_Z0, 'orlZ': allZ[2*i]})
                sio.savemat(opt.test_dir + ('_%s' % (2*i + 1)),{'outZ': pred_Z1, 'orlZ': allZ[2*i+1]})
                # 测试结果可视化
                X_show = allX[:, :, :, 2*i]
                toshow = np.hstack((X_show / 200,
                                    np.tensordot(allZ[2*i], colorMatrix, [2, 0]),
                                    np.tensordot(pred_Z0, colorMatrix, [2, 0])))
                toshow1 = np.hstack((X_show / 200,
                                        np.tensordot(allZ[2*i][:, :, 0:1], colorMatrix[0:1, :], [2, 0]),
                                        np.tensordot(pred_Z0[:, :, 0:1], colorMatrix[0:1, :], [2, 0])))
                toshow2 = np.hstack((X_show / 200,
                                        np.tensordot(allZ[2*i][:, :, 1:2], colorMatrix[1:2, :], [2, 0]),
                                        np.tensordot(pred_Z0[:, :, 1:2], colorMatrix[1:2, :], [2, 0])))
                toshow3 = np.hstack((X_show / 200,
                                        np.tensordot(allZ[2*i][:, :, 2:3], colorMatrix[2:3, :], [2, 0]),
                                        np.tensordot(pred_Z0[:, :, 2:3], colorMatrix[2:3, :], [2, 0])))
                toshow4 = np.hstack((X_show / 200,
                                        np.tensordot(allZ[2*i][:, :, 3:4], colorMatrix[3:4, :], [2, 0]),
                                        np.tensordot(pred_Z0[:, :, 3:4], colorMatrix[3:4, :], [2, 0])))
                toshow5 = np.hstack((X_show / 200,
                                        np.tensordot(allZ[2*i][:, :, 4:5], colorMatrix[4:5, :], [2, 0]),
                                        np.tensordot(pred_Z0[:, :, 4:5], colorMatrix[4:5, :], [2, 0])))

                ML.imwrite(np.vstack((toshow, toshow1, toshow2, toshow3, toshow4, toshow5)), ('tempIm_test/test%d.png' % (2*i)))
                print('the_%s' % (i + 1) + ' done!')
    Roc.RocCall(opt.test_dir)

def hpadding(inX):
    inX = np.hstack((inX[:, 4:0:-1, :, :], inX, inX[:, -2:-6:-1, :, :]))
    return inX

def dehpadding(inX):
    inX = inX[:, :, 4:656 - 4]
    return inX

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data = sio.loadmat('iniMuSig_ddr')
    inimu = torch.FloatTensor(data['mu']).cuda().to(device)
    inisig = torch.FloatTensor(np.sqrt(data['sigma'])).cuda().to(device)
    inimu = inimu.permute(0,3,4,1,2)
    inisig = inisig.permute(0,3,1,2)
    network = EM.preEMnet(opt.subnetL, opt.theKM, inimu, inisig).to(device)
    checkpoint_dir = opt.train_dir + '/latest.pth'
    network.load_state_dict(torch.load(checkpoint_dir,map_location='cuda:0'))
    testAll(network)


