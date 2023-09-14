# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as sio
import random


def all_train_data_in():
    data = sio.loadmat('../eyeAdata_new/train/X')
    allDataX = data['X']
    allDataZ = []
    for j in range(54):
        data = sio.loadmat(('../eyeAdata_new/train/Z_%s' % (j + 1)))
        tempZ = data['theZ']
        allDataZ.append(tempZ)
    return allDataX, allDataZ



def all_test_data_in():
    data = sio.loadmat('../eyeAdata_new/test/X')
    allDataX = data['X']
    allDataZ = []
    for j in range(27):
        data = sio.loadmat(('../eyeAdata_new/test/Z_%s' % (j + 1)))
        tempZ = data['theZ']
        allDataZ.append(tempZ)
    return allDataX, allDataZ


def train_data_in(allX, allZ, sizeI, batch_size, channel=3, dataNum=54):
    batch_X = np.zeros((batch_size, sizeI, sizeI, channel), 'f')  # 定好小块的大小
    batch_Z = np.zeros((batch_size, sizeI, sizeI, 5), 'f')

    for i in range(batch_size):
        ind = random.randint(0, dataNum - 1)  # 随机取一个数据的数据号
        X = allX[:, :, :, ind]
        Z = allZ[ind]
        px = random.randint(0, 712 - sizeI)
        py = random.randint(0, 1072 - sizeI)
        subX = X[px:px + sizeI:1, py:py + sizeI:1, :]
        subZ = Z[px:px + sizeI:1, py:py + sizeI:1, :]

        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        for j in range(rotTimes):
            subX = np.rot90(subX)
            subZ = np.rot90(subZ)
        for j in range(vFlip):
            subX = subX[:, ::-1, :]
            subZ = subZ[:, ::-1, :]
        for j in range(hFlip):
            subX = subX[::-1, :, :]
            subZ = subZ[::-1, :, :]
        batch_X[i, :, :, :] = subX
        batch_Z[i, :, :, :] = subZ
    return batch_X, batch_Z








