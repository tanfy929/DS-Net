# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
# import MyLib as ML
import os
import cv2


def normalized(X):
    maxX = np.max(X)
    minX = np.min(X)
    X = (X - minX) / (np.maximum(maxX - minX, 0.0001))
    return X


def normalized_Band(X):
    for i in range(3):
        maxX = np.max(X[:, :, i])
        minX = np.min(X[:, :, i])
        X[:, :, i] = np.minimum((X[:, :, i] - minX) / (np.maximum(maxX - minX, 0.0001)) * 1.5, 1)
    return X


def setRange(X, maxX=1, minX=0):
    X = (X - minX) / (maxX - minX + 0.0001)
    return X


def get3band_of_tensor(outX, nbanch=0, nframe=[0, 1, 2]):
    X = outX[:, :, :, nframe]
    X = X[nbanch, :, :, :]
    return X


def imshow(X):
    #    X = ML.normalized(X)
    X = np.maximum(X, 0)
    X = np.minimum(X, 1)
    plt.imshow(X)
    plt.axis('off')
    plt.show()


def imwrite(X, saveload='tempIm'):
    X = np.maximum(X, 0)
    X = np.minimum(X, 1)
    plt.imsave(saveload, X)
    plt.close()
    # print("X shape",X.shape)
    # cv2.imwrite(saveload,X)


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")
        print("---  " + path + "  ---")
    else:
        print("---  There is " + path + " !  ---")

