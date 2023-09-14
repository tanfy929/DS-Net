
import cv2
import os
import numpy as np
import scipy.io as scio
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
import MyLib as ML

from matplotlib import pyplot as plt


# figurePath = os.getcwd()
# figurePath = figurePath + '/'
# print("figure path: "+figurePath)

def FPTP(yTrue, yScore, test_dir, Tname):
    fpr, tpr, thresholds = roc_curve(yTrue, yScore)

    print("\nArea under the FPR curve: " + str(fpr))
    print("\nArea under the TPR curve: " + str(tpr))
    return fpr, tpr


def roc_evaluate(yTrue, yScore, test_dir, Tname):
    # Area under the ROC curve
    fpr, tpr, thresholds = roc_curve(yTrue, yScore)
    auc = roc_auc_score(yTrue, yScore)
    # print("\nArea under the ROC curve: " + str(auc))
    print("\nArea under ROC curve of " + Tname + ": " + str(auc))
    rocCurve = plt.figure()
    plt.plot(fpr, tpr, '-', label='Area Under the Curve (AUC = %0.4f)' % auc)
    plt.title(Tname + '  Precision - Recall curve')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower right")
    plt.savefig(test_dir + Tname + "ROC.png")
    return auc


def precision_recall_evaluate(yTrue, yScore, test_dir, Tname):
    # Precision-recall curve
    precision, recall, thresholds = precision_recall_curve(yTrue, yScore)
    # print('Precision of '+Tname+':'+str(precision))
    # print('precision',precision.shape)
    # print('Recall of '+Tname+':'+str(recall))
    # print('thresholds of '+Tname+':'+str(thresholds))
    precision = np.fliplr([precision])[0]  # so the array is increasing (you won't get negative AUC)
    recall = np.fliplr([recall])[0]  # so the array is increasing (you won't get negative AUC)
    auc = np.trapz(precision, recall)
    print("\nArea under Precision-Recall curve of " + Tname + ": " + str(auc))
    prCurve = plt.figure()
    plt.plot(recall, precision, '-', label='Area Under the Curve (AUC = %0.4f)' % auc)
    plt.title(Tname + '  Precision - Recall curve')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower right")
    plt.savefig(test_dir + Tname + "Precision_recall.png")
    return auc


def RocCall(test_dir):
    ifLarge = 0

    for i in range(225):
        data = scio.loadmat(test_dir + ('_%s' % (i)))

        print(test_dir + ('_%s' % (i)))
        outZ = data['outZ']
        # print('outZ',outZ.shape) # (712,1072,5)
        orlZ = data['orlZ']
        if ifLarge == 1:
            outZ = cv2.resize(outZ, (4288, 2848), interpolation=cv2.INTER_CUBIC)
            data = scio.loadmat(('D:\Documents\MATLAB\YH-EMnet-plus-large/eyeAdata/test/Z_%s' % (i + 1)))
            orlZ = data['theZ']

            #        orlZ = cv2.resize(orlZ, (4288, 2848), interpolation=cv2.INTER_CUBIC)
            if i == 0:
                outZall = np.reshape(outZ, [2848 * 4288, 5])
                truZall = np.reshape(orlZ, [2848 * 4288, 5])

            else:
                outZall = np.vstack((outZall, np.reshape(outZ, [2848 * 4288, 5])))
                truZall = np.vstack((truZall, np.reshape(orlZ, [2848 * 4288, 5])))
        else:

            if i == 0:
                outZall = np.reshape(outZ, [432*648, 5])
                truZall = np.reshape(orlZ, [432*648, 5])

            else:
                outZall = np.vstack((outZall, np.reshape(outZ, [432*648, 5])))
                truZall = np.vstack((truZall, np.reshape(orlZ, [432*648, 5])))

    Tname = ['EX', 'HE', 'MA', 'SE']
    all_auc = 0
    for i in range(4):
        auc = precision_recall_evaluate(truZall[:, i + 1], outZall[:, i + 1], test_dir, Tname[i] + 'ori')
        all_auc += auc
    print("\nMean Area under Precision-Recall curve " + ": " + str(all_auc / 4))


