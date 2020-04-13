import numpy as np
import struct
import matplotlib.pyplot as plt
from skimage import morphology,draw

from sklearn import svm
import time

import pickle

train_images_idx3_ubyte_file = '.\\minist\\train-images-idx3-ubyte\\train-images.idx3-ubyte'
train_label_idx1_ubyte_file = '.\\minist\\train-labels-idx1-ubyte\\train-labels.idx1-ubyte'
test_images_idx3_ubyte_file = '.\\minist\\t10k-images-idx3-ubyte\\t10k-images.idx3-ubyte'
test_label_idx1_ubyte_file = '.\\minist\\t10k-labels-idx1-ubyte\\t10k-labels.idx1-ubyte'

def decode_idx3_ubyte(inputFile):
    binaryData = open(inputFile, 'rb').read()
    offest = 0
    head = '>iiii'
    _, imageNum, imageRow, imageCol = struct.unpack_from(head, binaryData, offest)
    print("读取文件：{}， 图片数：{}，规格：{}*{}".format(inputFile, imageNum, imageRow, imageCol))
    iamgeSize = imageCol * imageRow
    offest += struct.calcsize(head)
    imageFmt = '>' + str(iamgeSize) + 'B'
    images = np.empty((imageNum, imageRow * imageCol))
    for i in range(imageNum):
        tmp = np.array(struct.unpack_from(imageFmt, binaryData, offest))
        #images[i] = np.array(tmp.reshape(imageRow, imageCol))
        images[i] = np.array(tmp)
        offest += struct.calcsize(imageFmt)
    print('Load image done')
    return images

def decode_idx1_ubyte(inputFile):
    binaryData = open(inputFile, 'rb').read()
    head = '>ii'
    offset = 0
    _, labelNum = struct.unpack_from(head, binaryData, offset)
    print('读取文件：{}， 标签数：{}'.format(inputFile, labelNum))
    labelHead = '>B'
    offset += struct.calcsize(head)
    labels = np.empty(labelNum)
    for i in range(labelNum):
        labels[i] = struct.unpack_from(labelHead, binaryData, offset)[0]
        offset += struct.calcsize(labelHead)
    return labels

def getFrame(data):
    imageNum = data.shape[0]
    for i in range(imageNum):
        tmp = data[i].reshape(28,28)
        tmp = morphology.skeletonize(tmp)
        #plt.imshow(tmp)
        #plt.show()
        data[i] = tmp.reshape(28*28)
    return data
def conv(data):
    mask = np.zeros((3, 3))
    mask[0][0] = mask[0][2] = mask[1][1] = mask[2][0] = mask[2][2] = 1
    images = np.empty((data.shape[0], 26*26))
    for i in range(data.shape[0]):
        tmp = data[i]
        #tmp.reshape(28, 28)
        res = np.zeros((26, 26))
        for j in range(26):
            for k in range(26):
                for ii in range(3):
                    for jj in range(3):
                        res[j][k] += tmp[(j+ii)*26 + k+jj]*mask[ii][jj]
        for j in range(26):
            for k in range(26):
                if res[j][k] != 0:
                    res[j][k] = 1
        images[i] = res.reshape((1, 26 * 26))
    return images

def normalize(data):
    for i in range(data.shape[0]):
        for j in range(data[i].shape[0]):
            if data[i][j]!=0:
                data[i][j]=1
    return data

def reNormalize(data):
    for i in range(data.shape[0]):
        for j in range(data[i].shape[0]):
            if data[i][j]==0:
                data[i][j]=1
            else:
                data[i][j]=0
    return data

def svmTrain(flag=0):
    clf = svm.SVC(C=100.0, kernel='rbf', gamma=0.03, verbose=1)
    if flag == 0:
        trainImages = getFrame(normalize(decode_idx3_ubyte(train_images_idx3_ubyte_file)))
        #trainImages = conv(normalize(decode_idx3_ubyte(train_images_idx3_ubyte_file)))
        trainLabels = decode_idx1_ubyte(train_label_idx1_ubyte_file)
        print(trainImages.shape)
        print('开始训练')
        clf = svm.SVC(C=100.0, kernel='rbf', gamma=0.03, verbose=1)
        clf.fit(trainImages, trainLabels)
        save = pickle.dumps(clf)
        f = open('.\\model_frame.txt', 'wb')
        f.write(save)
        f.close()
        print('训练结束')
    else:
        clf = loadModel()
    return clf

def showTrainImage():
    trainImages = decode_idx3_ubyte(train_images_idx3_ubyte_file)
    trainLabels = decode_idx1_ubyte(train_label_idx1_ubyte_file)
    for i in range(100):
        tmp = trainImages[i].reshape(28, 28)
        plt.imshow(tmp)
        #plt.show()
        plt.savefig(".\\images\\{}_{}.png".format(int(trainLabels[i]), i))

def svmEvaluator():
    clf = loadModel()
    testImages = getFrame(normalize(decode_idx3_ubyte(test_images_idx3_ubyte_file)))
    testLabels = decode_idx1_ubyte(test_label_idx1_ubyte_file)

    right = 0
    for i in range(10000):
        tmp = clf.predict([testImages[i,:]])
        #print((tmp, testLabels[i]))
        if int(tmp[0]) == int(testLabels[i]):
            right += 1
    print(right * 1.0 / 10000)

def loadModel():
    data = open('.\\model.txt', 'rb')
    tmp = data.read()
    model = pickle.loads(tmp)
    return model

if __name__ == '__main__':
    showTrainImage()
    #svmEvaluator()