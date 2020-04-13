#任务：比较不同的kernel的结果差异，并画出相应的曲线来直观的表示
import struct
import numpy as np
import time
from sklearn.svm import SVC#C-Support Vector Classification

def read_image(file_name):
    #先用二进制方式把文件都读进来
    file_handle=open(file_name,"rb")  #以二进制打开文档
    file_content=file_handle.read()   #读取到缓冲区中
    offset=0
    head = struct.unpack_from('>IIII', file_content, offset)  # 取前4个整数，返回一个元组
    offset += struct.calcsize('>IIII')
    imgNum = head[1]  #图片数
    rows = head[2]   #宽度
    cols = head[3]  #高度
    images=np.empty((imgNum , 784))#empty，是它所常见的数组内的所有元素均为空，没有实际意义，它是创建数组最快的方法
    image_size=rows*cols#单个图片的大小
    fmt='>' + str(image_size) + 'B'#单个图片的format

    for i in range(imgNum):
        images[i] = np.array(struct.unpack_from(fmt, file_content, offset))
        # images[i] = np.array(struct.unpack_from(fmt, file_content, offset)).reshape((rows, cols))
        offset += struct.calcsize(fmt)
    return images

#读取标签
def read_label(file_name):
    file_handle = open(file_name, "rb")  # 以二进制打开文档
    file_content = file_handle.read()  # 读取到缓冲区中

    head = struct.unpack_from('>II', file_content, 0)  # 取前2个整数，返回一个元组
    offset = struct.calcsize('>II')

    labelNum = head[1]  # label数
    # print(labelNum)
    bitsString = '>' + str(labelNum) + 'B'  # fmt格式：'>47040000B'
    label = struct.unpack_from(bitsString, file_content, offset)  # 取data数据，返回一个元组
    return np.array(label)

def  normalize(data):#图片像素二值化，变成0-1分布
    m=data.shape[0]
    n=np.array(data).shape[1]
    for i in range(m):
        for j in range(n):
            if data[i,j]!=0:
                data[i,j]=1
            else:
                data[i,j]=0
    return data

#另一种归一化的方法，就是将特征值变成[0,1]区间的数
def  normalize_new(data):
    m=data.shape[0]
    n=np.array(data).shape[1]
    for i in range(m):
        for j in range(n):
            data[i,j]=float(data[i,j])/255
    return data

def loadDataSet():
    train_x_filename = '.\\minist\\train-images-idx3-ubyte\\train-images.idx3-ubyte'
    train_y_filename = '.\\minist\\train-labels-idx1-ubyte\\train-labels.idx1-ubyte'
    test_x_filename = '.\\minist\\t10k-images-idx3-ubyte\\t10k-images.idx3-ubyte'
    test_y_filename = '.\\minist\\t10k-labels-idx1-ubyte\\t10k-labels.idx1-ubyte'

    train_x=read_image(train_x_filename)#60000*784 的矩阵
    train_y=read_label(train_y_filename)#60000*1的矩阵
    test_x=read_image(test_x_filename)#10000*784
    test_y=read_label(test_y_filename)#10000*1

    #可以比较这两种预处理的方式最后得到的结果
    # train_x=normalize(train_x)
    # test_x=normalize(test_x)

    # train_x=normalize_new(train_x)
    # test_x=normalize_new(test_x)

    return train_x, test_x, train_y, test_y

if __name__=='__main__':
    classNum=10
    score_train=0.0
    score=0.0
    temp=0.0
    temp_train=0.0
    print("Start reading data...")
    time1=time.time()
    train_x, test_x, train_y, test_y=loadDataSet()
    time2=time.time()
    print("read data cost",time2-time1,"second")

    print("Start training data...")
    # clf=SVC(C=1.0,kernel='poly')#多项式核函数
    clf = SVC(C=0.01,kernel='rbf',verbose=50)#高斯核函数

    #由于每6000个中的每个类的数量都差不多相等，所以直接按照整批划分的方法
    print(train_x.shape)
    for i in range(classNum):
       clf.fit(train_x[i*6000:(i+1)*6000,:],train_y[i*6000:(i+1)*6000])
       temp=clf.score(test_x[i*1000:(i+1)*1000,:], test_y[i*1000:(i+1)*1000])
       # print(temp)
       temp_train=clf.score(train_x[i*6000:(i+1)*6000,:],train_y[i*6000:(i+1)*6000])
       print(temp_train)
       score+=(clf.score(test_x[i*1000:(i+1)*1000,:], test_y[i*1000:(i+1)*1000])/classNum)
       score_train+=(temp_train/classNum)

    time3 = time.time()
    print("score:{:.6f}".format(score))
    print("score:{:.6f}".format(score_train))
    print("train data cost", time3 - time2, "second")