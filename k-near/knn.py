import numpy as np
import operator
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
#自己创建一个简单的数据集
def creatDataSet():
    group=np.array([[1.0,1.1],[1.0,1.0],[0.0,0.01],[0.0,0.0]])
    label=['A','A','B','B']
    return group,label

#给定一个向量来预测起类别的简单的分类器
def classify0(X,dataset,label,k):
    reps=dataset.shape[0]
    #print(reps)
    #print(X)
    diffMat=np.tile(X,(reps,1))-dataset#待测向量与样本数据的差，tile是将向量X沿行的行方向赋值
    squar=diffMat**2
    squarDist=squar.sum(axis=1)
    dist=squarDist**0.5
    disIndiceIndex=dist.argsort()#获得距离排序的序号，从小到大排（按距离）
    #print(disIndiceIndex)
    classcount={}
    for i in range (k):
        voteLabel=label[disIndiceIndex[i]]
        classcount[voteLabel]=classcount.get(voteLabel,0)+1#get(,default=None),如果不在字典中则返回None,此处的作用是如果第一次出现的label则返回0
    #print(classcount)
    #print(classcount.items())
    sortClassCount=sorted(classcount.items(),key=operator.itemgetter(1),reverse=True)#items,返回可遍历的元组数组,key=operator.itemgetter(1)按照第一维排序，reverse=True按照降序排列
    #print(sortClassCount)
    return sortClassCount[0][0]

#将一个文本转换为简单的数据集
def textToDataSet(fileName):
    fd=open(fileName)
    arrayLine=fd.readlines()#按行读取所有的数据，每行都为一个字符串，所有行构成一个列表
    numeLines=len(arrayLine)#得到一共有多少行
    data=np.zeros((numeLines,3))# 建立特征数组
    #label=np.zeros(numeLines)#数组方式
    label=[]#列表方式
    #print(type(label))
    index=0
    for line in arrayLine:
        line=line.strip().split('\t')#将每一行的数据按照'\t'分割成三部分从而而组成一个新的列表
        data[index,0:3]=line[0:3]#不包含最后一个
        #label[index]=line[-1]
        label.append(int(line[3]))#必须明确告知是int，不然会当做字符型处理
        index+=1
    #print(type(arrayLine))
    #print(arrayLine)
    #print(numeLines)
    return data,label

#分析数据，创建散点图
def showData(data,label):
    ax = plt.subplot(111, projection='3d')#相当于创建一个工程
    ax.scatter(data[:,0],data[:,1],data[:,2],c=15.0*np.array(label))
    #ax = plt.subplot(111)#相当于创建一个工程
    #ax.scatter(data[:,1],data[:,2],15.0*np.array(label),15.0*np.array(label))
    plt.show()

#将数据进行归一化处理
def dataNorm(data):
    min=data.min(0)
    max=data.max(0)
    diff=max-min
    print(diff)
    numLine=data.shape[0]
    normData=(data-np.tile(min,(numLine,1)))/np.tile(diff,(numLine,1))
    #print(min)
    #print(max)
    print(normData)
    return normData,diff,min

#对数据集的%10的数据测试，得到每次的预测值及错误率
def datingClassTest():
    ratio=0.1
    data,label=textToDataSet('datingTestSet2.txt')
    normData=dataNorm(data)[0]
    print(normData)
    num=data.shape[0]
    numTest=int(num*ratio)
    trainData=normData[numTest:num,:]#训练集特征
    trainLabel=label[numTest:num]#训练集数据
    #print(dataNorm(normData[1,:]))
    errCount=0
    for i in range(numTest):
        predict=classify0(normData[i],trainData,trainLabel,3)
        print('The predict is %d\tThe real label is %d'%(predict,label[i]))
        if(predict!=label[i]):
            errCount+=1
    errorRate=errCount/numTest
    print(errCount)
    print('The error rate is %f'%errorRate)

#测试一个人是否是helan的约会对象
def appiontTest():
    ffmiles=float(input('frequent flier miles per year:'))
    print(type(ffmiles))
    potime=float(input('percentage of time spent playing game and watching video:'))
    iceCream=float(input('liters of icecream consumed per yuear:'))
    results=['dislike','smallLike','largeLike']
    data, label = textToDataSet('datingTestSet2.txt')
    normData,diff,min = dataNorm(data)
    testX=np.array([ffmiles,potime,iceCream])
    testXNorm=(testX-min)/diff
    predict=classify0(testXNorm,normData,label,3)
    print('You will probably like this person',results[predict-1])

#识别手写字#

#将图像文本转换为1个向量
def wordToVector(fileName):
    fd=open(fileName)
    vect=np.zeros(1024)
    for i in range(32):
        line=fd.readline()
        for j in range(32):
            vect[32*i+j]=line[j]
    return vect

#识别手写字
def handWriting():
    trainFileList=os.listdir('trainingDigits')
    numTrain=len(trainFileList)
    trainData=np.zeros((numTrain,1024))
    trainLabel=[]
    for i in range(numTrain):
        trainData[i,:]=wordToVector('trainingDigits/%s'%trainFileList[i])
        trainLabel.append(int(trainFileList[i].split('_')[0]))
    testFileList = os.listdir('testDigits')
    numTest = len(testFileList)
    testData = np.zeros((numTest, 1024))
    errCount=0
    for i in range(numTest):
        X=wordToVector('testDigits/%s'%testFileList[i])
        #print(testFileList[i])
        y=int(testFileList[i].split('_')[0])
        #print(y)
        predict = classify0(X, trainData, trainLabel, 3)
        print('The predict is %d\tThe real label is %d' % (predict, y))
        if (predict != y):
            errCount += 1
    errorRate=errCount/numTest
    print(errCount)
    print('The error rate is %f'%errorRate)
