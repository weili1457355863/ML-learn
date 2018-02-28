import numpy as np
import operator
def creatDataSet():
    group=np.array([[1.0,1.1],[1.0,1.0],[0.0,0.01],[0.0,0.0]])
    label=['A','A','B','B']
    return group,label

def classify0(X,dataset,label,k):
    reps=dataset.shape[0]
    diffMat=np.tile(X,(reps,1))-dataset#待测向量与样本数据的差，tile是将向量X沿行的行方向赋值
    squar=diffMat**2
    squarDist=squar.sum(axis=1)
    dist=squarDist**0.5
    disIndiceIndex=dist.argsort()#获得距离排序的序号，从小到大排（按距离）
    print(disIndiceIndex)
    classcount={}
    for i in range (k):
        voteLabel=label[disIndiceIndex[i]]
        classcount[voteLabel]=classcount.get(voteLabel,0)+1#get(,default=None),如果不在字典中则返回None,此处的作用是如果第一次出现的label则返回0
    print(classcount)
    print(classcount.items())
    sortClassCount=sorted(classcount.items(),key=operator.itemgetter(1),reverse=True)#items,返回可遍历的元组数组,key=operator.itemgetter(1)按照第一维排序，reverse=True按照降序排列
    print(sortClassCount)
    return sortClassCount[0][0]
