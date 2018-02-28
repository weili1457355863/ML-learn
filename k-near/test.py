from knn import *

if __name__ == '__main__':
    group,label=creatDataSet()
    X=[0,0]
    predict=classify0(X,group,label,3)
    print(predict)
