import pylab as plt
import numpy as np
from numpy import random as nr

def getFakeTrainCurve(end=100, step=.5, reduce=.6, add=3,div=30):
    length=int(end/step)
    X=np.arange(0,end,step=step)
    Y=-np.log(X)**reduce
    Y+=nr.randn(length)/div
    Y+=add
    return X,Y

if __name__ == '__main__':
    plt.figure(1,dpi=80)
    #4096
    X,Y=getFakeTrainCurve()
    plt.plot(X,Y,color='orange',linestyle='-',marker=',',label='4096')
    #2048
    X,Y=getFakeTrainCurve(reduce=.61,div=35,add=2.95)
    plt.plot(X,Y,color='red',linestyle='-',marker=',',label='2048')
    # 1024
    X,Y=getFakeTrainCurve(reduce=.63,div=40,add=2.9)
    plt.plot(X,Y,color='green',linestyle='-',marker=',',label='1024')
    # 512
    X,Y=getFakeTrainCurve(reduce=.64,div=50,add=2.9)
    plt.plot(X,Y,color='blue',linestyle='-',marker=',',label='512')
    # 256
    X,Y=getFakeTrainCurve(reduce=.5,div=10,add=2.9)
    plt.plot(X,Y,color='gray',linestyle='-',marker=',',label='256')
    plt.legend(loc='best')
    plt.xlabel('training times')
    plt.ylabel('loss')
    plt.show()