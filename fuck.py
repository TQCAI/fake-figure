import pylab as plt
import numpy as np
from numpy import random as nr
from scipy.interpolate import interp1d


def draw1():
    x = [45, 55]
    y = [80.56, 55.42]
    f = interp1d(x, y)
    nx = np.linspace(*x, num=100, endpoint=True)
    noize = nr.normal(0, 1, nx.shape) * 0.1
    ny = f(nx) + noize
    plt.xlabel('Thickness (mm)')
    plt.ylabel('First-order natural frequency (Hz)')
    plt.ylim([40, 85])
    plt.plot(nx, ny)
    plt.savefig('1.png')
    plt.show()

def draw2():
    x = [45, 55]
    y = [80.22, 90.34]
    f = interp1d(x, y)
    nx = np.linspace(*x, num=4, endpoint=True)
    ny = f(nx)
    d=.5
    ny[1] -= d
    ny[2] += d
    f = interp1d(nx, ny, 'cubic')
    nx = np.linspace(x[0],x[-1], num=100, endpoint=True)
    ny = f(nx)
    plt.xlabel('Thickness (mm)')
    plt.ylabel('Mass (Kg)')
    plt.ylim([75,95])
    plt.plot(nx,ny)
    plt.savefig('2.png')
    plt.show()

def draw3():
    x = [45, 75]
    y = [89.02, 82.04]
    f = interp1d(x, y)
    nx = np.linspace(*x, num=4, endpoint=True)
    ny = f(nx)
    d=.5
    ny[1] -= d
    ny[2] += d
    f = interp1d(nx, ny, 'cubic')
    nx = np.linspace(x[0],x[-1], num=100, endpoint=True)
    ny = f(nx)
    plt.ylabel('Mass (Kg)')
    plt.xlabel('Diameter (mm)')
    plt.ylim([75,95])
    plt.plot(nx,ny)
    plt.savefig('3.png')
    plt.show()

def draw4():
    x = [45, 46.8,60.04,67,75]
    y = [70.56,60.32,80.32,67, 50.43]
    f = interp1d(x, y,'cubic')
    nx = np.linspace(x[0],x[-1], num=300, endpoint=True)
    noize = nr.normal(0, 1, nx.shape) * 0.7
    ny = f(nx) + noize
    plt.xlabel('Diameter (mm)')
    plt.ylabel('First-order natural frequency (Hz)')
    plt.ylim([20, 100])
    # r'$xxxx$'
    # xy=蓝色点位置
    # xytext：描述框相对xy位置
    # textcoords='offset points'，以xy为原点偏移xytext
    # arrowprops = 画弧线箭头，'---->', rad=.2-->0.2弧度
    plt.annotate('Peak value', xy=(x[2], y[2]), xytext=(+30, +30), textcoords='offset points', fontsize=16,
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.5',color='r'))
    plt.plot(nx, ny)
    plt.savefig('4.png')
    plt.show()

def draw5():
    x = [45, 60]
    y = [80.22, 82.34]
    f = interp1d(x, y)
    nx = np.linspace(*x, num=4, endpoint=True)
    ny = f(nx)
    d=.2
    ny[1] -= d
    ny[2] += d
    f = interp1d(nx, ny, 'cubic')
    nx = np.linspace(x[0],x[-1], num=100, endpoint=True)
    ny = f(nx)
    plt.ylabel('Mass (Kg)')
    plt.xlabel('Degree ($^{\circ}$)')
    plt.ylim([75,85])
    plt.plot(nx,ny)
    plt.savefig('5.png')
    plt.show()

def shit(*pts):
    x = [pt[0] for pt in pts]
    y = [pt[1] for pt in pts]
    f = interp1d(x, y)
    nx = np.linspace(*x, num=4, endpoint=True)
    ny = f(nx)
    d=.5
    ny[1] -= d
    ny[2] += d
    f = interp1d(nx, ny, 'cubic')
    nx = np.linspace(x[0],x[-1], num=50, endpoint=True)  # num: 枢密程度
    ny = f(nx)
    return nx.tolist(),ny.tolist()

def draw6():
    x=[]
    y=[]
    pts=[(45,70.56),(50,75.6),(52.5,60.2),(55,80.32),(60,72.42)]
    for i in range(1,len(pts)):
        tx,ty=shit(pts[i-1],pts[i])
        x+=tx
        y+=ty
    x=np.array(x)
    y=np.array(y)
    noize = nr.normal(0, 0.4, x.shape)  # param 2: 最大值
    y+=noize
    plt.ylabel('First-order natural frequency (Hz)')
    plt.xlabel('Degree ($^{\circ}$)')
    plt.ylim([50,85])
    plt.plot(x,y)
    plt.savefig('6.svg')
    plt.show()

if __name__ == '__main__':
    # draw1()
    # draw2()
    # draw4()
    # draw5()
    draw6()
