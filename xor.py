import matplotlib.pyplot as plt
import numpy as np



def relu(X):
    return np.maximum(0, X)


w1 = np.array([[1.5,0.5],
               [1.2,1.2]])
b1 = np.array([-0.8,-1.2])
w2 = np.array([1,-5])
b2 = -0.1

def p(X):
    a1 = np.dot(X,w1) + b1
    h1 = relu(a1)
    a2 = np.dot(h1,w2) + b2
    return (a2>=0).astype(int)


X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
z = np.array([0, 1, 1, 0])
plt.gca(aspect=1)
plt.scatter(X[:, 0], X[:, 1], 100, c=z, edgecolor='r', marker='D', cmap='gray')
plt.show()


plt.figure(figsize=[4,4])
mx,my = np.meshgrid(np.linspace(-0.5,1.5,200),np.linspace(-0.5,1.5,200))
mX = np.array([mx,my]).T
mz = p(mX)
plt.gca(aspect=1)
plt.contourf(mx,my,mz,cmap='summer')
plt.scatter(X[:,0],X[:,1],100,c=z,edgecolor='r',marker='D',cmap='gray')
plt.show()

ma1 = np.dot(mX,w1) + b1
ma1 = ma1
mh1 = relu(ma1)
plt.figure(figsize=[6.4,5.2])
for i in [0,1]:
    mam = np.abs(ma1[:,:,i]).max()
    v = np.linspace(-mam,mam,30)
    for j in [0,1]:
        plt.subplot(221+i+2*j,aspect=1)
        if(j):
            plt.title('$h_{1,%d}$'%(i+1))
            plt.contourf(mx,my,mh1[:,:,i],v,cmap='PuOr')
        else:
            plt.title('$a_{1,%d}$'%(i+1))
            plt.contourf(mx,my,ma1[:,:,i],v,cmap='PuOr')
        plt.colorbar(pad=0.01,aspect=40)
        plt.scatter(X[:,0],X[:,1],100,c=z,edgecolor='r',marker='D',cmap='gray')
plt.tight_layout()
plt.show()

######## แสดงค่า a2
ma2 = np.dot(mh1,w2) + b2
plt.figure(figsize=[4.4,3.6])
plt.gca(aspect=1)
mam = np.abs(ma2).max()
plt.contour(mx,my,ma2,[0],cmap='hot')
plt.contourf(mx,my,ma2,30,cmap='PuOr',vmin=-mam,vmax=mam)
plt.colorbar(pad=0.01,aspect=40)
plt.scatter(X[:,0],X[:,1],100,c=z,edgecolor='r',marker='D',cmap='gray')
plt.show()