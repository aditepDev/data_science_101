import numpy as np
import matplotlib.pyplot as plt


def ha_entropy(z, h):
    return -(np.log(h[z] + 1e-10)).mean()


def ha_1h(z, n):
    return (z[:, None] == range(n))


def softmax(x):
    exp_x = np.exp(x.T - x.max(1))
    return (exp_x / exp_x.sum(0)).T


class Prasat:
    def __init__(self, eta):
        self.eta = eta

    def rianru(self, X, z, n_thamsam):
        self.kiklum = int(z.max() + 1)
        Z = ha_1h(z, self.kiklum)
        self.w = np.zeros([X.shape[1], self.kiklum])
        self.b = np.zeros(self.kiklum)
        self.entropy = []
        self.khanaen = []
        for i in range(n_thamsam):
            a = self.ha_a(X)
            h = softmax(a)
            J = ha_entropy(Z, h)
            ga = (h - Z) /len(z)
            self.w -= self.eta * np.dot(X.T, ga)
            self.b -= self.eta * ga.sum(0)
            self.entropy.append(J)
            khanaen = (h.argmax(1) == z).mean()
            self.khanaen.append(khanaen)

    def thamnai(self, X):
        return self.ha_a(X).argmax(1)

    def ha_a(self, X):
        return np.dot(X, self.w) + self.b

np.random.seed(2)
X = np.random.normal(0,0.7,[45,2])
X[:15] += 2
X[30:,0] += 4
z = np.arange(3).repeat(15)
plt.gca(aspect=1)
plt.scatter(X[:,0],X[:,1],100,c=z,edgecolor='k',cmap='coolwarm')
plt.show()

prasat = Prasat(eta=0.1)
prasat.rianru(X,z,n_thamsam=250)
mx,my = np.meshgrid(np.linspace(X[:,0].min(),X[:,0].max(),200),np.linspace(X[:,1].min(),X[:,1].max(),200))
mX = np.array([mx.ravel(),my.ravel()]).T
mz = prasat.thamnai(mX).reshape(200,-1)
plt.gca(aspect=1,xlim=(X[:,0].min(),X[:,0].max()),ylim=(X[:,1].min(),X[:,1].max()))
plt.contourf(mx,my,mz,cmap='coolwarm',alpha=0.2)
plt.scatter(X[:,0],X[:,1],100,c=z,edgecolor='k',cmap='coolwarm')
plt.show()

plt.subplot(211,xticks=[])
plt.plot(prasat.entropy,'C9')
plt.ylabel(u'เอนโทรปี',family='Tahoma',size=12)
plt.subplot(212)
plt.plot(prasat.khanaen,'C9')
plt.ylabel(u'คะแนน',family='Tahoma',size=12)
plt.xlabel(u'จำนวนรอบ',family='Tahoma',size=12)
plt.show()

#############################################################################################
##### ข้อมูลรูปภาพใช้ในการเรียนรู้
from glob import glob
d = 25
X_ = np.array([plt.imread(x) for x in sorted(glob('ruprang-raisi-25x25x1000x5/*/*.png'))])
X = X_.reshape(-1,d*d)
z = np.arange(5).repeat(1000)
print(X)
print(z)

##### นำข้อมูลเเข้าไปเรียนรู้
prasat = Prasat(eta=0.02)
prasat.rianru(X,z,n_thamsam=1000)
print(prasat.khanaen[-1])

##### นำข้อมูลที่เทรนมาแสดง
plt.plot(prasat.khanaen,'C8')
plt.ylabel(u'คะแนน',family='Tahoma',size=12)
plt.xlabel(u'จำนวนรอบ',family='Tahoma',size=12)
plt.show()

##### สร้างฟังชั่นเอง
def confusion_matrix(z1,z2):
    n = max(z1.max(),z2.max())+1
    return np.dot((z1==np.arange(n)[:,None]).astype(int),(z2[:,None]==np.arange(n)).astype(int))

print(confusion_matrix(prasat.thamnai(X),z))

##### ใช้  sklearn.metrics
from sklearn.metrics import confusion_matrix
conma = confusion_matrix(prasat.thamnai(X),z)
[print(c) for c in conma]

##############################################################################################################
conma = confusion_matrix(prasat.thamnai(X),z)
def plotconma(conma,log=0):
    n = len(conma)
    plt.figure(figsize=[9,8])
    plt.gca(xticks=np.arange(n),xticklabels=np.arange(n),yticks=np.arange(n),yticklabels=np.arange(n))
    plt.xlabel(u'ทายได้',fontname='Tahoma',size=16)
    plt.ylabel(u'คำตอบ',fontname='Tahoma',size=16)
    for i in range(n):
        for j in range(n):
            plt.text(j,i,conma[i,j],ha='center',va='center',size=14)
    if(log):
        plt.imshow(conma,cmap='autumn_r')
    else:
        plt.imshow(conma,cmap='autumn_r')
    plt.colorbar(pad=0.01)
    plt.show()

plotconma(conma,log=1)