class Tuaprae(object):
    def __init__(self, kha, thima=None):
        self.kha = kha  # ค่า
        self.thima = thima  # ออบเจ็กต์ Chan ที่สร้างตัวแปรนี้ขึ้นมา

    def phraeyon(self, g=1):
        if (self.thima is None):
            return
        g = self.thima.yon(g)
        for tpt in self.thima.tuapraeton:
            tpt.phraeyon(g)


class Chan:
    def __init__(self):
        'รอนิยามในคลาสย่อย'

    def __call__(self, *tuaprae):
        self.tuapraeton = []  # เก็บออบเจ็กต์ Tuaprae ที่ป้อนเข้ามา
        kha_tuapraeton = []
        for tp in tuaprae:
            if (type(tp) == Tuaprae):
                self.tuapraeton.append(tp)
                kha_tuapraeton.append(tp.kha)
            else:
                kha_tuapraeton.append(tp)
        kha_tuapraetam = self.pai(*kha_tuapraeton)  # คำนวนไปข้างหน้า
        tuapraetam = Tuaprae(kha_tuapraetam, thima=self)  # สร้างออบเจ็กต์ของตัวแปรตามที่คำนวณได้
        return tuapraetam

    def pai(self):
        'รอนิยามในคลาสย่อย'

    def yon(self):
        'รอนิยามในคลาสย่อย'


import numpy as np


class Sigmoid(Chan):
    def pai(self, a):
        self.h = 1 / (1 + np.exp(-a))
        return self.h

    def yon(self, g):
        return g * (1. - self.h) * self.h


class Relu(Chan):
    def pai(self, x):
        self.krong = (x > 0)
        return np.where(self.krong, x, 0)

    def yon(self, g):
        return np.where(self.krong, g, 0)


class Softmax_entropy(Chan):
    def pai(self, a, Z):
        self.Z = Z
        exp_a = np.exp(a.T - a.max(1))
        self.h = (exp_a / exp_a.sum(0)).T
        return -(np.log(self.h[Z] + 1e-10)).mean()

    def yon(self, g):
        return g * (self.h - self.Z) / len(self.h)


class Param:
    def __init__(self, kha):
        self.kha = kha  # ค่า
        self.g = 0  # อนุพันธ์


class Affin(Chan):
    def __init__(self, m0, m1, sigma=0.1):
        self.m = m0, m1
        self.param = [Param(np.random.normal(0, sigma, self.m)),
                      Param(np.zeros(m1))]

    def pai(self, X):
        self.X = X
        return np.dot(X, self.param[0].kha) + self.param[1].kha

    def yon(self, g):
        self.param[0].g += np.dot(self.X.T, g)
        self.param[1].g += g.sum(0)
        return np.dot(g, self.param[0].kha.T)

def ha_1h(z, n):
    return z[:, None] == range(n)

class PrasatMLP:
    def __init__(self, m, s=1, eta=0.1, kratun='relu'):
        self.m = m
        self.eta = eta
        self.chan = []
        for i in range(len(m) - 1):
            self.chan.append(Affin(m[i], m[i + 1], s))
            if (i < len(m) - 2):
                if (kratun == 'relu'):
                    self.chan.append(Relu())
                else:
                    self.chan.append(Sigmoid())
        self.chan.append(Softmax_entropy())


    def rianru(self, X, z, n_thamsam):
        Z = ha_1h(z, self.m[-1])
        self.entropy = []
        for i in range(n_thamsam):
            entropy = self.ha_entropy(X, Z)
            entropy.phraeyon()
            self.prap_para()
            self.entropy.append(entropy.kha)

    def ha_entropy(self, X, Z):
        for c in self.chan[:-1]:
            X = c(X)
        return self.chan[-1](X, Z)

    def prap_para(self):
        for c in self.chan:
            if (not hasattr(c, 'param')):
                continue
            for p in c.param:
                p.kha -= self.eta * p.g
                p.g = 0

    def thamnai(self, X):
        for c in self.chan[:-1]:
            X = c(X)
        return X.kha.argmax(1)
import matplotlib.pyplot as plt
np.random.seed(1)
r = np.tile(np.sqrt(np.linspace(0.5,25,200)),4)
t = np.random.normal(np.sqrt(r*5),0.3)
z = np.arange(4).repeat(200)
t += z*np.pi/2
X = np.array([r*np.cos(t),r*np.sin(t)]).T

plt.scatter(X[:,0],X[:,1],50,c=z,edgecolor='k',cmap='coolwarm')
plt.show()
# 2 ชั้น
# prasat = PrasatMLP(m=[2,50,4],eta=0.01,kratun='relu')
# 4 ชั้น
prasat = PrasatMLP(m=[2,50,50,50,4],eta=0.01,kratun='relu')
prasat.rianru(X,z,n_thamsam=5000)

mx,my = np.meshgrid(np.linspace(X[:,0].min(),X[:,0].max(),200),np.linspace(X[:,1].min(),X[:,1].max(),200))
mX = np.array([mx.ravel(),my.ravel()]).T
mz = prasat.thamnai(mX).reshape(200,-1)
plt.gca(aspect=1,xlim=(X[:,0].min(),X[:,0].max()),ylim=(X[:,1].min(),X[:,1].max()))
plt.contourf(mx,my,mz,cmap='coolwarm',alpha=0.2)
plt.scatter(X[:,0],X[:,1],50,c=z,edgecolor='k',cmap='coolwarm')
plt.show()