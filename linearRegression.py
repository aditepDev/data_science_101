from unagi import Chan, Affin, Relu, Sigmoid, Adam
import numpy as np
import matplotlib.pyplot as plt

class Mse(Chan): #ช ั้นของค่าความต่างกำลังสองเฉลี่ย
    def pai(self, h, z):
        self.z = z[:, None]
        self.h = h
        return ((self.z - h) ** 2).mean()

    def yon(self, g):
        return g * 2 * (self.h - self.z) / len(self.z)

class ThotthoiChoengsen:
    def rianru(self,X,z,n_thamsam):
        self.chan = [Affin(X.shape[1],1,0),Mse()]
        self.opt = Adam(self.chan[0].param)
        for o in range(n_thamsam):
            h = self.chan[0](X)
            mse = self.chan[1](h,z)
            mse.phraeyon()
            self.opt()

    def thamnai(self, X):
        h = self.chan[0].pai(X)
        return h.ravel()

x = np.random.uniform(0,2,30)
X = x[:,None]
z = 2*x-3 + np.random.normal(0,0.4,30)
tc = ThotthoiChoengsen()
tc.rianru(X,z,10000)
x_ = np.linspace(-0.1,2.1,101)
X_ = x_[:,None]
z_ = tc.thamnai(X_)
plt.scatter(x,z,c='c',edgecolor='k')
plt.plot(x_,z_,'r')
plt.show()


class PrasatThotthoi:
    def __init__(self, m1, m2, eta=0.001):
        self.m1 = m1
        self.m2 = m2
        self.eta = eta
        self.chan = [None,
                     Relu(),
                     Affin(m1, m2, np.sqrt(2. / m1)),
                     Relu(),
                     Affin(m2, 1, 0),
                     Mse()]

    def rianru(self, X, z, n_thamsam):
        m0 = X.shape[1]
        self.chan[0] = Affin(m0, self.m1, np.sqrt(2. / m0))
        self.opt = Adam(self.param(), eta=self.eta)
        for o in range(n_thamsam):
            mse = self.ha_mse(X, z)
            mse.phraeyon()
            self.opt()

    def ha_mse(self, X, z):
        for c in self.chan[:-1]:
            X = c(X)
        return self.chan[-1](X, z)

    def param(self):
        p = []
        for c in self.chan:
            if (hasattr(c, 'param')):
                p.extend(c.param)
        return p

    def thamnai(self, X):
        for c in self.chan[:-1]:
            X = c(X)
        return X.kha.ravel()


np.random.seed(0)
x = np.random.uniform(-1, 1, 60)
X = x[:, None]
z = np.sin(x * 3) + np.random.normal(0, 0.2, 60)

m1, m2 = 20, 30
ps = PrasatThotthoi(m1, m2, eta=0.005)
ps.rianru(X, z, 1000)
x_ = np.linspace(-1.2, 1.2, 101)
X_ = x_[:, None]
z_ = ps.thamnai(X_)
plt.scatter(x, z, c='c', edgecolor='k')
plt.plot(x_, z_, 'r')
plt.show()