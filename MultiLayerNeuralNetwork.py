import numpy as np
import matplotlib.pyplot as plt

np.random.seed(4)
r = np.hstack([np.random.normal(0.8, 0.25, 100),
               np.random.normal(2, 0.3, 100),
               np.random.normal(3.6, 0.4, 100),
               np.random.uniform(0.2, 4.5, 100)])
t = np.hstack([np.random.uniform(0, np.pi, 300),
               np.random.uniform(-np.pi, 0, 100)])
X = np.array([r * np.cos(t), r * np.sin(t)]).T
z = np.arange(4).repeat(100)
plt.scatter(X[:, 0], X[:, 1], 50, c=z, alpha=0.7, edgecolor='g', cmap='plasma')
plt.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    exp_x = np.exp(x.T - x.max(1))
    return (exp_x / exp_x.sum(0)).T


def ha_1h(z, n):
    return (z[:, None] == range(n)).astype(int)


def ha_entropy(z, h):
    return -(np.log(h[z == 1] + 1e-10)).mean()


class Prasat2chan:
    def __init__(self, m, eta):
        self.m = m
        self.eta = eta

    def rianru(self, X, z, n_thamsam):
        self.kiklum = int(z.max() + 1)
        Z = ha_1h(z, self.kiklum)
        self.w1 = np.random.normal(0, 1, [X.shape[1], self.m])
        self.b1 = np.zeros(self.m)
        self.w2 = np.random.normal(0, 1, [self.m, self.kiklum])
        self.b2 = np.zeros(self.kiklum)
        self.entropy = []
        self.khanaen = []
        for o in range(n_thamsam):
            a1 = self.ha_a1(X)
            h1 = sigmoid(a1)
            a2 = self.ha_a2(h1)
            h2 = softmax(a2)
            J = ha_entropy(Z,h2)
            ga2 = (h2-Z)/len(z)
            gh1 = np.dot(ga2,self.w2.T)
            ga1 = gh1*h1*(1-h1)
            self.w2 -= self.eta*np.dot(h1.T,ga2)
            self.b2 -= self.eta*ga2.sum(0)
            self.w1 -= self.eta*np.dot(X.T,ga1)
            self.b1 -= self.eta*ga1.sum(0)
            self.entropy.append(J)
            khanaen = ((a2).argmax(1) == z).mean()
            self.khanaen.append(khanaen)
            if (o % 100 == 99):
                print(u'ผ่านไป %d รอบ คะแนน %.3f' % (o + 1, khanaen))

    def thamnai(self, X):
        a1 = self.ha_a1(X)
        h1 = sigmoid(a1)
        a2 = self.ha_a2(h1)
        h2 = softmax(a2)
        return h2.argmax(1)

    def ha_a1(self, X):
        return np.dot(X, self.w1) + self.b1

    def ha_a2(self, X):
        return np.dot(X, self.w2) + self.b2

prasat = Prasat2chan(m=10, eta=0.5)
prasat.rianru(X, z, n_thamsam=2000)

mm = X.max() * 1.05
mx, my = np.meshgrid(np.linspace(-mm, mm, 200), np.linspace(-mm, mm, 200))
mX = np.array([mx.ravel(), my.ravel()]).T
mz = prasat.thamnai(mX).reshape(200, -1)
plt.gca(aspect=1, xlim=(-mm, mm), ylim=(-mm, mm))
plt.contourf(mx, my, mz, cmap='plasma', alpha=0.1)
plt.scatter(X[:, 0], X[:, 1], 50, c=z, alpha=0.7, edgecolor='g', cmap='plasma')
plt.show()


from glob import glob
n = 1000
X = np.array([plt.imread(x) for x in sorted(glob('ruprang-raisi-25x25x1000x5/*/*.png'))])
X = X.reshape(-1,25*25)
z = np.arange(5).repeat(n)

prasat = Prasat2chan(m=50,eta=0.25)
prasat.rianru(X,z,n_thamsam=12000)

plt.subplot(211,xticks=[])
plt.plot(prasat.entropy,'#77aa77')
plt.ylabel(u'เอนโทรปี',family='Tahoma')
plt.subplot(212)
plt.plot(prasat.khanaen,'#77aa77')
plt.ylabel(u'คะแนน',family='Tahoma')
plt.xlabel(u'จำนวนรอบ',family='Tahoma')
print(prasat.khanaen[-1])
plt.show()