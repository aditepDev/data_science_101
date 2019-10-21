import numpy as np


class Affin:
    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.gw = 0
        self.gb = 0

    def pai(self, X):
        self.X = X
        return np.dot(X, self.w) + self.b

    def yon(self, g):
        self.gw += np.dot(self.X.T, g)
        self.gb += g.sum(0)
        return np.dot(g, self.w.T)


af = Affin(np.random.randint(0, 9, [3, 4]), np.random.randint(0, 9, 4))
x = np.random.randint(0, 9, [2, 3])
print(x)
print(af.w)
print(af.b)
a = af.pai(x)
print(a)
gx = af.yon(np.ones([2,4]))
print(gx)
print(af.gw)
print(af.gb)