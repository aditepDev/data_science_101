import numpy as np
import matplotlib.pyplot as plt


class Affin:
    # ขาเข้า (m0) ขาออก (m1) ความกว้างการแจกแจงค่า (sigma)
    def __init__(self, m0, m1, sigma):
        self.m = m0, m1
        self.w = np.random.normal(0, sigma, self.m)
        self.b = np.zeros(m1)
        self.gw = 0
        self.gb = 0

    def pai(self, X):
        self.X = X
        return np.dot(X, self.w) + self.b

    def yon(self, g):
        self.gw += np.dot(self.X.T, g)
        self.gb += g.sum(0)
        return np.dot(g, self.w.T)


class Sigmoid:
    def pai(self, a):
        self.h = 1 / (1 + np.exp(-a))
        return self.h

    def yon(self, g):
        return g * (1 - self.h) * self.h


class Softmax_entropy: # ซอฟต์แม็กซ์แล้วต่อด้วยเอนโทรปีไขว้
    def pai(self, a, Z):
        self.Z = Z
        exp_a = np.exp(a.T - a.max(1))
        self.h = (exp_a / exp_a.sum(0)).T
        return -(np.log(self.h[Z] + 1e-10)).mean()

    def yon(self, g):
        return g * (self.h - self.Z) / len(self.h)


def ha_1h(z, n):
    return z[:, None] == range(n)


class Prasat2chan:
    def __init__(self, m, eta):
        self.m = m
        self.eta = eta

    def rianru(self, X, z, n_thamsam):
        self.kiklum = int(z.max() + 1)
        Z = ha_1h(z, self.kiklum)  # แปลเป็นวันฮ็อต
        # ชั้นทั้งหมดของโครงข่ายประสาท
        self.chan = [Affin(X.shape[1], self.m, 1), Sigmoid(),
                     Affin(self.m, self.kiklum, 1),
                     Softmax_entropy()]
        self.khanaen = []
        self.entropy = []
        for o in range(n_thamsam):
            # คำนวณไปข้างหน้า
            a = X
            for c in self.chan[:-1]:
                a = c.pai(a)
            # คำนวณค่าเสียหายจากชั้นสุดท้าย
            entropy = self.chan[-1].pai(a, Z)
            self.entropy.append(entropy)
            khanaen = (a.argmax(1) == z).mean()
            self.khanaen.append(khanaen)
            # แพร่ย้อนกลับ
            g = 1
            for c in reversed(self.chan):
                g = c.yon(g)
                # ปรับพาราม0ิเตอร์ขากค่าอนุพันธ์ที่เก็บไว้หลังแฟร่ย้อน
            for i in [0, 2]:
                self.chan[i].w -= self.eta * self.chan[i].gw
                self.chan[i].b -= self.eta * self.chan[i].gb
                self.chan[i].gw = 0
                self.chan[i].gb = 0
    def thamnai(self,X):
        for c in self.chan[:-1]:
            X = c.pai(X)
        return X.argmax(1)

np.random.seed(7)
r = np.tile(np.sqrt(np.linspace(0.5,25,200)),3)
t = np.random.normal(r,0.4)
z = np.arange(3).repeat(200)
t += z*np.pi/3*2
X = np.array([r*np.cos(t),r*np.sin(t)]).T
plt.scatter(X[:,0],X[:,1],50,c=z,edgecolor='k',cmap='winter')
plt.show()

prasat = Prasat2chan(m=47,eta=0.5)
prasat.rianru(X,z,n_thamsam=2000)
mx,my = np.meshgrid(np.linspace(X[:,0].min(),X[:,0].max(),200),np.linspace(X[:,1].min(),X[:,1].max(),200))
mX = np.array([mx.ravel(),my.ravel()]).T
mz = prasat.thamnai(mX).reshape(200,-1)
plt.gca(aspect=1,xlim=(X[:,0].min(),X[:,0].max()),ylim=(X[:,1].min(),X[:,1].max()))
plt.contourf(mx,my,mz,cmap='winter',alpha=0.2)
plt.scatter(X[:,0],X[:,1],50,c=z,edgecolor='k',cmap='winter')
plt.figure()
plt.subplot(211,xticks=[])
plt.plot(prasat.entropy,'#8899dd')
plt.ylabel(u'เอนโทรปี',family='Tahoma')
plt.subplot(212)
plt.plot(prasat.khanaen,'#8899dd')
plt.ylabel(u'คะแนน',family='Tahoma')
plt.xlabel(u'จำนวนรอบ',family='Tahoma')
plt.show()