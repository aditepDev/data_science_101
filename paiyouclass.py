class KhunW_BuakB:
    def __init__(self,w,b):
        self.w = w
        self.b = b
        self.gw = 0
        self.gb = 0
    def pai(self,x):
        self.x = x
        return self.w*x+self.b
    def yon(self,g):
        self.gw += g*self.x
        self.gb += g
        return g*self.w
    
f1 = KhunW_BuakB(2,1)
f2 = KhunW_BuakB(3,4)
x = 3
y = f1.pai(x)
print(y) # 3*2+1 = 7
z = f2.pai(y)
print(z) # 7*3+4 = 25
gy = f2.yon(1)
print(gy) # 1*3 = 3
print(f2.gw,f2.gb) # 1*7 = 7, 1
gx = f1.yon(gy)
print(gx) # 3*2 = 6
print(f1.gw,f1.gb) # 3*3 = 9, 3