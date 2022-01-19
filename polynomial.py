import numpy as np
import pandas as pd

class PolynomialRegression:
    def __init__(self,x,y,n):
        self.x = np.array([x])
        self.y = np.array([y]).T
        self.n = n
        self.f = 0.0

    def fit(self):
        l = []
        xn = []
        for j in range(self.n+1):
            for i in self.x[0]:
                l.append(i**j)
            xn.append(l)
            l = []
        xn = np.array(xn)
        self.f = np.matmul((np.linalg.inv(np.matmul(xn,xn.T))),(np.matmul(xn,self.y))).tolist()
        return self.f

    def predict(self,x):
        p = 0
        for i in range(1,self.n+1):
            p += self.f[i][0]*(x**i)
        return p+self.f[0][0]
