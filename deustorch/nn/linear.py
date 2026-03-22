import numpy as np

def __init__(self, in_features, out_features):
    self.W = np.random.randn(out_features, in_features) * 0.01    
    self.b = np.random.randn(out_features, 1) * 0.01    

def forward(self, A):
    self.N = A.shape[0]                      
    self.Ones = np.ones((self.N, 1))        
    self.A = A                               
    Z = A @ self.W.T + self.Ones @ self.b.T
    return Z

def backward(self, dLdZ):
    dLdA = dLdZ @ self.W
    self.dLdW = (dLdZ.T @ self.A) / self.N
    self.dLdb = (dLdZ.T @ self.Ones) / self.N

    return dLdA