import numpy as np
import deustorch
import deustorch.nn
from deustorch.nn.linear import Linear
def __init__(self):
    self.layers = [
        Linear(784, 512),
        Linear(512, 256),
        Linear(256, 128),
        Linear(128, 64),
        Linear(64, 10)
    ]
    self.f = [deustorch.nn.ReLU() for _ in range(5)]

def forward(self, A):
    L = len(self.layers)
    for i in range(L):
        Z = self.layers[i].forward(A)
        A = self.f[i].forward(Z)
    return A

def backward(self, dLdA):
    L = len(self.layers)
    for i in reversed(range(L)):
        dAdZ = self.f[i].backward()       
        dLdZ = dLdA * dAdZ                  
        dLdA = self.layers[i].backward(dLdZ)
    return None