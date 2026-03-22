import numpy as np

# Sigmoid
def forward(self, Z):
    self.A = 1 / (1 + np.exp(-Z))
    return self.A

def backward(self):
    dAdZ = self.A * (1 - self.A)
    return dAdZ


# Tanh
def forward(self, Z):
    self.A = np.tanh(Z)
    return self.A

def backward(self):
    dAdZ = 1 - self.A ** 2
    return dAdZ


# ReLU
def forward(self, Z):
    self.A = np.maximum(0, Z)
    return self.A

def backward(self):
    dAdZ = (self.A > 0).astype(float)  
    return dAdZ