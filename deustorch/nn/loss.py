import numpy as np

#MSELoss
def forward(self, A, Y):
    N, C = A.shape
    self.A = A
    self.Y = Y
    L = (1 / (2 * N * C)) * np.sum((A - Y) ** 2)
    return L

def backward(self):
    N, C = self.A.shape
    dLdA = (self.A - self.Y) / (N * C)
    return dLdA


# CrossEntropyLoss
def forward(self, A, Y):
    N = A.shape[0]
    self.Y = Y
    exp_A = np.exp(A - np.max(A, axis=1, keepdims=True))
    self.A_hat = exp_A / np.sum(exp_A, axis=1, keepdims=True)
    L = -(1 / N) * np.sum(Y * np.log(self.A_hat))
    return L

def backward(self):
    N = self.A_hat.shape[0]
    dLdA = (self.A_hat - self.Y) / N
    return dLdA
