import numpy as np

def step(self):
    for i in range(len(self.l)):
        if self.mu == 0:
            self.l[i].W = self.l[i].W - self.lr * self.l[i].dLdW
            self.l[i].b = self.l[i].b - self.lr * self.l[i].dLdb
        else:
            self.l[i].v_W = self.mu * self.l[i].v_W + self.l[i].dLdW
            self.l[i].v_b = self.mu * self.l[i].v_b + self.l[i].dLdb
            self.l[i].W = self.l[i].W - self.lr * self.l[i].v_W
            self.l[i].b = self.l[i].b - self.lr * self.l[i].v_b