
# Data: x has shape (N, C, H, W) where N is the number of samples, C is the number of channels, H is the height, and W is the width.

"""
Each layer contains forward(x) and backward(dout) methods.
Forward will cache the information necessary for backward pass, where backward pass recieves the gradient of the loss with
respect to layer output and returns gradient with respect to layer input (as well as dW, db)
"""

import time

import numpy as np

class Flatten:                  # Flattens the input tensor into a 2D array (N, D) where D is the product of C, H, and W.
    def forward (self, x):
        self.in_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward (self, dout):
        return dout.reshape(self.in_shape)

class Dense:
    def __init__(self, d_in, d_out, seed=int(time.time())):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((d_in, d_out)) * np.sqrt(2.0 / d_in)   # He (Kaiming) initialization
                                                                            # sets variance of weight for each step to 2 / n_in
                                                                            # to stabilize forward and backward signal flow
        self.b = np.zeros(d_out)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW = self.x.T @ dout
        self.db = dout.sum(axis=0)
        return dout @ self.W.T

class SoftmaxCE:
    def forward(self, z, y):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        self.p = e / e.sum(axis=1, keepdims=True)                   # p = e^(z_k) / sum_j {e^(z_j)}
        self.y = y
        N = z.shape[0]
        return -np.log(self.p[np.arange(N), y] + 1e-12).mean()      # loss = -mean(log(p[i, y_i]))

    def backward(self):
        N = self.p.shape[0]
        dz = self.p.copy()
        dz[np.arange(N), self.y] -= 1
        return dz / N

def numerical_grad(f, w, eps=1e-5):
    g = np.zeros_like(w)
    it = np.nditer(w, flags=['multi_index'])
    for _ in it:
        i = it.multi_index
        old = w[i]
        w[i] = old + eps; fp = f()
        w[i] = old - eps; fm = f()
        w[i] = old
        g[i] = (fp - fm) / (2 * eps)
    return g

def rel_err(a, b):
    return np.abs(a - b).max() / (np.abs(a).max() + np.abs(b).max() + 1e-12)

if __name__ == "__main__":

    pass

