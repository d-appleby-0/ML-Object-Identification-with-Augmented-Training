
# Data: x has shape (N, C, H, W) where N is the number of samples, C is the number of channels, H is the height, and W is the width.

"""
Each layer contains forward(x) and backward(dout) methods.
Forward will cache the information necessary for backward pass, where backward pass recieves the gradient of the loss with
respect to layer output and returns gradient with respect to layer input (as well as dW, db)
"""

import numpy as np

class Flatten:                  # Flattens the input tensor into a 2D array (N, D) where D is the product of C, H, and W.
    def forward (self, x):
        self.in_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward (self, dout):
        return dout.reshape(self.in_shape)

class Dense:
    def __init__(self, d_in, d_out, seed=None):
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

class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask

class Conv2D:
    def __init__(self, c_in, c_out, k, stride=1, pad=0, seed=None):
        rng = np.random.default_rng(seed)
        fan_in = c_in * k * k
        self.W = rng.standard_normal((c_out, c_in, k, k)) * np.sqrt(2.0 / fan_in)  # He init
        self.b = np.zeros(c_out)
        self.k, self.stride, self.pad = k, stride, pad

    def forward(self, x):
        N, C, H, W = x.shape
        F, k, s, p = self.W.shape[0], self.k, self.stride, self.pad
        xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
        Ho = (H + 2 * p - k) // s + 1
        Wo = (W + 2 * p - k) // s + 1
        out = np.zeros((N, F, Ho, Wo))

        for i in range(Ho):
            for j in range(Wo):
                patch = xp[:, :, i*s:i*s+k, j*s:j*s+k]               # (N, C, k, k)
                out[:, :, i, j] = np.tensordot(patch, self.W, axes=([1, 2, 3], [1, 2, 3])) + self.b

        self.xp, self.x_shape = xp, x.shape
        return out

    def backward(self, dout):
        N, C, H, W = self.x_shape
        _, F, Ho, Wo = dout.shape
        k, s, p = self.k, self.stride, self.pad

        dxp = np.zeros_like(self.xp)
        self.dW = np.zeros_like(self.W)
        self.db = dout.sum(axis=(0, 2, 3))

        for i in range(Ho):
            for j in range(Wo):
                hs, ws = i * s, j * s
                patch = self.xp[:, :, hs:hs+k, ws:ws+k]              # (N, C, k, k)
                d = dout[:, :, i, j]                                 # (N, F)
                self.dW += np.tensordot(d, patch, axes=([0], [0]))   # (F, C, k, k)
                dxp[:, :, hs:hs+k, ws:ws+k] += np.tensordot(d, self.W, axes=([1], [0]))  # (N, C, k, k)

        return dxp[:, :, p:p+H, p:p+W]                               # strip the padding