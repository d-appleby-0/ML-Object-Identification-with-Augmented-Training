
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

class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask



# ----------------- Replacement Conv2D and MaxPool2D w/ im2col ---------------------------

def im2col(x, k, stride, pad):
    # (N,C,H,W) -> (N*Ho*Wo, C*k*k) - loops over kernel offsets (k*k iterations) instead of positions
    N, C, H, W = x.shape
    Ho = (H + 2 * pad - k) // stride + 1
    Wo = (W + 2 * pad - k) // stride + 1
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    cols = np.zeros((N, C, k, k, Ho, Wo))
    for u in range(k):
        for v in range(k):
            cols[:, :, u, v] = xp[:, :, u:u + stride * Ho:stride, v:v + stride * Wo:stride]
    return cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * Ho * Wo, C * k * k)

def col2im(cols, x_shape, k, stride, pad):
    # Adjoint of im2col that scatter-adds columns back into an image of shape x_shape
    N, C, H, W = x_shape
    Ho = (H + 2 * pad - k) // stride + 1
    Wo = (W + 2 * pad - k) // stride + 1
    cols = cols.reshape(N, Ho, Wo, C, k, k).transpose(0, 3, 4, 5, 1, 2)   # (N,C,k,k,Ho,Wo)
    xp = np.zeros((N, C, H + 2 * pad, W + 2 * pad))
    for u in range(k):
        for v in range(k):
            xp[:, :, u:u + stride * Ho:stride, v:v + stride * Wo:stride] += cols[:, :, u, v]
    return xp[:, :, pad:pad + H, pad:pad + W]


class Conv2D:
    def __init__(self, c_in, c_out, k, stride=1, pad=0, seed=None):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((c_out, c_in, k, k)) * np.sqrt(2.0 / (c_in * k * k))
        self.b = np.zeros(c_out)
        self.k, self.stride, self.pad = k, stride, pad

    def forward(self, x):
        N, C, H, W = x.shape
        F, k, s, p = self.W.shape[0], self.k, self.stride, self.pad
        Ho = (H + 2 * p - k) // s + 1
        Wo = (W + 2 * p - k) // s + 1
        self.cols = im2col(x, k, s, p)                          # (N*Ho*Wo, C*k*k)
        out = self.cols @ self.W.reshape(F, -1).T + self.b      # (N*Ho*Wo, F)
        self.x_shape = x.shape
        return out.reshape(N, Ho, Wo, F).transpose(0, 3, 1, 2)

    def backward(self, dout):
        F = self.W.shape[0]
        dmat = dout.transpose(0, 2, 3, 1).reshape(-1, F)        # (N*Ho*Wo, F)
        self.db = dmat.sum(axis=0)
        self.dW = (dmat.T @ self.cols).reshape(self.W.shape)
        dcols = dmat @ self.W.reshape(F, -1)
        return col2im(dcols, self.x_shape, self.k, self.stride, self.pad)


class MaxPool2D:
    def __init__(self, k=2, stride=None):
        self.k = k
        self.stride = stride if stride is not None else k

    def forward(self, x):
        N, C, H, W = x.shape
        k, s = self.k, self.stride
        Ho, Wo = (H - k) // s + 1, (W - k) // s + 1
        # pool each channel independently: treat every (n, c) as its own 1-channel image
        cols = im2col(x.reshape(N * C, 1, H, W), k, s, 0)       # (N*C*Ho*Wo, k*k)
        self.argmax = cols.argmax(axis=1)
        self.x_shape = x.shape
        out = cols[np.arange(cols.shape[0]), self.argmax]
        return out.reshape(N, C, Ho, Wo)

    def backward(self, dout):
        N, C, H, W = self.x_shape
        dcols = np.zeros((dout.size, self.k * self.k))
        dcols[np.arange(dout.size), self.argmax] = dout.ravel()
        dx = col2im(dcols, (N * C, 1, H, W), self.k, self.stride, 0)
        return dx.reshape(self.x_shape)




class Dropout:
    def __init__(self, p=0.5, seed=None):
        assert 0.0 <= p < 1.0
        self.p = p
        self.rng = np.random.default_rng(seed)
        self.training = True
        self.freeze = False            # test hook - reuse the previous mask so numerical gradients are deterministic
        self.mask = None

    def forward(self, x):
        if not self.training or self.p == 0.0:
            self.mask = None
            return x
        if not (self.freeze and self.mask is not None and self.mask.shape == x.shape):
            self.mask = (self.rng.random(x.shape) >= self.p) / (1.0 - self.p)
        return x * self.mask

    def backward(self, dout):
        return dout if self.mask is None else dout * self.mask



# Alpha implementation of Conv2D and MaxPool2D used as reference for replacement with im2col protocol

class Conv2DNaive:
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
                patch = self.xp[:, :, hs:hs+k, ws:ws+k]                                     # (N, C, k, k)
                d = dout[:, :, i, j]                                                        # (N, F)
                self.dW += np.tensordot(d, patch, axes=([0], [0]))                          # (F, C, k, k)
                dxp[:, :, hs:hs+k, ws:ws+k] += np.tensordot(d, self.W, axes=([1], [0]))     # (N, C, k, k)

        return dxp[:, :, p:p+H, p:p+W]                  # remove padding to return gradient 

class MaxPool2DNaive:
    def __init__(self, k=2, stride=None):
        self.k = k
        self.stride = stride if stride != None else k

    def forward(self, x):
        N, C, H, W = x.shape
        k, s = self.k, self.stride
        Ho = (H - k) // s + 1
        Wo = (W - k) // s + 1
        out = np.zeros((N, C, Ho, Wo))
        self.argmax = np.zeros((N, C, Ho, Wo), dtype=int)   # flat index within each window

        for i in range(Ho):
            for j in range(Wo):
                win = x[:, :, i*s:i*s+k, j*s:j*s+k]          # (N, C, k, k)
                flat = win.reshape(N, C, k * k)
                self.argmax[:, :, i, j] = flat.argmax(axis=2)
                out[:, :, i, j] = flat.max(axis=2)

        self.x_shape = x.shape
        return out

    def backward(self, dout):
        N, C, H, W = self.x_shape
        _, _, Ho, Wo = dout.shape
        k, s = self.k, self.stride
        dx = np.zeros(self.x_shape)

        n_idx, c_idx = np.meshgrid(np.arange(N), np.arange(C), indexing="ij")   # (N, C) each

        for i in range(Ho):
            for j in range(Wo):
                a = self.argmax[:, :, i, j]                  # (N, C)
                u, v = a // k, a % k                         # row/col offset inside the window
                dx[n_idx, c_idx, i*s + u, j*s + v] += dout[:, :, i, j]

        return dx


# ------------------- Batch Normalization methods and helper ------------------------------

def get_param_names(layer):
    if hasattr(layer, "param_names"):
        return layer.param_names
    return ("W", "b") if hasattr(layer, "W") else ()


class BatchNorm:
    # Per-channel batch norm. Input (N,C,H,W) reduces over (N,H,W), input (N,C) reduces over N.
    param_names = ("gamma", "beta")
    buffer_names = ("running_mean", "running_var")

    def __init__(self, channels, momentum=0.9, eps=1e-5):
        self.gamma = np.ones(channels)
        self.beta = np.zeros(channels)
        self.running_mean = np.zeros(channels)
        self.running_var = np.ones(channels)
        self.momentum, self.eps = momentum, eps
        self.training = True

    def forward(self, x):
        if x.ndim == 4:
            axes, shp = (0, 2, 3), (1, -1, 1, 1)
        elif x.ndim == 2:
            axes, shp = (0,), (1, -1)
        else:
            raise ValueError(f"BatchNorm expects 2D or 4D input, got {x.ndim}D")

        if self.training:
            mu, var = x.mean(axis=axes), x.var(axis=axes)           # biased - divides by m
            self.running_mean[...] = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var[...] = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            mu, var = self.running_mean, self.running_var

        self.istd = 1.0 / np.sqrt(var + self.eps)
        self.xhat = (x - mu.reshape(shp)) * self.istd.reshape(shp)
        self.axes, self.shp = axes, shp
        return self.gamma.reshape(shp) * self.xhat + self.beta.reshape(shp)

    def backward(self, dout):
        axes, shp = self.axes, self.shp
        self.dgamma = (dout * self.xhat).sum(axis=axes)
        self.dbeta = dout.sum(axis=axes)
        dxhat = dout * self.gamma.reshape(shp)
        if not self.training:                                       # fixed affine map in eval mode
            return dxhat * self.istd.reshape(shp)
        m = dout.size // self.gamma.size                            # elements per channel
        return (self.istd.reshape(shp) / m) * (
            m * dxhat
            - dxhat.sum(axis=axes, keepdims=True)
            - self.xhat * (dxhat * self.xhat).sum(axis=axes, keepdims=True))