import numpy as np

class SoftmaxCE:
    def __init__(self, class_weights=None):
        self.cw = None if class_weights is None else np.asarray(class_weights, dtype=float)

    def forward(self, z, y):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        self.p = e / e.sum(axis=1, keepdims=True)                   # p = e^(z_k) / sum_j {e^(z_j)}
        self.y = y
        N = z.shape[0]
        self.w = np.ones(N) if self.cw is None else self.cw[y]
        self.wsum = self.w.sum()
        return -(self.w * np.log(self.p[np.arange(N), y] + 1e-12)).sum() / self.wsum   # loss = -mean(log(p[i, y_i]))

    def backward(self):
        N = self.p.shape[0]
        dz = self.p.copy()
        dz[np.arange(N), self.y] -= 1
        return dz * (self.w / self.wsum)[:, None]