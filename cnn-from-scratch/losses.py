import numpy as np

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