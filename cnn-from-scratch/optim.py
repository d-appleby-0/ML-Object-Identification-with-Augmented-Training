import numpy as np

from layers import get_param_names

# Ensure that loaded optimizer checkpoint is compatible with the current model parameters
def _check_keys(expected, got):
    if set(expected) != set(got):
        raise ValueError(f"optimizer state mismatch: missing {sorted(set(expected) - set(got))}, "
                         f"unexpected {sorted(set(got) - set(expected))}")


# underlying class
class Optimizer:
    def __init__(self, layers):
        self.params = [(l, n) for l in layers for n in get_param_names(l)]
    def _iter(self):
        for l, n in self.params:
            yield getattr(l, n), getattr(l, "d" + n)        # (param array, fresh gradient)


class SGD(Optimizer):
    def __init__(self, layers, lr=0.1, momentum=0.0):
        super().__init__(layers)
        self.lr, self.mu = lr, momentum
        self.v = [np.zeros_like(getattr(l, n)) for l, n in self.params]

    def step(self):
        for i, (p, g) in enumerate(self._iter()):
            if self.mu:
                self.v[i] = self.mu * self.v[i] + g         # accumulation of past gradients
                g = self.v[i]
            p -= self.lr * g                                # update

    def state_dict(self):
        return {f"v.{i}": v for i, v in enumerate(self.v)} if self.mu else {}       # plain SGD has no state

    def load_state_dict(self, state):
        _check_keys(self.state_dict(), state)
        for i in range(len(state)):
            self.v[i] = state[f"v.{i}"].copy()


# Adam optimizer - uses moving averages of the gradient and its square to adapt the learning rate for each parameter

class Adam(Optimizer):
    def __init__(self, layers, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(layers)
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = [np.zeros_like(getattr(l, n)) for l, n in self.params]
        self.v = [np.zeros_like(getattr(l, n)) for l, n in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, (p, g) in enumerate(self._iter()):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g          # mean
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g      # uncentered variance
            mhat = self.m[i] / (1 - self.b1 ** self.t)                   # bias correction with both starting at 0
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * mhat / (np.sqrt(vhat) + self.eps)

    def state_dict(self):
        d = {"t": np.array(self.t)}
        d.update({f"m.{i}": m for i, m in enumerate(self.m)})
        d.update({f"v.{i}": v for i, v in enumerate(self.v)})
        return d

    def load_state_dict(self, state):
        _check_keys(self.state_dict(), state)
        self.t = int(state["t"])
        self.m = [state[f"m.{i}"].copy() for i in range(len(self.m))]
        self.v = [state[f"v.{i}"].copy() for i in range(len(self.v))]