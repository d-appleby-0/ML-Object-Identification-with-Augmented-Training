from optim import SGD, Adam
import numpy as np

class _P:                                    # minimal pseudolayer object for optimizer tests
    def __init__(self, w):
        self.W, self.b = w.copy(), np.zeros(1)
        self.dW, self.db = np.zeros_like(w), np.zeros(1)

def optim_test():
    rng = np.random.default_rng(5)
    target = rng.standard_normal((3, 4))

    # every optimizer should minimize the simple quadratic loss ||W - target||^2
    for name, make in [("sgd", lambda L: SGD(L, 0.3)),
                       ("momentum", lambda L: SGD(L, 0.05, momentum=0.9)),
                       ("adam", lambda L: Adam(L, 0.05))]:
        p = _P(rng.standard_normal((3, 4)))
        opt = make([p])
        for _ in range(1000):
            p.dW = p.W - target
            opt.step()
        err = np.abs(p.W - target).max()
        print(f"  {name:<9} final error {err:.1e}")
        assert err < 1e-3, f"{name} did not converge"

    # Adam's first step is ~ lr * sign(g), since the bias-corrected mhat/sqrt(vhat) is +-1
    g = rng.standard_normal((3, 4))
    p = _P(np.zeros((3, 4)))
    p.dW = g
    Adam([p], lr=0.01).step()
    assert np.allclose(p.W, -0.01 * np.sign(g), atol=1e-6), "Adam first step wrong"

    # momentum with a constant gradient - after 2 steps, W = -lr*(g + (mu*g + g))
    p = _P(np.zeros((3, 4)))
    opt = SGD([p], lr=0.1, momentum=0.9)
    for _ in range(2):
        p.dW = g
        opt.step()
    assert np.allclose(p.W, -0.1 * g * (2 + 0.9)), "momentum update wrong"
    print("optimizer tests passed")

if __name__ == "__main__":
    optim_test()