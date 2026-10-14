import numpy as np

from layers import Flatten, Dense, SoftmaxCE, numerical_grad, rel_err

# test on small data
def init_test():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((4, 2, 3, 3))
    y = rng.integers(0, 5, size=4)
    flat, fc, loss = Flatten(), Dense(18, 5), SoftmaxCE()

    def run():
        return loss.forward(fc.forward(flat.forward(x)), y)

    run()
    dx = flat.backward(fc.backward(loss.backward()))

    print("W:", rel_err(fc.dW, numerical_grad(run, fc.W)))
    print("b:", rel_err(fc.db, numerical_grad(run, fc.b)))



from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def test_digits():
    X, Y = load_digits(return_X_y=True)
    X = (X / 16.0).reshape(-1, 8, 8)           # (N, 8, 8)
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=0)

    flat, fc, loss = Flatten(), Dense(64, 10), SoftmaxCE()
    lr, bs = 0.1, 64

    for epoch in range(30):
        perm = np.random.permutation(len(Xtr))
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i+bs]
            l = loss.forward(fc.forward(flat.forward(Xtr[idx])), Ytr[idx])
            fc.backward(loss.backward())
            fc.W -= lr * fc.dW
            fc.b -= lr * fc.db
        acc = (fc.forward(flat.forward(Xte)).argmax(1) == Yte).mean()
        print(f"epoch {epoch:2d}  loss {l:.3f}  test acc {acc:.3f}")

if __name__ == "__main__":
    init_test()
    test_digits()
