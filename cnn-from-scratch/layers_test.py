import numpy as np
from layers import Flatten, Dense, ReLU, SoftmaxCE, Conv2D
from gradcheck import check_layer, check_model, numerical_grad, rel_err

#   Constants
learning_rate = 0.05    # defaulted at 0.01, lowered to 0.05 due to jumpy losses
                        # running conv with no maxpooling 


def run_gradchecks():
    digits = load_digits()
    # small slice: numerical grads cost 2 forward passes per parameter
    x = (digits.data[:8] / 16.0).reshape(-1, 1, 8, 8)      # (N, C, H, W)
    y = digits.target[:8]
    ok = True

    print("ReLU alone (random input):")
    xr = np.random.default_rng(1).standard_normal((4, 10))
    ok &= check_layer(ReLU(), xr)

    print("Dense alone:")
    ok &= check_layer(Dense(10, 5, seed=0), xr)

    print("Full model, Flatten -> Dense -> ReLU -> Dense -> SoftmaxCE:")
    layers = [Flatten(), Dense(64, 16, seed=0), ReLU(), Dense(16, 10, seed=1)]

    ok &= check_model(layers, SoftmaxCE(), x, y)
    return ok

def conv_forward_reference(x, W, b, stride, pad):
    # verify forward pass of Conv2D layer against this loop
    N, C, H, Wd = x.shape
    F, _, k, _ = W.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    Ho = (H + 2 * pad - k) // stride + 1
    Wo = (Wd + 2 * pad - k) // stride + 1
    out = np.zeros((N, F, Ho, Wo))
    for n in range(N):
        for f in range(F):
            for i in range(Ho):
                for j in range(Wo):
                    win = xp[n, :, i*stride:i*stride+k, j*stride:j*stride+k]
                    out[n, f, i, j] = (win * W[f]).sum() + b[f]
    return out

def conv_test():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((2, 3, 6, 6))

    for stride, pad in [(1, 0), (1, 1), (2, 1), (2, 0)]:
        print(f"Conv2D stride={stride} pad={pad}:")
        conv = Conv2D(3, 4, 3, stride=stride, pad=pad, seed=0)
        conv.b = rng.standard_normal(4)                   # nonzero bias so db is tested

        ref = conv_forward_reference(x, conv.W, conv.b, stride, pad)
        assert np.allclose(conv.forward(x), ref), "forward does not match reference"

        assert check_layer(conv, x), "gradient check failed"

    print("Conv -> ReLU -> Flatten -> Dense:")
    layers = [Conv2D(2, 3, 3, pad=1, seed=0), ReLU(), Flatten(), Dense(3 * 4 * 4, 5, seed=1)]
    xs = rng.standard_normal((3, 2, 4, 4))
    ys = rng.integers(0, 5, size=3)
    assert check_model(layers, SoftmaxCE(), xs, ys)



# Helpers for any stack of layers:

def forward_all(layers, x):
    for l in layers:
        x = l.forward(x)
    return x

def backward_all(layers, dout):
    for l in reversed(layers):
        dout = l.backward(dout)
    return dout

def sgd_step(layers, lr):
    for l in layers:
        if hasattr(l, "W"):
            l.W -= lr * l.dW
            l.b -= lr * l.db




# test on small data

def init_test():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((4, 2, 3, 3))
    y = rng.integers(0, 5, size=4)

    layers = [Flatten(), Dense(18, 8, seed=0), ReLU(), Dense(8, 5, seed=1)]
    print("Full model gradient check:")
    assert check_model(layers, SoftmaxCE(), x, y), "gradient check failed"

    print("ReLU alone:")
    assert check_layer(ReLU(), rng.standard_normal((4, 10)))




from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def test_digits():
    X, Y = load_digits(return_X_y=True)
    X = (X / 16.0).reshape(-1, 1, 8, 8)        # (N, C, H, W), ready for conv later
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=0)

    layers = [Conv2D(1, 8, 3, pad=1, seed=0), ReLU(), Flatten(), Dense(8 * 8 * 8, 10, seed=1)] # temp test with Conv2D before MaxPool
    # layers = [Flatten(), Dense(64, 32, seed=0), ReLU(), Dense(32, 10, seed=1)]
    loss = SoftmaxCE()
    lr, bs = learning_rate, 64
    rng = np.random.default_rng(0)

    for epoch in range(30):
        perm = rng.permutation(len(Xtr))
        losses = []
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i+bs]
            logits = forward_all(layers, Xtr[idx])
            losses.append(loss.forward(logits, Ytr[idx]))
            backward_all(layers, loss.backward())
            sgd_step(layers, lr)
        acc = (forward_all(layers, Xte).argmax(1) == Yte).mean()
        print(f"epoch {epoch:2d}  loss {np.mean(losses):.3f}  test acc {acc:.3f}")



if __name__ == "__main__":
    assert run_gradchecks(), "gradient check failed, fix before training"
    conv_test()
    init_test()
    test_digits()
