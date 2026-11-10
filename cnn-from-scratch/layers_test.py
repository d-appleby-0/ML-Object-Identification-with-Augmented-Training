import numpy as np
from layers import BatchNorm, Flatten, Dense, ReLU, Conv2D, MaxPool2D, Conv2DNaive, MaxPool2DNaive, Dropout, get_param_names
from losses import SoftmaxCE
from gradcheck import check_layer, check_model, numerical_grad, rel_err
from train import build_xray_model

#   *******     Constants     *******
learning_rate = 0.05    # defaulted at 0.1, lowered to 0.05 due to jumpy losses
                        # running conv with no maxpooling 

# ----------------- BatchNorm tests ---------------------------

def batchnorm_test():
    rng = np.random.default_rng(7)

    for shape in [(6, 3, 4, 4), (8, 5)]:
        print(f"BatchNorm input {shape}:")
        bn = BatchNorm(shape[1])
        bn.gamma[...] = 1.0 + 0.3 * rng.standard_normal(shape[1])      # nonuniform, so dgamma/dbeta are tested
        bn.beta[...] = 0.3 * rng.standard_normal(shape[1])
        x = 2.0 * rng.standard_normal(shape) + 1.5                     # nonzero mean, non-unit variance
        assert check_layer(bn, x), "gradient check failed (train mode)"
        bn.training = False
        assert check_layer(bn, x), "gradient check failed (eval mode)"

    # training output is normalized per channel
    bn = BatchNorm(3)
    x = 2.0 * rng.standard_normal((16, 3, 5, 5)) + 1.5
    out = bn.forward(x)
    assert np.allclose(out.mean(axis=(0, 2, 3)), 0, atol=1e-10)
    assert np.allclose(out.var(axis=(0, 2, 3)), 1, atol=1e-3)

    # shift invariance - sum of dx over each channel is exactly zero
    dx = bn.backward(rng.standard_normal(out.shape))
    assert np.allclose(dx.sum(axis=(0, 2, 3)), 0, atol=1e-10), "dx is not shift-invariant"

    # running stats converge to the batch stats, and eval must not update them
    bn = BatchNorm(3, momentum=0.9)
    for _ in range(300):
        train_out = bn.forward(x)
    before = bn.running_mean.copy()
    bn.training = False
    assert np.allclose(bn.forward(x), train_out, atol=1e-6), "eval output != train output at convergence"
    assert np.array_equal(bn.running_mean, before), "eval mode modified running stats"
    print("  batchnorm tests passed")



# ----------------- Dropout tests ---------------------------

def dropout_test():
    rng = np.random.default_rng(6)

    # 1) ~p of units dropped, expectation preserved (inverted scaling)
    d = Dropout(0.3, seed=0)
    out = d.forward(np.ones((1000, 1000)))
    print(f"  dropped fraction {np.mean(out == 0):.3f} (want ~0.300), mean output {out.mean():.3f} (want ~1.000)")
    assert abs(np.mean(out == 0) - 0.3) < 0.01 and abs(out.mean() - 1.0) < 0.01

    # 2) eval mode is the identity, forward and backward
    d.training = False
    x = rng.standard_normal((4, 10))
    assert np.array_equal(d.forward(x), x) and np.array_equal(d.backward(x), x)

    # 3) backward is exactly zero where forward dropped
    d.training = True
    out = d.forward(x)
    dx = d.backward(np.ones_like(x))
    assert np.all(dx[out == 0] == 0) and np.all(dx[out != 0] != 0)

    # 4) numerical gradient check with the mask frozen
    d = Dropout(0.4, seed=1)
    d.freeze = True
    assert check_layer(d, rng.standard_normal((5, 8))), "gradient check failed"

    # 5) Sequential.predict must run in eval mode and restore train mode
    m = build_xray_model(size=16, channels=(4,), p_drop=0.5, bn=False)
    X = rng.random((6, 1, 16, 16))
    drop = next(l for l in m.layers if isinstance(l, Dropout))
    assert np.array_equal(m.predict(X), m.predict(X)), "predict is stochastic: dropout still active"
    assert m.training and drop.training, "mode was not restored after predict"
    print("  dropout tests passed")


# ----------------- Conv2D tests -------------------------------

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



# ----------------- MaxPool2D tests ---------------------------

def maxpool_forward_reference(x, k, s):
    N, C, H, W = x.shape
    Ho, Wo = (H - k) // s + 1, (W - k) // s + 1
    out = np.zeros((N, C, Ho, Wo))
    for n in range(N):
        for c in range(C):
            for i in range(Ho):
                for j in range(Wo):
                    out[n, c, i, j] = x[n, c, i*s:i*s+k, j*s:j*s+k].max()
    return out

def maxpool_test():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 3, 6, 6))

    for k, s in [(2, 2), (3, 3), (3, 1), (2, 1)]:
        print(f"MaxPool2D k={k} stride={s}:")
        pool = MaxPool2D(k, s)
        ref = maxpool_forward_reference(x, k, s)
        assert np.allclose(pool.forward(x), ref), "forward does not match reference"
        assert check_layer(pool, x), "gradient check failed"

    print("Conv -> ReLU -> MaxPool -> Flatten -> Dense:")
    layers = [Conv2D(1, 2, 3, pad=1, seed=0), ReLU(), MaxPool2D(2),
              Flatten(), Dense(2 * 4 * 4, 5, seed=1)]
    xs = rng.standard_normal((3, 1, 8, 8))
    ys = rng.integers(0, 5, size=3)
    assert check_model(layers, SoftmaxCE(), xs, ys)



# ----------------- im2col tests (Conv2D and MaxPool2D tests still apply) ---------------------------

def im2col_equivalence_test():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((3, 2, 7, 7))

    for stride, pad in [(1, 0), (1, 1), (2, 1), (2, 0)]:
        print(f"Conv2D vs naive, stride={stride} pad={pad}:")
        fast = Conv2D(2, 4, 3, stride=stride, pad=pad, seed=0)
        fast.b = rng.standard_normal(4)
        slow = Conv2DNaive(2, 4, 3, stride=stride, pad=pad, seed=0)
        slow.W, slow.b = fast.W.copy(), fast.b.copy()

        of, os_ = fast.forward(x), slow.forward(x)
        assert np.allclose(of, os_), "forward mismatch"
        dout = rng.standard_normal(of.shape)
        dxf, dxs = fast.backward(dout), slow.backward(dout)
        assert np.allclose(dxf, dxs), "dx mismatch"
        assert np.allclose(fast.dW, slow.dW), "dW mismatch"
        assert np.allclose(fast.db, slow.db), "db mismatch"
        assert check_layer(fast, x), "gradient check failed"

    for k, s in [(2, 2), (3, 3), (3, 1), (2, 1)]:
        print(f"MaxPool2D vs naive, k={k} stride={s}:")
        fast, slow = MaxPool2D(k, s), MaxPool2DNaive(k, s)
        of, os_ = fast.forward(x), slow.forward(x)
        assert np.allclose(of, os_), "forward mismatch"
        dout = rng.standard_normal(of.shape)
        assert np.allclose(fast.backward(dout), slow.backward(dout)), "dx mismatch"
        assert check_layer(fast, x), "gradient check failed"

import time
def bench(conv, x, reps=3):
    t0 = time.time()
    for _ in range(reps):
        out = conv.forward(x)
        conv.backward(np.ones_like(out))
    return (time.time() - t0) / reps

def im2col_speed_test():
    x = np.random.default_rng(0).standard_normal((32, 16, 32, 32))
    fast = Conv2D(16, 32, 3, pad=1, seed=0)
    slow = Conv2DNaive(16, 32, 3, pad=1, seed=0)
    print(f"naive {bench(slow, x):.3f}s   im2col {bench(fast, x):.3f}s")



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
        for n in get_param_names(l):
            getattr(l, n)[...] -= lr * getattr(l, "d" + n)




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


# ***************** obsolete version of scikit learn digits test **************

"""
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
"""

"""
# ----------------- Full sci-kit learn digits test ---------------------------

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def test_digits():
    X, Y = load_digits(return_X_y=True)
    X = (X / 16.0).reshape(-1, 1, 8, 8)        # (N, C, H, W), ready for conv later
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=0)

    layers = [Conv2D(1, 8, 3, pad=1, seed=0), ReLU(), Flatten(), Dense(8 * 8 * 8, 10, seed=1)]     # temp test with Conv2D before MaxPool
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
"""

if __name__ == "__main__":
    
    # speed test of Conv2D vs Conv2DNaive
    # im2col_speed_test()


    # assert run_gradchecks(), "gradient check failed, fix before training"
    im2col_equivalence_test()
    conv_test()
    maxpool_test()
    init_test()
    # test_digits()
    dropout_test()
    batchnorm_test()
