
import numpy as np

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

def _compare(checks, tol, verbose, atol = 1e-8):
    # checks: list of (label, numerical, analytic).
    ok = True
    for label, num, ana in checks:
        err = rel_err(num, ana)
        abs_err = np.abs(num - ana).max()
        passed = err < tol or abs_err < atol
        ok &= passed
        if verbose:
            print(f"  {'PASS' if passed else 'FAIL'}  {label:<26} rel err {err:.2e}  abs {abs_err:.1e}")
    return ok

def check_layer(layer, x, tol=1e-6, verbose=True):
    # check one layer in isolation using scalar L = sum(out * r)
    r = np.random.default_rng(0).standard_normal(layer.forward(x).shape)
    f = lambda: (layer.forward(x) * r).sum()

    f()
    dx = layer.backward(r)                      # dL/dout = r
    checks = [("input x", numerical_grad(f, x), dx)]
    for name in ("W", "b"):
        if hasattr(layer, name):
            checks.append((name, numerical_grad(f, getattr(layer, name)),
                           getattr(layer, "d" + name)))
    return _compare(checks, tol, verbose)

def check_model(layers, loss, x, y, tol=1e-6, verbose=True):
    # check a full stack of layers against the total loss
    def f():
        out = x
        for l in layers:
            out = l.forward(out)
        return loss.forward(out, y)

    f()
    g = loss.backward()
    for l in reversed(layers):
        g = l.backward(g)

    checks = [("input x", numerical_grad(f, x), g)]
    for i, l in enumerate(layers):
        for name in ("W", "b"):
            if hasattr(l, name):
                checks.append((f"layer {i} {type(l).__name__}.{name}",
                               numerical_grad(f, getattr(l, name)),
                               getattr(l, "d" + name)))
    return _compare(checks, tol, verbose)