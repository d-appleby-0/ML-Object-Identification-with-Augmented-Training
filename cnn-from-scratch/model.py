import numpy as np
from layers import get_param_names

class Sequential:
    def __init__(self, layers):
        self.layers = layers
        self.training = True

    def train(self, mode=True):
        self.training = mode
        for l in self.layers:
            if hasattr(l, "training"):
                l.training = mode
        return self

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x

    def backward(self, dout):
        for l in reversed(self.layers):
            dout = l.backward(dout)
        return dout

    def trainable(self):
        return [l for l in self.layers if get_param_names(l)]

    def eval(self):
        return self.train(False)

    def predict(self, x, batch_size=256):
        was = self.training
        self.eval()
        try:
            preds = [self.forward(x[i:i + batch_size]).argmax(axis=1)
                     for i in range(0, len(x), batch_size)]
        finally:
            self.train(was)
        return np.concatenate(preds)

    def predict_proba(self, x):
        was = self.training
        self.eval()
        try:
            z = self.forward(x)
        finally:
            self.train(was)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def summary(self, input_shape, verbose=True):
        was = self.training
        self.eval()
        try:
            x = np.zeros((1, *input_shape))
            total = 0
            for l in self.layers:
                x = l.forward(x)
                n = sum(getattr(l, a).size for a in get_param_names(l))
                total += n
                if verbose:
                    print(f"{type(l).__name__:<10} out {str(x.shape[1:]):<12} params {n}")
            if verbose:
                print(f"total params {total}")
        finally:
            self.train(was)
        return total