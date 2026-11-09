import numpy as np

class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x

    def backward(self, dout):
        for l in reversed(self.layers):
            dout = l.backward(dout)
        return dout

    def predict(self, x, batch_size=256):
        # Class predictions batched to avoid memory failure on large data
        preds = [self.forward(x[i:i + batch_size]).argmax(axis=1)
                 for i in range(0, len(x), batch_size)]
        return np.concatenate(preds)

    def trainable(self):
        return [l for l in self.layers if hasattr(l, "W")]