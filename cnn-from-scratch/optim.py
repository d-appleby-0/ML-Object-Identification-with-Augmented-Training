class SGD:
    def __init__(self, layers, lr=0.05):
        self.layers = layers          # pass model.trainable()
        self.lr = lr

    def step(self):
        for l in self.layers:
            l.W -= self.lr * l.dW
            l.b -= self.lr * l.db

# to be extended with momentum and adam