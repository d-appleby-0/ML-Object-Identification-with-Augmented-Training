import time
import numpy as np

from layers import Conv2D, ReLU, MaxPool2D, Flatten, Dense
from losses import SoftmaxCE
from model import Sequential
from optim import SGD
from data import load_data, batches
from gradcheck import check_model

#   ********     Constants     *******
def_learning_rate = 0.3
num_epochs = 60
batch_size = 32

def build_model(seed=0):
    return Sequential([
        Conv2D(1, 8, 3, pad=1, seed=seed),      # (N,1,8,8) -> (N,8,8,8)
        ReLU(),
        MaxPool2D(2),                           # -> (N,8,4,4)
        Flatten(),                              # -> (N,128)
        Dense(8 * 4 * 4, 10, seed=seed + 1),    # -> (N,10) logits
    ])

def main(epochs=num_epochs, lr=def_learning_rate, bs=batch_size, seed=0, verbose=True):
    Xtr, Xte, ytr, yte = load_data()
    model, loss = build_model(seed), SoftmaxCE()
    opt = SGD(model.trainable(), lr)
    rng = np.random.default_rng(seed)

    """
    pre = Conv2D(1, 8, 3, pad=1, seed=0).forward(Xtr[:4])
    print((pre == 0).mean())    # possibly a large fraction
    """
    

    #                   ************************* rechack after changes otherwise omit *************************
    """
    # gradient check on a model with nonzero biases and slightly noisy digits
    gc_rng = np.random.default_rng(123)
    gc_model = build_model(seed)
    gc_model.layers[0].b = 0.1 * gc_rng.standard_normal(8)
    xg = Xtr[:4] + 0.05 * gc_rng.standard_normal((4, 1, 8, 8))
    print("Gradient check, assembled model:")
    assert check_model(gc_model.layers, SoftmaxCE(), xg, ytr[:4]), "gradcheck failed"
    
    # loss should start near ln(10) ~ 2.303 with a reasonable init
    print(f"\ninitial loss: {loss.forward(model.forward(Xtr[:256]), ytr[:256]):.3f}\n")
    """

    # train
    for epoch in range(epochs):
        t0, losses = time.time(), []
        for xb, yb in batches(Xtr, ytr, bs, rng):
            losses.append(loss.forward(model.forward(xb), yb))
            model.backward(loss.backward())
            opt.step()
        tr_acc = (model.predict(Xtr) == ytr).mean()
        te_acc = (model.predict(Xte) == yte).mean()
        if verbose:
            print(f"epoch {epoch:2d}  loss {np.mean(losses):.3f}  "
                    f"train {tr_acc:.3f}  test {te_acc:.3f}  ({time.time() - t0:.1f}s)")
    return te_acc

if __name__ == "__main__":
    
    # main()




    # ------------------ Learning rate exploration -------------------

    """
    for lr in [0.02, 0.05, 0.1, 0.2]:
        print(lr, main(epochs=60, lr=lr, verbose=False))
                        # During intial testing, these lr were still below the edge case of divergence
                        # with 0.2 yielding the highest accuracy of 0.975
    
    # To find the limit at this stage:
    
    for lr in [0.05, 0.1, 0.2, 0.5, 1.0]:
        accs = [main(epochs=60, lr=lr, seed=s, verbose=False) for s in range(3)]
        print(f"lr {lr:<4} mean {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
    
    
    for lr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            accs = [main(epochs=60, lr=lr, seed=s, verbose=False) for s in range(3)]
            print(f"lr {lr:<4} mean {np.mean(accs):.3f} +/- {np.std(accs):.3f}")


    accs = [main(epochs=60, lr=0.3, seed=s, verbose=False) for s in range(3)]
    print(f"lr {0.3:<4} mean {np.mean(accs):.3f} +/- {np.std(accs):.3f}")       # benchmarked against previous run after im2col implementation
                                                                                # with identical results (good) but threefold faster epoch time
                                                                                # 0.3s -> 0.1s
    """