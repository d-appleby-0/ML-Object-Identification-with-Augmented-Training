import time
import numpy as np
import os
import json

from layers import Conv2D, ReLU, MaxPool2D, Flatten, Dense, Dropout, BatchNorm, get_param_names
from losses import SoftmaxCE
from model import Sequential
from optim import SGD, Adam
from data import load_cache, xray_batches
from augment import Augmenter
from metrics import score_dataset, auc, pick_threshold, binary_metrics, bootstrap_ci
from gradcheck import check_model
from checkpoint import CheckpointManager, load_checkpoint
from pathlib import Path
from plot import plot_training_metrics

#   ********     Constants     *******
def_learning_rate = 1e-3
def_momentum = 0.9
def_p_drop = 0.3
def_epochs = 30

batch_size = 32

DATA = "C:/data/xray_cache"



def recalibrate_bn(model, X, boxes, bs=64):
    # Re-estimate BatchNorm running stats on clean (non-augmented) training images
    bns = [l for l in model.layers if isinstance(l, BatchNorm)]
    was_training, old_mom = model.training, [l.momentum for l in bns]
    model.eval()                                      # dropout etc. stay off
    for l in bns:
        l.training = True                             # only BN goes back to batch-statistics mode
        l.running_mean[...] = 0.0
        l.running_var[...] = 1.0
    for k, (xb, _) in enumerate(xray_batches(X, boxes, bs, train=False)):
        for l in bns:
            l.momentum = k / (k + 1)                  # makes the running average a plain mean over batches
        model.forward(xb)
    for l, m in zip(bns, old_mom):
        l.momentum = m
    model.train(was_training)

def build_xray_model(size=128, channels=(8, 16, 32, 32), seed=0, p_drop=0.0, bn=True, n_classes=2):
    assert size % 2 ** len(channels) == 0, "size must be divisible by 2**len(channels)"
    layers, c_in = [], 1
    for i, c in enumerate(channels):
        layers.append(Conv2D(c_in, c, 3, pad=1, seed=seed + i))
        if bn:
            layers.append(BatchNorm(c))
        layers += [ReLU(), MaxPool2D(2)]
        c_in = c
    side = size // 2 ** len(channels)                       # spatial size after all the pooling
    layers.append(Flatten())
    if p_drop > 0:
        layers.append(Dropout(p_drop, seed=seed + 100))
    layers.append(Dense(c_in * side * side, n_classes, seed=seed + len(channels)))
    return Sequential(layers)

def make_opt(name, layers, lr):
    if name == "sgd":      return SGD(layers, lr)
    if name == "momentum": return SGD(layers, lr, momentum=def_momentum)
    if name == "adam":     return Adam(layers, lr)
    raise ValueError(name)

"""
def build_model(arch="cnn1", seed=0, p_drop=def_p_drop, bn=False):
    def block(c_in, c_out, s):
        b = [Conv2D(c_in, c_out, 3, pad=1, seed=s)]
        if bn:
            b.append(BatchNorm(c_out))
        return b + [ReLU(), MaxPool2D(2)]
    
    if arch == "cnn1":
        layers, head = block(1, 8, seed), [Dense(8 * 4 * 4, 10, seed=seed + 1)]
    elif arch == "cnn2":
        layers = block(1, 8, seed) + block(8, 16, seed + 1)
        head = [Dense(16 * 2 * 2, 10, seed=seed + 2)]
    else:
        raise ValueError(arch)
    layers.append(Flatten())
    if p_drop > 0:
        layers.append(Dropout(p_drop, seed=seed + 100))
    return Sequential(layers + head)
"""

def gradcheck_gate(p_drop=def_p_drop, bn=True, class_weights=None, seed=0):
    rng = np.random.default_rng(123)
    m = build_xray_model(size=16, channels=(4, 4), seed=seed, p_drop=p_drop, bn=bn)
    for l in m.trainable():
        for n in get_param_names(l):
            p = getattr(l, n)
            if n in ("b", "beta"):
                p[...] = 0.1 * rng.standard_normal(p.shape)
            elif n == "gamma":
                p[...] = 1.0 + 0.1 * rng.standard_normal(p.shape)
    for l in m.layers:
        if isinstance(l, Dropout):
            l.freeze = True
    x, y = rng.random((4, 1, 16, 16)), np.array([0, 1, 1, 0])
    print("Gradient check (small instance of the same architecture):")
    assert check_model(m.layers, SoftmaxCE(class_weights), x, y), "gradcheck failed"


# ------------------ evaluation metrics -------------------

"""
def evaluate(model, X, boxes, bs=64):
    preds, ys = [], []
    for xb, yb in xray_batches(X, boxes, bs, train=False):
        preds.append(model.predict(xb)); ys.append(yb)
    p, y = np.concatenate(preds), np.concatenate(ys)
    sens = np.sum((p == 1) & (y == 1)) / max(1, np.sum(y == 1))     # tumors correctly found
    spec = np.sum((p == 0) & (y == 0)) / max(1, np.sum(y == 0))     # healthy correctly cleared
    return sens, spec
"""

def report(name, scores, labels, thr):
    m = binary_metrics(scores, labels, thr)
    a = auc(scores, labels)
    a_lo, a_hi = bootstrap_ci(scores, labels, auc)
    s_lo, s_hi = bootstrap_ci(scores, labels, lambda s, l: binary_metrics(s, l, thr)["sensitivity"])
    print(f"\n{name} (threshold {thr:.3f})")
    print(f"  confusion    TP {m['tp']}  FN {m['fn']}  FP {m['fp']}  TN {m['tn']}")
    print(f"  sensitivity  {m['sensitivity']:.3f}  (95% CI {s_lo:.3f}-{s_hi:.3f})    specificity {m['specificity']:.3f}")
    print(f"  precision    {m['precision']:.3f}    F1 {m['f1']:.3f}    AUC {a:.3f}  (95% CI {a_lo:.3f}-{a_hi:.3f})")
    return dict(m, auc=a)


# ------------------ main training loop! -------------------

def main(epochs=def_epochs, lr=def_learning_rate, bs=32, seed=0, size=128, channels=(8, 16, 32, 32), 
         opt_name="adam", p_drop=def_p_drop, bn=True, augment=True, balance=True, min_sensitivity=0.9, 
         data_dir=DATA, ckpt_dir=None, resume=False, verbose=True):
    Xtr, btr = load_cache(f"{data_dir}/train")
    Xva, bva = load_cache(f"{data_dir}/val")
    Xte, bte = load_cache(f"{data_dir}/test")
    assert Xtr.shape[1:] == (size, size), f"cache images are {Xtr.shape[1:]}, expected {(size, size)}"

    ytr = np.array([int(len(b) > 0) for b in btr])
    pos = ytr.mean()
    cw = [0.5 / (1 - pos), 0.5 / pos] if balance and 0 < pos < 1 else None     # equalizes the classes' total weight

    if verbose:
        for name, b in (("train", btr), ("val", bva), ("test", bte)):
            n_pos = sum(len(x) > 0 for x in b)
            print(f"{name:<5} {len(b):>6} images, {n_pos} with tumor ({n_pos / len(b):.1%})")
        gradcheck_gate(p_drop, bn, cw, seed)

    model, loss = build_xray_model(size, channels, seed, p_drop, bn), SoftmaxCE(cw)
    model.train()
    model.summary((1, size, size), verbose=verbose)       # (1, 8, 8): fails fast if shapes don't line up
    opt = make_opt(opt_name, model.trainable(), lr)
    rng = np.random.default_rng(seed)
    aug = Augmenter() if augment else None
    mgr = CheckpointManager(ckpt_dir) if ckpt_dir else None
    calib_rng = np.random.default_rng(seed + 1)               # separate generator: leaves training randomness untouched
    calib_idx = np.sort(calib_rng.choice(len(Xtr), min(2000, len(Xtr)), replace=False))
    Xcal, bcal = Xtr[calib_idx], [btr[i] for i in calib_idx]  # small in-memory copy (~33 MB at 128x128)


    start = 0
    if mgr and resume and mgr.latest():
        start = load_checkpoint(mgr.latest(), model, opt, rng)["epoch"] + 1
        print(f"resumed after epoch {start - 1}")

    
    history = {
        "epoch": [],
        "train_loss": [],
        "val_auc": [],
        "val_sens": [],
        "val_spec": [],
        "val_f1": []
    }

    log_file = Path(ckpt_dir) / "training_log.txt" if ckpt_dir else None
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w") as f:
            f.write("Epoch | Train Loss | Val AUC | Val Sens | Val Spec | Val F1 | Time (s)\n")
            f.write("-" * 65 + "\n")

    for epoch in range(start, epochs):
        opt.lr = lr * 0.5 * (1 + np.cos(np.pi * epoch / epochs))     # cosine decay to ~0
        t0, losses = time.time(), []
        for xb, yb in xray_batches(Xtr, btr, bs, rng, augment=aug):
            losses.append(loss.forward(model.forward(xb), yb))
            model.backward(loss.backward())
            opt.step()
        if bn:
            recalibrate_bn(model, Xcal, bcal)
        sv, yv = score_dataset(model, Xva, bva)
        va_auc, m = auc(sv, yv), binary_metrics(sv, yv, 0.5)
        mean_loss = float(np.mean(losses))
        elapsed = time.time() - t0
        # Record metrics
        history["epoch"].append(epoch)
        history["train_loss"].append(mean_loss)
        history["val_auc"].append(float(va_auc))
        history["val_sens"].append(float(m["sensitivity"]))
        history["val_spec"].append(float(m["specificity"]))
        history["val_f1"].append(float(m["f1"]))
        # Log format
        log_str = (f"{epoch:5d} | {mean_loss:10.4f} | {va_auc:7.4f} | "
                   f"{m['sensitivity']:8.4f} | {m['specificity']:8.4f} | {m['f1']:6.4f} | {elapsed:7.1f}")
        if verbose:
            print(log_str)
        if log_file:
            with open(log_file, "a") as f:
                f.write(log_str + "\n")
        if mgr:
            mgr.save(model, opt, epoch, rng, metric=va_auc)
    if ckpt_dir:
        with open(Path(ckpt_dir) / "metrics_history.json", "w") as f:
            json.dump(history, f, indent=4)
    if mgr and os.path.exists(os.path.join(ckpt_dir, "best.npz")):
        load_checkpoint(os.path.join(ckpt_dir, "best.npz"), model)       # restores weights and BN running stats
    if bn:
        if verbose:
            s0, y0 = score_dataset(model, Xva, bva)
            print(f"val spec at 0.5 before recalibration: {binary_metrics(s0, y0, 0.5)['specificity']:.3f}")
        idx = np.sort(rng.choice(len(Xtr), min(2000, len(Xtr)), replace=False))
        recalibrate_bn(model, Xtr[idx], [btr[i] for i in idx])
        if verbose:
            s1, y1 = score_dataset(model, Xva, bva)
            print(f"val spec at 0.5 after recalibration:  {binary_metrics(s1, y1, 0.5)['specificity']:.3f}")
    sv, yv = score_dataset(model, Xva, bva)
    if verbose:
        print("\noperating points (validation):")
        for target in (0.80, 0.90, 0.95, 0.99):
            t = pick_threshold(sv, yv, target)
            m = binary_metrics(sv, yv, t)
            print(f"  sens >= {target:.2f}: threshold {t:.3f} -> "
                  f"specificity {m['specificity']:.3f}  precision {m['precision']:.3f}")
    thr = pick_threshold(sv, yv, min_sensitivity)
    report("validation", sv, yv, thr)
    st, yt = score_dataset(model, Xte, bte)
    return report("TEST", st, yt, thr), history



if __name__ == "__main__":


    ckpt_directory = "C:/ckpts/xray_5_epoch_test_w_plots"
    test_report, history = main(epochs=30, ckpt_dir=ckpt_directory)
    
    # Plot results
    plot_training_metrics(history, save_path=f"{ckpt_directory}/learning_curves.png")


    
    # main(epochs=2, ckpt_dir="C:/ckpts/xray_test_2_epochs")
    #           test on 2 epochs - yielded positive results
    
    # main(ckpt_dir="C:/ckpts/xray_run1")




    # bn and lr sweep
    """
    for bn in (False, True):
        for lr in (0.05, 0.3, 1.0, 3.0):
            res = [main(arch="cnn2", epochs=30, lr=lr, seed=s, verbose=False, bn=bn) for s in range(8)]
            accs, losses = zip(*res)
            print(f"bn={bn!s:<5} lr {lr:<4} acc {np.mean(accs):.3f} +/- {np.std(accs):.3f}   loss {np.mean(losses):.3f}")
    """


    # working bn check
    """
    for p in [0.0, 0.2, 0.3, 0.5]:
        res = [main(arch="cnn2", epochs=30, lr=0.05, seed=s, verbose=True, p_drop=p, bn=True) for s in range(3)]
        accs, losses = zip(*res)
        print(f"cnn2 p_drop {p:<4} acc {np.mean(accs):.3f} +/- {np.std(accs):.3f}   train-mode loss {np.mean(losses):.3f}")
    """

    
    # main(epochs=30, verbose=True, opt_name="adam")
    
    
    
    # comparative cnn test
    """
    configs = [("cnn1", [0.01, 0.03, 0.05, 0.08]),
               ("cnn2", [0.01, 0.03, 0.05, 0.08])]
    
    for cnn, lrs in configs:
        for lr in lrs:
            res = [main(arch=cnn, epochs=30, lr=lr, seed=s, verbose=False, opt_name="adam") for s in range(3)]
            accs, losses = zip(*res)
            print(f"{cnn:<9} lr {lr:<6} acc {np.mean(accs):.3f} +/- {np.std(accs):.3f}   loss {np.mean(losses):.3f}")
    """

    # checkpoint verification
    """
    main(arch="cnn1", epochs=10, ckpt_dir="C:/ckpts/run1")                 # run, interrupt it
    main(arch="cnn1", epochs=30, ckpt_dir="C:/ckpts/run1", resume=True)    # continues from epoch 10
    """

    """
    configs = [("sgd", [0.3]),
           ("momentum", [0.01, 0.03, 0.1]),
           ("adam", [0.001, 0.003, 0.01, 0.03])]

    for name, lrs in configs:
        for lr in lrs:
            res = [main(epochs=20, lr=lr, seed=s, verbose=False, opt_name=name) for s in range(3)]
            accs, losses = zip(*res)
            print(f"{name:<9} lr {lr:<6} acc {np.mean(accs):.3f} +/- {np.std(accs):.3f}   loss {np.mean(losses):.3f}")
                                # Adam reaches loss of 0.001 after 20 epochs and has greatest accuracy behind momentum at 0.1 lr
                                # momentum @ 0.1 lr 98% vs Adam @ 0.03 lr 97.9% is within noise with comparable std dev but 5x lower loss
    """

    # ------------------ Learning rate exploration prior to train.py tuple refactor -------------------

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






#           **************** obsolete main that lacks checkpoint loading
"""
def main(arch="cnn1", epochs=num_epochs, lr=def_learning_rate, bs=batch_size, seed=0, verbose=True, opt_name="sgd"):
    Xtr, Xte, ytr, yte = load_data()
    model, loss = build_model(seed), SoftmaxCE()
    opt = make_opt(opt_name, model.trainable(), lr)
    rng = np.random.default_rng(seed)

    
    pre = Conv2D(1, 8, 3, pad=1, seed=0).forward(Xtr[:4])
    print((pre == 0).mean())    # possibly a large fraction
    
    

    #                   ************************* rechack after changes otherwise omit *************************
    
    # gradient check on a model with nonzero biases and slightly noisy digits
    gc_rng = np.random.default_rng(123)
    gc_model = build_model(seed)
    gc_model.layers[0].b = 0.1 * gc_rng.standard_normal(8)
    xg = Xtr[:4] + 0.05 * gc_rng.standard_normal((4, 1, 8, 8))
    print("Gradient check, assembled model:")
    assert check_model(gc_model.layers, SoftmaxCE(), xg, ytr[:4]), "gradcheck failed"
    
    # loss should start near ln(10) ~ 2.303 with a reasonable init
    print(f"\ninitial loss: {loss.forward(model.forward(Xtr[:256]), ytr[:256]):.3f}\n")
    

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
    return te_acc, float(np.mean(losses))
"""