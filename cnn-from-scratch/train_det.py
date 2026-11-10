import os, time
import numpy as np

from layers import Conv2D, BatchNorm, ReLU, MaxPool2D, get_param_names
from model import Sequential
from data import load_cache, xray_det_batches
from augment import Augmenter
from detect import YoloLoss, decode, mean_ap, image_scores, detection_counts, n_out
from metrics import auc, binary_metrics, pick_threshold
from checkpoint import CheckpointManager, load_checkpoint
from gradcheck import check_model
from train import make_opt, recalibrate_bn
from viz import save_examples
from pathlib import Path
from detect import map_coco                                  # added to the existing detect import
from logger import TrainLogger, DET_COLUMNS
from plot_det import plot_det_metrics

DATA = "C:/data/xray_cache"


def evaluate_split(model, X, boxes, nc, loss_fn, bs=32):
    """One eval-mode pass over a split: returns detections and the mean loss."""
    preds, total, n = [], 0.0, 0
    for xb, bl in xray_det_batches(X, boxes, bs, train=False):
        out = model.infer(xb)
        total += float(loss_fn.forward(out, bl)) * len(bl)
        n += len(bl)
        preds += decode(out, nc, conf_thr=0.01)
    return preds, total / n



def build_det_model(size=128, channels=(8, 16, 32), extra=(32, 32), nc=1, seed=0, bn=True):
    assert size % 2 ** len(channels) == 0, "size must be divisible by 2**len(channels)"
    layers, c_in, s = [], 1, seed

    def block(ci, co, pool):
        nonlocal s
        ls = [Conv2D(ci, co, 3, pad=1, seed=s)]
        s += 1
        if bn:
            ls.append(BatchNorm(co))
        ls.append(ReLU())
        return ls + ([MaxPool2D(2)] if pool else [])

    for c in channels:                                   # downsampling blocks
        layers += block(c_in, c, True); c_in = c
    for c in extra:                                      # same-resolution blocks: widen the receptive field
        layers += block(c_in, c, False); c_in = c
    head = Conv2D(c_in, n_out(nc), 1, seed=s)
    head.b[0] = -4.0                                     # start with objectness ~2%: most cells are background
    layers.append(head)
    return Sequential(layers)

def receptive_field(model):
    rf, jump = 1, 1
    for l in model.layers:
        if isinstance(l, (Conv2D, MaxPool2D)):
            rf += (l.k - 1) * jump
            jump *= l.stride
    return rf, jump

def randomize_params(model, rng):
    for l in model.trainable():
        for n in get_param_names(l):
            p = getattr(l, n)
            if n in ("b", "beta"):
                p[...] = 0.1 * rng.standard_normal(p.shape)
            elif n == "gamma":
                p[...] = 1.0 + 0.1 * rng.standard_normal(p.shape)

def gradcheck_gate_det(nc=2, bn=True, seed=0):
    rng = np.random.default_rng(123)
    m = build_det_model(16, (4,), (4,), nc, seed, bn)                    # 16x16 input -> 8x8 grid
    randomize_params(m, rng)
    x = rng.random((2, 1, 16, 16))                                       # two images
    boxes = [np.array([[0, .30, .55, .20, .25], [nc - 1, .80, .20, .10, .10]]),
             np.array([[nc - 1, .52, .48, .30, .30]])]                   # one list entry per image
    print("Gradient check, detection model + YoloLoss:")
    assert check_model(m.layers, YoloLoss(nc), x, boxes, eps=1e-6), "gradcheck failed"

def predict_dataset(model, X, boxes, nc, bs=16, conf_thr=0.01):
    preds = []
    for xb, _ in xray_det_batches(X, boxes, bs, train=False):
        preds += decode(model.infer(xb), nc, conf_thr=conf_thr)
    return preds

def report_det(name, preds, gts, thr, nc):
    ap = mean_ap(preds, gts, nc)
    s, y = image_scores(preds), np.array([int(len(g) > 0) for g in gts])
    m = binary_metrics(s, y, thr)
    tp, fp, fn = detection_counts(preds, gts, thr)
    print(f"\n{name} (confidence threshold {thr:.3f})")
    print(f"  AP@0.5 {ap:.3f}    image-level AUC {auc(s, y):.3f}    "
          f"sensitivity {m['sensitivity']:.3f}    specificity {m['specificity']:.3f}")
    print(f"  tumors: {tp} found, {fn} missed, {fp} false boxes ({fp / len(gts):.2f} per image)    "
          f"recall {tp / max(1, tp + fn):.3f}    precision {tp / max(1, tp + fp):.3f}")
    return dict(ap50=ap, tp=tp, fp=fp, fn=fn, sensitivity=m["sensitivity"], specificity=m["specificity"])


def main(epochs=60, lr=1e-3, bs=32, seed=0, size=128, channels=(8, 16, 32), extra=(32, 32), nc=1,
         bn=True, augment=True, min_sensitivity=0.90, log_thr=0.25, lambda_noobj=0.5, n_calib=500,
         patience=None, abort_after=None, init_from=None, eval_test=True, viz_min_conf=0.15,
         class_names=None, data_dir=DATA, ckpt_dir=None, resume=False, verbose=True):
    Xtr, btr = load_cache(f"{data_dir}/train")
    Xva, bva = load_cache(f"{data_dir}/val")
    Xte, bte = load_cache(f"{data_dir}/test")
    assert Xtr.shape[1:] == (size, size), f"cache images are {Xtr.shape[1:]}, expected {(size, size)}"

    if verbose:
        for nm, b in (("train", btr), ("val", bva), ("test", bte)):
            print(f"{nm:<5} {len(b):>6} images, {sum(len(x) > 0 for x in b)} with tumors, {sum(len(x) for x in b)} boxes")
        gradcheck_gate_det(max(nc, 2), bn, seed)

    model = build_det_model(size, channels, extra, nc, seed, bn)
    loss, vloss = YoloLoss(nc, lambda_noobj=lambda_noobj), YoloLoss(nc, lambda_noobj=lambda_noobj)
    model.train()
    model.summary((1, size, size), verbose=verbose)
    rf, jump = receptive_field(model)
    print(f"grid {size // jump}x{size // jump}, cell stride {jump}px, receptive field {rf}px ({rf / size:.0%} of the image)")
    opt = make_opt("adam", model.trainable(), lr)
    rng = np.random.default_rng(seed)
    aug = Augmenter() if augment else None
    mgr = CheckpointManager(ckpt_dir) if ckpt_dir else None

    calib_rng = np.random.default_rng(seed + 1)
    ci = np.sort(calib_rng.choice(len(Xtr), min(n_calib, len(Xtr)), replace=False))
    Xcal, bcal = Xtr[ci], [btr[i] for i in ci]

    start = 0
    if mgr and resume and mgr.latest():
        start = load_checkpoint(mgr.latest(), model, opt, rng)["epoch"] + 1
        print(f"resumed after epoch {start - 1}")
    elif init_from:
        load_checkpoint(init_from, model)                                       # weights + BN stats only; fresh optimizer
        print(f"initialized weights from {init_from}")
    logger = TrainLogger(ckpt_dir, DET_COLUMNS, resume_from=start or None, verbose=verbose)
    y_val = np.array([int(len(g) > 0) for g in bva])

    for epoch in range(start, epochs):
        opt.lr = lr * min(1.0, (epoch + 1) / 3) * 0.5 * (1 + np.cos(np.pi * epoch / epochs))   # warmup, then cosine
        t0, totals, parts = time.time(), [], []
        for xb, bl in xray_det_batches(Xtr, btr, bs, rng, augment=aug):
            l = loss.forward(model.forward(xb), bl)
            model.backward(loss.backward())
            opt.step()
            totals.append(float(l))
            parts.append(loss.parts)
        if bn:
            recalibrate_bn(model, Xcal, bcal)

        pv, v_loss = evaluate_split(model, Xva, bva, nc, vloss)
        v_ap = mean_ap(pv, bva, nc)
        m = binary_metrics(image_scores(pv), y_val, log_thr)
        tp, fp, fn = detection_counts(pv, bva, log_thr)
        if mgr:
            mgr.save(model, opt, epoch, rng, metric=v_ap)                      # best = highest validation mAP@0.5
        avg = {k: float(np.mean([p[k] for p in parts])) for k in parts[0]}
        logger.log(epoch=epoch, lr=opt.lr, train_loss=np.mean(totals), val_loss=v_loss,
                   l_obj=avg["obj"], l_xy=avg["xy"], l_wh=avg["wh"], l_cls=avg["cls"],
                   val_map50=v_ap, val_map5095=map_coco(pv, bva, nc),
                   val_auc=auc(image_scores(pv), y_val), val_sens=m["sensitivity"], val_spec=m["specificity"],
                   val_f1=m["f1"], val_recall=tp / max(1, tp + fn), val_prec=tp / max(1, tp + fp),
                   val_fppi=fp / len(bva), time=time.time() - t0)

        h = logger.history                                                      # stop conditions
        if abort_after and epoch + 1 >= abort_after and max(h["val_map50"]) < 0.01 and max(h["val_auc"]) < 0.6:
            print(f"stopping: no learning signal after {epoch + 1} epochs")
            break
        if patience and epoch - h["epoch"][int(np.nanargmax(h["val_map50"]))] >= patience:
            print(f"stopping: val mAP50 has not improved for {patience} epochs")
            break

    if ckpt_dir and logger.history["epoch"]:
        plot_det_metrics(logger.history, thr=log_thr, save_path=str(Path(ckpt_dir) / "training_curves.png"), show=False)

    # ---- final: best-by-validation checkpoint; confidence cutoff chosen on validation; test scored once (if asked) ----
    if mgr and os.path.exists(os.path.join(ckpt_dir, "best.npz")):
        load_checkpoint(os.path.join(ckpt_dir, "best.npz"), model)
    if bn:
        recalibrate_bn(model, Xcal, bcal)                                       # stats stored in best.npz are stale
    pv = predict_dataset(model, Xva, bva, nc)
    thr = pick_threshold(image_scores(pv), y_val, min_sensitivity)
    report_det("validation", pv, bva, thr, nc)
    res, splits = {}, [("val", Xva, bva, pv)]
    if eval_test:
        pt = predict_dataset(model, Xte, bte, nc)
        res = report_det("TEST", pt, bte, thr, nc)
        splits.append(("test", Xte, bte, pt))
    if ckpt_dir:
        for nm, X, g, p in splits:
            fp_ = f"{data_dir}/{nm}_files.npy"
            files = np.load(fp_) if os.path.exists(fp_) else None
            save_examples(nm, X, g, p, files, max(thr, viz_min_conf), ckpt_dir, class_names)
    return dict(history=logger.history, **res)

if __name__ == "__main__":
    main(epochs=2, ckpt_dir="C:/ckpts/det_run1")