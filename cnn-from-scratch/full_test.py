import tempfile
import numpy as np

from detect import encode_targets, decode, average_precision, nms, to_xyxy, iou_matrix, YoloLoss
from train_det import build_det_model, gradcheck_gate_det, predict_dataset, receptive_field
from data import synthetic_det_data, xray_det_batches, save_boxes
from optim import Adam
from train_det import main

from gradcheck import numerical_grad, rel_err
from detect import YoloLoss

def loss_gradcheck(nc=2):
    rng = np.random.default_rng(0)
    S = 8
    boxes = [np.array([[0, .30, .55, .20, .25], [nc - 1, .80, .20, .10, .10]]),
             np.zeros((0, 5)), np.array([[nc - 1, .52, .48, .30, .30]])]
    pred = rng.standard_normal((3, 5 + (nc if nc > 1 else 0), S, S))
    loss = YoloLoss(nc)
    f = lambda: loss.forward(pred, boxes)
    f()
    ana = loss.backward()                                  # computed before the numerical pass overwrites the cache
    num = numerical_grad(f, pred)
    names = ["obj", "tx", "ty", "tw", "th"] + [f"cls{k}" for k in range(nc if nc > 1 else 0)]
    for c, n in enumerate(names):
        print(f"  {n:<5} rel err {rel_err(num[:, c], ana[:, c]):.2e}")

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def detect_tests():
    boxes = [np.array([[0, .30, .55, .20, .25], [1, .80, .20, .10, .10]]), np.zeros((0, 5)),
             np.array([[1, .52, .48, .30, .30]])]
    S, nc = 8, 2

    # 1) encode -> "perfect" network output -> decode must recover the ground truth (checks every convention)
    obj, box, cls = encode_targets(boxes, S)
    pred = np.zeros((3, 5 + nc, S, S))
    pred[:, 0] = np.where(obj > 0.5, 12.0, -12.0)
    pred[:, 1:5] = logit(box)
    pred[:, 5:] = np.where(cls[:, None] == np.arange(nc).reshape(1, -1, 1, 1), 12.0, -12.0)
    out = decode(pred, nc, conf_thr=0.5)
    for g, d in zip(boxes, out):
        assert len(d) == len(g), "wrong number of decoded boxes"
        if len(g):
            ious = iou_matrix(to_xyxy(g[:, 1:5]), d[:, 2:6])
            assert np.all(ious.max(axis=1) > 0.99), "decoded box does not match ground truth"
            assert set(g[:, 0].astype(int)) == set(d[:, 0].astype(int)), "wrong classes"
    print("  encode/decode round trip OK")

    # 2) NMS keeps the more confident of two overlapping boxes, and keeps boxes of a different class
    b = np.array([[0, 0, .5, .5], [.02, .02, .52, .52], [.6, .6, .9, .9], [0, 0, .5, .5]])
    keep = nms(b, np.array([.9, .8, .7, .6]), np.array([0, 0, 0, 1]), 0.5)
    assert sorted(keep.tolist()) == [0, 2, 3], keep

    # 3) AP: perfect detections -> 1, displaced detections -> 0
    perfect = [np.column_stack([g[:, 0], np.full(len(g), .9), to_xyxy(g[:, 1:5])]) if len(g) else np.zeros((0, 6))
               for g in boxes]
    assert average_precision(perfect, boxes, 0) == 1.0 and average_precision(perfect, boxes, 1) == 1.0
    moved = [np.column_stack([p[:, :2], p[:, 2:6] * 0 + 0.97]) if len(p) else p for p in perfect]
    assert average_precision(moved, boxes, 0) == 0.0
    print("  NMS and AP OK")

    # 4) numerical gradient check of network + YoloLoss, with the class branch active
    gradcheck_gate_det(nc=2, bn=True)

def overfit_test(steps=300):
    # A working detector must be able to memorize 8 images - if not, something upstream is wrong
    X, boxes = synthetic_det_data(8, 32, seed=0)
    boxes = [b if len(b) else np.array([[0, .5, .5, .25, .25]]) for b in boxes]       # make sure every image has a target
    for i, b in enumerate(boxes):                                                    # draw the target if we added one
        if X[i].max() < 180:
            X[i, 12:20, 12:20] = 255
    model = build_det_model(32, (4, 8), (8,), 1, seed=0, bn=True)
    loss, opt = YoloLoss(1), Adam(model.trainable(), 3e-3)
    xb = X.astype(np.float64)[:, None] / 255.0
    for step in range(steps):
        l = loss.forward(model.forward(xb), boxes)
        model.backward(loss.backward())
        opt.step()
        if step % 50 == 0:
            print(f"  step {step:3d}  loss {l:.3f}")
    pred = decode(model.infer(xb), 1, conf_thr=0.01)
    ap = average_precision(pred, boxes, 0)
    print(f"  AP@0.5 after memorizing 8 images: {ap:.3f}")
    assert ap > 0.7, "cannot overfit 8 images (try 600 steps before suspecting a bug)"

def smoke_test_det():
    with tempfile.TemporaryDirectory() as d:
        for split, n, seed in (("train", 300, 0), ("val", 60, 1), ("test", 60, 2)):
            X, boxes = synthetic_det_data(n, 64, seed)
            np.save(f"{d}/{split}_x.npy", X)
            save_boxes(f"{d}/{split}", boxes)
        main(epochs=25, size=64, channels=(4, 8, 16), extra=(16,), nc=1,
             data_dir=d, ckpt_dir="smoke_det_out", lr=2e-3)

if __name__ == "__main__":
    loss_gradcheck()
    for s in range(4):
        try:
            gradcheck_gate_det(2, True, seed=s)
        except AssertionError:
            print(f"seed {s}: FAILED\n")
    detect_tests()
    overfit_test()
    smoke_test_det()