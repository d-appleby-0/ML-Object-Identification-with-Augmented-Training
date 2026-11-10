import numpy as np

from augment import Augmenter

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)

def augment_box_test():
    rng = np.random.default_rng(0)
    aug = Augmenter(contrast=(1, 1), brightness=0, gamma=(1, 1), noise=0)      # geometry only
    H = W = 128
    worst = 1.0
    for _ in range(200):
        img = np.zeros((H, W))
        x1, y1 = rng.integers(30, 60, 2)
        x2, y2 = x1 + rng.integers(15, 30), y1 + rng.integers(15, 30)
        img[y1:y2, x1:x2] = 1.0
        box = np.array([[0, (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H]])
        out_img, out_box = aug(img, box, rng)
        assert len(out_box) == 1
        ys, xs = np.nonzero(out_img > 0.5)
        got = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]) / [W, H, W, H]
        b = out_box[0]
        want = np.array([b[1] - b[3] / 2, b[2] - b[4] / 2, b[1] + b[3] / 2, b[2] + b[4] / 2])
        worst = min(worst, iou(got, want))
    print(f"  worst IoU over 200 augmentations: {worst:.3f}")
    assert worst > 0.8

if __name__ == "__main__":
    augment_box_test()