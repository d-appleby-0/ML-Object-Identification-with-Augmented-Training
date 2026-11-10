import numpy as np
from pathlib import Path

def build_cache(img_dir, label_dir, prefix, size=128):
    from PIL import Image                                  # pip install pillow (DICOM would need pydicom)
    files = sorted(p for p in Path(img_dir).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    X = np.lib.format.open_memmap(prefix + "_x.npy", mode="w+", dtype=np.uint8, shape=(len(files), size, size))
    boxes = []
    for i, f in enumerate(files):
        X[i] = np.asarray(Image.open(f).convert("L").resize((size, size), Image.BILINEAR))
        lab = Path(label_dir) / (f.stem + ".txt")
        has = lab.exists() and lab.stat().st_size > 0      # negatives often have an empty or missing label file
        boxes.append(read_yolo_label(lab) if has else np.zeros((0, 5), dtype=np.float32))
    X.flush()
    save_boxes(prefix, boxes)

def load_cache(prefix):
    X = np.load(prefix + "_x.npy", mmap_mode="r")
    z = np.load(prefix + "_boxes.npz")
    off, allb = z["offsets"], z["boxes"]
    return X, [allb[off[i]:off[i + 1]] for i in range(len(X))]

def xray_batches(X, boxes, batch_size, rng=None, augment=None, train=True):
    order = rng.permutation(len(X)) if train else np.arange(len(X))
    for i in range(0, len(order), batch_size):
        xs, ys = [], []
        for j in order[i:i + batch_size]:
            img, b = X[j].astype(np.float64) / 255.0, boxes[j]
            if augment is not None:
                img, b = augment(img, b, rng)
            xs.append(img[None])
            ys.append(int(len(b) > 0))               # image-level label from surviving boxes
        yield np.stack(xs), np.array(ys)


def read_yolo_label(path):
    # (K, 5) rows [cls, xc, yc, w, h], polygon lines (YOLO segmentation) become their bounding box
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            v = line.split()
            if len(v) == 5:
                rows.append([float(x) for x in v])
            elif len(v) > 5 and len(v) % 2 == 1:
                xy = np.array(v[1:], dtype=float).reshape(-1, 2)
                x0, y0 = xy.min(0); x1, y1 = xy.max(0)
                rows.append([float(v[0]), (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0])
    return np.array(rows).reshape(-1, 5)



# Helper to save/load boxes in a single file with offsets for each image

def save_boxes(prefix, boxes):
    np.savez(prefix + "_boxes.npz",
             boxes=np.concatenate(boxes) if len(boxes) else np.zeros((0, 5)),
             offsets=np.cumsum([0] + [len(b) for b in boxes]))


# Test data

def synthetic_xray_data(n=64, size=16, seed=0):
    # Random noise images with a bright square 'tumor' in every other image, tests and smoke runs only
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 120, (n, size, size)).astype(np.uint8)
    boxes = []
    for i in range(n):
        if i % 2:
            s = size // 4
            x0, y0 = rng.integers(2, size - s - 2, 2)
            X[i, y0:y0 + s, x0:x0 + s] = 255
            boxes.append(np.array([[0, (x0 + s / 2) / size, (y0 + s / 2) / size, s / size, s / size]]))
        else:
            boxes.append(np.zeros((0, 5)))
    return X, boxes

"""
# Temp refactor for testing:

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def load_data(test_size=0.2, seed=0):
    X, y = load_digits(return_X_y=True)
    X = (X / 16.0).reshape(-1, 1, 8, 8)            # (N, C, H, W)
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

def batches(X, y, batch_size, rng):
    perm = rng.permutation(len(X))
    for i in range(0, len(X), batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]

"""