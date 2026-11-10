import numpy as np
from scipy.ndimage import affine_transform               # scipy ships with scikit-learn

def clip_boxes(boxes, min_keep=0.5):
    # Clip normalized [cls, xc, yc, w, h] boxes to the frame, drop those that lose too much area
    if len(boxes) == 0:
        return boxes
    x1, x2 = boxes[:, 1] - boxes[:, 3] / 2, boxes[:, 1] + boxes[:, 3] / 2
    y1, y2 = boxes[:, 2] - boxes[:, 4] / 2, boxes[:, 2] + boxes[:, 4] / 2
    cx1, cx2, cy1, cy2 = (np.clip(v, 0, 1) for v in (x1, x2, y1, y2))
    kept = np.clip(cx2 - cx1, 0, None) * np.clip(cy2 - cy1, 0, None)
    out = np.stack([boxes[:, 0], (cx1 + cx2) / 2, (cy1 + cy2) / 2, cx2 - cx1, cy2 - cy1], axis=1)
    return out[kept >= min_keep * (x2 - x1) * (y2 - y1)]

class Augmenter:
    def __init__(self, hflip=True, shift=0.08, scale=(0.9, 1.1),
                 contrast=(0.85, 1.15), brightness=0.08, gamma=(0.8, 1.25), noise=0.01):
        self.hflip, self.shift, self.scale = hflip, shift, scale
        self.contrast, self.brightness, self.gamma, self.noise = contrast, brightness, gamma, noise

    def __call__(self, img, boxes, rng):
        # img: (H, W) float in [0, 1]. boxes: (K, 5) - returns a new (img, boxes) pair
        H, W = img.shape
        boxes = boxes.copy()

        if self.hflip and rng.random() < 0.5:
            img = img[:, ::-1]
            boxes[:, 1] = 1.0 - boxes[:, 1]

        # random scale + translate as one affine warp (output pixel o samples input at o/s + c - (c+t)/s)
        s = rng.uniform(*self.scale)
        tx, ty = rng.uniform(-self.shift, self.shift, 2) * (W, H)          # pixels
        c, t = np.array([H / 2, W / 2]), np.array([ty, tx])
        img = affine_transform(img, np.diag([1 / s, 1 / s]), offset=c - (c + t) / s, order=1, mode="nearest")
        boxes[:, 1] = (boxes[:, 1] - 0.5) * s + 0.5 + tx / W
        boxes[:, 2] = (boxes[:, 2] - 0.5) * s + 0.5 + ty / H
        boxes[:, 3:5] *= s
        boxes = clip_boxes(boxes)

        # intensity (X-ray exposure and detector differences): contrast, brightness, gamma, noise
        m = img.mean()
        img = (img - m) * rng.uniform(*self.contrast) + m + rng.uniform(-self.brightness, self.brightness)
        img = np.clip(img, 0, 1) ** rng.uniform(*self.gamma)
        img = img + self.noise * rng.standard_normal(img.shape)
        return np.clip(img, 0, 1), boxes