import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))

def to_xyxy(b):
    # (K, 4) [xc, yc, w, h] -> (K, 4) [x1, y1, x2, y2]
    return np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                     b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], axis=1)

def iou_matrix(a, b):
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-12)

def n_out(nc):
    return 5 + (nc if nc > 1 else 0)

def encode_targets(boxes_list, S):
    # Per image - which cell owns each box and what it should predict,  one box per cell (the larger wins)
    N = len(boxes_list)
    obj = np.zeros((N, S, S)); box = np.zeros((N, 4, S, S)); cls = np.zeros((N, S, S), dtype=int)
    area = np.zeros((N, S, S))
    for n, b in enumerate(boxes_list):
        for c, xc, yc, w, h in b:
            gi = int(np.clip(xc * S, 0, S - 1)); gj = int(np.clip(yc * S, 0, S - 1))    # column (x), row (y)
            if w * h >= area[n, gj, gi]:
                area[n, gj, gi] = w * h
                obj[n, gj, gi] = 1.0
                box[n, :, gj, gi] = (xc * S - gi, yc * S - gj, w, h)
                cls[n, gj, gi] = int(c)
    return obj, box, cls


class YoloLoss:
    def __init__(self, nc=1, lambda_coord=5.0, lambda_noobj=0.5):
        self.nc, self.lc, self.ln = nc, lambda_coord, lambda_noobj
        self.parts = {}

    def forward(self, pred, boxes_list):
        N, _, S, _ = pred.shape
        obj_t, box_t, cls_t = encode_targets(boxes_list, S)
        pos = obj_t > 0.5
        o = pred[:, 0]
        wgt = np.where(pos, 1.0, self.ln)
        bce = np.maximum(o, 0) - o * obj_t + np.log1p(np.exp(-np.abs(o)))
        sx, sy, sw, sh = (sigmoid(pred[:, k]) for k in (1, 2, 3, 4))
        rw, rh = np.sqrt(sw), np.sqrt(sh)
        tw, th = np.sqrt(box_t[:, 2]), np.sqrt(box_t[:, 3])

        l_obj = (wgt * bce).sum() / N
        l_xy = self.lc * (pos * ((sx - box_t[:, 0]) ** 2 + (sy - box_t[:, 1]) ** 2)).sum() / N
        l_wh = self.lc * (pos * ((rw - tw) ** 2 + (rh - th) ** 2)).sum() / N
        l_cls, p, onehot = 0.0, None, None
        if self.nc > 1:
            z = pred[:, 5:5 + self.nc]
            z = z - z.max(axis=1, keepdims=True)
            e = np.exp(z)
            p = e / e.sum(axis=1, keepdims=True)
            onehot = cls_t[:, None] == np.arange(self.nc).reshape(1, -1, 1, 1)
            l_cls = -(pos[:, None] * onehot * np.log(p + 1e-12)).sum() / N

        self.parts = dict(obj=float(l_obj), xy=float(l_xy), wh=float(l_wh), cls=float(l_cls))
        self.c = (pred, N, pos, wgt, obj_t, box_t, sx, sy, sw, sh, rw, rh, tw, th, p, onehot)
        return l_obj + l_xy + l_wh + l_cls

    def backward(self):
        pred, N, pos, wgt, obj_t, box_t, sx, sy, sw, sh, rw, rh, tw, th, p, onehot = self.c
        d = np.zeros_like(pred)
        d[:, 0] = wgt * (sigmoid(pred[:, 0]) - obj_t) / N
        d[:, 1] = self.lc * 2 * (sx - box_t[:, 0]) * sx * (1 - sx) * pos / N
        d[:, 2] = self.lc * 2 * (sy - box_t[:, 1]) * sy * (1 - sy) * pos / N
        d[:, 3] = self.lc * (rw - tw) * rw * (1 - sw) * pos / N          # d/dt of (sqrt(sigmoid(t)) - c)^2
        d[:, 4] = self.lc * (rh - th) * rh * (1 - sh) * pos / N
        if self.nc > 1:
            d[:, 5:5 + self.nc] = (p - onehot) * pos[:, None] / N
        return d


def nms(boxes, scores, classes, iou_thr=0.5):
    order, keep = np.argsort(-scores), []
    while len(order):
        i = order[0]
        keep.append(i)
        rest = order[1:]
        if len(rest) == 0:
            break
        ious = iou_matrix(boxes[i][None], boxes[rest])[0]
        order = rest[~((ious >= iou_thr) & (classes[rest] == classes[i]))]
    return np.array(keep, dtype=int)

def decode(pred, nc=1, conf_thr=0.05, iou_thr=0.5, max_det=20):
    # pred (N, 5+, S, S) -> list of (M, 6) arrays [cls, conf, x1, y1, x2, y2], normalized coordinates.
    N, _, S, _ = pred.shape
    gx, gy = np.meshgrid(np.arange(S), np.arange(S))                    # gx[j, i] = column i, gy[j, i] = row j
    obj = sigmoid(pred[:, 0])
    xc, yc = (sigmoid(pred[:, 1]) + gx) / S, (sigmoid(pred[:, 2]) + gy) / S
    w, h = sigmoid(pred[:, 3]), sigmoid(pred[:, 4])
    if nc > 1:
        z = pred[:, 5:5 + nc]
        e = np.exp(z - z.max(axis=1, keepdims=True))
        cp = e / e.sum(axis=1, keepdims=True)
        cls, conf = cp.argmax(axis=1), obj * cp.max(axis=1)
    else:
        cls, conf = np.zeros((N, S, S), dtype=int), obj
    out = []
    for n in range(N):
        k = conf[n] >= conf_thr
        boxes = np.clip(np.stack([xc[n][k] - w[n][k] / 2, yc[n][k] - h[n][k] / 2,
                                  xc[n][k] + w[n][k] / 2, yc[n][k] + h[n][k] / 2], axis=1), 0, 1)
        sc, cl = conf[n][k], cls[n][k]
        keep = nms(boxes, sc, cl, iou_thr)[:max_det] if len(sc) else np.zeros(0, dtype=int)
        out.append(np.column_stack([cl[keep], sc[keep], boxes[keep]]) if len(keep) else np.zeros((0, 6)))
    return out


def average_precision(preds, gts, cls=0, iou_thr=0.5):
    # VOC-style AP for one class: sort detections by confidence, match greedily to ground truth
    gt = [to_xyxy(g[g[:, 0] == cls][:, 1:5]) if len(g) else np.zeros((0, 4)) for g in gts]
    n_gt = sum(len(b) for b in gt)
    dets = [(r[1], i, r[2:6]) for i, p in enumerate(preds) for r in p[p[:, 0] == cls]]
    dets.sort(key=lambda d: -d[0])
    used = [np.zeros(len(b), bool) for b in gt]
    tp, fp = np.zeros(len(dets)), np.zeros(len(dets))
    for k, (_, i, box) in enumerate(dets):
        if len(gt[i]):
            ious = iou_matrix(box[None], gt[i])[0]
            j = ious.argmax()
            if ious[j] >= iou_thr and not used[i][j]:
                used[i][j], tp[k] = True, 1
                continue
        fp[k] = 1
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    rec, prec = ctp / max(1, n_gt), ctp / np.maximum(1, ctp + cfp)
    mrec, mpre = np.concatenate([[0], rec, [1]]), np.concatenate([[0], prec, [0]])
    for k in range(len(mpre) - 2, -1, -1):
        mpre[k] = max(mpre[k], mpre[k + 1])                             # monotone precision envelope
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

def mean_ap(preds, gts, nc=1, iou_thr=0.5):
    present = [c for c in range(nc) if any(len(g) and np.any(g[:, 0] == c) for g in gts)]
    return float(np.mean([average_precision(preds, gts, c, iou_thr) for c in present])) if present else 0.0

def image_scores(preds):
    # Image-level tumor score = highest detection confidence (0 if nothing detected)
    return np.array([p[:, 1].max() if len(p) else 0.0 for p in preds])

def detection_counts(preds, gts, conf_thr, iou_thr=0.5):
    # Tumor-level true positives, false positives and misses at a confidence threshold
    tp = fp = fn = 0
    for p, g in zip(preds, gts):
        p = p[p[:, 1] >= conf_thr]
        gb = to_xyxy(g[:, 1:5]) if len(g) else np.zeros((0, 4))
        gc = g[:, 0] if len(g) else np.zeros(0)
        used = np.zeros(len(gb), bool)
        for row in p[np.argsort(-p[:, 1])]:
            hit = False
            if len(gb):
                ious = np.where((gc == row[0]) & ~used, iou_matrix(row[None, 2:6], gb)[0], -1.0)
                j = ious.argmax()
                if ious[j] >= iou_thr:
                    used[j], hit = True, True
            tp += hit
            fp += not hit
        fn += int((~used).sum())
    return tp, fp, fn

def map_coco(preds, gts, nc=1):
    # mAP averaged over IoU thresholds 0.50, 0.55, ..., 0.95. Rewards tight boxes, not just roughly right ones
    return float(np.mean([mean_ap(preds, gts, nc, t) for t in np.arange(0.5, 0.96, 0.05)]))