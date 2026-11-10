import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from detect import detection_counts

def pick_examples(preds, gts, conf_thr, per_group=4, iou_thr=0.5):
    groups = {"found": [], "missed": [], "false alarm": [], "clean": []}
    for i, (p, g) in enumerate(zip(preds, gts)):
        _, fp, fn = detection_counts([p], [g], conf_thr, iou_thr)
        top = float(p[:, 1].max()) if len(p) else 0.0
        key = ("false alarm" if fp else "clean") if len(g) == 0 else ("missed" if fn else "found")
        groups[key].append((i, top))
    picks = []
    for name, items in groups.items():
        items.sort(key=lambda t: -t[1])                                 # most confident first
        picks += [(i, name) for i, _ in items[:per_group]]
    return picks

def show_detections(images, gts, preds, titles, path=None, cols=4, conf_thr=0.25, class_names=None, show=False):
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, img, g, p, t in zip(axes.ravel(), images, gts, preds, titles):
        H, W = img.shape
        ax.imshow(img, cmap="gray", vmin=0, vmax=255 if img.dtype == np.uint8 else 1.0)
        ax.set_title(t, fontsize=9)
        for c, xc, yc, w, h in g:                                       # ground truth - green dashed
            ax.add_patch(Rectangle(((xc - w / 2) * W, (yc - h / 2) * H), w * W, h * H,
                                   fill=False, edgecolor="lime", linestyle="--", linewidth=1.5))
        for c, conf, x1, y1, x2, y2 in (p[p[:, 1] >= conf_thr] if len(p) else []):     # predictions - red
            ax.add_patch(Rectangle((x1 * W, y1 * H), (x2 - x1) * W, (y2 - y1) * H,
                                   fill=False, edgecolor="red", linewidth=2))
            name = class_names[int(c)] if class_names else "tumor"
            ax.text(x1 * W, max(0.0, y1 * H - 2), f"{name} {conf:.2f}", color="white", fontsize=8,
                    va="bottom", bbox=dict(facecolor="red", alpha=0.75, pad=1, edgecolor="none"))
    fig.suptitle(f"green dashed = ground truth   red = prediction (confidence >= {conf_thr:.2f})")
    if path:
        fig.savefig(path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

def save_examples(name, X, gts, preds, files, conf_thr, out_dir, class_names=None):
    # Original full-resolution images when the file list is available (boxes are normalized, so they transfer)
    from PIL import Image
    picks = pick_examples(preds, gts, conf_thr)
    images, titles = [], []
    for i, tag in picks:
        if files is not None:
            im = Image.open(files[i]).convert("L")
            im.thumbnail((512, 512))
            images.append(np.asarray(im))
        else:
            images.append(np.asarray(X[i]))
        titles.append(f"{tag} | {Path(str(files[i])).name if files is not None else i}")
    show_detections(images, [gts[i] for i, _ in picks], [preds[i] for i, _ in picks], titles,
                    path=str(Path(out_dir) / f"detections_{name}.png"), conf_thr=conf_thr, class_names=class_names)