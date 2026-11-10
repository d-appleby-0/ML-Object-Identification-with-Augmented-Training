import numpy as np
import matplotlib.pyplot as plt

def plot_det_metrics(history, thr=0.25, save_path=None, show=True):
    e = np.array(history["epoch"])
    best = e[int(np.nanargmax(history["val_map50"]))]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    ax = axes.ravel()

    def draw(a, key, label, color, marker, **kw):
        a.plot(e, history[key], marker=marker, color=color, label=label, **kw)

    def finish(a, title, ylabel, ylim=None, legend=True):
        a.axvline(best, color="gray", linestyle=":", alpha=0.7)           # best epoch
        a.set_title(title); a.set_xlabel("Epoch"); a.set_ylabel(ylabel)
        if ylim:
            a.set_ylim(*ylim)
        a.grid(True, linestyle="--", alpha=0.6)
        if legend:
            a.legend(loc="best")

    draw(ax[0], "train_loss", "Train (augmented)", "crimson", "o")
    draw(ax[0], "val_loss", "Val (clean, eval mode)", "navy", "s")
    finish(ax[0], "Total loss", "Loss")

    for key, label, color in (("l_obj", "objectness", "crimson"), ("l_xy", "center xy", "teal"),
                              ("l_wh", "size wh", "darkorange"), ("l_cls", "class", "purple")):
        if max(history[key]) > 0:                                          # skip the class term when nc == 1
            draw(ax[1], key, label, color, ".")
    finish(ax[1], "Training loss components", "Loss")

    draw(ax[2], "val_map50", "mAP@0.5", "navy", "s")
    draw(ax[2], "val_map5095", "mAP@[0.5:0.95]", "teal", "^")
    finish(ax[2], "Detection mAP (validation)", "mAP", (0, 1))

    draw(ax[3], "val_auc", "Image AUC (max confidence)", "navy", "s")
    draw(ax[3], "val_f1", f"F1 (conf >= {thr})", "teal", "^")
    finish(ax[3], "Image-level: does the image contain a tumor?", "Score", (0, 1))

    draw(ax[4], "val_sens", "Sensitivity", "darkorange", "o")
    draw(ax[4], "val_spec", "Specificity", "purple", "d")
    finish(ax[4], f"Sensitivity vs Specificity (conf >= {thr})", "Rate", (0, 1))

    draw(ax[5], "val_recall", "Tumor recall", "darkorange", "o")
    draw(ax[5], "val_prec", "Box precision", "navy", "s")
    twin = ax[5].twinx()
    twin.plot(e, history["val_fppi"], color="crimson", linestyle="--", marker="x", label="False boxes / image")
    twin.set_ylabel("False boxes per image")
    finish(ax[5], f"Tumor-level (IoU >= 0.5, conf >= {thr})", "Rate", (0, 1), legend=False)
    h1, l1 = ax[5].get_legend_handles_labels()
    h2, l2 = twin.get_legend_handles_labels()
    ax[5].legend(h1 + h2, l1 + l2, loc="best")

    fig.suptitle(f"best epoch {best} (val mAP@0.5 {max(history['val_map50']):.3f})")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
        print(f"Saved metric plots to {save_path}")
    if show:
        plt.show()
    plt.close(fig)