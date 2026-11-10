import json
from pathlib import Path
import numpy as np

class TrainLogger:
    # columns: list of (key, header, format). resume_from: first epoch about to be (re)run, or None for a fresh log
    def __init__(self, out_dir, columns, resume_from=None, verbose=True):
        self.columns, self.verbose, self._header_printed = columns, verbose, False
        self.history = {k: [] for k, _, _ in columns}
        self.dir = Path(out_dir) if out_dir else None
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)
            js = self.dir / "metrics_history.json"
            if resume_from and js.exists():                          # continue an interrupted run
                old = json.loads(js.read_text())
                keep = [i for i, e in enumerate(old["epoch"]) if e < resume_from]
                for k in self.history:
                    col = old.get(k, [float("nan")] * len(old["epoch"]))
                    self.history[k] = [col[i] for i in keep]

    def _header(self):
        return " | ".join(h for _, h, _ in self.columns)

    def _line(self, i):
        return " | ".join(f.format(self.history[k][i]) for k, _, f in self.columns)

    def log(self, **row):
        for k, _, _ in self.columns:
            v = row[k]
            self.history[k].append(int(v) if isinstance(v, (int, np.integer)) else float(v))
        if self.verbose:
            if not self._header_printed:
                print(self._header())
                self._header_printed = True
            print(self._line(len(self.history["epoch"]) - 1))
        self._flush()

    def _flush(self):                                                # rewritten whole each epoch: tiny, and always consistent
        if not self.dir:
            return
        n = len(self.history["epoch"])
        lines = [self._header(), "-" * 150] + [self._line(i) for i in range(n)]
        (self.dir / "training_log.txt").write_text("\n".join(lines) + "\n")
        (self.dir / "metrics_history.json").write_text(json.dumps(self.history, indent=4))


DET_COLUMNS = [
    ("epoch", "Epoch", "{:5d}"), ("lr", "LR", "{:.2e}"),
    ("train_loss", "Train Loss", "{:10.4f}"), ("val_loss", "Val Loss", "{:8.4f}"),
    ("l_obj", "obj", "{:6.3f}"), ("l_xy", "xy", "{:6.3f}"), ("l_wh", "wh", "{:6.3f}"), ("l_cls", "cls", "{:6.3f}"),
    ("val_map50", "mAP50", "{:6.4f}"), ("val_map5095", "mAP50-95", "{:8.4f}"),
    ("val_auc", "img AUC", "{:7.4f}"), ("val_sens", "Sens", "{:6.4f}"), ("val_spec", "Spec", "{:6.4f}"),
    ("val_f1", "F1", "{:6.4f}"),
    ("val_recall", "Recall", "{:6.4f}"), ("val_prec", "Prec", "{:6.4f}"), ("val_fppi", "FP/img", "{:6.3f}"),
    ("time", "Time (s)", "{:7.1f}"),
]