import glob, json, os, tempfile
import numpy as np

from layers import get_param_names

def model_state(model):
    # references to every parameter, keyed by position and layer type
    state = {}
    for i, l in enumerate(model.layers):
        for n in get_param_names(l) + getattr(l, "buffer_names", ()):
            state[f"layers.{i}.{type(l).__name__}.{n}"] = getattr(l, n)
    return state

def load_model_state(model, state):
    expected = model_state(model)
    if set(expected) != set(state):
        raise ValueError("architecture mismatch:\n"
                         f"  missing in checkpoint: {sorted(set(expected) - set(state))}\n"
                         f"  unexpected in checkpoint: {sorted(set(state) - set(expected))}")
    for k, arr in expected.items():                       # validate everything first
        if arr.shape != state[k].shape:
            raise ValueError(f"shape mismatch for {k}: model {arr.shape} vs checkpoint {state[k].shape}")
    for k, arr in expected.items():                       # write in place
        arr[...] = state[k]

def save_checkpoint(path, model, opt, epoch, rng, extra=None):
    arrays = {f"model.{k}": v for k, v in model_state(model).items()}
    arrays.update({f"optim.{k}": v for k, v in opt.state_dict().items()})
    meta = {"epoch": epoch, "rng": rng.bit_generator.state, "extra": extra or {},
            "layer_rng": {str(i): l.rng.bit_generator.state
                          for i, l in enumerate(model.layers) if hasattr(l, "rng")}}
    arrays["meta"] = np.array(json.dumps(meta))

    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")      # same directory -> same filesystem
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez(f, **arrays)                         # uncompressed
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)                             # atomic swap
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise

def load_checkpoint(path, model, opt=None, rng=None):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        m = {k[len("model."):]: z[k] for k in z.files if k.startswith("model.")}
        o = {k[len("optim."):]: z[k] for k in z.files if k.startswith("optim.")}
    saved = meta.get("layer_rng", {})
    have = {str(i) for i, l in enumerate(model.layers) if hasattr(l, "rng")}
    if set(saved) != have:
        raise ValueError(f"layer RNG mismatch: checkpoint has {sorted(saved)}, model has {sorted(have)}")
    load_model_state(model, m)
    if opt is not None:
        opt.load_state_dict(o)
    if rng is not None:
        rng.bit_generator.state = meta["rng"]
    for i, st in saved.items():
        model.layers[int(i)].rng.bit_generator.state = st
    return meta


class CheckpointManager:
    # Keeps the last N epoch checkpoints plus best.npz (higher metric is better)
    def __init__(self, directory, keep_last=3):
        self.dir, self.keep_last = directory, keep_last
        os.makedirs(directory, exist_ok=True)
        self.best_metric = -np.inf
        best = os.path.join(directory, "best.npz")
        if os.path.exists(best):                          # survive restarts without losing a better best
            with np.load(best, allow_pickle=False) as z:
                self.best_metric = json.loads(str(z["meta"]))["extra"].get("metric", -np.inf)

    def _all(self):
        return sorted(glob.glob(os.path.join(self.dir, "ckpt_*.npz")))

    def latest(self):
        files = self._all()
        return files[-1] if files else None

    def save(self, model, opt, epoch, rng, metric=None):
        extra = {"metric": float(metric)} if metric is not None else {}
        save_checkpoint(os.path.join(self.dir, f"ckpt_{epoch:04d}.npz"), model, opt, epoch, rng, extra)
        for old in self._all()[:-self.keep_last]:
            os.remove(old)
        if metric is not None and metric > self.best_metric:
            self.best_metric = float(metric)
            save_checkpoint(os.path.join(self.dir, "best.npz"), model, opt, epoch, rng, extra)