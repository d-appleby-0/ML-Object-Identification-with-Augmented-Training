import tempfile
import os
import numpy as np

from checkpoint import save_checkpoint, load_checkpoint, model_state
from train import build_xray_model
from losses import SoftmaxCE
from augment import Augmenter
from data import synthetic_xray_data, xray_batches
from optim import Adam
from layers import get_param_names

def checkpoint_test():
    X, boxes = synthetic_xray_data(64, 16)
    aug, loss = Augmenter(), SoftmaxCE()
    mk_model = lambda seed, ch=(4, 4): build_xray_model(16, ch, seed, p_drop=0.3, bn=True)
    mk_opt = lambda m: Adam(m.trainable(), 0.01)

    def run(model, opt, rng, start, stop):
        for _ in range(start, stop):
            for xb, yb in xray_batches(X, boxes, 16, rng, augment=aug):
                loss.forward(model.forward(xb), yb)
                model.backward(loss.backward())
                opt.step()



    mA = mk_model(0); oA = mk_opt(mA); rA = np.random.default_rng(0)
    run(mA, oA, rA, 0, 6)                                 # uninterrupted

    mB = mk_model(0); oB = mk_opt(mB); rB = np.random.default_rng(0)
    run(mB, oB, rB, 0, 3)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "c.npz")
        save_checkpoint(path, mB, oB, 2, rB)

        mC = mk_model(123); oC = mk_opt(mC); rC = np.random.default_rng(999)   # all different on purpose
        meta = load_checkpoint(path, mC, oC, rC)
        run(mC, oC, rC, meta["epoch"] + 1, 6)

        for la, lc in zip(mA.layers, mC.layers):
            for n in get_param_names(la) + getattr(la, "buffer_names", ()):
                assert np.allclose(getattr(la, n), getattr(lc, n), rtol=0, atol=1e-12), f"{n} diverged"
        print("  resume matches uninterrupted run")

        # wrong architecture must be rejected without modifying the model
        m2 = mk_model(0, ch=(4, 4, 4))
        before = {k: v.copy() for k, v in model_state(m2).items()}      # was [l.W.copy() for l in m2.trainable()]
        try:
            load_checkpoint(path, m2, mk_opt(m2))
            raise AssertionError("architecture mismatch not detected")
        except ValueError as e:
            print("  correctly rejected:", str(e).splitlines()[0])
        after = model_state(m2)
        assert all(np.array_equal(before[k], after[k]) for k in before), "failed load modified the model"

if __name__ == "__main__":
    checkpoint_test()