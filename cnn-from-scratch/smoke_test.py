# smoke_test.py
import tempfile
import numpy as np

from data import synthetic_xray_data, save_boxes
from train import main

def smoke_test(**kw):
    with tempfile.TemporaryDirectory() as d:
        for split, n, seed in (("train", 256, 0), ("val", 64, 1), ("test", 64, 2)):
            X, boxes = synthetic_xray_data(n, 32, seed)
            np.save(f"{d}/{split}_x.npy", X)
            save_boxes(f"{d}/{split}", boxes)
        res = main(epochs=15, size=32, channels=(4, 8), data_dir=d, ckpt_dir=f"{d}/ckpt", **kw)
        assert res["auc"] > 0.9, "pipeline cannot learn a trivial task: bug upstream of the data"
    print("\nsmoke test passed")

if __name__ == "__main__":
    smoke_test(augment=False)
    smoke_test(bn=False)
    smoke_test(p_drop=0.0)