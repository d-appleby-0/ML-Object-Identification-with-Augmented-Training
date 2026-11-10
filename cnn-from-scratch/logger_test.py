import json, tempfile
from pathlib import Path
from logger import TrainLogger

def logger_test():
    cols = [("epoch", "Epoch", "{:5d}"), ("x", "x", "{:6.3f}")]
    with tempfile.TemporaryDirectory() as d:
        lg = TrainLogger(d, cols, verbose=False)
        for e in range(4):
            lg.log(epoch=e, x=e * 0.5)
        lg2 = TrainLogger(d, cols, resume_from=2, verbose=False)            # as if resuming at epoch 2
        assert lg2.history["epoch"] == [0, 1], "history was not trimmed to the resume point"
        lg2.log(epoch=2, x=9.0)
        saved = json.loads((Path(d) / "metrics_history.json").read_text())
        assert saved["epoch"] == [0, 1, 2] and saved["x"][2] == 9.0
        assert len((Path(d) / "training_log.txt").read_text().splitlines()) == 2 + 3
    print("  logger resume OK")

if __name__ == "__main__":
    logger_test()