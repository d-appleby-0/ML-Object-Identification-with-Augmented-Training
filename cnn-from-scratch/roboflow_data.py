import os
from pathlib import Path
from data import build_cache
from dotenv import load_dotenv
from data import read_yolo_label


def download_roboflow(APIkey, workspace, project, version=1, location="C:/data/xrays"):
    from roboflow import Roboflow
    rf = Roboflow(api_key=APIkey)
    ds = rf.workspace(workspace).project(project).version(version).download("yolov8", location=location)
    return Path(ds.location)

SPLITS = {"train": "train", "valid": "val", "test": "test"}    # Roboflow calls the validation split "valid"

def build_all_caches(root, out_dir="C:/data/xray_cache", size=128):
    root, out = Path(root), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for src, dst in SPLITS.items():
        img_dir, lab_dir = root / src / "images", root / src / "labels"
        if not img_dir.exists():
            print(f"skip {src}: {img_dir} not found")
            continue
        build_cache(img_dir, lab_dir, str(out / dst), size=size)
        print(f"{src}: cached")


# verification functions

import numpy as np

EXTS = {".png", ".jpg", ".jpeg"}

def check_layout(root):
    root, total = Path(root), 0
    for split in ("train", "valid", "test"):
        imgs = {p.stem for p in (root / split / "images").glob("*") if p.suffix.lower() in EXTS}
        labs = {p.stem for p in (root / split / "labels").glob("*.txt")}
        total += len(imgs)
        print(f"{split:<6} images {len(imgs):>5}  labels {len(labs):>5}  "
              f"images without label file {len(imgs - labs)}  labels without image {len(labs - imgs)}")
    print("total images:", total)
    print((root / "data.yaml").read_text())             # class names and count

def box_stats(root):
    # Tumor size relative to the image. Decides whether 128 px can resolve them
    w, n_empty, n = [], 0, 0
    for lp in Path(root).glob("*/labels/*.txt"):
        rows = read_yolo_label(lp)
        n += 1
        n_empty += len(rows) == 0
        w += list(rows[:, 3])
    w = np.array(w)
    print(f"{n} label files, {n_empty} with no boxes ({n_empty / n:.1%}); {len(w)} boxes")
    print(f"box width as fraction of image: median {np.median(w):.3f}, 10th percentile {np.percentile(w, 10):.3f}")



if __name__ == "__main__":
    """
    load_dotenv()
    APIkey = os.getenv("ROBOFLOW_API_KEY")
    workspace = os.getenv("Roboflow_workspace")
    project = os.getenv("Roboflow_project_name")
    version = 1
    root = download_roboflow(APIkey, workspace, project, version=version)
    check_layout(root)
    box_stats(root)
    """
    build_all_caches("C:/data/xrays", size=128)
    