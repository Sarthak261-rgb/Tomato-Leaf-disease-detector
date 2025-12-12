import os, shutil, random
from pathlib import Path

random.seed(42)
SOURCE_DIR = Path("PlantVillage")          # Folder that has class subfolders
OUT_DIR    = Path("data_tomato_splits")    # Output split folder
SPLITS     = {"train": 0.70, "val": 0.20, "test": 0.10}

def list_images(d: Path):
    return [p for p in d.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]

# clean/create directories
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
for s in SPLITS:
    (OUT_DIR / s).mkdir(parents=True, exist_ok=True)

# detect classes
classes = [d.name for d in SOURCE_DIR.iterdir() if d.is_dir()]
for s in SPLITS:
    for c in classes:
        (OUT_DIR/s/c).mkdir(parents=True, exist_ok=True)

print("Detected classes:", classes)

# split and copy images
for c in classes:
    imgs = list_images(SOURCE_DIR / c)
    imgs.sort(); random.shuffle(imgs)
    n = len(imgs)
    n_tr = int(round(n * SPLITS["train"]))
    n_va = int(round(n * SPLITS["val"]))
    if n_tr + n_va > n:
        n_va = n - n_tr
    n_te = n - n_tr - n_va

    for p in imgs[:n_tr]:
        shutil.copy2(p, OUT_DIR/"train"/c/p.name)
    for p in imgs[n_tr:n_tr+n_va]:
        shutil.copy2(p, OUT_DIR/"val"/c/p.name)
    for p in imgs[n_tr+n_va:]:
        shutil.copy2(p, OUT_DIR/"test"/c/p.name)

    print(f"{c}: {n_tr} train, {n_va} val, {n_te} test")

print("✅ Split complete → ./data_tomato_splits")
