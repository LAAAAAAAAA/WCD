
from pathlib import Path
import random
import shutil

__all__ = ["split_dataset"]

def _set_seed(seed: int = 42):
    random.seed(seed)

def split_dataset(src_root: str, dst_root: str, ratio=(0.8, 0.1, 0.1), seed: int = 42):
    """Split *src_root*/{class}/images into train/val/test folders under *dst_root*.

    Args:
        src_root: Folder containing one sub‑folder per class (e.g. lensed_data/, unlensed_data/).
        dst_root: Destination folder where splits/train|val|test/{class}/ will be created.
        ratio: Tuple of proportions for (train, val, test). Must sum to 1.
        seed: Random seed for reproducibility.
    """
    assert abs(sum(ratio) - 1.0) < 1e-6, "Split ratios must add up to 1.0"
    _set_seed(seed)

    src_root, dst_root = Path(src_root), Path(dst_root)
    classes = [d.name for d in src_root.iterdir() if d.is_dir()]

    if not classes:
        raise ValueError(f"No class sub‑directories found in {src_root}")

    # Create destination directories
    for split in ("train", "val", "test"):
        for cls in classes:
            (dst_root / split / cls).mkdir(parents=True, exist_ok=True)

    # Shuffle & copy
    for cls in classes:
        images = list((src_root / cls).iterdir())
        # --- If there are more than 1000 lensed_data, randomly select 1000 ---
        if cls == "lensed_data" and len(images) > 1000:
            images = random.sample(images, 1000)
        # --------------------------------------------------------

        random.shuffle(images)
        n = len(images)
        n_train = int(ratio[0] * n)
        n_val = int(ratio[1] * n)

        splits = {
            "train": images[:n_train],
            "val":   images[n_train:n_train + n_val],
            "test":  images[n_train + n_val:],
        }

        for split, imgs in splits.items():
            for img in imgs:
                shutil.copy(img, dst_root / split / cls / img.name)
