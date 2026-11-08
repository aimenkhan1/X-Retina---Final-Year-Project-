"""PyTorch Dataset and DataLoader factory with augmentations. Owner: Maheen (Step 2, Week 9).

Input: Aiman's processed data (Step 1) on the shared Drive
    processed/xretina_<dataset>_512_partNNN.zip   512x512 JPEGs (crop + CLAHE + circle mask)
    processed/manifest.csv                        one row per image, with the frozen `split` column

Rules (from notebook 02):
- Use the `split` column as it is. Never re-split.
- Augmentation happens here, on TRAIN rows only. Val/test are only resized and normalised.
- No oversampling inside the files. Imbalance handling (Focal Loss, SMOTE) is Step 4.
"""
import glob
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)   # EfficientNet-B4 is pretrained on ImageNet
IMAGENET_STD = (0.229, 0.224, 0.225)
LOCAL_PROCESSED = "/content/data/processed"


# ---------------------------------------------------------------- loading the manifest
def extract_shards(processed_dir, local_dir=LOCAL_PROCESSED, verbose=True) -> Path:
    """Unzip every shard from Drive to the fast local Colab disk (skips shards already done).

    Reading 38k small files straight from Drive is very slow, so we copy once per session.
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    zips = sorted(glob.glob(str(Path(processed_dir) / "xretina_*_part*.zip")))
    assert zips, f"No shard zips found in {processed_dir}. Has notebook 02 (Aiman) finished its full run?"
    for i, z in enumerate(zips, 1):
        marker = local_dir / f".done_{Path(z).stem}"
        if marker.exists():
            continue
        with zipfile.ZipFile(z) as f:
            f.extractall(local_dir)
        marker.touch()
        if verbose:
            print(f"  extracted {i}/{len(zips)}: {Path(z).name}")
    if verbose:
        print(f"All {len(zips)} shards ready in {local_dir}")
    return local_dir


def load_manifest(processed_dir, local_dir=LOCAL_PROCESSED, datasets=None, check_files=True) -> pd.DataFrame:
    """Read manifest.csv and add an absolute `path` column. Optionally keep only some datasets."""
    man = pd.read_csv(Path(processed_dir) / "manifest.csv", dtype={"image_id": str, "patient_id": str})
    need = {"image_id", "dataset", "label", "split", "rel_path"}
    assert need <= set(man.columns), f"manifest.csv is missing columns: {need - set(man.columns)}"
    assert set(man["split"].unique()) <= {"train", "val", "test"}, "unexpected values in the split column"
    if datasets is not None:
        datasets = [datasets] if isinstance(datasets, str) else list(datasets)
        man = man[man["dataset"].isin(datasets)].reset_index(drop=True)
    man["label"] = man["label"].astype(int)
    man["path"] = [str(Path(local_dir) / r) for r in man["rel_path"]]
    if check_files:
        sample = man.sample(min(200, len(man)), random_state=0)
        missing = [p for p in sample["path"] if not Path(p).exists()]
        assert not missing, f"{len(missing)} of 200 sampled images are missing locally, e.g. {missing[0]}. Run extract_shards first."
    return man


def stratified_subset(df: pd.DataFrame, per_class: int, seed: int = 42) -> pd.DataFrame:
    """Small subset with up to `per_class` images of each grade (for smoke tests only)."""
    parts = [g.sample(min(per_class, len(g)), random_state=seed) for _, g in df.groupby("label")]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------- augmentations
def get_transforms(img_size: int = 380, train: bool = True) -> A.Compose:
    """Train: light, label-safe augmentation. Val/test: resize + normalise only.

    Chosen to be safe for fundus images: flips and rotation do not change the DR grade,
    and brightness/contrast changes are kept small because CLAHE was already applied.
    Similar to [7] (flips, rotation up to 25 degrees, colour jitter) and [11] (flip, rotation, brightness).
    """
    if train:
        steps = [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=25, border_mode=cv2.BORDER_CONSTANT, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        ]
    else:
        steps = [A.Resize(img_size, img_size)]
    steps += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(steps)


# ---------------------------------------------------------------- dataset
class RetinaDataset(Dataset):
    """Returns (image tensor 3xHxW, grade 0-4). One row of the manifest = one sample."""

    def __init__(self, df: pd.DataFrame, transform=None):
        self.paths = df["path"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.imread(self.paths[i])
        if img is None:
            raise IOError(f"cannot read {self.paths[i]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)          # OpenCV reads BGR, models expect RGB
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img, torch.tensor(self.labels[i], dtype=torch.long)


def class_weights(labels, num_classes: int = 5) -> torch.Tensor:
    """Inverse-frequency weights (mean = 1). Can be passed to the loss or used for sampling."""
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=num_classes).astype(float)
    w = np.where(counts > 0, counts.sum() / (num_classes * np.maximum(counts, 1)), 0.0)
    return torch.tensor(w, dtype=torch.float32)


def build_dataloaders(manifest: pd.DataFrame, img_size: int = 380, batch_size: int = 16,
                      num_workers: int = 2, weighted_sampler: bool = False, seed: int = 42) -> dict:
    """Make train/val/test DataLoaders from the manifest's split column.

    weighted_sampler=True draws rare grades more often (train only). Off by default: the
    baseline uses the natural class mix, and imbalance methods are compared in Step 4.
    """
    loaders = {}
    g = torch.Generator().manual_seed(seed)
    for split in ("train", "val", "test"):
        df = manifest[manifest["split"] == split]
        if len(df) == 0:
            continue
        is_train = split == "train"
        ds = RetinaDataset(df, get_transforms(img_size, train=is_train))
        sampler = None
        if is_train and weighted_sampler:
            w = class_weights(ds.labels)[ds.labels]
            sampler = WeightedRandomSampler(w.double(), num_samples=len(ds), replacement=True, generator=g)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=is_train and sampler is None, sampler=sampler,
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
            drop_last=is_train and len(ds) > batch_size, generator=g if is_train else None,
            persistent_workers=num_workers > 0)
    return loaders


def denormalize(t: torch.Tensor) -> np.ndarray:
    """Tensor 3xHxW (normalised) -> HxWx3 uint8 RGB, for showing a batch."""
    x = t.detach().cpu().numpy().transpose(1, 2, 0)
    x = x * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)
