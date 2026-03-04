"""
CIFAR-10 data loading utilities.

Supports:
  - Standard CIFAR-10 (torchvision or CSV/image-folder format)
  - CIFAR-10-C corrupted test sets (all 15 corruption types × 5 severities)
"""

import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]


# ──────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────
def get_train_transform():
    """Standard CIFAR-10 training augmentation."""
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])


def get_test_transform():
    """Deterministic test transform (normalize only)."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])


# ──────────────────────────────────────────────
# CSV-based CIFAR-10 (Kaggle format)
# ──────────────────────────────────────────────
class CIFAR10CSV(Dataset):
    """
    CIFAR-10 dataset stored as individual PNGs with a CSV label file
    (the format used by the Kaggle CIFAR-10 competition).
    """

    def __init__(self, img_dir, labels_csv, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        df = pd.read_csv(labels_csv)
        image_ids = set(
            int(f.split(".")[0]) for f in os.listdir(img_dir) if f.endswith(".png")
        )
        df = df[df["id"].isin(image_ids)].sort_values("id").reset_index(drop=True)

        self.ids = df["id"].tolist()
        labels_str = [s.rstrip(",") for s in df["label"].tolist()]
        classes = sorted(set(labels_str))
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.labels = [self.class_to_idx[c] for c in labels_str]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img = Image.open(
            os.path.join(self.img_dir, f"{self.ids[idx]}.png")
        ).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ──────────────────────────────────────────────
# CIFAR-10-C (numpy .npy format)
# ──────────────────────────────────────────────
class CIFAR10C(Dataset):
    """
    Loads a single (corruption, severity) slice from CIFAR-10-C.

    Expects the standard directory layout:
        data_dir/{corruption}.npy   — shape (50000, 32, 32, 3)
        data_dir/labels.npy         — shape (50000,)
    Each severity contains 10 000 images (indices 0-9999 = sev 1, etc.)
    """

    def __init__(self, data_dir, corruption, severity, transform=None):
        assert corruption in CORRUPTIONS, f"Unknown corruption: {corruption}"
        assert 1 <= severity <= 5, f"Severity must be 1-5, got {severity}"

        self.transform = transform

        images = np.load(os.path.join(data_dir, f"{corruption}.npy"))
        labels = np.load(os.path.join(data_dir, "labels.npy"))

        start = (severity - 1) * 10_000
        end = severity * 10_000
        self.images = images[start:end]
        self.labels = labels[start:end].astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        if self.transform:
            img = self.transform(img)
        return img, int(self.labels[idx])


# ──────────────────────────────────────────────
# Convenient loader factories
# ──────────────────────────────────────────────
def get_cifar10_train_loader(data_dir="./data", batch_size=128, num_workers=4,
                             csv_path=None, img_dir=None):
    """
    Returns a training DataLoader.
    If csv_path and img_dir are given → uses CIFAR10CSV (Kaggle format).
    Otherwise → uses torchvision CIFAR-10 (auto-download).
    """
    transform = get_train_transform()
    if csv_path and img_dir:
        ds = CIFAR10CSV(img_dir, csv_path, transform=transform)
    else:
        ds = datasets.CIFAR10(data_dir, train=True, download=True,
                              transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True)


def get_cifar10c_loader(data_dir, corruption, severity, batch_size=200,
                        num_workers=4, shuffle=False):
    """Returns a DataLoader for one (corruption, severity) slice of CIFAR-10-C."""
    ds = CIFAR10C(data_dir, corruption, severity, transform=get_test_transform())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)
