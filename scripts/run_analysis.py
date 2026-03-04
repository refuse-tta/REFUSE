"""
Run CKA similarity and feature drift analysis across the 4 FiLM
insertion variants, comparing to the base model on clean data.

Usage:
    python scripts/run_analysis.py \
        --checkpoint checkpoints/resnet26_cifar10.pth \
        --adapted_dir checkpoints/adapted/ \
        --data_dir data/CIFAR-10-C \
        --corruption gaussian_noise \
        --severity 5 \
        --save_dir results/

Produces:
    results/cka_comparison.png
    results/drift_comparison.png
    (+ printed tables)
"""

import argparse
import copy
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision import datasets

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.resnet26 import ResNet26
from models.film_variants import ResNet26_FiLM_General, INSERTION_POINTS
from data.corruptions import get_cifar10c_loader, get_test_transform
from analysis.features import collect_features, strip_prefix
from analysis.cka import layerwise_cka
from analysis.drift import compute_layerwise_drift
from utils import evaluate


LAYER_NAMES_BASE = ["prep", "layer1", "layer2", "layer3"]
LAYER_NAMES_FILM = ["m.prep", "m.layer1", "m.layer2", "m.layer3"]
LAYER_NAMES = LAYER_NAMES_BASE  # canonical names for display


def parse_args():
    p = argparse.ArgumentParser(description="CKA + drift analysis")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Pre-trained ResNet-26 checkpoint")
    p.add_argument("--adapted_dir", type=str, required=True,
                   help="Directory with film_prep.pth, film_layer1.pth, etc.")
    p.add_argument("--data_dir", type=str, required=True,
                   help="Path to CIFAR-10-C directory")
    p.add_argument("--cifar10_dir", type=str, default="./data")
    p.add_argument("--corruption", type=str, default="gaussian_noise")
    p.add_argument("--severity", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--max_images", type=int, default=256,
                   help="Number of images for feature collection")
    p.add_argument("--save_dir", type=str, default="results/")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # ── Load base model ──────────────────────
    base = ResNet26(num_classes=10).to(device)
    base_state = torch.load(args.checkpoint, map_location=device)
    base.load_state_dict(base_state)
    base.eval()

    # ── Data loaders ─────────────────────────
    dl_clean = torch.utils.data.DataLoader(
        datasets.CIFAR10(args.cifar10_dir, train=False, download=True,
                         transform=get_test_transform()),
        batch_size=args.batch_size, shuffle=False, num_workers=4,
    )
    dl_corrupt = get_cifar10c_loader(
        args.data_dir, args.corruption, args.severity,
        batch_size=args.batch_size,
    )

    # ── Collect base features ────────────────
    print("Collecting base-clean features...")
    base_clean_feats = collect_features(
        base, dl_clean, LAYER_NAMES_BASE, args.max_images, device,
    )
    print("Collecting base-corrupt features...")
    base_corr_feats = collect_features(
        base, dl_corrupt, LAYER_NAMES_BASE, args.max_images, device,
    )

    # ── Load adapted FiLM variants & collect features ──
    insertion_points = list(INSERTION_POINTS.keys())
    film_feats_corr = {}  # insert_after → feats dict

    for insert_after in insertion_points:
        ckpt_path = os.path.join(args.adapted_dir, f"film_{insert_after}.pth")
        if not os.path.exists(ckpt_path):
            print(f"  [SKIP] {ckpt_path} not found")
            continue

        print(f"Loading adapted FiLM@{insert_after}...")

        # Rebuild model and load adapted weights
        base.load_state_dict(copy.deepcopy(base_state))
        base.eval()
        film_model = ResNet26_FiLM_General(base, insert_after).to(device)
        film_model.load_state_dict(torch.load(ckpt_path, map_location=device))
        film_model.eval()

        # Evaluate accuracy
        acc_corrupt = evaluate(film_model, dl_corrupt, device)
        acc_clean = evaluate(film_model, dl_clean, device)
        print(f"  FiLM@{insert_after}: {acc_corrupt:.2f}% (corrupt) "
              f"| {acc_clean:.2f}% (clean)")

        # Collect features (use "m." prefix for wrapped backbone)
        feats_corr = collect_features(
            film_model, dl_corrupt, LAYER_NAMES_FILM, args.max_images, device,
        )
        film_feats_corr[insert_after] = strip_prefix(feats_corr, "m.")

    # ── Compute CKA ──────────────────────────
    print("\n=== CKA similarity to BASE-CLEAN (higher is better) ===")
    cka_results = {"base": layerwise_cka(base_clean_feats, base_corr_feats)}
    for ip in insertion_points:
        if ip in film_feats_corr:
            cka_results[ip] = layerwise_cka(base_clean_feats, film_feats_corr[ip])

    header = f"{'Layer':>8s} | {'Base':>6s}"
    for ip in insertion_points:
        if ip in cka_results:
            header += f" | FiLM@{ip}:{'':<1s}"
    print(header)
    print("-" * len(header))
    for lname in LAYER_NAMES:
        row = f"{lname:>8s} | {cka_results['base'][lname]:.3f}"
        for ip in insertion_points:
            if ip in cka_results:
                row += f" |    {cka_results[ip][lname]:.3f}   "
        print(row)

    # ── Compute drift ────────────────────────
    print("\n=== Drift vs BASE-CLEAN (lower is better) ===")
    drift_results = {"base": compute_layerwise_drift(base_clean_feats, base_corr_feats)}
    for ip in insertion_points:
        if ip in film_feats_corr:
            drift_results[ip] = compute_layerwise_drift(
                base_clean_feats, film_feats_corr[ip],
            )

    for lname in LAYER_NAMES:
        row = f"{lname:>8s} | Base:{drift_results['base'][lname][0]:.3f}"
        for ip in insertion_points:
            if ip in drift_results:
                row += f" | FiLM@{ip}:{drift_results[ip][lname][0]:.3f}"
        print(row)

    # ── Plot CKA ─────────────────────────────
    plt.rcParams.update({"font.family": "serif", "font.size": 11, "figure.dpi": 300})

    x = np.arange(len(LAYER_NAMES))
    width = 0.16
    labels_map = {"base": "Base"}
    for ip in insertion_points:
        labels_map[ip] = f"FiLM@{ip.capitalize()}"

    fig, ax = plt.subplots(figsize=(6.5, 4))
    keys = [k for k in ["base"] + insertion_points if k in cka_results]
    for i, key in enumerate(keys):
        vals = [cka_results[key][l] for l in LAYER_NAMES]
        ax.bar(x + (i - len(keys) / 2) * width, vals, width,
               label=labels_map.get(key, key))

    ax.set_ylabel("CKA Similarity")
    ax.set_xlabel("Network Layers")
    ax.set_title("Layer-wise Representation Analysis (CKA)")
    ax.set_xticks(x)
    ax.set_xticklabels(LAYER_NAMES, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=False, ncol=2)
    plt.tight_layout()

    cka_path = os.path.join(args.save_dir, "cka_comparison.png")
    fig.savefig(cka_path)
    print(f"\nCKA plot saved → {cka_path}")
    plt.close(fig)

    # ── Plot drift ───────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    keys = [k for k in ["base"] + insertion_points if k in drift_results]
    for i, key in enumerate(keys):
        vals = [drift_results[key][l][0] for l in LAYER_NAMES]  # mean drift
        ax.bar(x + (i - len(keys) / 2) * width, vals, width,
               label=labels_map.get(key, key), alpha=0.9)

    ax.set_ylabel("Representation Drift (↓ better)")
    ax.set_xlabel("Layer")
    ax.set_title("Layer-wise Drift vs Base-Clean")
    ax.set_xticks(x)
    ax.set_xticklabels(LAYER_NAMES)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=False)
    plt.tight_layout()

    drift_path = os.path.join(args.save_dir, "drift_comparison.png")
    fig.savefig(drift_path)
    print(f"Drift plot saved → {drift_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
