"""
Evaluate TTA baselines (TENT, EATA, SAR, BN-Adapt, Buffer) on CIFAR-10-C.

Usage:
    # All baselines, all corruptions, severity 5
    python scripts/run_baselines.py \
        --checkpoint checkpoints/resnet26_cifar10.pth \
        --data_dir data/CIFAR-10-C \
        --severity 5

    # Specific baselines
    python scripts/run_baselines.py \
        --checkpoint checkpoints/resnet26_cifar10.pth \
        --data_dir data/CIFAR-10-C \
        --methods tent sar \
        --severity 5
"""

import argparse
import copy
import os
import sys

import torch
from torchvision import datasets

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.resnet26 import ResNet26
from data.corruptions import CORRUPTIONS, get_cifar10c_loader, get_test_transform
from methods.tent import TENT
from methods.eata import EATA
from methods.sar import SAR
from methods.bn_adapt import BNAdapt
from methods.buffer import Buffer
from utils import evaluate


ALL_METHODS = ["tent", "eata", "sar", "bn_adapt", "buffer"]


def parse_args():
    p = argparse.ArgumentParser(description="Run TTA baselines on CIFAR-10-C")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_dir", type=str, required=True,
                   help="Path to CIFAR-10-C directory")
    p.add_argument("--cifar10_dir", type=str, default="./data",
                   help="Root for torchvision CIFAR-10 (clean test set)")
    p.add_argument("--methods", nargs="+", default=ALL_METHODS,
                   choices=ALL_METHODS, help="Baselines to run")
    p.add_argument("--corruptions", nargs="+", default=None)
    p.add_argument("--severity", type=int, default=5, choices=[1, 2, 3, 4, 5])
    p.add_argument("--batch_size", type=int, default=256)
    return p.parse_args()


def build_adapter(method_name, model, clean_loader, device):
    """Factory: instantiate the adapter for a given method name."""
    if method_name == "tent":
        return TENT(model)
    elif method_name == "eata":
        return EATA(model, clean_loader)
    elif method_name == "sar":
        return SAR(model)
    elif method_name == "bn_adapt":
        return BNAdapt(model)
    elif method_name == "buffer":
        return Buffer(copy.deepcopy(model.state_dict()), device)
    else:
        raise ValueError(f"Unknown method: {method_name}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load base model ──────────────────────
    base = ResNet26(num_classes=10).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    base.load_state_dict(state)
    base.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # ── Clean test loader ────────────────────
    clean_ds = datasets.CIFAR10(
        root=args.cifar10_dir, train=False, download=True,
        transform=get_test_transform(),
    )
    dl_clean = torch.utils.data.DataLoader(
        clean_ds, batch_size=args.batch_size, shuffle=False, num_workers=4,
    )

    corruptions = args.corruptions if args.corruptions else CORRUPTIONS

    # ── Run each method ──────────────────────
    for method_name in args.methods:
        print(f"\n{'=' * 88}")
        print(f"  Method: {method_name.upper()}")
        print(f"{'=' * 88}")

        results = {}
        for corr in corruptions:
            dl_corrupt = get_cifar10c_loader(
                args.data_dir, corr, args.severity,
                batch_size=args.batch_size,
            )

            # Before adaptation (source model)
            base.load_state_dict(copy.deepcopy(state))
            base.eval()
            acc_before = evaluate(base, dl_corrupt, device)

            # Build fresh adapter (re-load base for methods that modify it)
            base.load_state_dict(copy.deepcopy(state))
            base.eval()
            adapter = build_adapter(method_name, base, dl_clean, device)

            # Adapt
            adapter.adapt_loader(dl_corrupt)

            # Evaluate after adaptation
            model_to_eval = adapter.model
            acc_after = evaluate(model_to_eval, dl_corrupt, device)
            acc_clean = evaluate(model_to_eval, dl_clean, device)

            results[corr] = {
                "before": acc_before,
                "after_corrupt": acc_after,
                "clean_after": acc_clean,
            }
            print(
                f"[{corr:>22s}]  Before: {acc_before:6.2f}%  |  "
                f"{method_name.upper()}: {acc_after:6.2f}%  |  "
                f"Clean: {acc_clean:6.2f}%"
            )

        # Summary
        print(f"\n{'-' * 88}")
        print(f"{'Corruption':>22s} | {'Before':>8s} | {method_name.upper():>8s} | {'Δ':>7s} | {'Clean↓':>8s}")
        print(f"{'-' * 88}")

        sum_b, sum_a = 0.0, 0.0
        for corr in corruptions:
            r = results[corr]
            delta = r["after_corrupt"] - r["before"]
            sum_b += r["before"]
            sum_a += r["after_corrupt"]
            print(
                f"{corr:>22s} | {r['before']:7.2f}% | {r['after_corrupt']:7.2f}% | "
                f"{delta:+6.2f}% | {r['clean_after']:7.2f}%"
            )

        n = len(corruptions)
        print(f"{'-' * 88}")
        print(
            f"{'AVERAGE':>22s} | {sum_b/n:7.2f}% | {sum_a/n:7.2f}% | "
            f"{(sum_a-sum_b)/n:+6.2f}% |"
        )


if __name__ == "__main__":
    main()
