"""
Evaluate REFUSE on all CIFAR-10-C corruptions.

Usage:
    python scripts/run_refuse.py \
        --checkpoint checkpoints/resnet26_cifar10.pth \
        --data_dir data/CIFAR-10-C \
        --severity 5

    # Single corruption
    python scripts/run_refuse.py \
        --checkpoint checkpoints/resnet26_cifar10.pth \
        --data_dir data/CIFAR-10-C \
        --corruptions gaussian_noise fog \
        --severity 5
"""

import argparse
import os
import sys

import torch
from torchvision import datasets

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.resnet26 import ResNet26
from data.corruptions import CORRUPTIONS, get_cifar10c_loader, get_test_transform
from methods.refuse import REFUSE
from utils import evaluate


def parse_args():
    p = argparse.ArgumentParser(description="Run REFUSE TTA on CIFAR-10-C")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to pre-trained ResNet-26 checkpoint")
    p.add_argument("--data_dir", type=str, required=True,
                   help="Path to CIFAR-10-C directory (contains .npy files)")
    p.add_argument("--cifar10_dir", type=str, default="./data",
                   help="Root for torchvision CIFAR-10 (clean test set)")
    p.add_argument("--corruptions", nargs="+", default=None,
                   help="Subset of corruptions (default: all 15)")
    p.add_argument("--severity", type=int, default=5, choices=[1, 2, 3, 4, 5])
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--h0_floor", type=float, default=0.1)
    p.add_argument("--lambda_kl", type=float, default=0.16)
    p.add_argument("--lambda_stab", type=float, default=3e-4)
    p.add_argument("--lambda_gate", type=float, default=0.55)
    p.add_argument("--save_adapted", type=str, default=None,
                   help="If set, save final adapted model to this path")
    return p.parse_args()


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

    # ── Clean test loader (for monitoring) ───
    clean_ds = datasets.CIFAR10(
        root=args.cifar10_dir, train=False, download=True,
        transform=get_test_transform(),
    )
    dl_clean = torch.utils.data.DataLoader(
        clean_ds, batch_size=args.batch_size, shuffle=False, num_workers=4,
    )

    # ── Corruption list ──────────────────────
    corruptions = args.corruptions if args.corruptions else CORRUPTIONS

    # ── Run ───────────────────────────────────
    results = {}
    for corr in corruptions:
        # Fresh adapter for each corruption
        adapter = REFUSE(
            base_model=base,
            steps=args.steps,
            lr=args.lr,
            h0_floor=args.h0_floor,
            lambda_kl=args.lambda_kl,
            lambda_stab=args.lambda_stab,
            lambda_gate=args.lambda_gate,
        )
        adapter.reset_base()

        dl_corrupt = get_cifar10c_loader(
            args.data_dir, corr, args.severity,
            batch_size=args.batch_size,
        )

        # Before adaptation
        acc_before = evaluate(base, dl_corrupt, device)

        # Adapt (continual over batches)
        adapter.adapt_loader(dl_corrupt)

        # After adaptation
        acc_after = evaluate(adapter.model, dl_corrupt, device)
        acc_clean = evaluate(adapter.model, dl_clean, device)

        results[corr] = {
            "before": acc_before,
            "after_corrupt": acc_after,
            "clean_after": acc_clean,
        }
        print(
            f"[{corr:>22s}]  Before: {acc_before:6.2f}%  |  "
            f"REFUSE: {acc_after:6.2f}%  |  Clean: {acc_clean:6.2f}%"
        )

    # ── Summary ──────────────────────────────
    print("\n" + "=" * 88)
    print(f"{'Corruption':>22s} | {'Before':>8s} | {'REFUSE':>8s} | {'Δ':>7s} | {'Clean↓':>8s}")
    print("-" * 88)

    sum_before, sum_after = 0.0, 0.0
    for corr in corruptions:
        r = results[corr]
        delta = r["after_corrupt"] - r["before"]
        sum_before += r["before"]
        sum_after += r["after_corrupt"]
        print(
            f"{corr:>22s} | {r['before']:7.2f}% | {r['after_corrupt']:7.2f}% | "
            f"{delta:+6.2f}% | {r['clean_after']:7.2f}%"
        )

    n = len(corruptions)
    avg_before = sum_before / n
    avg_after = sum_after / n
    print("-" * 88)
    print(
        f"{'AVERAGE':>22s} | {avg_before:7.2f}% | {avg_after:7.2f}% | "
        f"{avg_after - avg_before:+6.2f}% |"
    )
    print("=" * 88)

    # ── Optionally save adapted model ────────
    if args.save_adapted:
        os.makedirs(os.path.dirname(args.save_adapted) or ".", exist_ok=True)
        torch.save(adapter.model.state_dict(), args.save_adapted)
        print(f"\nAdapted model saved to {args.save_adapted}")


if __name__ == "__main__":
    main()
