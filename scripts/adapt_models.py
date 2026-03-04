"""
Adapt ResNet-26 with FiLM inserted at 4 different positions and save
the adapted checkpoints for downstream analysis (CKA, drift).

Usage:
    python scripts/adapt_models.py \
        --checkpoint checkpoints/resnet26_cifar10.pth \
        --data_dir data/CIFAR-10-C \
        --corruption gaussian_noise \
        --severity 5 \
        --save_dir checkpoints/adapted/

This produces:
    checkpoints/adapted/film_prep.pth
    checkpoints/adapted/film_layer1.pth
    checkpoints/adapted/film_layer2.pth
    checkpoints/adapted/film_layer3.pth
"""

import argparse
import copy
import math
import os
import sys

import torch
import torch.nn.functional as F
from torchvision import datasets

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.resnet26 import ResNet26
from models.film_variants import ResNet26_FiLM_General, INSERTION_POINTS
from methods.refuse import FiLM, norm_entropy_from_logits, kl_mean_to_uniform
from data.corruptions import get_cifar10c_loader, get_test_transform
from utils import evaluate


def parse_args():
    p = argparse.ArgumentParser(
        description="Adapt 4 FiLM insertion variants and save checkpoints"
    )
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_dir", type=str, required=True,
                   help="Path to CIFAR-10-C directory")
    p.add_argument("--cifar10_dir", type=str, default="./data")
    p.add_argument("--corruption", type=str, default="gaussian_noise")
    p.add_argument("--severity", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--save_dir", type=str, default="checkpoints/adapted/")
    # REFUSE hyper-parameters
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--h0_floor", type=float, default=0.1)
    p.add_argument("--lambda_kl", type=float, default=0.16)
    p.add_argument("--lambda_stab", type=float, default=3e-4)
    p.add_argument("--lambda_gate", type=float, default=0.55)
    return p.parse_args()


def adapt_film_variant(film_model, film_params, loader, device, args):
    """
    Run REFUSE-style adaptation on a FiLM-wrapped model.
    Identical loss to the main method, just on a different insertion point.
    """
    theta0 = [p.detach().clone() for p in film_params]

    def stability_term():
        s = 0.0
        for p, p0 in zip(film_params, theta0):
            s = s + (p - p0).pow(2).mean()
        return s

    optimizer = torch.optim.Adam(film_params, lr=args.lr)

    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)

        # Check entropy before adapting
        film_model.eval()
        with torch.no_grad():
            logits0 = film_model(xb)
            _, ent_n0 = norm_entropy_from_logits(logits0)
            h0 = ent_n0.mean().item()

        if h0 < args.h0_floor:
            continue  # skip confident batches

        for _ in range(args.steps):
            film_model.eval()
            logits = film_model(xb)
            probs, ent_n = norm_entropy_from_logits(logits)

            h_mean = ent_n.mean()
            gated = F.relu(h_mean - args.h0_floor)
            kl = kl_mean_to_uniform(probs)
            stab = stability_term()

            loss = (
                args.lambda_gate * gated
                + args.lambda_kl * kl
                + args.lambda_stab * stab
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # ── Load base model ──────────────────────
    base = ResNet26(num_classes=10).to(device)
    base_state = torch.load(args.checkpoint, map_location=device)
    base.load_state_dict(base_state)
    base.eval()
    print(f"Loaded: {args.checkpoint}")

    # ── Data ─────────────────────────────────
    dl_corrupt = get_cifar10c_loader(
        args.data_dir, args.corruption, args.severity,
        batch_size=args.batch_size,
    )
    dl_clean = torch.utils.data.DataLoader(
        datasets.CIFAR10(args.cifar10_dir, train=False, download=True,
                         transform=get_test_transform()),
        batch_size=args.batch_size, shuffle=False, num_workers=4,
    )

    # ── Adapt each insertion variant ─────────
    for insert_after in INSERTION_POINTS:
        print(f"\n{'─' * 60}")
        print(f"Adapting FiLM @ {insert_after} "
              f"({INSERTION_POINTS[insert_after]} channels)")
        print(f"{'─' * 60}")

        # Reset base to clean state
        base.load_state_dict(copy.deepcopy(base_state))
        base.eval()

        # Build FiLM wrapper
        film_model = ResNet26_FiLM_General(base, insert_after).to(device).eval()
        with torch.no_grad():
            film_model.film.gamma.fill_(1.0)
            film_model.film.beta.zero_()

        # Freeze everything, unfreeze only FiLM
        for p in film_model.parameters():
            p.requires_grad = False
        film_params = []
        for mod in film_model.modules():
            if isinstance(mod, FiLM):
                for p in mod.parameters():
                    p.requires_grad = True
                film_params.extend(mod.parameters())

        n_params = sum(p.numel() for p in film_params)
        print(f"  Trainable FiLM params: {n_params}")

        # Evaluate before
        acc_before = evaluate(film_model, dl_corrupt, device)
        print(f"  Before adaptation: {acc_before:.2f}%")

        # Adapt
        adapt_film_variant(film_model, film_params, dl_corrupt, device, args)

        # Evaluate after
        acc_corrupt = evaluate(film_model, dl_corrupt, device)
        acc_clean = evaluate(film_model, dl_clean, device)
        print(f"  After adaptation:  {acc_corrupt:.2f}% (corrupt)  "
              f"{acc_clean:.2f}% (clean)")

        # Save
        save_path = os.path.join(args.save_dir, f"film_{insert_after}.pth")
        torch.save(film_model.state_dict(), save_path)
        print(f"  Saved → {save_path}")


if __name__ == "__main__":
    main()
