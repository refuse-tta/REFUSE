"""
Train ResNet-26 on CIFAR-10.

Usage:
    python scripts/train.py --data_dir ./data --epochs 200 --batch_size 128
    python scripts/train.py --csv_path labels.csv --img_dir imgs/  # Kaggle format
"""

import argparse
import os
import sys

import torch
from torch import nn
from torch.amp import GradScaler
from tqdm import tqdm

# allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.resnet26 import ResNet26
from data.corruptions import get_cifar10_train_loader


def parse_args():
    p = argparse.ArgumentParser(description="Train ResNet-26 on CIFAR-10")
    p.add_argument("--data_dir", type=str, default="./data",
                   help="Root dir for torchvision CIFAR-10 download")
    p.add_argument("--csv_path", type=str, default=None,
                   help="Path to trainLabels.csv (Kaggle format)")
    p.add_argument("--img_dir", type=str, default=None,
                   help="Image folder for Kaggle format")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--save_path", type=str, default="checkpoints/resnet26_cifar10.pth")
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────
    train_loader = get_cifar10_train_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        csv_path=args.csv_path,
        img_dir=args.img_dir,
    )
    print(f"Training samples: {len(train_loader.dataset)}")

    # ── Model ─────────────────────────────────
    model = ResNet26(num_classes=10).to(device)

    # ── Optimiser ─────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler("cuda")

    # ── Training loop ─────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total += labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            loss_sum += loss.item() * labels.size(0)

        scheduler.step()
        acc = 100.0 * correct / total
        avg_loss = loss_sum / total
        print(f"  → Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")

    # ── Save ──────────────────────────────────
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"\nModel saved to {args.save_path}")


if __name__ == "__main__":
    train(parse_args())
