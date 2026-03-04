"""Shared evaluation helpers for TTA experiments."""

import torch


@torch.no_grad()
def evaluate(model, loader, device=None):
    """
    Compute top-1 accuracy (%).

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader
    device : torch.device or None (inferred from model)
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        preds = model(imgs).argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total
