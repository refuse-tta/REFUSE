"""
EATA: Efficient Test-Time Model Adaptation using Fisher Regularization.
Reference: Niu et al., ICML 2022.

Key additions over TENT:
  1. Sample-adaptive filtering — skip high-entropy (unreliable) samples.
  2. Anti-forgetting — Fisher-weighted L2 regularisation toward source BN params.
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# Default hyper-parameters
# ──────────────────────────────────────────────
EATA_LR = 0.001
EATA_EPOCHS = 1
FISHER_ALPHA = 2000.0          # Fisher regularisation weight


def _collect_bn_params(model):
    """Return list of (dotted_name, param) for every BN affine parameter."""
    params = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for pn, p in m.named_parameters():
                params.append((f"{nm}.{pn}", p))
    return params


def configure_eata(model):
    """Same BN-only setup as TENT. Returns trainable param list."""
    model.eval()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()
            m.weight.requires_grad_(True)
            m.bias.requires_grad_(True)
    return [p for p in model.parameters() if p.requires_grad]


def compute_fishers(model, loader, device, num_classes=10, num_samples=2000):
    """
    Diagonal Fisher Information for BN affine params, computed on
    source / clean data using low-entropy samples only.
    """
    e0 = 0.4 * math.log(num_classes)

    model.eval()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()
            m.weight.requires_grad_(True)
            m.bias.requires_grad_(True)

    bn_params = _collect_bn_params(model)
    fishers = {nm: torch.zeros_like(p) for nm, p in bn_params}

    seen = 0
    for imgs, _ in loader:
        if seen >= num_samples:
            break
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        probs = F.softmax(logits, dim=1)
        ent = -(probs * probs.log()).sum(dim=1)
        mask = ent < e0
        if mask.sum() == 0:
            seen += imgs.size(0)
            continue

        loss = ent[mask].mean()
        loss.backward()

        for nm, p in bn_params:
            if p.grad is not None:
                fishers[nm] += (p.grad.detach() ** 2) * mask.sum().item()
                p.grad = None
        seen += imgs.size(0)

    for nm in fishers:
        fishers[nm] /= max(seen, 1)

    model.zero_grad()
    model.requires_grad_(False)
    return fishers


class EATA:
    """
    Stateful EATA adapter.

    Parameters
    ----------
    model : nn.Module
        Pre-trained backbone.
    clean_loader : DataLoader
        Source / clean data for computing Fisher Information.
    lr : float
    epochs : int
    fisher_alpha : float
        Weighting for the Fisher regularisation term.
    num_classes : int
    """

    def __init__(self, model, clean_loader, lr=EATA_LR, epochs=EATA_EPOCHS,
                 fisher_alpha=FISHER_ALPHA, num_classes=10):
        self.device = next(model.parameters()).device
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.fisher_alpha = fisher_alpha
        self.num_classes = num_classes
        self.e0 = 0.4 * math.log(num_classes)
        self.base_state = copy.deepcopy(model.state_dict())

        # Pre-compute Fisher & anchor params
        self.fishers = compute_fishers(
            model, clean_loader, self.device,
            num_classes=num_classes,
        )
        self.anchor_params = {}
        for nm, p in _collect_bn_params(model):
            self.anchor_params[nm] = p.detach().clone()

        # Restore state after Fisher computation
        self.model.load_state_dict(copy.deepcopy(self.base_state))

    def reset(self):
        self.model.load_state_dict(copy.deepcopy(self.base_state))

    def adapt_loader(self, loader):
        params = configure_eata(self.model)
        optimizer = torch.optim.Adam(params, lr=self.lr)

        for _ in range(self.epochs):
            bn_params = _collect_bn_params(self.model)
            for imgs, _ in loader:
                imgs = imgs.to(self.device, non_blocking=True)
                logits = self.model(imgs)
                probs = F.softmax(logits, dim=1)
                ent = -(probs * probs.log()).sum(dim=1)

                # Sample-adaptive filtering
                mask = ent < self.e0
                ent_loss = ent[mask].mean() if mask.sum() > 0 else torch.tensor(0.0, device=self.device)

                # Anti-forgetting Fisher regularisation
                fisher_loss = torch.tensor(0.0, device=self.device)
                for nm, p in bn_params:
                    if nm in self.fishers:
                        fisher_loss += (self.fishers[nm] * (p - self.anchor_params[nm]) ** 2).sum()

                loss = ent_loss + self.fisher_alpha * fisher_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
