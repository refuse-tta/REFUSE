"""
SAR: Towards Stable Test-Time Adaptation in Dynamic Wild World.
Reference: Niu et al., ICLR 2023.

Key mechanisms:
  1. Reliable Entropy Minimisation (REM) — filter unreliable samples.
  2. Sharpness-Aware Minimisation (SAM) style two-step update.
  3. Model recovery — reset BN to source when EMA entropy exceeds threshold.
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# Default hyper-parameters
# ──────────────────────────────────────────────
SAR_LR = 0.001
SAR_EPOCHS = 1
SAM_RHO = 0.05                 # perturbation radius
RESET_CONSTANT = 0.2           # ρ: reset if EMA entropy > ρ·ln(C)


def configure_sar(model):
    """Same BN-only setup as TENT. Returns trainable param list."""
    model.eval()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()
            m.weight.requires_grad_(True)
            m.bias.requires_grad_(True)
    return [p for p in model.parameters() if p.requires_grad]


class SAR:
    """
    Stateful SAR adapter.

    Parameters
    ----------
    model : nn.Module
    lr : float
    epochs : int
    num_classes : int
    reset_constant : float
        ρ in the paper — controls model recovery threshold.
    sam_rho : float
        Perturbation radius for the SAM ascent step.
    """

    def __init__(self, model, lr=SAR_LR, epochs=SAR_EPOCHS,
                 num_classes=10, reset_constant=RESET_CONSTANT,
                 sam_rho=SAM_RHO):
        self.device = next(model.parameters()).device
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.num_classes = num_classes
        self.e_margin = 0.4 * math.log(num_classes)
        self.reset_thresh = reset_constant * math.log(num_classes)
        self.sam_rho = sam_rho
        self.base_state = copy.deepcopy(model.state_dict())

    def reset(self):
        self.model.load_state_dict(copy.deepcopy(self.base_state))

    def adapt_loader(self, loader):
        params = configure_sar(self.model)
        optimizer = torch.optim.SGD(params, lr=self.lr, momentum=0.9)
        source_state = copy.deepcopy(self.base_state)
        ema_ent = None

        for _ in range(self.epochs):
            for imgs, _ in loader:
                imgs = imgs.to(self.device, non_blocking=True)

                # ── First forward ──
                logits = self.model(imgs)
                probs = F.softmax(logits, dim=1)
                ent = -(probs * probs.log()).sum(dim=1)

                mask = ent < self.e_margin
                if mask.sum() == 0:
                    continue

                loss = ent[mask].mean()

                # ── SAM ascent step ──
                optimizer.zero_grad()
                loss.backward()

                grads_norm = 0.0
                for p in optimizer.param_groups[0]["params"]:
                    if p.grad is not None:
                        grads_norm += p.grad.data.norm(2).item() ** 2
                grads_norm = max(grads_norm ** 0.5, 1e-12)

                param_cache = []
                for p in optimizer.param_groups[0]["params"]:
                    if p.grad is not None:
                        eps = p.grad.data / grads_norm
                        param_cache.append((p.data.clone(), eps))
                        p.data.add_(eps, alpha=self.sam_rho)
                    else:
                        param_cache.append((p.data.clone(), None))

                # ── Second forward at perturbed point ──
                logits2 = self.model(imgs)
                probs2 = F.softmax(logits2, dim=1)
                ent2 = -(probs2 * probs2.log()).sum(dim=1)
                mask2 = ent2 < self.e_margin
                loss2 = ent2[mask2].mean() if mask2.sum() > 0 else torch.tensor(0.0, device=self.device)

                optimizer.zero_grad()
                loss2.backward()

                # ── Restore original params, then descend ──
                for p, (orig, _) in zip(optimizer.param_groups[0]["params"], param_cache):
                    p.data.copy_(orig)
                optimizer.step()

                # ── Model recovery via EMA of entropy ──
                with torch.no_grad():
                    mean_ent = ent.mean().item()
                    ema_ent = mean_ent if ema_ent is None else 0.9 * ema_ent + 0.1 * mean_ent

                    if ema_ent > self.reset_thresh:
                        for nm, m in self.model.named_modules():
                            if isinstance(m, nn.BatchNorm2d):
                                m.weight.data.copy_(source_state[f"{nm}.weight"])
                                m.bias.data.copy_(source_state[f"{nm}.bias"])
                                m.running_mean.copy_(source_state[f"{nm}.running_mean"])
                                m.running_var.copy_(source_state[f"{nm}.running_var"])
                        ema_ent = None
