"""
REFUSE: Normalization-free TTA with Entropy-regularized Feature Recalibration.

Core idea:
  - Attach a lightweight FiLM (Feature-wise Linear Modulation) layer after the
    prep stage of a frozen backbone.
  - At test time, adapt *only* FiLM parameters using a composite loss:
        L = λ_gate · ReLU(H̄_n − H₀) + λ_kl · KL(p̄ ‖ U) + λ_stab · ‖θ − θ₀‖²
  - Entropy gating skips adaptation when the batch is already confident.
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# FiLM layer
# ──────────────────────────────────────────────
class FiLM(nn.Module):
    """Feature-wise Linear Modulation: x → γ ⊙ x + β."""

    def __init__(self, num_channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        g = self.gamma.view(1, -1, 1, 1)
        b = self.beta.view(1, -1, 1, 1)
        return x * g + b


# ──────────────────────────────────────────────
# Wrapped backbone with FiLM
# ──────────────────────────────────────────────
class ResNet26_FiLM(nn.Module):
    """
    Wraps a frozen ResNet-26 and inserts a FiLM layer after the prep stage.

    Only FiLM parameters are trainable; everything else is frozen.
    """

    def __init__(self, base_model, film_channels=32):
        super().__init__()
        self.m = base_model
        self.film = FiLM(C=film_channels)

    def forward(self, x):
        m = self.m
        x = m.prep(x)
        x = self.film(x)
        x = m.layer1(x)
        x = m.layer2(x)
        x = m.layer3(x)
        x = m.pool(x)
        x = torch.flatten(x, 1)
        return m.fc(x)


# ──────────────────────────────────────────────
# Loss utilities
# ──────────────────────────────────────────────
def norm_entropy_from_logits(logits, eps=1e-8):
    """
    Returns (softmax probs, normalised entropy per sample).
    Normalised entropy ∈ [0, 1] where 1 = uniform.
    """
    probs = F.softmax(logits, dim=1)
    ent = -(probs * torch.log(probs + eps)).sum(dim=1)
    ent_n = ent / math.log(probs.size(1))
    return probs, ent_n


def kl_mean_to_uniform(probs, eps=1e-8):
    """KL( p̄ ‖ U ) where p̄ is the batch-mean prediction."""
    p_bar = probs.mean(dim=0)
    K = p_bar.numel()
    u = torch.full_like(p_bar, 1.0 / K)
    p_bar = torch.clamp(p_bar, eps, 1.0)
    p_bar = p_bar / p_bar.sum()
    return (p_bar * (torch.log(p_bar) - torch.log(u + eps))).sum()


# ──────────────────────────────────────────────
# REFUSE adapter
# ──────────────────────────────────────────────
class REFUSE:
    """
    Stateful adapter that wraps a base model with FiLM and runs
    entropy-gated adaptation at test time.

    Parameters
    ----------
    base_model : nn.Module
        Pre-trained backbone (e.g. ResNet26).  Its weights are frozen.
    steps : int
        Number of gradient steps per batch when entropy gate is open.
    lr : float
        Learning rate for FiLM parameters.
    h0_floor : float
        Entropy floor — batches with mean normalised entropy below this
        are skipped (already confident).
    lambda_kl : float
        Weight for the KL-to-uniform diversity term.
    lambda_stab : float
        Weight for the L2 stability term ‖θ − θ₀‖².
    lambda_gate : float
        Weight for the gated entropy term ReLU(H̄ − H₀).
    film_channels : int
        Number of channels for the FiLM layer (must match prep output).
    """

    # Default hyper-parameters (paper values)
    DEFAULT_STEPS = 120
    DEFAULT_LR = 5e-4
    DEFAULT_H0_FLOOR = 0.1
    DEFAULT_LAMBDA_KL = 0.16
    DEFAULT_LAMBDA_STAB = 3e-4
    DEFAULT_LAMBDA_GATE = 0.55

    def __init__(
        self,
        base_model,
        steps=DEFAULT_STEPS,
        lr=DEFAULT_LR,
        h0_floor=DEFAULT_H0_FLOOR,
        lambda_kl=DEFAULT_LAMBDA_KL,
        lambda_stab=DEFAULT_LAMBDA_STAB,
        lambda_gate=DEFAULT_LAMBDA_GATE,
        film_channels=32,
    ):
        self.device = next(base_model.parameters()).device
        self.steps = steps
        self.h0_floor = h0_floor
        self.lambda_kl = lambda_kl
        self.lambda_stab = lambda_stab
        self.lambda_gate = lambda_gate

        # Freeze base, build FiLM wrapper
        self.base_state = copy.deepcopy(base_model.state_dict())
        self.film_model = ResNet26_FiLM(base_model, film_channels).to(self.device)

        # Freeze everything, unfreeze only FiLM
        for p in self.film_model.parameters():
            p.requires_grad = False

        self.film_params = []
        for module in self.film_model.modules():
            if isinstance(module, FiLM):
                for p in module.parameters():
                    p.requires_grad = True
                self.film_params.extend(module.parameters())

        # Snapshot of initial FiLM state (for stability term)
        self.theta0 = [p.detach().clone() for p in self.film_params]

        # Optimiser
        self.optimizer = torch.optim.Adam(self.film_params, lr=lr)

    # ── public API ───────────────────────────

    def reset(self):
        """Reset FiLM parameters to identity (γ=1, β=0) and rebuild optimiser."""
        with torch.no_grad():
            self.film_model.film.gamma.fill_(1.0)
            self.film_model.film.beta.zero_()
        self.theta0 = [p.detach().clone() for p in self.film_params]
        self.optimizer = torch.optim.Adam(
            self.film_params, lr=self.optimizer.defaults["lr"]
        )

    def reset_base(self):
        """Reload the original base-model weights (use between corruptions)."""
        self.film_model.m.load_state_dict(copy.deepcopy(self.base_state))
        self.film_model.m.eval()
        self.reset()

    def adapt_batch(self, x):
        """
        Adapt FiLM on a single batch *in-place* (continual).

        Returns the mean normalised entropy before adaptation (for logging).
        """
        x = x.to(self.device, non_blocking=True)

        # Check entropy before adapting
        self.film_model.eval()
        with torch.no_grad():
            logits0 = self.film_model(x)
            _, ent_n0 = norm_entropy_from_logits(logits0)
            h_before = ent_n0.mean().item()

        # Entropy gate
        if h_before >= self.h0_floor:
            for _ in range(self.steps):
                self._step(x)

        return h_before

    def adapt_loader(self, loader):
        """Run continual adaptation over an entire DataLoader."""
        for x, _ in loader:
            self.adapt_batch(x)

    @property
    def model(self):
        """Return the adapted model (for evaluation)."""
        return self.film_model

    # ── internals ────────────────────────────

    def _stability_term(self):
        s = 0.0
        for p, p0 in zip(self.film_params, self.theta0):
            s = s + (p - p0).pow(2).mean()
        return s

    def _step(self, x):
        """Single gradient step on the composite REFUSE loss."""
        self.film_model.eval()           # BN stays in eval mode
        logits = self.film_model(x)
        probs, ent_n = norm_entropy_from_logits(logits)

        h_mean = ent_n.mean()
        gated = F.relu(h_mean - self.h0_floor)
        kl = kl_mean_to_uniform(probs)
        stab = self._stability_term()

        loss = (
            self.lambda_gate * gated
            + self.lambda_kl * kl
            + self.lambda_stab * stab
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
