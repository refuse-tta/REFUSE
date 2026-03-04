"""
TENT: Fully Test-Time Adaptation by Entropy Minimization.
Reference: Wang et al., ICLR 2021.

Adapts only BatchNorm affine parameters (γ, β) by minimising
the marginal entropy of softmax predictions on each test batch.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# Default hyper-parameters
# ──────────────────────────────────────────────
TENT_LR = 0.001
TENT_EPOCHS = 1


def configure_tent(model):
    """
    Prepare model for TENT:
      - Eval mode globally (freeze everything).
      - BN layers switched to train mode (batch stats).
      - Only BN affine params (weight, bias) get gradients.

    Returns list of trainable parameters.
    """
    model.eval()
    model.requires_grad_(False)

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()
            m.weight.requires_grad_(True)
            m.bias.requires_grad_(True)

    return [p for p in model.parameters() if p.requires_grad]


class TENT:
    """
    Stateful TENT adapter.

    Parameters
    ----------
    model : nn.Module
        Pre-trained backbone.
    lr : float
        Adam learning rate for BN affine params.
    epochs : int
        Number of passes over the test loader per adaptation call.
    """

    def __init__(self, model, lr=TENT_LR, epochs=TENT_EPOCHS):
        self.device = next(model.parameters()).device
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.base_state = copy.deepcopy(model.state_dict())

    def reset(self):
        """Restore model to pre-adaptation state."""
        self.model.load_state_dict(copy.deepcopy(self.base_state))

    def adapt_loader(self, loader):
        """Run TENT adaptation over the full loader."""
        params = configure_tent(self.model)
        optimizer = torch.optim.Adam(params, lr=self.lr)

        for _ in range(self.epochs):
            for imgs, _ in loader:
                imgs = imgs.to(self.device, non_blocking=True)
                logits = self.model(imgs)
                probs = F.softmax(logits, dim=1)
                loss = -(probs * probs.log()).sum(dim=1).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
