"""
BN-Adapt: Test-Time Batch Normalization Adaptation.
Reference: Schneider et al., NeurIPS 2020.

No parameters are optimised — BN layers simply switch to
batch statistics at test time instead of stored running stats.
"""

import copy

import torch
import torch.nn as nn


def configure_bn_adapt(model):
    """
    Set BN layers to train mode (batch stats) while keeping
    everything else in eval. No parameters are trainable.
    """
    model.eval()
    model.requires_grad_(False)

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()
            m.running_mean.zero_()
            m.running_var.fill_(1.0)
            m.momentum = 0.1

    return model


class BNAdapt:
    """
    Stateful BN-Adapt wrapper.

    This is the simplest TTA baseline — just replace BN running
    stats with batch stats at test time. No gradient updates.
    """

    def __init__(self, model):
        self.device = next(model.parameters()).device
        self.model = model
        self.base_state = copy.deepcopy(model.state_dict())

    def reset(self):
        self.model.load_state_dict(copy.deepcopy(self.base_state))

    def adapt_loader(self, loader):
        """Configure BN for batch stats and do a forward pass to accumulate them."""
        configure_bn_adapt(self.model)
        # Single forward pass so BN running stats converge on target domain
        with torch.no_grad():
            for imgs, _ in loader:
                imgs = imgs.to(self.device, non_blocking=True)
                self.model(imgs)
