"""
Buffer Layers for Test-Time Adaptation.
Reference: Kim et al., NeurIPS 2025.

Inserts lightweight parallel adaptation modules (dual-path 1×1 + 3×3 conv
with learnable scaling α) after early and middle stages. Only buffer
parameters are updated at test time via entropy minimisation.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet26 import BasicBlock


# ──────────────────────────────────────────────
# Default hyper-parameters
# ──────────────────────────────────────────────
BUFFER_LR = 1e-3
BUFFER_EPOCHS = 1
ALPHA_INIT = 1e-2


# ──────────────────────────────────────────────
# Buffer layer
# ──────────────────────────────────────────────
class BufferLayer(nn.Module):
    """
    Lightweight parallel adaptation module.
    BufferLayer(x) = α · (Conv1×1(x) + Conv3×3(x))
    """

    def __init__(self, channels, alpha_init=ALPHA_INIT):
        super().__init__()
        self.conv1x1 = nn.Conv2d(channels, channels, 1, bias=False)
        self.conv3x3 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

        nn.init.kaiming_normal_(self.conv1x1.weight)
        nn.init.kaiming_normal_(self.conv3x3.weight)

    def forward(self, x):
        return self.alpha * (self.conv1x1(x) + self.conv3x3(x))


# ──────────────────────────────────────────────
# Modified ResNet-26 with Buffer layers
# ──────────────────────────────────────────────
class ResNet26WithBuffer(nn.Module):
    """
    ResNet-26 with BufferLayer inserted after layer1 (32 ch) and
    layer2 (64 ch), added residually.
    """

    def __init__(self, num_classes=10, alpha_init=ALPHA_INIT):
        super().__init__()
        self.prep = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(32, 32, 4, stride=1)
        self.layer2 = self._make_layer(32, 64, 4, stride=2)
        self.layer3 = self._make_layer(64, 128, 4, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

        self.buffer1 = BufferLayer(32, alpha_init=alpha_init)
        self.buffer2 = BufferLayer(64, alpha_init=alpha_init)

    @staticmethod
    def _make_layer(in_c, out_c, blocks, stride):
        layers = [BasicBlock(in_c, out_c, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_c, out_c, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = x + self.buffer1(x)
        x = self.layer2(x)
        x = x + self.buffer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def build_buffer_model(base_state_dict, device, alpha_init=ALPHA_INIT):
    """
    Create ResNet26WithBuffer and load matching backbone weights
    from a standard ResNet-26 checkpoint (buffer layers stay random).
    """
    model = ResNet26WithBuffer(num_classes=10, alpha_init=alpha_init).to(device)
    model_dict = model.state_dict()
    pretrained = {k: v for k, v in base_state_dict.items()
                  if k in model_dict and "buffer" not in k}
    model_dict.update(pretrained)
    model.load_state_dict(model_dict)
    return model


def configure_buffer(model):
    """
    Freeze backbone (including BN affine), set BN to train mode
    (batch stats), unfreeze only buffer parameters.
    """
    model.eval()
    model.requires_grad_(False)

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()

    for name, p in model.named_parameters():
        if "buffer" in name:
            p.requires_grad_(True)

    return [p for p in model.parameters() if p.requires_grad]


class Buffer:
    """
    Stateful Buffer adapter.

    Parameters
    ----------
    base_state_dict : dict
        State dict of the pre-trained ResNet-26 (no buffer layers).
    device : torch.device
    lr : float
    epochs : int
    alpha_init : float
    """

    def __init__(self, base_state_dict, device, lr=BUFFER_LR,
                 epochs=BUFFER_EPOCHS, alpha_init=ALPHA_INIT):
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.alpha_init = alpha_init
        self.base_state_dict = copy.deepcopy(base_state_dict)
        self.model = None  # built fresh per corruption

    def reset(self):
        """Build a fresh buffer model from the source checkpoint."""
        self.model = build_buffer_model(
            self.base_state_dict, self.device, self.alpha_init,
        )

    def adapt_loader(self, loader):
        """Build fresh model, then adapt buffer params via entropy min."""
        self.reset()
        params = configure_buffer(self.model)
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
