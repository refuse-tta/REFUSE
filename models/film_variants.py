"""
Generalised FiLM-wrapped ResNet-26 with configurable insertion point.

Used by the ablation study (scripts/adapt_models.py) to compare FiLM
placement after different stages of the backbone.
"""

import torch
import torch.nn as nn

from methods.refuse import FiLM


# Mapping from insertion-point name → (channel width, where to insert)
INSERTION_POINTS = {
    "prep":   32,   # after prep / stem
    "layer1": 32,   # after stage 1
    "layer2": 64,   # after stage 2
    "layer3": 128,  # after stage 3
}


class ResNet26_FiLM_General(nn.Module):
    """
    Wraps a frozen ResNet-26 and inserts a FiLM layer at a specified stage.

    Parameters
    ----------
    base_model : nn.Module
        Pre-trained ResNet-26 backbone.
    insert_after : str
        One of 'prep', 'layer1', 'layer2', 'layer3'.
    """

    def __init__(self, base_model, insert_after="prep"):
        super().__init__()
        assert insert_after in INSERTION_POINTS, (
            f"insert_after must be one of {list(INSERTION_POINTS.keys())}, "
            f"got '{insert_after}'"
        )
        self.m = base_model
        self.insert_after = insert_after
        self.film = FiLM(INSERTION_POINTS[insert_after])

    def forward(self, x):
        m = self.m

        x = m.prep(x)
        if self.insert_after == "prep":
            x = self.film(x)

        x = m.layer1(x)
        if self.insert_after == "layer1":
            x = self.film(x)

        x = m.layer2(x)
        if self.insert_after == "layer2":
            x = self.film(x)

        x = m.layer3(x)
        if self.insert_after == "layer3":
            x = self.film(x)

        x = m.pool(x)
        x = torch.flatten(x, 1)
        return m.fc(x)
