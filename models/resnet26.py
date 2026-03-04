"""
ResNet-26 for CIFAR-10 (TENT-compatible variant with BatchNorm).

Architecture: 3 stages × 4 BasicBlocks = 24 conv layers + prep conv + FC = 26 layers.
"""

import torch.nn as nn


class BasicBlock(nn.Module):
    """Standard residual block with two 3×3 convolutions and BN."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x)).relu_()
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return out.relu_()


class ResNet26(nn.Module):
    """
    ResNet-26 tailored for CIFAR-10 (32×32 inputs).

    Stage widths: 32 → 64 → 128, 4 blocks per stage.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.prep = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(32, 32, num_blocks=4, stride=1)
        self.layer2 = self._make_layer(32, 64, num_blocks=4, stride=2)
        self.layer3 = self._make_layer(64, 128, num_blocks=4, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    @staticmethod
    def _make_layer(in_c, out_c, num_blocks, stride):
        layers = [BasicBlock(in_c, out_c, stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_c, out_c, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)
