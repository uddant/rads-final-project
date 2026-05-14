"""Small convolutional neural network for MNIST."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    """A compact CNN that trains quickly in a federated simulation.

    The model has roughly ~190k trainable parameters:
        conv1: 32 * 1 * 5 * 5 + 32 = 832
        conv2: 64 * 32 * 5 * 5 + 64 = 51,264
        fc1:   1024 * 128 + 128 = 131,200
        fc2:   128 * num_classes + num_classes = varies
        The parameter count is modest enough for CPU experiments, yet large enough
        that compressing update vectors with Count Sketch is meaningful.
    """

    def __init__(self, num_classes: int = 10) -> None:
        """Create the convolutional and fully connected layers."""
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a batch of MNIST images through the network.

        Args:
            x: Tensor of shape ``(batch, 1, 28, 28)``.

        Returns:
            Raw class logits of shape ``(batch, 10)``.
        """
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
