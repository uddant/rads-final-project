"""Client-side local training and update-vector utilities."""

from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader


def local_train(
    model: nn.Module,
    dataloader: DataLoader,
    lr: float,
    epochs: int,
    device: torch.device,
) -> nn.Module:
    """Train a client model on its private local data.

    Args:
        model: Copy of the current global model.
        dataloader: Loader over one client's private subset.
        lr: SGD learning rate.
        epochs: Number of local epochs.
        device: CPU or GPU device.

    Returns:
        The trained model.  The input model is modified in place and returned
        for convenience.
    """
    model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

    return model


def compute_update_vector(
    old_state_dict: Mapping[str, torch.Tensor],
    new_state_dict: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Flatten (new_params - old_params) into one update vector.

    Args:
        old_state_dict: Parameters before local client training.
        new_state_dict: Parameters after local client training.

    Returns:
        A one-dimensional tensor containing all parameter deltas.

    Federated learning usually sends updates rather than full parameters because
    all clients already know the starting global model for the round.  Flattening
    all tensors into one vector lets Count Sketch treat the whole neural network
    update uniformly, independent of layer shapes.
    """
    deltas = []
    for name in old_state_dict.keys():
        old_tensor = old_state_dict[name]
        new_tensor = new_state_dict[name]
        deltas.append((new_tensor - old_tensor).reshape(-1))
    return torch.cat(deltas)


def clone_state_dict(state_dict: Mapping[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    """Make a detached copy of a model state dict.

    Args:
        state_dict: Model parameters to copy.

    Returns:
        OrderedDict with cloned tensors so later training cannot mutate the
        saved reference parameters.
    """
    return OrderedDict((name, tensor.detach().clone()) for name, tensor in state_dict.items())
