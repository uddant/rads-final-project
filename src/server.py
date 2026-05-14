"""Server-side aggregation for FedAvg and Count Sketch FL."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from .sketch import CountSketch


def fedavg_aggregate(update_vectors: list[torch.Tensor]) -> torch.Tensor:
    """Average full client update vectors.

    Args:
        update_vectors: List of flattened client updates.

    Returns:
        The element-wise mean update.  This is the standard FedAvg baseline
        when each participating client has equal weight.
    """
    return torch.stack(update_vectors, dim=0).mean(dim=0)


def sketch_aggregate(sketches: list[torch.Tensor], sketcher: CountSketch) -> torch.Tensor:
    """Average Count Sketches and reconstruct an update vector.

    Args:
        sketches: List of client sketch matrices, all with shape ``(d, w)``.
        sketcher: CountSketch object that knows the hash/sign functions needed
            to unsketch the averaged matrix.

    Returns:
        Approximate averaged update vector.

    Core idea:
        Count Sketch is linear, so summing sketch matrices equals sketching the
        sum of the original vectors.  That lets the server aggregate compressed
        client messages without first reconstructing each client update.
    """
    mean_sketch = torch.stack(sketches, dim=0).mean(dim=0)
    return sketcher.unsketch(mean_sketch)


def apply_update_vector(model: nn.Module, update_vector: torch.Tensor) -> None:
    """Unflatten and add an update vector to a model's parameters.

    Args:
        model: Global model to modify in place.
        update_vector: Flattened update whose layout matches ``model.parameters()``.
    """
    pointer = 0
    with torch.no_grad():
        for parameter in model.parameters():
            num_values = parameter.numel()
            update_slice = update_vector[pointer : pointer + num_values]
            parameter.add_(update_slice.view_as(parameter))
            pointer += num_values

    if pointer != update_vector.numel():
        raise ValueError("Update vector length did not match model parameters.")


def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters in a model."""
    return sum(parameter.numel() for parameter in model.parameters())
