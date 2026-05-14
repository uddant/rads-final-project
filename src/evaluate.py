"""Evaluation and diagnostic helpers."""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader


def evaluate_accuracy(model: nn.Module, test_loader: DataLoader, device: torch.device) -> float:
    """Compute classification accuracy on the test set.

    Args:
        model: Model to evaluate.
        test_loader: DataLoader over the MNIST test set.
        device: CPU or GPU device.

    Returns:
        Accuracy as a fraction in ``[0, 1]``.
    """
    model.to(device)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.numel()

    return correct / total


def get_sample_predictions(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    num_samples: int = 20,
) -> List[Tuple[torch.Tensor, int, int]]:
    """Collect example images, labels, and model predictions.

    Args:
        model: Trained model.
        test_loader: Loader over test images.
        device: CPU or GPU device.
        num_samples: Number of examples to return.

    Returns:
        List of ``(image_tensor_cpu, true_label, predicted_label)`` tuples.
    """
    model.to(device)
    model.eval()
    samples: List[Tuple[torch.Tensor, int, int]] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images_device = images.to(device)
            preds = model(images_device).argmax(dim=1).cpu()
            for image, true_label, pred_label in zip(images, labels, preds):
                samples.append((image.cpu(), int(true_label), int(pred_label)))
                if len(samples) >= num_samples:
                    return samples
    return samples


def capture_single_round_gradient(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    """Capture a true flattened gradient from one small supervised batch.

    Args:
        model: Model whose gradient should be measured.
        dataloader: DataLoader providing at least one mini-batch.
        device: CPU or GPU device.

    Returns:
        A one-dimensional tensor containing the exact gradient for one batch.
        This is useful for visualizing how Count Sketch reconstruction compares
        against a known vector.
    """
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()

    images, labels = next(iter(dataloader))
    images = images.to(device)
    labels = labels.to(device)

    model.zero_grad(set_to_none=True)
    loss = criterion(model(images), labels)
    loss.backward()

    gradients = []
    for parameter in model.parameters():
        if parameter.grad is None:
            gradients.append(torch.zeros_like(parameter).reshape(-1))
        else:
            gradients.append(parameter.grad.detach().reshape(-1))
    return torch.cat(gradients)
