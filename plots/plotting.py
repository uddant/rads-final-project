"""Plotting utilities for the Count Sketch federated learning project."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.evaluate import get_sample_predictions


sns.set_theme(style="whitegrid", context="talk")


def plot_accuracy_vs_compression(results_dict: Dict[float, dict], save_path: Path) -> None:
    """Plot final test accuracy as compression increases.

    Args:
        results_dict: Mapping from compression ratio to result dictionaries.
            Ratio ``1`` is expected to represent the uncompressed FedAvg run.
        save_path: Destination PNG path.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fedavg_acc = results_dict[1]["accuracy_history"][-1]
    cs_ratios = sorted(r for r in results_dict if r != 1)
    cs_accs = [results_dict[r]["accuracy_history"][-1] for r in cs_ratios]

    plt.figure(figsize=(9, 6))
    plt.plot(cs_ratios, cs_accs, marker="o", label="Count Sketch")
    plt.axhline(fedavg_acc, linestyle="--", label=f"FedAvg ({fedavg_acc:.3f})")
    plt.xscale("log")
    plt.xlabel("Compression ratio (log scale; 1 = uncompressed FedAvg)")
    plt.ylabel("Final test accuracy")
    plt.title("EMNIST accuracy under Count Sketch compression")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_prediction_grid(
    models_dict: Dict[str, nn.Module],
    test_loader: DataLoader,
    save_path: Path,
    label_map: Dict[int, str] | None = None,
) -> None:
    """Show example MNIST/EMNIST predictions for several trained models.

    Args:
        models_dict: Ordered mapping from method name to trained model.  The
            intended rows are FedAvg, CS@10, CS@50, and CS@100.
        test_loader: Loader over test examples.
        save_path: Destination PNG path.
        label_map: Optional mapping from integer class index to display string.
            When None, the raw integer is shown (correct for MNIST digits).
            For EMNIST, pass the result of ``get_emnist_label_map(split)``
            so that class 23 shows as ``'n'`` rather than ``23``.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    def _label_str(idx: int) -> str:
        return label_map[idx] if label_map is not None else str(idx)

    method_names = list(models_dict.keys())
    num_cols = 10
    num_rows = len(method_names)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(1.7 * num_cols, 2.0 * num_rows))

    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, method_name in enumerate(method_names):
        model = models_dict[method_name]
        device = next(model.parameters()).device
        samples = get_sample_predictions(model, test_loader, device, num_samples=num_cols)

        for col_idx, (image, true_label, pred_label) in enumerate(samples):
            ax = axes[row_idx, col_idx]
            ax.imshow(image.squeeze().numpy(), cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            color = "green" if true_label == pred_label else "red"
            ax.set_xlabel(
                f"pred {_label_str(pred_label)}\ntrue {_label_str(true_label)}",
                color=color,
                fontsize=10,
            )
            if col_idx == 0:
                ax.set_ylabel(method_name, rotation=0, labelpad=55, fontsize=12, va="center")

    fig.suptitle("Sample predictions: errors become visible at extreme compression", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_gradient_recovery(
    true_gradient: torch.Tensor,
    sketch_estimates_dict: Dict[float, torch.Tensor],
    save_path: Path,
) -> None:
    """Compare true gradient coordinates against Count Sketch estimates.

    Args:
        true_gradient: Exact flattened gradient vector.
        sketch_estimates_dict: Mapping from compression ratio to reconstructed
            gradient estimate.
        save_path: Destination PNG path.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    ratios = sorted(sketch_estimates_dict.keys())
    n = len(ratios)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    x_full = true_gradient.detach().cpu().numpy()
    rng = np.random.RandomState(0)
    # Plotting every coordinate can be slow and visually saturated.  A fixed
    # subsample preserves the trend and keeps the figure readable.
    sample_size = min(4000, x_full.shape[0])
    sample_idx = rng.choice(x_full.shape[0], size=sample_size, replace=False)
    x = x_full[sample_idx]

    min_val = float(np.percentile(x, 1))
    max_val = float(np.percentile(x, 99))

    for ax, ratio in zip(axes, ratios):
        y = sketch_estimates_dict[ratio].detach().cpu().numpy()[sample_idx]
        ax.scatter(x, y, s=8, alpha=0.35)
        ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=1)
        ax.set_title(f"CS compression {ratio}x")
        ax.set_xlabel("True gradient")
        ax.set_ylabel("Recovered estimate")

    fig.suptitle("Count Sketch gradient reconstruction quality")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
