"""Run the complete MNIST federated learning experiment.

This script is intentionally linear and readable.  It performs exactly the
steps described in the project prompt: seed everything, train FedAvg, train
Count Sketch variants, save result files, and generate the three required
figures.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src import config
from src.data import load_mnist
from src.evaluate import capture_single_round_gradient
from src.model import SmallCNN
from src.server import count_parameters
from src.sketch import CountSketch
from src.trainer import federated_train
from plots.plotting import (
    plot_accuracy_vs_compression,
    plot_gradient_recovery,
    plot_prediction_grid,
)


def set_all_seeds(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms can make some GPU operations slower, but this
    # project favors clarity/reproducibility over peak throughput.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _result_path(method: str, ratio: int | None = None) -> Path:
    """Return the saved-result path for a training run."""
    if method == "fedavg":
        return config.RESULTS_DIR / "fedavg_baseline.pt"
    return config.RESULTS_DIR / f"countsketch_ratio_{ratio}.pt"


def _load_model_from_result(result: dict) -> SmallCNN:
    """Rebuild a SmallCNN and load a saved result state dict."""
    model = SmallCNN(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    model.load_state_dict(result["final_model_state_dict"])
    model.eval()
    return model


def main() -> None:
    """Run all experiments and generate all figures."""
    set_all_seeds(config.SEED)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Detected device: {config.DEVICE}")
    print("Starting FedAvg baseline...")
    fedavg_result = federated_train(method="fedavg", config=config)
    print("Finished FedAvg baseline.")

    all_results: dict[float, dict] = {1: fedavg_result}

    for ratio in config.SKETCH_COMPRESSION_RATIOS:
        print(f"Starting Count Sketch run at target compression {ratio}x...")
        result = federated_train(
            method="countsketch",
            sketch_compression_ratio=ratio,
            config=config,
        )
        all_results[ratio] = result
        print(f"Finished Count Sketch run at target compression {ratio}x.")

    print("Generating plots...")
    plot_accuracy_vs_compression(
        all_results,
        config.FIGURES_DIR / "accuracy_vs_compression.png",
    )

    # Build the prediction-grid models requested in the prompt.  If a particular
    # ratio is absent because the configuration was changed, we skip it rather
    # than crashing after a long experiment.
    _, test_dataset = load_mnist()
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
    models_for_grid = {"FedAvg": _load_model_from_result(fedavg_result)}
    for ratio in config.SKETCH_COMPRESSION_RATIOS:
        if ratio in all_results:
            models_for_grid[f"CS@{ratio}"] = _load_model_from_result(all_results[ratio])
    plot_prediction_grid(
        models_for_grid,
        test_loader,
        config.FIGURES_DIR / "prediction_grid.png",
    )

    # Visualization 3 uses one exact gradient vector and several reconstructed
    # versions to show how sketch size controls approximation quality.
    diagnostic_model = _load_model_from_result(fedavg_result)
    small_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    true_gradient = capture_single_round_gradient(diagnostic_model, small_loader, config.DEVICE)

    vector_dim = count_parameters(diagnostic_model)
    sketch_estimates = {}
    for ratio in config.SKETCH_COMPRESSION_RATIOS:
        width = max(1, int(round(vector_dim / (config.SKETCH_ROWS * ratio))))
        sketcher = CountSketch(
            num_rows=config.SKETCH_ROWS,
            num_cols=width,
            vector_dim=vector_dim,
            seed=config.SEED + 999,
        )
        sketch_estimates[ratio] = sketcher.unsketch(sketcher.sketch(true_gradient))

    plot_gradient_recovery(
        true_gradient,
        sketch_estimates,
        config.FIGURES_DIR / "gradient_recovery.png",
    )

    print(f"Saved results to: {config.RESULTS_DIR}")
    print(f"Saved figures to: {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
