"""Script to run plot_gradient_recovery with 3 plots per row."""

import math
from pathlib import Path

import numpy as np
import torch

from plots.plotting import plot_gradient_recovery
from src.config import SKETCH_COMPRESSION_RATIOS

if __name__ == "__main__":
    # Create example data with compression ratios from config
    true_grad = torch.randn(10000)
    compression_ratios = [float(r) for r in SKETCH_COMPRESSION_RATIOS]

    sketch_estimates = {}
    for ratio in compression_ratios:
        # Simulate sketch estimates with decreasing quality at higher compression
        noise = torch.randn_like(true_grad) * (0.1 * np.log(ratio))
        sketch_estimates[ratio] = true_grad + noise

    # Calculate rows with max 3 columns per row
    num_ratios = len(compression_ratios)
    cols_per_row = 3
    num_rows = math.ceil(num_ratios / cols_per_row)
    rows_cols_per_row = [cols_per_row] * (num_rows - 1)
    # Last row gets remaining plots
    rows_cols_per_row.append(num_ratios - (num_rows - 1) * cols_per_row)

    output_path = Path("plots/gradient_recovery_custom_grid.png")

    plot_gradient_recovery(
        true_grad,
        sketch_estimates,
        output_path,
        rows_cols_per_row=rows_cols_per_row,
    )

    print(f"Plot saved to {output_path}")
    print(f"Layout: {rows_cols_per_row}")
