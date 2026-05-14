"""Count Sketch compression for high-dimensional model updates.

Count Sketch is a randomized linear sketch for compressing a long vector into a
much smaller matrix.  For each coordinate of the input vector, every sketch row
chooses two things: a bucket index and a random sign.  The coordinate's value is
multiplied by that sign and added into the chosen bucket.  Many coordinates may
collide in the same bucket, so reconstruction is approximate rather than exact.

Count Sketch is closely related to Count-Min Sketch, but it is the right tool
for gradients and neural-network updates because model updates contain positive
and negative values.  Count-Min Sketch assumes nonnegative counts and recovers
using a minimum operation.  Count Sketch adds random signs and recovers by taking
a median across rows.  The signs make each coordinate estimate unbiased even
when other coordinates collide with it.

The key property for federated learning is linearity:

    sketch(a + b) = sketch(a) + sketch(b)

This means clients can transmit compressed update sketches, the server can add
and average those sketches directly, and only then reconstruct an approximate
average update.  The server never needs each client's full update vector.

Informally, reconstruction error decreases as sketch width grows.  For a vector
v, collision noise variance scales like ||v||^2 / w, where w is the number of
columns per row.  More columns mean fewer collisions; more rows make the median
more robust to unlucky collisions.

This implementation pre-computes all hash buckets and signs as tensors.  The
same object can be moved to CPU or GPU, and sketching uses tensor operations
rather than Python loops over coordinates.
"""

from __future__ import annotations

import numpy as np
import torch


class CountSketch:
    """Randomized Count Sketch compressor for one-dimensional tensors."""

    def __init__(self, num_rows: int, num_cols: int, vector_dim: int, seed: int = 0):
        """Initialize independent hash and sign functions.

        Args:
            num_rows: Number of independent sketch rows, often called ``d``.
            num_cols: Number of buckets per row, often called ``w``.
            vector_dim: Length of vectors that will be sketched.
            seed: Base random seed for reproducible hash functions.
        """
        self.num_rows = int(num_rows)
        self.num_cols = int(num_cols)
        self.vector_dim = int(vector_dim)
        self.seed = int(seed)

        buckets = []
        signs = []
        for row in range(self.num_rows):
            # Independent RandomState objects make it explicit that each row has
            # its own hash functions while still being reproducible.
            rng = np.random.RandomState(self.seed + row)
            buckets.append(rng.randint(0, self.num_cols, size=self.vector_dim))
            signs.append(rng.choice([-1.0, 1.0], size=self.vector_dim))

        # Store on CPU initially.  sketch()/unsketch() move these tensors to the
        # input's device so the class works naturally with CPU or GPU tensors.
        self.buckets = torch.as_tensor(np.stack(buckets), dtype=torch.long)
        self.signs = torch.as_tensor(np.stack(signs), dtype=torch.float32)

    def sketch(self, vector: torch.Tensor) -> torch.Tensor:
        """Compress a vector into a ``(num_rows, num_cols)`` sketch matrix.

        Args:
            vector: One-dimensional tensor of length ``vector_dim``.

        Returns:
            A sketch matrix.  Each row contains signed bucket sums.
        """
        if vector.ndim != 1 or vector.numel() != self.vector_dim:
            raise ValueError(f"Expected 1D vector of length {self.vector_dim}.")

        device = vector.device
        buckets = self.buckets.to(device)
        signs = self.signs.to(device=device, dtype=vector.dtype)

        sketch_matrix = torch.zeros(
            self.num_rows, self.num_cols, device=device, dtype=vector.dtype
        )

        # For each row, add signed vector entries into their assigned buckets.
        # scatter_add_ is the tensor equivalent of:
        #   sketch[row, h_row[j]] += s_row[j] * vector[j]
        for row in range(self.num_rows):
            sketch_matrix[row].scatter_add_(0, buckets[row], signs[row] * vector)

        return sketch_matrix

    def unsketch(self, sketch_matrix: torch.Tensor) -> torch.Tensor:
        """Recover an approximate vector from a sketch matrix.

        Args:
            sketch_matrix: Tensor with shape ``(num_rows, num_cols)``.

        Returns:
            One-dimensional tensor estimating the original vector.  For each
            coordinate, each row gives a signed estimate; the coordinate-wise
            median across rows reduces the impact of collisions.
        """
        expected_shape = (self.num_rows, self.num_cols)
        if tuple(sketch_matrix.shape) != expected_shape:
            raise ValueError(f"Expected sketch matrix of shape {expected_shape}.")

        device = sketch_matrix.device
        buckets = self.buckets.to(device)
        signs = self.signs.to(device=device, dtype=sketch_matrix.dtype)

        row_estimates = []
        for row in range(self.num_rows):
            # Gather the bucket value assigned to each original coordinate, then
            # multiply by the same sign used during sketching.  This reverses the
            # random sign for the target coordinate while leaving collision noise
            # with random signs, producing an unbiased estimate.
            estimates = sketch_matrix[row, buckets[row]] * signs[row]
            row_estimates.append(estimates)

        stacked = torch.stack(row_estimates, dim=0)
        return torch.median(stacked, dim=0).values

    def compression_ratio(self) -> float:
        """Return ``original_length / sketch_size``.

        Returns:
            A float greater than 1 when the sketch uses fewer numbers than the
            original vector.
        """
        return self.vector_dim / float(self.num_rows * self.num_cols)
