# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Geometry-aware importance sampling of mesh cells.

Uniform (area-weighted) cell sampling spends the sample budget evenly over a
surface, which under-resolves the small, high-curvature regions where physical
fields (e.g. CFD pressure / shear gradients) vary most. This module provides
two composable pieces:

- :func:`compute_curvature_sampling_weights` derives a per-cell importance
  weight from surface curvature.
- :func:`sample_cells_by_weight` draws cell indices with probability
  proportional to any per-cell weight vector.

The drawn indices feed directly into
:func:`~physicsnemo.mesh.sampling.random_point_sampling.sample_random_points_on_cells`
to obtain importance-sampled points on the surface.

Both functions are pure PyTorch with no optional dependencies; curvature is
computed with the native :mod:`physicsnemo.mesh.curvature` module.
"""

import torch
from jaxtyping import Float, Int

from physicsnemo.mesh.mesh import Mesh


def sample_cells_by_weight(
    weights: Float[torch.Tensor, " n_cells"],
    num_samples: int,
    *,
    replacement: bool = True,
    generator: torch.Generator | None = None,
) -> Int[torch.Tensor, " num_samples"]:
    """Draw cell indices with probability proportional to per-cell weights.

    Parameters
    ----------
    weights : torch.Tensor
        Non-negative per-cell importance weights of shape ``(n_cells,)``. Need
        not be normalized; the values are treated as unnormalized probabilities.
    num_samples : int
        Number of cell indices to draw.
    replacement : bool, optional, default=True
        Whether to sample with replacement. When ``False``, ``num_samples`` must
        not exceed the number of cells with non-zero weight.
    generator : torch.Generator or None, optional, default=None
        Optional random generator for reproducible sampling.

    Returns
    -------
    torch.Tensor
        Sampled cell indices of shape ``(num_samples,)`` and dtype ``int64``,
        on the same device as ``weights``. Repeated indices are expected when
        ``replacement=True``.

    Raises
    ------
    ValueError
        If ``weights`` is not 1-D, ``num_samples`` is not positive, or any
        weight is negative.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.mesh.sampling import sample_cells_by_weight
    >>> weights = torch.tensor([0.0, 0.0, 1.0])
    >>> idx = sample_cells_by_weight(weights, 4, generator=torch.manual_seed(0))
    >>> bool((idx == 2).all())
    True
    """
    if weights.ndim != 1:
        raise ValueError(f"weights must be 1-D, got shape {tuple(weights.shape)}")
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if torch.any(weights < 0):
        raise ValueError("weights must be non-negative")

    probabilities = weights.to(torch.float64)
    total = probabilities.sum()
    if total <= 0:
        raise ValueError("weights must contain at least one positive value")

    return torch.multinomial(
        probabilities / total,
        num_samples,
        replacement=replacement,
        generator=generator,
    )


def _smooth_vertex_scalar(
    cells: Int[torch.Tensor, "n_cells cell_size"],
    values: Float[torch.Tensor, " n_points"],
    n_points: int,
    iterations: int,
) -> Float[torch.Tensor, " n_points"]:
    """Umbrella (neighbor-average) smoothing of a per-vertex scalar field."""
    if iterations <= 0:
        return values

    cell_size = cells.shape[1]
    pairs = [
        cells[:, [i, j]]
        for i in range(cell_size)
        for j in range(i + 1, cell_size)
    ]
    edges = torch.cat(pairs, dim=0)
    src = torch.cat([edges[:, 0], edges[:, 1]])
    dst = torch.cat([edges[:, 1], edges[:, 0]])

    smoothed = values.to(torch.float64)
    ones = torch.ones(src.shape[0], dtype=torch.float64, device=values.device)
    counts = torch.zeros(n_points, dtype=torch.float64, device=values.device)
    counts.index_add_(0, src, ones)
    has_neighbors = counts > 0
    safe_counts = counts.clamp_min(1.0)
    for _ in range(iterations):
        summed = torch.zeros(n_points, dtype=torch.float64, device=values.device)
        summed.index_add_(0, src, smoothed[dst])
        neighbor_mean = summed / safe_counts
        smoothed = torch.where(has_neighbors, neighbor_mean, smoothed)
    return smoothed


def _cell_curvature_weights(
    mesh: "Mesh",
    clip_percentile: float,
    smoothing_iterations: int,
) -> Float[torch.Tensor, " n_cells"]:
    """Normalized ``[0, 1]`` per-cell maximum-curvature importance.

    The per-vertex maximum principal curvature magnitude is reconstructed from
    the native mean curvature ``H`` and Gaussian curvature ``K`` via
    ``k_max = |H| + sqrt(max(H**2 - K, 0))``, umbrella-smoothed, averaged to
    cells, clipped at a percentile, and normalized to ``[0, 1]``.
    """
    from physicsnemo.mesh.curvature import (
        gaussian_curvature_vertices,
        mean_curvature_vertices,
    )

    mean_h = mean_curvature_vertices(mesh).abs().to(torch.float64)
    gauss_k = gaussian_curvature_vertices(mesh).to(torch.float64)
    point_curvature = mean_h + torch.sqrt(
        torch.clamp(mean_h * mean_h - gauss_k, min=0.0)
    )
    point_curvature = torch.nan_to_num(
        point_curvature, nan=0.0, posinf=0.0, neginf=0.0
    )

    point_curvature = _smooth_vertex_scalar(
        mesh.cells, point_curvature, mesh.n_points, smoothing_iterations
    )

    cell_curvature = point_curvature[mesh.cells].mean(dim=1)
    clip_value = torch.quantile(cell_curvature, clip_percentile / 100.0)
    if clip_value > 0:
        cell_curvature = cell_curvature.clamp(max=clip_value)
    max_curvature = cell_curvature.max()
    if max_curvature > 0:
        return cell_curvature / max_curvature
    return torch.zeros_like(cell_curvature)


def compute_curvature_sampling_weights(
    mesh: "Mesh",
    *,
    curvature_weight: float = 4.0,
    curvature_exponent: float = 1.0,
    curvature_clip_percentile: float = 95.0,
    curvature_smoothing_iterations: int = 5,
    density_exponent: float = 1.0,
    min_weight: float = 0.3,
) -> Float[torch.Tensor, " n_cells"]:
    """Per-cell importance weights from surface curvature.

    The returned weights bias cell sampling toward geometrically salient regions
    (high curvature). Each cell starts from a uniform baseline of ``1.0``; a
    curvature term is added on top, and a ``min_weight`` floor guarantees every
    cell keeps a non-zero chance of being sampled::

        weight = (1 + curvature_weight * curvature ** curvature_exponent)
                 ** density_exponent
        weight = max(min_weight, weight)

    where ``curvature`` is normalized to ``[0, 1]``. Setting ``curvature_weight``
    to ``0`` recovers a uniform (per-cell) weighting.

    Only triangular surface meshes (2-manifold embedded in 3D) are supported.

    Parameters
    ----------
    mesh : Mesh
        Triangular surface mesh with ``n_manifold_dims == 2`` and
        ``n_spatial_dims == 3``.
    curvature_weight : float, optional, default=4.0
        Strength of the curvature contribution. ``0`` disables curvature and
        yields uniform weights.
    curvature_exponent : float, optional, default=1.0
        Exponent applied to the normalized curvature to sharpen or soften its
        contrast.
    curvature_clip_percentile : float, optional, default=95.0
        Percentile at which per-cell curvature is clipped before normalization,
        limiting the influence of a few extreme cells.
    curvature_smoothing_iterations : int, optional, default=5
        Number of umbrella-smoothing passes applied to the per-vertex curvature
        to denoise faceted geometry. ``0`` disables smoothing.
    density_exponent : float, optional, default=1.0
        Global exponent applied to the combined weight to tune overall contrast.
    min_weight : float, optional, default=0.3
        Lower bound applied to every weight so no cell is ever excluded.

    Returns
    -------
    torch.Tensor
        Per-cell sampling weights of shape ``(n_cells,)`` and dtype ``float32``,
        on the same device as ``mesh``.

    Raises
    ------
    ValueError
        If the mesh is not a triangular 2-manifold embedded in 3D.

    Examples
    --------
    >>> from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
    >>> from physicsnemo.mesh.sampling import (
    ...     compute_curvature_sampling_weights,
    ...     sample_cells_by_weight,
    ...     sample_random_points_on_cells,
    ... )
    >>> mesh = sphere_icosahedral.load(subdivisions=2)
    >>> weights = compute_curvature_sampling_weights(mesh)
    >>> cells = sample_cells_by_weight(weights, 5000)
    >>> points = sample_random_points_on_cells(mesh, cells)
    >>> points.shape
    torch.Size([5000, 3])
    """
    if mesh.n_manifold_dims != 2 or mesh.n_spatial_dims != 3:
        raise ValueError(
            "compute_curvature_sampling_weights requires a 2-manifold mesh in 3D "
            f"space, got n_manifold_dims={mesh.n_manifold_dims}, "
            f"n_spatial_dims={mesh.n_spatial_dims}"
        )

    weights = torch.ones(mesh.n_cells, dtype=torch.float64, device=mesh.device)
    if curvature_weight > 0:
        curvature = _cell_curvature_weights(
            mesh, curvature_clip_percentile, curvature_smoothing_iterations
        )
        weights = weights + curvature_weight * curvature.pow(curvature_exponent)
    if density_exponent != 1.0:
        weights = weights.pow(density_exponent)
    weights = weights.clamp(min=min_weight)

    return weights.to(torch.float32)

