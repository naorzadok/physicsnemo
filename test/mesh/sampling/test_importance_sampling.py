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

"""Tests for geometry-aware importance sampling of mesh cells.

Validates the pure-torch weighted cell sampler and the native curvature weight
computation, including uniform recovery on a sphere and the boosting of weights
in high-curvature regions of a cone.
"""

import pytest
import torch

from physicsnemo.mesh.sampling import (
    sample_cells_by_weight,
    sample_random_points_on_cells,
)

### sample_cells_by_weight (pure torch, no optional dependencies) ###


class TestSampleCellsByWeight:
    """Tests for sample_cells_by_weight."""

    def test_draws_only_nonzero_weight_cells(self):
        """All draws land on the single cell with non-zero weight."""
        weights = torch.tensor([0.0, 0.0, 1.0, 0.0])
        idx = sample_cells_by_weight(weights, 32, generator=torch.manual_seed(0))
        assert idx.shape == (32,)
        assert idx.dtype == torch.int64
        assert bool((idx == 2).all())

    def test_proportional_frequencies(self):
        """Empirical draw frequencies track the normalized weights."""
        weights = torch.tensor([1.0, 3.0])
        idx = sample_cells_by_weight(weights, 20000, generator=torch.manual_seed(0))
        fraction_one = float((idx == 1).float().mean())
        assert abs(fraction_one - 0.75) < 0.02

    def test_without_replacement(self):
        """Sampling without replacement yields unique indices."""
        weights = torch.ones(10)
        idx = sample_cells_by_weight(
            weights, 10, replacement=False, generator=torch.manual_seed(0)
        )
        assert idx.shape == (10,)
        assert len(torch.unique(idx)) == 10

    def test_reproducible_with_generator(self):
        """Identical seeds produce identical draws."""
        weights = torch.rand(50) + 0.1
        a = sample_cells_by_weight(weights, 100, generator=torch.manual_seed(7))
        b = sample_cells_by_weight(weights, 100, generator=torch.manual_seed(7))
        assert bool((a == b).all())

    @pytest.mark.parametrize(
        "weights, num_samples",
        [
            (torch.ones(2, 2), 4),  # not 1-D
            (torch.ones(4), 0),  # non-positive count
            (torch.tensor([1.0, -1.0]), 2),  # negative weight
            (torch.zeros(4), 2),  # all-zero weights
        ],
    )
    def test_invalid_inputs_raise(self, weights, num_samples):
        """Invalid weight tensors or sample counts raise ValueError."""
        with pytest.raises(ValueError):
            sample_cells_by_weight(weights, num_samples)


### compute_curvature_sampling_weights (pure torch, native curvature) ###


@pytest.fixture
def sphere_mesh():
    """A triangular unit-sphere surface mesh (uniform curvature)."""
    from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

    return sphere_icosahedral.load(subdivisions=3)


@pytest.fixture
def cone_mesh():
    """A triangular cone surface with strongly varying curvature (apex/rim)."""
    from physicsnemo.mesh.primitives.surfaces import cone

    return cone.load()


class TestCurvatureSamplingWeights:
    """Tests for compute_curvature_sampling_weights."""

    def test_shape_dtype_and_floor(self, sphere_mesh):
        """Weights are per-cell float32 and respect the min_weight floor."""
        from physicsnemo.mesh.sampling import compute_curvature_sampling_weights

        weights = compute_curvature_sampling_weights(sphere_mesh, min_weight=0.3)
        assert weights.shape == (sphere_mesh.n_cells,)
        assert weights.dtype == torch.float32
        assert float(weights.min()) >= 0.3

    def test_uniform_recovery(self, sphere_mesh):
        """Disabling the curvature term recovers a uniform per-cell weight."""
        from physicsnemo.mesh.sampling import compute_curvature_sampling_weights

        weights = compute_curvature_sampling_weights(
            sphere_mesh, curvature_weight=0.0
        )
        assert torch.allclose(weights, torch.ones_like(weights))

    def test_uniform_curvature_is_flat(self, sphere_mesh):
        """A constant-curvature sphere yields near-constant weights."""
        from physicsnemo.mesh.sampling import compute_curvature_sampling_weights

        weights = compute_curvature_sampling_weights(sphere_mesh)
        assert float(weights.max()) - float(weights.min()) < 0.1

    def test_curvature_boosts_high_curvature_cells(self, cone_mesh):
        """High-curvature cells (cone apex/rim) receive above-baseline weight."""
        from physicsnemo.mesh.sampling import compute_curvature_sampling_weights

        weights = compute_curvature_sampling_weights(
            cone_mesh, curvature_weight=4.0
        )
        assert float(weights.max()) > 1.0 + 1e-3
        assert float(weights.max()) - float(weights.min()) > 1e-3

    def test_rejects_non_triangular_3d_mesh(self):
        """A 2-manifold mesh in 2D space is rejected."""
        from physicsnemo.mesh.mesh import Mesh
        from physicsnemo.mesh.sampling import compute_curvature_sampling_weights

        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [1.5, 0.5]])
        cells = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)
        with pytest.raises(ValueError):
            compute_curvature_sampling_weights(mesh)

    def test_composes_with_point_sampling(self, cone_mesh):
        """Weights feed the sampler and produce points on the surface."""
        from physicsnemo.mesh.sampling import compute_curvature_sampling_weights

        weights = compute_curvature_sampling_weights(cone_mesh)
        idx = sample_cells_by_weight(weights, 256, generator=torch.manual_seed(0))
        points = sample_random_points_on_cells(cone_mesh, idx)
        assert points.shape == (256, 3)
        assert int(idx.max()) < cone_mesh.n_cells
