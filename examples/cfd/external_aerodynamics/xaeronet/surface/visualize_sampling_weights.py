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

"""
Standalone utility to visualize the geometry-aware sampling weights used by
``preprocessor3d.sample_boundary_from_mesh``.

Given an STL (or any surface readable by PyVista), it computes the per-cell
curvature importance, feature-edge proximity, combined sampling weight and the
resulting draw probability, attaches them as cell data, and writes a ``.vtp``
that can be opened in ParaView. This lets you sweep the ``sampling`` knobs and
see where points will concentrate without running the full preprocessing.

Example
-------
    python visualize_sampling_weights.py drivaer_data/run_1/drivaer_1.stl \
        --output sampling_weights.vtp \
        --curvature-weight 4.0 --feature-edge-weight 2.0 --density-exponent 1.5
"""

import argparse

import numpy as np
import pyvista as pv

from preprocessor3d import (
    compute_cell_curvature_weights,
    compute_cell_edge_weights,
    compute_cell_sampling_weights,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stl", help="Path to the input STL / surface mesh.")
    parser.add_argument(
        "--output",
        default="sampling_weights.vtp",
        help="Path to the output .vtp file (default: sampling_weights.vtp).",
    )
    parser.add_argument("--curvature-weight", type=float, default=4.0)
    parser.add_argument("--curvature-exponent", type=float, default=1.0)
    parser.add_argument("--curvature-clip-percentile", type=float, default=95.0)
    parser.add_argument("--curvature-smoothing-iterations", type=int, default=5)
    parser.add_argument("--feature-edge-weight", type=float, default=2.0)
    parser.add_argument("--feature-edge-angle", type=float, default=30.0)
    parser.add_argument("--feature-edge-falloff", type=float, default=0.02)
    parser.add_argument("--density-exponent", type=float, default=1.0)
    parser.add_argument("--min-weight", type=float, default=0.3)
    return parser.parse_args()


def main():
    """Compute sampling weights for a mesh and save them for visualization."""
    args = parse_args()

    sampling_cfg = {
        "strategy": "curvature",
        "curvature_weight": args.curvature_weight,
        "curvature_exponent": args.curvature_exponent,
        "curvature_clip_percentile": args.curvature_clip_percentile,
        "curvature_smoothing_iterations": args.curvature_smoothing_iterations,
        "feature_edge_weight": args.feature_edge_weight,
        "feature_edge_angle": args.feature_edge_angle,
        "feature_edge_falloff": args.feature_edge_falloff,
        "density_exponent": args.density_exponent,
        "min_weight": args.min_weight,
    }

    mesh = pv.read(args.stl).triangulate()
    mesh = mesh.compute_normals(
        cell_normals=True, point_normals=False, auto_orient_normals=True
    )

    areas = np.asarray(
        mesh.compute_cell_sizes(length=False, volume=False)["Area"]
    )
    curvature_importance = compute_cell_curvature_weights(mesh, sampling_cfg)
    edge_importance = compute_cell_edge_weights(mesh, sampling_cfg)
    weights = compute_cell_sampling_weights(mesh, sampling_cfg)

    # Draw probability actually used by the sampler: proportional to area * weight.
    draw_weights = areas * weights
    probability = draw_weights / draw_weights.sum()

    # Relative density vs. pure area weighting (1.0 == unchanged).
    relative_density = weights / weights.mean()

    mesh.cell_data["curvature_importance"] = curvature_importance
    mesh.cell_data["edge_importance"] = edge_importance
    mesh.cell_data["sampling_weight"] = weights
    mesh.cell_data["draw_probability"] = probability
    mesh.cell_data["relative_density"] = relative_density

    mesh.save(args.output)
    print(
        f"Wrote {args.output} with {mesh.n_cells} cells. "
        f"Relative density range: [{relative_density.min():.2f}, "
        f"{relative_density.max():.2f}]."
    )


if __name__ == "__main__":
    main()
