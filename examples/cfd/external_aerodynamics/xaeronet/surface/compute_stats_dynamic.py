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
Compute global normalization statistics for the dynamic (on-the-fly) XAeroNet-S
pipeline directly from the raw DoMINO surface Zarr stores.

Because the dynamic pipeline stores only raw per-point tensors (no pre-built
graphs), this script:
  * accumulates per-point statistics for the node fields (coordinates, normals,
    area, pressure, shear_stress) over each full point cloud, and
  * estimates the edge-feature statistics (key ``"x"``: relative displacement +
    norm) by building the kNN graph exactly as ``DynamicGraphBuilder`` does, on
    a subsample of each cloud, from **raw** coordinates.
Optional per-sample global (parametric) scalars (``global_features``) are
accumulated one row per sample when present.

The output JSON matches the schema consumed by the training loop / builder:
``{"mean": {...}, "std_dev": {...}}`` keyed by the node fields, ``"x"`` and
(optionally) ``"global_features"``.
"""

import json
import os
import sys

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tqdm import tqdm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from dataloader_dynamic import (
    NODE_FIELDS,
    find_zarr_stores,
    load_raw_surface_sample,
    surface_point_count,
)
from dynamic_graph_datapipe import build_knn_graph, build_multilevel_knn_graph


class RunningStats:
    """Streaming mean / std accumulator (weighted by element count)."""

    def __init__(self):
        self.sum: np.ndarray | None = None
        self.sumsq: np.ndarray | None = None
        self.count: int = 0

    def update(self, data: np.ndarray) -> None:
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 1:
            data = data[:, None]
        s = data.sum(axis=0)
        ssq = (data**2).sum(axis=0)
        if self.sum is None:
            self.sum = s
            self.sumsq = ssq
        else:
            self.sum += s
            self.sumsq += ssq
        self.count += data.shape[0]

    def finalize(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self.count == 0 or self.sum is None:
            return None
        mean = self.sum / self.count
        var = self.sumsq / self.count - mean**2
        std = np.sqrt(np.maximum(var, 0.0))
        # Guard constant features (zero variance) against divide-by-zero.
        std = np.where(std < 1e-8, 1.0, std)
        return mean, std


def compute_stats(
    stores: list[str],
    node_degree: int,
    num_sampled_nodes: int | None,
    device: torch.device,
    level_sizes: list[int] | None = None,
) -> tuple[dict, dict]:
    """Accumulate node, edge and global statistics across all stores.

    When ``level_sizes`` is given the edge (``"x"``) statistics are gathered from
    a multi-resolution kNN graph (matching ``DynamicGraphBuilder``), so the
    normalization sees the same long-range edge distribution used at train time.
    """
    fields = list(NODE_FIELDS) + ["x", "global_features"]
    stats = {f: RunningStats() for f in fields}

    # Number of nodes drawn for the edge-statistics subsample.
    if level_sizes is not None:
        edge_sample = int(sum(level_sizes))
    else:
        edge_sample = num_sampled_nodes

    for store in tqdm(stores, desc="Computing stats", unit="sample"):
        # Node-field statistics over the full cloud.
        raw, _ = load_raw_surface_sample(store)
        for name in NODE_FIELDS:
            stats[name].update(raw[name])

        # One global-feature row per sample (identical across the cloud).
        if "global_features" in raw:
            stats["global_features"].update(raw["global_features"])

        # Edge-feature statistics from a kNN graph built on a raw subsample,
        # matching DynamicGraphBuilder (edges are built from raw coordinates).
        n = surface_point_count(store)
        if edge_sample is not None and edge_sample < n:
            sel = np.random.default_rng(0).choice(n, size=edge_sample, replace=False)
            sel.sort()
            coords = raw["coordinates"][sel]
        else:
            coords = raw["coordinates"]

        coords_t = torch.as_tensor(coords, dtype=torch.float32, device=device)
        if level_sizes is not None:
            _, edge_attr = build_multilevel_knn_graph(
                coords_t, level_sizes, node_degree
            )
        else:
            _, edge_attr = build_knn_graph(coords_t, node_degree)
        stats["x"].update(edge_attr.detach().cpu().numpy())

    mean: dict = {}
    std: dict = {}
    for name, acc in stats.items():
        result = acc.finalize()
        if result is None:
            continue
        mean[name], std[name] = result
    return mean, std


def save_stats_to_json(mean: dict, std: dict, output_file: str) -> None:
    """Serialize the statistics dicts to JSON (lists for array values)."""
    payload = {
        "mean": {k: v.tolist() for k, v in mean.items()},
        "std_dev": {k: v.tolist() for k, v in std.items()},
    }
    with open(output_file, "w") as f:
        json.dump(payload, f, indent=4)


@hydra.main(version_base="1.3", config_path="conf", config_name="config_dynamic")
def main(cfg: DictConfig) -> None:
    stores = find_zarr_stores(to_absolute_path(cfg.zarr_train_path))
    if not stores:
        raise FileNotFoundError(
            f"No .zarr stores found under {to_absolute_path(cfg.zarr_train_path)}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Computing statistics from {len(stores)} store(s) on {device}")

    # Multi-resolution level sizes (edge stats must match the training graph).
    # Enabled automatically when num_nodes lists more than one level.
    level_sizes = list(cfg.num_nodes) if len(cfg.num_nodes) > 1 else None

    mean, std = compute_stats(
        stores,
        node_degree=cfg.node_degree,
        num_sampled_nodes=cfg.get("num_sampled_nodes", None),
        device=device,
        level_sizes=level_sizes,
    )

    output_file = to_absolute_path(cfg.stats_file)
    save_stats_to_json(mean, std, output_file)

    print("Global Mean:", {k: np.round(v, 4).tolist() for k, v in mean.items()})
    print("Global Std:", {k: np.round(v, 4).tolist() for k, v in std.items()})
    print(f"Statistics saved to {output_file}")


if __name__ == "__main__":
    main()
