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
Inference for the dynamic (on-the-fly) XAeroNet-S graph pipeline.

This is the dynamic-graph counterpart of ``inference.py``: instead of reading
pre-built partitioned ``.bin`` graphs it reads the *raw* DoMINO surface Zarr
stores and rebuilds the graph on the GPU with the exact same
``DynamicGraphBuilder`` used during training. For each test store it

  1. loads the raw point cloud (optionally CPU-subsampled to ``num_sampled_nodes``,
     exactly like training),
  2. builds the graph identically to training (multi-resolution / importance
     draw + spatial partitioning, per ``conf/config3d.yaml``) using a fixed seed
     so the run is reproducible,
  3. predicts pressure + wall-shear-stress, accumulating the owned
     (``inner_node``) predictions from every partition,
  4. denormalizes with ``global_stats.json`` and evaluates the fields directly
     against the ground truth stored in the Zarr (no raw ``.vtp`` needed), and
  5. optionally writes a per-sample prediction point cloud (``.vtp``).

Single-GPU by design (clean baseline). Aerodynamic-coefficient integration and
the associated fit metrics are layered on top of this script separately.

Run::

    python inference_dynGraph.py \
        checkpoint_filename=model_checkpoint.pth \
        zarr_test_path=/workspace/drivaer_zarr/val
"""

from __future__ import annotations

import os
import sys
import json

import hydra
import numpy as np
import pyvista as pv
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tabulate import tabulate

from physicsnemo.models.meshgraphnet import MeshGraphNet

# Make the parent package importable (mirrors inference.py / train3d_dynGraph.py).
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from dataloader_dynGraph import (
    find_zarr_stores,
    load_raw_surface_sample,
    surface_point_count,
)
from dynamic_graph_datapipe import DynamicGraphBuilder
from utils import count_trainable_params


def remove_module_prefix(state_dict: dict) -> dict:
    """Strip a leading ``module.`` (DDP) from every checkpoint key."""
    return {
        (k[len("module.") :] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }


def load_model_params(model: torch.nn.Module, filename: str) -> None:
    """Load model weights from a checkpoint file (no optimizer state needed)."""
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"No checkpoint found at {filename}")
    checkpoint = torch.load(filename, map_location="cpu", weights_only=False)
    state_dict = remove_module_prefix(checkpoint["model_state_dict"])
    model.load_state_dict(state_dict)
    print(f"Checkpoint loaded: {filename}")


def node_features(part) -> torch.Tensor:
    """Assemble the MeshGraphNet node-input tensor for one partition.

    Matches the feature layout used in ``train3d_dynGraph.py``: coordinates,
    normals and three Fourier bands (sin/cos at 2/4/8 pi), plus the broadcast
    graph-level (global) features when present.
    """
    coords = part.coordinates
    feats = [
        coords,
        part.normals,
        torch.sin(2 * np.pi * coords),
        torch.cos(2 * np.pi * coords),
        torch.sin(4 * np.pi * coords),
        torch.cos(4 * np.pi * coords),
        torch.sin(8 * np.pi * coords),
        torch.cos(8 * np.pi * coords),
    ]
    if "global_features" in part:
        feats.append(part.global_features.expand(part.num_nodes, -1))
    return torch.cat(feats, dim=1)


def relative_error(pred: torch.Tensor, true: torch.Tensor, p: int = 2) -> float:
    """Relative L``p`` error ``||pred - true||_p / ||true||_p`` as a float."""
    denom = torch.norm(true, p=p)
    if denom == 0:
        return float("nan")
    return float(torch.norm(pred - true, p=p) / denom)


@torch.inference_mode()
def predict_sample(
    raw: dict[str, torch.Tensor],
    graph_builder: DynamicGraphBuilder,
    model: torch.nn.Module,
    generator: torch.Generator,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Build the graph like training and accumulate owned-node predictions.

    Returns a dict of ``(M, C)`` tensors (``M`` = number of drawn/owned nodes)
    on the CPU: ``pressure_pred``, ``shear_stress_pred``, ``pressure_true``,
    ``shear_stress_true``, ``coordinates``, ``normals``, ``area`` -- all still
    normalized (denormalization happens in the caller).
    """
    partitions = graph_builder(raw, generator=generator)

    ids_parts: list[torch.Tensor] = []
    pred_parts: list[torch.Tensor] = []
    true_parts: list[torch.Tensor] = []
    coord_parts: list[torch.Tensor] = []
    normal_parts: list[torch.Tensor] = []
    area_parts: list[torch.Tensor] = []

    for part in partitions:
        part = part.to(device)
        inner = part.inner_node
        ndata = node_features(part)
        with torch.autocast("cuda", enabled=True, dtype=amp_dtype):
            pred = model(ndata, part.edge_attr, part)[inner]
        target = torch.cat((part.pressure, part.shear_stress), dim=1)[inner]

        # ``part_node`` maps owned nodes back to the drawn-cloud index space.
        ids_parts.append(part.part_node[inner])
        pred_parts.append(pred.float())
        true_parts.append(target.float())
        coord_parts.append(part.coordinates[inner].float())
        normal_parts.append(part.normals[inner].float())
        area_parts.append(part.area[inner].float())

    ids = torch.cat(ids_parts)
    total = int(ids.max().item()) + 1

    def _scatter(parts: list[torch.Tensor], width: int) -> torch.Tensor:
        out = torch.zeros((total, width), dtype=torch.float32, device=device)
        out[ids] = torch.cat(parts)
        return out.cpu()

    pred_full = _scatter(pred_parts, 4)
    true_full = _scatter(true_parts, 4)
    return {
        "pressure_pred": pred_full[:, :1],
        "shear_stress_pred": pred_full[:, 1:],
        "pressure_true": true_full[:, :1],
        "shear_stress_true": true_full[:, 1:],
        "coordinates": _scatter(coord_parts, 3),
        "normals": _scatter(normal_parts, 3),
        "area": _scatter(area_parts, 1),
    }


def denormalize(
    fields: dict[str, torch.Tensor],
    mean: dict[str, torch.Tensor],
    std: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Undo standardization for every field (CPU tensors in, CPU tensors out)."""
    key_of = {
        "pressure_pred": "pressure",
        "shear_stress_pred": "shear_stress",
        "pressure_true": "pressure",
        "shear_stress_true": "shear_stress",
        "coordinates": "coordinates",
        "normals": "normals",
        "area": "area",
    }
    return {
        name: value * std[key_of[name]] + mean[key_of[name]]
        for name, value in fields.items()
    }


def save_point_cloud(fields: dict[str, torch.Tensor], path: str) -> None:
    """Write the denormalized predictions + ground truth to a ``.vtp``."""
    coords = fields["coordinates"].numpy()
    cloud = pv.PolyData(coords)
    cloud["coordinates"] = coords
    cloud["normals"] = fields["normals"].numpy()
    cloud["area"] = fields["area"].numpy()
    cloud["pressure_pred"] = fields["pressure_pred"].numpy()
    cloud["shear_stress_pred"] = fields["shear_stress_pred"].numpy()
    cloud["pressure_true"] = fields["pressure_true"].numpy()
    cloud["shear_stress_true"] = fields["shear_stress_true"].numpy()
    cloud.save(path)


@hydra.main(version_base="1.3", config_path="conf", config_name="config3d")
def main(cfg: DictConfig) -> None:
    torch.backends.cudnn.benchmark = cfg.enable_cudnn_benchmark
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16

    # Discover the raw DoMINO surface Zarr stores to run inference on.
    test_path = to_absolute_path(cfg.zarr_test_path)
    stores = find_zarr_stores(test_path)
    if not stores:
        raise FileNotFoundError(f"No .zarr stores found under {test_path}")
    print(f"Found {len(stores)} test store(s) under {test_path}")

    # Load the normalization statistics.
    with open(to_absolute_path(cfg.stats_file), "r") as f:
        stats = json.load(f)
    mean = {k: torch.tensor(v, dtype=torch.float32) for k, v in stats["mean"].items()}
    std = {k: torch.tensor(v, dtype=torch.float32) for k, v in stats["std_dev"].items()}
    num_global_features = len(stats["mean"].get("global_features", []))

    # Build the model exactly as in training and load the checkpoint.
    model = MeshGraphNet(
        input_dim_nodes=24 + num_global_features,
        input_dim_edges=4,
        output_dim=4,
        processor_size=cfg.num_message_passing_layers,
        aggregation="sum",
        hidden_dim_node_encoder=cfg.hidden_dim,
        hidden_dim_edge_encoder=cfg.hidden_dim,
        hidden_dim_node_decoder=cfg.hidden_dim,
        mlp_activation_fn=cfg.activation,
        do_concat_trick=cfg.use_concat_trick,
        num_processor_checkpoint_segments=cfg.checkpoint_segments,
    ).to(device)
    print(f"Number of trainable parameters: {count_trainable_params(model)}")
    load_model_params(model, to_absolute_path(cfg.checkpoint_filename))
    model.eval()

    # Graph builder configured identically to train3d_dynGraph: multi-resolution
    # is auto-enabled when num_nodes lists more than one level, and partitioning
    # is disabled when no_partition is set.
    level_sizes = list(cfg.num_nodes) if len(cfg.num_nodes) > 1 else None
    num_partitions_eff = 1 if cfg.get("no_partition", False) else cfg.num_partitions
    graph_builder = DynamicGraphBuilder(
        node_degree=cfg.node_degree,
        num_partitions=num_partitions_eff,
        halo_hops=cfg.num_message_passing_layers,
        num_target_nodes=None,
        level_sizes=level_sizes,
        mean=stats["mean"],
        std=stats["std_dev"],
    )

    # Subsample per cloud on the CPU exactly like training (None keeps all).
    num_sampled_nodes = cfg.get("num_sampled_nodes", None)
    save_clouds = cfg.get("save_inference_clouds", True)

    # Per-sample relative errors, aggregated at the end.
    metric_rows: list[dict[str, float]] = []

    for store_path in stores:
        # Seed per sample so the draw is reproducible yet varies across samples.
        seed = int(cfg.get("inference_seed", 0))
        generator = torch.Generator(device=device).manual_seed(seed)

        indices = None
        if num_sampled_nodes is not None:
            group_n = surface_point_count(store_path)
            if num_sampled_nodes < group_n:
                rng = np.random.default_rng(seed)
                indices = np.sort(
                    rng.choice(group_n, size=num_sampled_nodes, replace=False)
                )

        raw_np, run_id = load_raw_surface_sample(store_path, indices)
        raw = {
            k: torch.from_numpy(np.ascontiguousarray(v)).to(device)
            for k, v in raw_np.items()
        }

        fields = predict_sample(
            raw, graph_builder, model, generator, device, amp_dtype
        )
        fields = denormalize(fields, mean, std)

        # Field-level relative errors (denormalized), directly on the cloud.
        row = {
            "id": run_id,
            "num_nodes": int(fields["pressure_pred"].shape[0]),
            "p_l2": relative_error(fields["pressure_pred"], fields["pressure_true"], 2),
            "p_l1": relative_error(fields["pressure_pred"], fields["pressure_true"], 1),
        }
        for i, axis in enumerate("xyz"):
            row[f"wss_{axis}_l2"] = relative_error(
                fields["shear_stress_pred"][:, i], fields["shear_stress_true"][:, i], 2
            )
            row[f"wss_{axis}_l1"] = relative_error(
                fields["shear_stress_pred"][:, i], fields["shear_stress_true"][:, i], 1
            )
        metric_rows.append(row)

        table = [
            ["Pressure", f"{row['p_l2']:.4f}", f"{row['p_l1']:.4f}"],
            ["X-wall shear stress", f"{row['wss_x_l2']:.4f}", f"{row['wss_x_l1']:.4f}"],
            ["Y-wall shear stress", f"{row['wss_y_l2']:.4f}", f"{row['wss_y_l1']:.4f}"],
            ["Z-wall shear stress", f"{row['wss_z_l2']:.4f}", f"{row['wss_z_l1']:.4f}"],
        ]
        print(f"\nRelative errors for {run_id} ({row['num_nodes']} nodes):\n")
        print(tabulate(table, headers=["Quantity", "L2", "L1"], tablefmt="github"))

        if save_clouds:
            out_path = f"inference_cloud_{run_id}.vtp"
            save_point_cloud(fields, out_path)
            print(f"Saved {out_path}")

    # Aggregate mean errors across all samples.
    metric_keys = [k for k in metric_rows[0] if k not in ("id", "num_nodes")]
    aggregate = {
        k: float(np.nanmean([r[k] for r in metric_rows])) for k in metric_keys
    }
    summary = [
        ["Pressure", f"{aggregate['p_l2']:.4f}", f"{aggregate['p_l1']:.4f}"],
        ["X-wall shear stress", f"{aggregate['wss_x_l2']:.4f}", f"{aggregate['wss_x_l1']:.4f}"],
        ["Y-wall shear stress", f"{aggregate['wss_y_l2']:.4f}", f"{aggregate['wss_y_l1']:.4f}"],
        ["Z-wall shear stress", f"{aggregate['wss_z_l2']:.4f}", f"{aggregate['wss_z_l1']:.4f}"],
    ]
    print("\nMean relative errors across all samples:\n")
    print(
        tabulate(
            summary, headers=["Quantity", "Avg Rel L2", "Avg Rel L1"], tablefmt="github"
        )
    )

    out_json = "inference_dynGraph_errors.json"
    with open(out_json, "w") as f:
        json.dump({"per_sample": metric_rows, "aggregate": aggregate}, f, indent=2)
    print(f"\nWrote {out_json}")
    print("Inference complete")


if __name__ == "__main__":
    main()
