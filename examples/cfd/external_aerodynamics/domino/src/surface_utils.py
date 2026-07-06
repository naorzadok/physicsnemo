# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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
Specialized utility functions for surface-only DoMINO datasets.

This file contains modified versions of the original utility functions,
removing all references and logic related to volumetric data, and focusing
solely on surface fields, mesh data, and global parameters.
"""

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Literal, Any

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from physicsnemo.distributed import DistributedManager
from torch.distributed.tensor.placement_types import (
    Shard,
    Replicate,
)


# ==============================================================================
# Helper functions (Adapted for Surface-Only)
# ==============================================================================

def get_num_vars_surface_only(cfg: dict) -> tuple[int | None, int]:
    """Calculate the number of variables for surface and global features.

    This function analyzes the configuration to determine how many variables are needed
    for surface mesh data and global parameters.

    Args:
        cfg: Configuration object containing variable definitions for surface and
             global parameters with their types (scalar/vector).

    Returns:
        tuple: A 2-tuple containing:
            - num_surf_vars (int): Number of surface variables.
            - num_global_features (int): Number of global parameter features.
    """
    # Volume logic removed
    num_vol_vars = None

    num_surf_vars = 0
    surface_variable_names = list(cfg.variables.surface.solution.keys())
    for j in surface_variable_names:
        if cfg.variables.surface.solution[j] == "vector":
            num_surf_vars += 3
        else:
            num_surf_vars += 1

    num_global_features = 0
    global_params_names = list(cfg.variables.global_parameters.keys())
    for param in global_params_names:
        # Assuming the config structure for global parameters remains the same
        # i.e., it has a 'type' and 'reference' field.
        param_config = cfg.variables.global_parameters[param]
        if param_config.type == "vector":
            # For vector, use the length of the reference list
            if isinstance(param_config.reference, list):
                num_global_features += len(param_config.reference)
            else:
                # If it's a simple list in config, or otherwise structured
                # QUESTION: How are vector global parameters represented? If they are
                # always 3D, use 3. If they are based on 'reference', use its length.
                # Assuming simple 3-component vector for safety, but check the 'reference' field if available.
                num_global_features += 3
        elif param_config.type == "scalar":
            num_global_features += 1
        else:
            raise ValueError(f"Unknown global parameter type for {param}")

    return num_surf_vars, num_global_features


def get_keys_to_read_surface_only(
    cfg: dict,
    get_ground_truth: bool = True,
):
    """
    Configures the keys to read from the dataset for a surface-only model.

    Args:
        cfg: Configuration object.
        get_ground_truth: Boolean to include ground truth surface fields.

    Returns:
        tuple: A 2-tuple containing:
            - keys_to_read (list[str]): List of keys to be loaded from the dataset.
            - keys_to_read_if_available (dict): Dictionary of default values for
              keys that may be missing, like global parameters.
    """

    # Always read these keys:
    keys_to_read = ["stl_coordinates", "stl_centers", "stl_faces", "stl_areas"]

    # Global parameter defaults:
    cfg_params_vec = []
    for key in cfg.variables.global_parameters:
        param_config = cfg.variables.global_parameters[key]
        if param_config.type == "vector":
            # Assuming 'reference' is a list/tuple of values for vector params
            if isinstance(param_config.reference, (list, tuple)):
                cfg_params_vec.extend(param_config.reference)
            else:
                # Fallback, assuming a 3-component vector if reference is not a list
                cfg_params_vec.extend([param_config.reference] * 3)
        else:
            cfg_params_vec.append(param_config.reference)

    # NOTE: The exact key name for the values array ("global_params_values" or otherwise)
    # must be confirmed based on your dataset structure.
    keys_to_read_if_available = {
        "global_params_values": torch.tensor(cfg_params_vec).reshape(-1, 1),
        "global_params_reference": torch.tensor(cfg_params_vec).reshape(-1, 1),
    }

    # Surface keys:
    surface_keys = [
        "surface_mesh_centers",
        "surface_normals",
        "surface_areas",
    ]
    if get_ground_truth:
        surface_keys.append("surface_fields")

    keys_to_read.extend(surface_keys)

    return keys_to_read, keys_to_read_if_available


def coordinate_distributed_environment_surface_only(cfg: DictConfig):
    """
    Initialize the distributed env for DoMINO, focusing on surface data placements.

    Args:
        cfg: Configuration object containing the domain parallelism configuration.

    Returns:
        domain_mesh: torch.distributed.DeviceMesh: The domain mesh.
        data_mesh: torch.distributed.DeviceMesh: The data mesh.
        placements: dict[str, torch.distributed.tensor.Placement]: The placements.
    """

    if not DistributedManager.is_initialized():
        DistributedManager.initialize()
    dist = DistributedManager()

    domain_size = cfg.get("domain_parallelism", {}).get("domain_size", 1)

    if dist.world_size == 1:
        domain_mesh = None
        data_mesh = None
        placements = None
    else:
        # Initialize the device mesh:
        mesh = dist.initialize_mesh(
            mesh_shape=(-1, domain_size), mesh_dim_names=("ddp", "domain")
        )
        domain_mesh = mesh["domain"]
        data_mesh = mesh["ddp"]

        if domain_size > 1:
            shard_grid = cfg.get("domain_parallelism", {}).get("shard_grid", False)
            shard_points = cfg.get("domain_parallelism", {}).get("shard_points", False)

            if not shard_grid and not shard_points:
                raise ValueError(
                    "Either shard_grid or shard_points must be True if domain_size > 1"
                )

            if cfg.train.add_physics_loss:
                raise ValueError(
                    "Domain parallelism is not supported with physics loss"
                )

            if shard_points:
                point_like_placement = [
                    Shard(0),
                ]
            else:
                point_like_placement = [
                    Replicate(),
                ]

            # Define placements only for keys relevant to a surface model
            placements = {
                "stl_coordinates": point_like_placement,
                "stl_centers": point_like_placement,
                "stl_faces": point_like_placement,
                "stl_areas": point_like_placement,
                "surface_fields": point_like_placement,
                "surface_mesh_centers": point_like_placement,
                "surface_normals": point_like_placement,
                "surface_areas": point_like_placement,
            }
        else:
            domain_mesh = None
            placements = None

    return domain_mesh, data_mesh, placements


@dataclass
class ScalingFactors:
    """
    Data structure for storing scaling factors computed for DoMINO datasets.
    (No changes needed here as it is data agnostic)
    """

    mean: Dict[str, np.ndarray]
    std: Dict[str, np.ndarray]
    min_val: Dict[str, np.ndarray]
    max_val: Dict[str, np.ndarray]
    field_keys: list[str]

    def to_torch(
        self, device: Optional[torch.device] = None
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Convert numpy arrays to torch tensors for use in training/inference."""
        device = device or torch.device("cpu")

        return {
            "mean": {k: torch.from_numpy(v).to(device) for k, v in self.mean.items()},
            "std": {k: torch.from_numpy(v).to(device) for k, v in self.std.items()},
            "min_val": {
                k: torch.from_numpy(v).to(device) for k, v in self.min_val.items()
            },
            "max_val": {
                k: torch.from_numpy(v).to(device) for k, v in self.max_val.items()
            },
        }

    def save(self, filepath: str | Path) -> None:
        """Save scaling factors to pickle file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str | Path) -> "ScalingFactors":
        """Load scaling factors from pickle file."""
        with open(filepath, "rb") as f:
            factors = pickle.load(f)
        return factors

    def get_field_shapes(self) -> Dict[str, tuple]:
        """Get the shape of each field's statistics."""
        return {key: self.mean[key].shape for key in self.field_keys}

    def summary(self) -> str:
        """Generate a human-readable summary of the scaling factors."""
        summary = ["Scaling Factors Summary:"]
        summary.append(f"Field Keys: {self.field_keys}")

        for key in self.field_keys:
            mean_val = self.mean[key]
            std_val = self.std[key]
            min_val = self.min_val[key]
            max_val = self.max_val[key]

            summary.append(f"\n{key}:")
            summary.append(f"   Shape: {mean_val.shape}")
            summary.append(f"   Mean: {mean_val}")
            summary.append(f"   Std: {std_val}")
            summary.append(f"   Min: {min_val}")
            summary.append(f"   Max: {max_val}")

        return "\n".join(summary)


def load_scaling_factors_surface_only(
    cfg: DictConfig, logger=None
) -> torch.Tensor:
    """Load scaling factors from the configuration for surface fields only.

    NOTE: The return type is changed from a tuple to a single torch.Tensor
    containing the surface scaling factors.

    Args:
        cfg: Hydra configuration object.
        logger: Optional logger instance.

    Returns:
        torch.Tensor: Tensor containing surface field scaling factors
                      ([max/mean, min/std] x num_surface_vars).
    """
    pickle_path = os.path.join(cfg.data.scaling_factors)

    try:
        scaling_factors = ScalingFactors.load(pickle_path)
        if logger is not None:
            logger.info(f"Scaling factors loaded from: {pickle_path}")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Scaling factors not found at: {pickle_path}; please run compute_statistics.py to compute them."
        )
    
    # Check if 'surface_fields' key exists, which is mandatory for a surface model
    if 'surface_fields' not in scaling_factors.field_keys:
        raise ValueError(
            "Scaling factors file is missing 'surface_fields' key. Was it computed correctly?"
        )

    if cfg.model.normalization == "min_max_scaling":
        surf_factors = np.asarray(
            [
                scaling_factors.max_val["surface_fields"],
                scaling_factors.min_val["surface_fields"],
            ]
        )
    elif cfg.model.normalization == "mean_std_scaling":
        surf_factors = np.asarray(
            [
                scaling_factors.mean["surface_fields"],
                scaling_factors.std["surface_fields"],
            ]
        )
    else:
        raise ValueError(f"Invalid normalization mode: {cfg.model.normalization}")

    surf_factors_tensor = torch.from_numpy(surf_factors)

    dm = DistributedManager()
    surf_factors_tensor = surf_factors_tensor.to(dm.device, dtype=torch.float32)

    return surf_factors_tensor


def compute_l2_surface_only(
    pred_surface: torch.Tensor,
    batch: Dict[str, Any],
    dataloader,
) -> dict[str, torch.Tensor]:
    """
    Compute the L2 norm between surface prediction and target.

    Requires the dataloader to unscale back to original values.

    Args:
        pred_surface: The predicted surface fields (normalized).
        batch: The batch dictionary containing the target fields.
        dataloader: The dataloader instance with unscaling method.

    Returns:
        dict[str, torch.Tensor]: Dictionary of L2 metrics for surface components.
    """

    l2_dict = {}

    _, target_surface = dataloader.unscale_model_outputs(
        # Pass None for volume fields as they are not needed
        volume_fields=None, 
        surface_fields=batch["surface_fields"]
    )
    _, pred_surface = dataloader.unscale_model_outputs(
        volume_fields=None, 
        surface_fields=pred_surface
    )
    l2_surface = metrics_fn_surface(pred_surface, target_surface)
    l2_dict.update(l2_surface)

    return l2_dict

# NOTE: metrics_fn_surface remains unchanged as it is already surface-specific.
def metrics_fn_surface(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Computes L2 surface metrics between prediction and target.

    Args:
        pred: Predicted values (unscaled).
        target: Target values (unscaled).

    Returns:
        Dictionary of L2 surface metrics for pressure and shear components.
    """

    l2_num = (pred - target) ** 2
    l2_num = torch.sum(l2_num, dim=1)
    l2_num = torch.sqrt(l2_num)

    l2_denom = target**2
    l2_denom = torch.sum(l2_denom, dim=1)
    l2_denom = torch.sqrt(l2_denom)

    l2 = l2_num / l2_denom

    # NOTE: This assumes the order of surface variables in the tensor is:
    # [pMeanTrim (scalar), wallShearStressMeanTrim (vector x, y, z)]
    metrics = {
        "l2_surf_pressure": torch.mean(l2[:, 0]),
        "l2_shear_x": torch.mean(l2[:, 1]),
        "l2_shear_y": torch.mean(l2[:, 2]),
        "l2_shear_z": torch.mean(l2[:, 3]),
    }

    return metrics

# metrics_fn_volume is removed as it's not applicable.

def all_reduce_dict(
    metrics: dict[str, torch.Tensor], dm: DistributedManager
) -> dict[str, torch.Tensor]:
    """
    Reduces a dictionary of metrics across all distributed processes.
    (No changes needed here as it is general distributed utility)

    Args:
        metrics: Dictionary of metric names to torch.Tensor values.
        dm: DistributedManager instance for distributed context.

    Returns:
        Dictionary of reduced metrics.
    """
    # TODO - update this to use domains and not the full world

    if dm.world_size == 1:
        return metrics

    for key, value in metrics.items():
        dist.all_reduce(value)
        value = value / dm.world_size
        metrics[key] = value

    return metrics