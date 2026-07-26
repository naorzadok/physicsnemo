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

"""Config-driven global-feature datapipe for GeoTransolver (Option B).

This is a *local* extension for the ``transformer_models`` example.  It does not
modify the installed ``physicsnemo`` library; instead it subclasses
:class:`~physicsnemo.datapipes.cae.transolver_datapipe.TransolverDataPipe` and
overrides only the global-feature (``fx``) assembly so the set of global
features is driven by a config list instead of the two hardcoded names
``air_density`` / ``stream_velocity``.

Usage
-----
1. Add the feature names to ``data_keys`` in your data config so the zarr
   reader loads those scalar attributes, e.g.::

       data_keys:
         - "surface_fields"
         - "surface_mesh_centers"
         - ...
         - "alpha"
         - "beta"
         - "mach"
         - "reynolds"

2. Add a ``global_feature_keys`` list to the same data config::

       global_feature_keys: [alpha, beta, mach, reynolds]

3. Set the model's ``global_dim`` to ``len(global_feature_keys)`` (here, 4).

4. In ``train.py`` import :func:`create_global_features_transolver_dataset`
   from this module in place of ``create_transolver_dataset``.

5. add:
        from global_features_datapipe import (
            create_global_features_transolver_dataset as create_transolver_dataset,
        )
           
If ``global_feature_keys`` is not set (``None``/empty), behavior falls back to
the base pipeline (``air_density`` + ``stream_velocity`` when both are present),
so existing configs keep working unchanged.

Each listed key must resolve to a per-sample scalar attribute (the common case)
or a 1-D per-sample vector; scalars/vectors are concatenated in list order to
form the ``global_dim`` channel axis, then broadcast over points exactly like
the base pipeline.
"""

from __future__ import annotations

from typing import Literal, Sequence

import torch
from omegaconf import DictConfig

from physicsnemo.datapipes.cae.cae_dataset import CAEDataset
from physicsnemo.datapipes.cae.transolver_datapipe import TransolverDataPipe
from physicsnemo.distributed import DistributedManager


class GlobalFeaturesTransolverDataPipe(TransolverDataPipe):
    """TransolverDataPipe with a configurable list of global features.

    Overrides only the ``fx`` construction: the global embedding is stacked
    from ``global_feature_keys`` (any number of scalar/vector per-sample
    attributes) instead of the hardcoded ``air_density`` / ``stream_velocity``.
    All other preprocessing is inherited unchanged.
    """

    def __init__(
        self,
        input_path,
        model_type: Literal["surface", "volume", "combined"],
        global_feature_keys: Sequence[str] | None = None,
        **data_config_overrides,
    ) -> None:
        # ``global_feature_keys`` is intentionally kept off ``self.config`` so we
        # do not need to modify the library ``TransolverDataConfig`` dataclass.
        super().__init__(input_path, model_type, **data_config_overrides)
        self.global_feature_keys = (
            list(global_feature_keys) if global_feature_keys else None
        )

    def _build_global_fx(
        self, data_dict: dict[str, torch.Tensor], n_points: int
    ) -> torch.Tensor | None:
        """Assemble the ``fx`` global embedding from the configured keys.

        Returns a tensor of shape ``(n_points, global_dim)`` when
        ``broadcast_global_features`` is True, otherwise ``(1, global_dim)``.
        Returns ``None`` when no keys are configured.
        """
        keys = self.global_feature_keys
        if not keys:
            return None

        missing = [k for k in keys if k not in data_dict]
        if missing:
            raise KeyError(
                f"global_feature_keys {missing} not found in sample. Make sure "
                f"these names are written into the zarr (as scalar attributes) "
                f"and listed in the config's data_keys. Available keys: "
                f"{sorted(data_dict.keys())}"
            )

        # Flatten each feature to 1-D and concatenate along the channel axis so
        # both scalar (0-D) and vector (1-D) attributes are supported.
        channels = []
        for k in keys:
            v = data_dict[k]
            channels.append(v.reshape(1) if v.ndim == 0 else v.reshape(-1))
        fx = torch.cat(channels, dim=0)  # (global_dim,)

        if self.config.broadcast_global_features:
            fx = fx.broadcast_to(n_points, -1)  # (N, global_dim)
        else:
            fx = fx.unsqueeze(0)  # (1, global_dim)
        return fx

    def preprocess_surface_data(
        self,
        data_dict,
        center_of_mass: torch.Tensor | None = None,
        scale_factor: torch.Tensor | None = None,
    ):
        result = super().preprocess_surface_data(
            data_dict, center_of_mass, scale_factor
        )
        if self.global_feature_keys:
            fx = self._build_global_fx(data_dict, result["embeddings"].shape[0])
            if fx is not None:
                result["fx"] = fx
        return result

    def preprocess_volume_data(
        self,
        data_dict,
        center_of_mass: torch.Tensor | None = None,
        scale_factor: torch.Tensor | None = None,
    ):
        result = super().preprocess_volume_data(
            data_dict, center_of_mass, scale_factor
        )
        if self.global_feature_keys:
            fx = self._build_global_fx(data_dict, result["embeddings"].shape[0])
            if fx is not None:
                result["fx"] = fx
        return result


def create_global_features_transolver_dataset(
    cfg: DictConfig,
    phase: Literal["train", "val", "test"],
    surface_factors: dict[str, torch.Tensor] | None = None,
    volume_factors: dict[str, torch.Tensor] | None = None,
    device_mesh: torch.distributed.DeviceMesh | None = None,
    placements: dict[str, torch.distributed.tensor.Placement] | None = None,
) -> GlobalFeaturesTransolverDataPipe:
    """Drop-in replacement for ``create_transolver_dataset``.

    Mirrors the library factory but (a) reads an optional ``global_feature_keys``
    list from the data config and (b) instantiates
    :class:`GlobalFeaturesTransolverDataPipe`.
    """
    model_type = cfg.mode
    if phase == "train":
        input_path = cfg.train.data_path
    elif phase == "val":
        input_path = cfg.val.data_path
    else:
        raise ValueError(f"Invalid phase {phase}")

    keys_to_read = cfg.data_keys

    overrides = {}

    dm = DistributedManager()

    if torch.cuda.is_available():
        device = dm.device
        consumer_stream = torch.cuda.default_stream()
    else:
        device = torch.device("cpu")
        consumer_stream = None

    preload_depth = cfg.preload_depth if cfg.get("preload_depth", None) is not None else 1
    pin_memory = cfg.pin_memory if cfg.get("pin_memory", None) is not None else False

    # Optional config keys with sensible defaults on the config dataclass.
    optional_cfg_keys = [
        "include_normals",
        "include_sdf",
        "volume_sample_from_disk",
        "broadcast_global_features",
        "include_geometry",
        "geometry_sampling",
        "translational_invariance",
        "reference_origin",
        "scale_invariance",
        "reference_scale",
        "return_mesh_features",
    ]
    for optional_key in optional_cfg_keys:
        if cfg.get(optional_key, None) is not None:
            overrides[optional_key] = cfg[optional_key]

    # The one new, config-driven knob.
    global_feature_keys = cfg.get("global_feature_keys", None)
    if global_feature_keys is not None:
        global_feature_keys = list(global_feature_keys)

    dataset = CAEDataset(
        data_dir=input_path,
        keys_to_read=keys_to_read,
        keys_to_read_if_available={},
        output_device=device,
        preload_depth=preload_depth,
        pin_memory=pin_memory,
        device_mesh=device_mesh,
        placements=placements,
        consumer_stream=consumer_stream,
    )

    datapipe = GlobalFeaturesTransolverDataPipe(
        input_path,
        resolution=cfg.resolution,
        surface_factors=surface_factors,
        volume_factors=volume_factors,
        model_type=model_type,
        scaling_type="mean_std_scaling",
        global_feature_keys=global_feature_keys,
        **overrides,
    )

    datapipe.set_dataset(dataset)

    return datapipe
