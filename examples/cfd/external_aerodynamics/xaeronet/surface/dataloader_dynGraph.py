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
Raw-Zarr datapipe for the dynamic (on-the-fly) XAeroNet-S graph pipeline.

Unlike the offline pipeline (dataloader.py), this loader does **not** read
pre-built partitioned graphs. It reads the *raw* DoMINO surface Zarr stores
produced by ``vtk_to_zarr.py`` and returns un-normalized per-point tensors. The
kNN graph, edge features, normalization and partitioning are all built later, on
the GPU, by ``DynamicGraphBuilder`` (dynamic_graph_datapipe.py).

DoMINO surface schema consumed here (per ``<sample>.zarr``):
    surface_mesh_centers  (N, 3)  float32  -> coordinates
    surface_normals       (N, 3)  float32  -> normals
    surface_areas         (N,)    float32  -> area   (reshaped to (N, 1))
    surface_fields        (N, F)  float32  -> split via ``surface_fields_meta``
                                              into pressure (1) + shear_stress (3)
Optional per-sample global (parametric) scalars, written by the converter as
``global_params_values`` (K, 1); exposed here as ``global_features`` (1, K).
Absent in the provided example, so the pipeline stays global-feature-free unless
the stores contain them.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Per-point fields returned by this loader, matching NODE_FIELDS in
# dynamic_graph_datapipe.py and the keys expected by train3d.
NODE_FIELDS = ("coordinates", "normals", "area", "pressure", "shear_stress")

# How the columns of ``surface_fields`` map onto the model targets. The
# converter records the true layout in the ``surface_fields_meta`` attribute;
# these are the fallbacks if that attribute is missing.
_DEFAULT_FIELD_SLICES = {"pressure": (0, 1), "shear_stress": (1, 4)}
# Names the converter may use for the shear-stress block in surface_fields_meta.
_SHEAR_NAMES = {"wallshearstress", "shear_stress", "wallshearstressmean"}


def find_zarr_stores(data_path: str) -> list[str]:
    """Return the sorted list of ``*.zarr`` stores directly under ``data_path``."""
    if not os.path.isdir(data_path):
        return []
    stores = [
        os.path.join(data_path, d)
        for d in os.listdir(data_path)
        if d.endswith(".zarr") and os.path.isdir(os.path.join(data_path, d))
    ]
    return sorted(stores)


def _surface_field_slices(group: zarr.Group) -> dict[str, tuple[int, int]]:
    """Resolve pressure / shear_stress column ranges from surface_fields_meta."""
    meta = group.attrs.get("surface_fields_meta", None)
    if not meta:
        return dict(_DEFAULT_FIELD_SLICES)

    slices: dict[str, tuple[int, int]] = {}
    offset = 0
    for entry in meta:
        name = str(entry.get("name", "")).lower()
        comp = int(entry.get("components", 1))
        span = (offset, offset + comp)
        if name == "pressure":
            slices["pressure"] = span
        elif name in _SHEAR_NAMES:
            slices["shear_stress"] = span
        offset += comp
    # Fall back for anything the meta did not describe.
    for k, v in _DEFAULT_FIELD_SLICES.items():
        slices.setdefault(k, v)
    return slices


def _read_global_features(group: zarr.Group) -> np.ndarray | None:
    """Return per-sample global scalars as a ``(1, K)`` array, or ``None``.

    Supports the DoMINO ``global_params_values`` (K, 1) array written by
    ``vtk_to_zarr.py``. Returns ``None`` when the store has no global params, so
    the rest of the pipeline stays global-feature-free.
    """
    if "global_params_values" in set(group.array_keys()):
        vals = np.asarray(group["global_params_values"][:], dtype=np.float32)
        return vals.reshape(1, -1)
    return None


def load_raw_surface_sample(
    store_path: str, indices: np.ndarray | None = None
) -> tuple[dict[str, np.ndarray], str]:
    """Load one raw DoMINO surface sample as numpy arrays.

    Parameters
    ----------
    store_path
        Path to a ``<sample>.zarr`` store.
    indices
        Optional 1-D int array selecting a subset of the ``N`` surface points
        (used to subsample large clouds on the CPU before transfer). ``None``
        loads every point. Global features are never subsampled.

    Returns
    -------
    (raw, run_id)
        ``raw`` maps NODE_FIELDS (+ optional ``global_features``) to numpy
        arrays; ``run_id`` is the sample name.
    """
    group = zarr.open_group(store_path, mode="r")

    sel = slice(None) if indices is None else indices
    coordinates = np.asarray(group["surface_mesh_centers"][sel], dtype=np.float32)
    normals = np.asarray(group["surface_normals"][sel], dtype=np.float32)
    area = np.asarray(group["surface_areas"][sel], dtype=np.float32).reshape(-1, 1)
    fields = np.asarray(group["surface_fields"][sel], dtype=np.float32)

    slices = _surface_field_slices(group)
    p_lo, p_hi = slices["pressure"]
    s_lo, s_hi = slices["shear_stress"]

    raw: dict[str, np.ndarray] = {
        "coordinates": coordinates,
        "normals": normals,
        "area": area,
        "pressure": fields[:, p_lo:p_hi],
        "shear_stress": fields[:, s_lo:s_hi],
    }

    global_features = _read_global_features(group)
    if global_features is not None:
        raw["global_features"] = global_features

    run_id = str(group.attrs.get("sample_name", os.path.basename(store_path)))
    run_id = run_id.split(".")[0]
    return raw, run_id


def surface_point_count(store_path: str) -> int:
    """Number of surface points in a store (without reading the arrays)."""
    group = zarr.open_group(store_path, mode="r")
    return int(group["surface_mesh_centers"].shape[0])


class RawZarrSurfaceDataset(Dataset):
    """Streams raw (un-normalized) DoMINO surface samples for on-the-fly graphs.

    Parameters
    ----------
    file_list
        Paths to ``*.zarr`` stores.
    num_sampled_nodes
        Subsample each cloud to this many points on the CPU worker (fresh subset
        per call -> per-epoch resampling dynamics), minimizing the host->device
        transfer. ``None`` keeps every point.
    deterministic
        When ``True`` the subsample is seeded per-sample so it is identical
        across epochs (use for validation to keep metrics comparable).
    """

    def __init__(
        self,
        file_list: list[str],
        num_sampled_nodes: int | None = None,
        deterministic: bool = False,
    ):
        self.file_list = file_list
        self.num_sampled_nodes = num_sampled_nodes
        self.deterministic = deterministic

    def __len__(self) -> int:
        return len(self.file_list)

    def _select_indices(self, n: int, seed: int | None) -> np.ndarray | None:
        if self.num_sampled_nodes is None or self.num_sampled_nodes >= n:
            return None
        rng = np.random.default_rng(seed if self.deterministic else None)
        sel = rng.choice(n, size=self.num_sampled_nodes, replace=False)
        sel.sort()  # sorted indices keep Zarr chunk reads coherent
        return sel

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], str]:
        store_path = self.file_list[idx]
        n = surface_point_count(store_path)
        indices = self._select_indices(n, seed=idx)
        raw_np, run_id = load_raw_surface_sample(store_path, indices)
        raw = {
            k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in raw_np.items()
        }
        return raw, run_id

    @staticmethod
    def collate_fn(
        batch: list[tuple[dict[str, torch.Tensor], str]],
    ) -> tuple[list[dict[str, torch.Tensor]], list[str]]:
        raws, run_ids = zip(*batch)
        return list(raws), list(run_ids)


def create_raw_zarr_dataloader(
    file_list: list[str],
    num_sampled_nodes: int | None = None,
    batch_size: int = 1,
    shuffle: bool = False,
    use_ddp: bool = True,
    drop_last: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int | None = 2,
    deterministic: bool = False,
) -> DataLoader:
    """DataLoader over raw DoMINO surface Zarr stores.

    Mirrors ``dataloader.create_dataloader`` (DDP sampler, batch size 1, list
    collate) but yields raw per-point tensors instead of pre-built partitions.
    """
    if batch_size != 1:
        raise ValueError(f"Batch size must be 1 for now, but got {batch_size}")

    dataset = RawZarrSurfaceDataset(
        file_list, num_sampled_nodes=num_sampled_nodes, deterministic=deterministic
    )

    if use_ddp:
        from physicsnemo.distributed import DistributedManager

        dist = DistributedManager()
        world_size = dist.world_size
        rank = dist.rank
    else:
        world_size = 1
        rank = 0

    sampler = DistributedSampler(
        dataset,
        shuffle=shuffle,
        drop_last=drop_last,
        num_replicas=world_size,
        rank=rank,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=pin_memory,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=RawZarrSurfaceDataset.collate_fn,
    )
