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
REFERENCE SKELETON (not yet wired into training) for on-the-fly, GPU-side graph
construction for XAeroNet-S. See DYNAMIC_GRAPH_DESIGN.md for the full rationale,
performance analysis and trade-offs.

Pipeline overview
-----------------
Disk (Zarr, raw only)                CPU worker              GPU (per step)
---------------------                ----------              --------------
x/y/z + p + tau + normals + area  →  read chunk, pin      →  1. subsample nodes
(one dense point cloud per run)      memory, H2D async       2. build kNN/radius graph
                                                             3. compute edge features
                                                             4. normalize (node + edge)
                                                             5. spatial partition + halo
                                                             6. forward / backward

Only *raw, un-normalized per-point tensors* are stored on disk. The (large) edge
lists and the METIS partitioning are recomputed every step directly on the GPU,
which both removes the multi-GB `.bin` artifacts and gives a fresh graph topology
each epoch (node resampling) for free. Normalization is applied on the GPU from
the raw values (never baked into disk) because kNN connectivity and edge
displacements must be computed from raw, un-normalized coordinates.

This module deliberately depends only on primitives that already ship with
PhysicsNeMo:

    from physicsnemo.nn.functional import knn, radius_search

`radius_search` is backed by a Warp `wp.HashGrid`; `knn` dispatches to cuML/Warp
on CUDA tensors and falls back to torch elsewhere. No `torch_cluster` /
`pyg-lib` compiled wheels are required.
"""

from __future__ import annotations

import torch
import torch_geometric as pyg

from physicsnemo.nn.functional import knn

# Per-point node fields carried through the pipeline, in the order train3d.py
# expects them. Statistics in global_stats.json are keyed by these same names
# (edge features use the key "x").
NODE_FIELDS = ("coordinates", "normals", "area", "pressure", "shear_stress")


# --------------------------------------------------------------------------- #
# Partitions are plain torch_geometric.data.Data objects, exactly like the
# offline PartitionedGraph in dataloader.py. This matters: MeshGraphNet's
# processor requires a PyG Data instance (concat_efeat does
# `isinstance(graph, PyGData)` and reads `graph.edge_index`), so a custom
# container would raise "Unsupported graph type". Each partition carries:
#   coordinates, normals, area, pressure, shear_stress  (node attrs, normalized)
#   edge_index, edge_attr                                (edge attrs, normalized)
#   inner_node   -> positions of owned nodes within the partition
#   part_node    -> global node ids (used to scatter predictions back at val)
#   num_nodes
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# 1. Node resampling (dynamics come from here)
# --------------------------------------------------------------------------- #
def subsample_nodes(
    num_source: int,
    num_target: int,
    generator: torch.Generator | None = None,
    device: torch.device = "cuda",
) -> torch.Tensor:
    """Draw `num_target` node ids without replacement from a dense source cloud.

    Resampling a different subset every epoch is what replaces the fixed offline
    multi-resolution point clouds and provides free data augmentation.
    """
    perm = torch.randperm(num_source, generator=generator, device=device)
    return perm[:num_target]


# --------------------------------------------------------------------------- #
# 2. GPU graph construction (kNN) + 3. edge features
# --------------------------------------------------------------------------- #
def build_knn_graph(
    coords: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a symmetric kNN graph on the GPU and return (edge_index, edge_attr).

    Uses `physicsnemo.nn.functional.knn`, which runs on-device (cuML/Warp) and
    needs no compiled PyG extensions. Edge features match the offline pipeline:
    relative displacement (3) + displacement norm (1).
    """
    # indices: [N, k+1] including self at column 0.
    indices, _ = knn(coords, coords, k + 1)
    n = coords.shape[0]
    dst = indices[:, 1:].reshape(-1)  # drop self-loop column
    src = torch.arange(n, device=coords.device).repeat_interleave(k)

    # Make undirected + coalesce (dedupe) with pure-torch ops (no pyg required).
    edge_index = torch.stack([src, dst], dim=0)
    edge_index = _to_undirected_coalesce(edge_index, n)

    row, col = edge_index
    disp = coords[row] - coords[col]
    disp_norm = torch.linalg.norm(disp, dim=-1, keepdim=True)
    edge_attr = torch.cat((disp, disp_norm), dim=-1)
    return edge_index, edge_attr


def _to_undirected_coalesce(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Symmetrize and de-duplicate edges using a linear-index sort (GPU friendly)."""
    src, dst = edge_index
    src2 = torch.cat([src, dst])
    dst2 = torch.cat([dst, src])
    keys = src2.to(torch.int64) * num_nodes + dst2.to(torch.int64)
    keys = torch.unique(keys)
    src_u = keys // num_nodes
    dst_u = keys % num_nodes
    return torch.stack([src_u, dst_u], dim=0)


def _edge_attr_from_index(
    coords: torch.Tensor, edge_index: torch.Tensor
) -> torch.Tensor:
    """Relative displacement (3) + displacement norm (1) for each edge."""
    row, col = edge_index
    disp = coords[row] - coords[col]
    disp_norm = torch.linalg.norm(disp, dim=-1, keepdim=True)
    return torch.cat((disp, disp_norm), dim=-1)


def build_multilevel_knn_graph(
    coords: torch.Tensor, level_sizes: list[int], k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Multi-resolution kNN graph reproducing the offline preprocessor's scheme.

    ``coords`` must be ordered so that its nested prefixes are the resolution
    levels: level 1 is ``coords[:level_sizes[0]]``, level 2 adds the next
    ``level_sizes[1]`` points, and so on. A kNN graph is built independently on
    each *cumulative* prefix and the edge sets are unioned. Because the coarse
    prefixes are sparse over the whole geometry, their kNN edges span long
    distances, while finer prefixes add local detail — giving the multi-scale
    connectivity (long-range coupling) that lets the GNN model global physics.

    Everything runs on-device (``physicsnemo.nn.functional.knn`` + a linear-index
    union), so it preserves GPU acceleration. Equivalent to ``build_knn_graph``
    when ``level_sizes`` has a single entry covering the whole cloud.
    """
    n = coords.shape[0]
    srcs: list[torch.Tensor] = []
    dsts: list[torch.Tensor] = []

    cum = 0
    for size in level_sizes:
        cum = min(cum + int(size), n)
        if cum <= 1:
            continue
        sub = coords[:cum]
        kk = min(k + 1, cum)  # cannot ask for more neighbors than points
        indices, _ = knn(sub, sub, kk)
        deg = indices.shape[1] - 1  # neighbors excluding the self column
        dst = indices[:, 1:].reshape(-1)
        src = torch.arange(cum, device=coords.device).repeat_interleave(deg)
        srcs.append(src)
        dsts.append(dst)
        if cum >= n:
            break

    edge_index = torch.stack([torch.cat(srcs), torch.cat(dsts)], dim=0)
    edge_index = _to_undirected_coalesce(edge_index, n)
    edge_attr = _edge_attr_from_index(coords, edge_index)
    return edge_index, edge_attr


def single_graph(
    coords: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    node_fields: dict[str, torch.Tensor],
    global_features: torch.Tensor | None = None,
) -> list[pyg.data.Data]:
    """Wrap the whole geometry as one un-partitioned ``pyg.data.Data``.

    Returned as a single-element list so it is drop-in compatible with the
    partitioned path in ``train3d``: ``inner_node`` and ``part_node`` are the
    full identity range, so every node is "owned" and predictions scatter back
    directly. Use when ``num_partitions <= 1`` (whole geometry = one graph).
    """
    n = coords.shape[0]
    ids = torch.arange(n, device=coords.device)
    data = pyg.data.Data(
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=n,
        part_node=ids,
        inner_node=ids,
    )
    for name in NODE_FIELDS:
        setattr(data, name, node_fields[name])
    if global_features is not None:
        data.global_features = global_features
    return [data]


# --------------------------------------------------------------------------- #
# 4. Spatial partitioning with halo (replaces offline METIS)
# --------------------------------------------------------------------------- #
def _k_hop_subgraph(
    seed_nodes: torch.Tensor,
    num_hops: int,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """On-device equivalent of ``pyg.utils.k_hop_subgraph`` (relabel_nodes=True).

    Grows an ``num_hops``-hop neighborhood around ``seed_nodes`` over the
    (undirected) ``edge_index`` and returns the relabeled subgraph. Everything
    stays on the GPU; no PyG dependency.

    Returns
    -------
    subset : LongTensor [n_sub]
        Global ids of every node in the subgraph (inner seed + halo).
    sub_edge_index : LongTensor [2, e_sub]
        Subgraph edges, relabeled into the contiguous range ``[0, n_sub)``.
    inner_mapping : LongTensor [n_seed]
        Positions of ``seed_nodes`` within ``subset`` (the offline
        ``inner_node`` tensor consumed by train3d.py).
    edge_mask : BoolTensor [e]
        Mask selecting the retained edges from the input ``edge_index`` (used to
        slice ``edge_attr``).
    """
    row, col = edge_index
    node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index.device)

    # Breadth-first halo growth: each hop adds the 1-ring of the current front.
    subsets = [seed_nodes]
    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        hop_edge_mask = node_mask[row]
        subsets.append(col[hop_edge_mask])

    # Deduplicate to the final node set; `inv` maps the concatenated seeds back
    # into `subset`, so its first `n_seed` entries are the inner-node positions.
    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inner_mapping = inv[: seed_nodes.numel()]

    # Retain only edges whose both endpoints live in the subgraph.
    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]
    sub_edge_index = edge_index[:, edge_mask]

    # Relabel global ids -> contiguous local ids in [0, n_sub).
    relabel = torch.full(
        (num_nodes,), -1, dtype=torch.long, device=edge_index.device
    )
    relabel[subset] = torch.arange(subset.numel(), device=edge_index.device)
    sub_edge_index = relabel[sub_edge_index]

    return subset, sub_edge_index, inner_mapping, edge_mask


def spatial_partition(
    coords: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    node_fields: dict[str, torch.Tensor],
    num_parts: int,
    halo_hops: int,
    global_features: torch.Tensor | None = None,
) -> list[pyg.data.Data]:
    """Partition by a cheap spatial split, then grow `halo_hops`-hop halos.

    METIS is CPU-bound and cannot run every step. Because XAeroNet-S operates on
    a spatially-coherent surface point cloud, a recursive-coordinate-bisection /
    grid split along the longest axis gives balanced, locality-preserving parts
    at negligible GPU cost. Halo growth reuses the same edge_index via repeated
    neighborhood expansion, identical in spirit to `pyg.utils.k_hop_subgraph`
    but done on-device (see `_k_hop_subgraph`).

    ``global_features`` (shape ``[1, K]``), when provided, is a graph-level
    attribute identical across partitions and is attached unchanged to each
    partition (not indexed by node), matching the offline PartitionedGraph.

    Returns a list of `torch_geometric.data.Data` partitions, drop-in compatible
    with the offline PartitionedGraph consumed by train3d.py.
    """
    # Assign each node to a partition by bisecting the longest bounding-box axis.
    extent = coords.max(0).values - coords.min(0).values
    axis = int(torch.argmax(extent))
    order = torch.argsort(coords[:, axis])
    part_of_node = torch.empty(coords.shape[0], dtype=torch.long, device=coords.device)
    chunks = torch.chunk(order, num_parts)
    for pid, ch in enumerate(chunks):
        part_of_node[ch] = pid

    partitions: list[pyg.data.Data] = []
    for pid in range(num_parts):
        # Inner (owned) nodes of this partition, before halo growth.
        seed_nodes = (part_of_node == pid).nonzero(as_tuple=False).squeeze(-1)

        subset, sub_edge_index, inner_mapping, edge_mask = _k_hop_subgraph(
            seed_nodes, halo_hops, edge_index, coords.shape[0]
        )

        partition = pyg.data.Data(
            edge_index=sub_edge_index,
            edge_attr=edge_attr[edge_mask],
            num_nodes=int(subset.numel()),
            part_node=subset,
            inner_node=inner_mapping,
        )
        for name in NODE_FIELDS:
            setattr(partition, name, node_fields[name][subset])
        if global_features is not None:
            partition.global_features = global_features
        partitions.append(partition)

    return partitions


# --------------------------------------------------------------------------- #
# Top-level per-sample transform executed on the GPU each step.
# --------------------------------------------------------------------------- #
class DynamicGraphBuilder:
    """Turns a raw per-point sample (already on GPU) into partitioned graphs.

    Intended to be called inside the training step, after an async H2D copy of
    the raw tensors, so that graph construction overlaps the previous step's
    backward pass on a side CUDA stream.

    Normalization is applied here, on the GPU, from the raw (un-normalized)
    inputs. This is deliberate: the kNN connectivity and the edge displacement
    features must be computed from **raw** coordinates. Per-axis (anisotropic)
    coordinate normalization would distort distances and change the neighbor
    set, so the raw cloud is stored on disk and standardized on the fly. See
    DYNAMIC_GRAPH_DESIGN.md for the rationale.

    Parameters
    ----------
    node_degree, num_partitions, halo_hops
        Graph-construction knobs mirroring the offline preprocessor / config.
        ``num_partitions <= 1`` disables partitioning and emits the whole
        geometry as a single graph (halo growth is then skipped).
    num_target_nodes
        Number of nodes to subsample from the raw cloud before building the
        graph (single-resolution path). Set to ``None`` (or a value ``>=`` the
        cloud size) to use every node as-is — e.g. when the DataLoader already
        subsampled on the CPU to minimize the host->device transfer. Ignored
        when ``level_sizes`` is given.
    level_sizes
        Optional list of per-level point counts (e.g. ``[10000, 20000, 40000]``)
        enabling multi-resolution sampling: ``sum(level_sizes)`` nodes are drawn
        and connected with cumulative kNN (see ``build_multilevel_knn_graph``),
        adding long-range edges across resolutions. ``None`` keeps the
        single-resolution kNN graph.
    mean, std
        Optional statistics dicts (e.g. loaded from ``global_stats.json``),
        keyed by the entries of ``NODE_FIELDS`` plus ``"x"`` for edge features
        and, optionally, ``"global_features"``. Values may be
        lists/arrays/tensors; they are converted and cached to the input device
        on first use. When ``None``, no normalization is applied.

    Notes
    -----
    Importance sampling: if the raw sample carries a per-point ``sampling_weight``
    vector, node selection uses a weighted draw (``torch.multinomial``) instead
    of a uniform permutation, so high-importance points (e.g. high-curvature /
    feature-edge regions) are sampled more densely. The weights themselves are
    expected to be *precomputed* (stored alongside the raw cloud); computing
    mesh curvature on the fly would be CPU-bound and defeat the GPU pipeline.
    """

    def __init__(
        self,
        node_degree: int,
        num_partitions: int,
        halo_hops: int,
        num_target_nodes: int | None = None,
        level_sizes: list[int] | None = None,
        mean: dict | None = None,
        std: dict | None = None,
    ):
        self.node_degree = node_degree
        self.num_target_nodes = num_target_nodes
        self.level_sizes = list(level_sizes) if level_sizes else None
        self.num_partitions = num_partitions
        self.halo_hops = halo_hops
        self._mean_src = mean
        self._std_src = std
        # Device-resident stat tensors, cached on first call.
        self._mean: dict[str, torch.Tensor] | None = None
        self._std: dict[str, torch.Tensor] | None = None

    def _stats_on(self, device: torch.device) -> None:
        """Materialize (mean, std) as tensors on `device`, once."""
        if self._mean_src is None or self._std_src is None or self._mean is not None:
            return
        keys = list(NODE_FIELDS) + ["x"]
        if "global_features" in self._mean_src:
            keys.append("global_features")
        self._mean = {
            k: torch.as_tensor(self._mean_src[k], dtype=torch.float32, device=device)
            for k in keys
        }
        self._std = {
            k: torch.as_tensor(self._std_src[k], dtype=torch.float32, device=device)
            for k in keys
        }

    def _select_indices(
        self,
        num_source: int,
        total: int,
        weights: torch.Tensor | None,
        generator: torch.Generator | None,
        device: torch.device,
    ) -> torch.Tensor:
        """Pick ``total`` node ids, importance-weighted when weights are given.

        With ``weights`` present, a ``multinomial`` draw (without replacement)
        biases selection toward high-weight points. Otherwise falls back to a
        uniform permutation. When ``total >= num_source`` and no weights are
        given, every node is kept (identity order).
        """
        if weights is not None:
            w = torch.as_tensor(weights, device=device, dtype=torch.float32).reshape(-1)
            w = torch.clamp(w, min=0.0)
            total = min(total, int((w > 0).sum().item()))
            return torch.multinomial(
                w, total, replacement=False, generator=generator
            )
        if total >= num_source:
            return torch.arange(num_source, device=device)
        return subsample_nodes(num_source, total, generator=generator, device=device)

    def __call__(
        self,
        raw: dict[str, torch.Tensor],
        generator: torch.Generator | None = None,
        return_indices: bool = False,
    ) -> list[pyg.data.Data] | tuple[list[pyg.data.Data], torch.Tensor]:
        device = raw["coordinates"].device
        self._stats_on(device)

        num_source = raw["coordinates"].shape[0]
        weights = raw.get("sampling_weight")

        # How many nodes to draw: multi-resolution sums the level sizes; the
        # single-resolution path uses num_target_nodes (or the whole cloud).
        if self.level_sizes is not None:
            total = min(int(sum(self.level_sizes)), num_source)
        elif self.num_target_nodes is not None and self.num_target_nodes < num_source:
            total = self.num_target_nodes
        else:
            total = num_source

        # Subsample nodes (per-step dynamics), importance-weighted if available.
        idx = self._select_indices(num_source, total, weights, generator, device)

        # Build connectivity + edge features from RAW coordinates (normalization
        # is anisotropic and would change the neighbor set). Multi-resolution
        # relies on the draw order: the nested prefixes of idx are the levels.
        coords_raw = raw["coordinates"][idx]
        if self.level_sizes is not None:
            edge_index, edge_attr = build_multilevel_knn_graph(
                coords_raw, self.level_sizes, self.node_degree
            )
        else:
            edge_index, edge_attr = build_knn_graph(coords_raw, self.node_degree)

        # Normalize node fields and edge features on-device.
        node_fields: dict[str, torch.Tensor] = {}
        for name in NODE_FIELDS:
            v = raw[name][idx]
            if self._mean is not None:
                v = (v - self._mean[name]) / self._std[name]
            node_fields[name] = v
        if self._mean is not None:
            edge_attr = (edge_attr - self._mean["x"]) / self._std["x"]

        # Graph-level (global) features: identical for every node, so kept as a
        # single [1, K] row and attached unchanged to every partition.
        global_features = raw.get("global_features")
        if global_features is not None:
            global_features = global_features.to(device)
            if self._mean is not None and "global_features" in self._mean:
                global_features = (
                    global_features - self._mean["global_features"]
                ) / self._std["global_features"]

        # No partitioning requested: emit the whole geometry as one graph.
        if self.num_partitions is None or self.num_partitions <= 1:
            partitions = single_graph(
                node_fields["coordinates"],
                edge_index,
                edge_attr,
                node_fields,
                global_features=global_features,
            )
        else:
            # Partition using the (normalized) coordinates; the axis-sort split is
            # invariant to positive per-axis scaling, so this is equivalent to
            # splitting on raw coords.
            partitions = spatial_partition(
                node_fields["coordinates"],
                edge_index,
                edge_attr,
                node_fields,
                self.num_partitions,
                self.halo_hops,
                global_features=global_features,
            )

        # ``idx`` maps every drawn node (in draw / part_node order) back to its
        # row in the raw cloud, letting callers recover per-point quantities
        # (e.g. sampling_weight) that are not carried on the graph. With
        # multi-resolution the finest level is the whole drawn set, so ``idx``
        # also identifies the finest-level points used for coefficient integration.
        if return_indices:
            return partitions, idx
        return partitions

