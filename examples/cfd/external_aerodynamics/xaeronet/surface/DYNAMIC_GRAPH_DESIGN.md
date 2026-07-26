# Dynamic (On-the-Fly) Graph Construction for XAeroNet-S — Feasibility & Design

Investigation of replacing the offline `.bin` graph pipeline with raw Zarr
storage + GPU-side graph construction, for the `xaeronet/surface` workflow.

Companion reference skeleton: `dynamic_graph_datapipe.py`.

---

## 1. What the current pipeline actually does

| Stage | File | Cost |
|-------|------|------|
| Sample multi-res point clouds (100k+200k+400k pts) from STL | [preprocessor.py](preprocessor.py) `sample_boundary_from_mesh` | offline, per sample |
| Build kNN graph (`sklearn` `NearestNeighbors`, CPU, ball-tree) per level, superimpose | [preprocessor.py](preprocessor.py) `process_run` | offline, **CPU-bound** |
| Interpolate `pMeanTrim` / `wallShearStressMeanTrim` from `.vtp` via 1-NN | [preprocessor.py](preprocessor.py) | offline |
| Edge features = relative displacement + norm | [preprocessor.py](preprocessor.py) `add_edge_features` | offline |
| **METIS partition + `k_hop_subgraph` halos** | [dataloader.py](dataloader.py) `PartitionedGraph` | offline, **CPU-bound** |
| Save `torch.save(...)` → `graph_partitions_<id>.bin` | [preprocessor.py](preprocessor.py) | disk |
| Load `.bin`, normalize, hand partitions to trainer | [dataloader.py](dataloader.py) `GraphDataset` | per epoch, I/O |
| Train over partitions | [train3d.py](train3d.py) | per step |

Two facts drive the whole analysis:

1. **The graph is fixed after preprocessing.** `num_nodes = [100k, 200k, 400k]`
   (700k nodes) × `node_degree = 6`, made undirected + self-loops ⇒ order
   ~8–9M edges. Stored are `edge_index` (int64, 2×E), `edge_attr` (float32,
   E×4), plus 5 node fields, **replicated across `num_partitions = 3` halos**.
   That is the ~hundreds-of-MB-per-sample → **100s of GB–TB** footprint you are
   seeing, and the per-epoch read cost.

2. **The expensive-to-recompute parts are the kNN build and the METIS
   partition** — both currently CPU-bound (`sklearn`, PyG METIS).

So there are really *two* problems to solve to go dynamic: (a) graph
*construction*, and (b) graph *partitioning*. Most write-ups only address (a).

---

## 2. Feasibility: can we build the graph on the GPU every step?

**Yes, and most of it already exists in PhysicsNeMo — no `torch_cluster` needed.**

The repo already ships on-device neighbor primitives:

- `physicsnemo.nn.functional.knn` — dispatches to **cuML / Warp** on CUDA
  tensors, torch fallback otherwise. Returns `(indices[M,k], distances[M,k])`.
  (`physicsnemo/nn/functional/neighbors/knn/`)
- `physicsnemo.nn.functional.radius_search` — backed by a **Warp `wp.HashGrid`**
  (`physicsnemo/nn/functional/neighbors/radius_search/_warp_impl.py`), exactly
  the `wp.HashGrid` approach in your brainstorm, already wrapped as a torch
  custom op and `torch.compile`-compatible when `max_points` is set.

This means the "accelerated kernels" idea is a *reuse*, not a new dependency —
which is the strongly preferred path (avoids the fragile `pyg-lib` /
`torch_scatter` / `torch_cluster` wheel-matching documented in the README).

What has to be reimplemented on-device (all cheap, pure-torch, in the skeleton):

- `to_undirected` + `coalesce` → linear-index sort/unique (no PyG).
- edge features → gather + norm.
- **partitioning** → see §4, the one genuine design decision.

## 3. Performance trade-offs (GPU compute vs. disk I/O)

**Storage / I/O — decisively in favor of dynamic.**

| | Offline `.bin` | Raw Zarr |
|---|---|---|
| Per sample | node fields + `edge_index` + `edge_attr` × halos | node fields only |
| Rough size | ~10× raw (edges dominate; ~8–9M × (16+16) B) | `x,y,z,p,τ(3),n(3),area` ≈ 700k × 14 × 4 B ≈ **~40 MB** |
| 500 runs | 100s GB – TB | **~20 GB** |
| Per-epoch read | full edge lists | coords + fields only |

Edges are the bulk of the bytes *and* are pure derived data — regenerating them
on-device trades DRAM/PCIe traffic for GPU FLOPs, which is the right trade on
modern GPUs.

**Compute — the real question is whether construction hides behind the step.**

- kNN over 700k pts with a hash grid is milliseconds-to-low-tens-of-ms on a
  datacenter GPU; the model is 15 message-passing layers at `hidden_dim = 512`
  with activation checkpointing, so the **forward/backward dwarfs construction**.
- Correct engineering makes it *free*: build graph for step *t+1* on a **side
  CUDA stream** while step *t* backward runs (see §5). With Zarr chunk reads +
  pinned-memory async H2D, the DataLoader worker only moves ~40 MB/sample.

**Bottlenecks to watch (call these out honestly):**

1. **Partitioning, not construction.** METIS is CPU-bound and *cannot* run every
   step without erasing the I/O win. Must be replaced by a GPU-cheap spatial
   split (§4). This is the make-or-break item.
2. **Dynamic shapes.** Edge count varies per resample ⇒ `radius_search` with
   `max_points=None` breaks `torch.compile` and triggers CUDA re-allocations /
   cuDNN re-autotune (`enable_cudnn_benchmark: true`). Prefer **fixed-degree
   kNN** (static `k`) or `radius_search(max_points=...)` to keep shapes stable.
3. **Determinism / validation.** Offline graphs are reproducible; resampling is
   not. Validation must use a **fixed seed** (or a cached eval graph) so metrics
   are comparable across epochs.
4. **Field interpolation.** 1-NN interpolation from the `.vtp` currently happens
   offline. Keep it offline: pre-bake fields onto a **dense** source cloud once,
   store in Zarr; per-epoch subsampling just gathers, never re-interpolates.
5. **DDP + variable partitions.** `train3d.py` uses `static_graph=True` and
   `find_unused_parameters`. Changing node/edge counts per step is fine for the
   model (message passing is size-agnostic) but keep partition *count* fixed.

## 4. The one hard part: partitioning every step

METIS gives balanced, edge-cut-minimizing parts but is CPU-only and slow. For
XAeroNet-S it is overkill: the data is a **spatially coherent surface point
cloud**, so a **Recursive Coordinate Bisection** (sort along the longest
bounding-box axis, split into `num_partitions` contiguous chunks) yields
balanced, locality-preserving partitions at negligible GPU cost. Halos are then
grown by `halo_hops`-hop neighborhood expansion over `edge_index` (repeated
scatter/`index` over neighbors — the on-device analogue of
`pyg.utils.k_hop_subgraph`). This preserves the exact training semantics
(halo = no truncated message passing at borders) while dropping the CPU
dependency. Skeleton stubs this in `spatial_partition` with the halo body as a
documented TODO.

## 5. Proposed architecture

```
                        ┌──────────────────────── one-time ────────────────────────┐
STL + VTP  ──preprocess──▶  dense source cloud (x,y,z, p, τ, n, area)  ──▶  run_<id>.zarr
                          (bake fields via existing 1-NN interpolation, NO edges)
                        └───────────────────────────────────────────────────────────┘

per training step (per rank):
  DataLoader worker:  zarr chunk read ─▶ TensorDict ─▶ pin_memory
        │ async H2D (non_blocking) on a copy stream
        ▼
  GPU (side stream, overlaps prev backward):     DynamicGraphBuilder.__call__
        1. subsample N of M nodes  (+ optional coord jitter)      ← dynamics
        2. knn(coords, coords, k+1)  → symmetric coalesced edges  ← physicsnemo.nn.functional.knn
        3. edge_attr = [disp, |disp|]
        4. spatial_partition(..., halo_hops)                      ← replaces METIS
        ▼
  main stream:  MeshGraphNet forward/backward  (unchanged train3d.py loop)
```

Component responsibilities:

- **Offline (rewritten `preprocessor.py`)** — sample a single *dense* cloud,
  interpolate fields, write Zarr. Delete the kNN/METIS/`torch.save` path.
- **`RawZarrDataset` (rewritten `dataloader.py`)** — return raw per-point
  tensors as a `TensorDict`, `pin_memory=True`. Normalization can stay here
  (cheap) or move to GPU.
- **`DynamicGraphBuilder` (new, `dynamic_graph_datapipe.py`)** — the GPU
  transform above. Called from the training step, **not** the worker.
- **`train3d.py`** — insert one call to the builder per step; the existing
  `for part in graph_partitions:` loop is otherwise unchanged.

## 6. Recommendation

Pursue the dynamic pipeline — it is feasible and well-supported by existing
PhysicsNeMo primitives — with these guardrails:

1. **Reuse `physicsnemo.nn.functional.knn` / `radius_search`**; do not add
   `torch_cluster`/`pyg-lib`.
2. **Keep static shapes**: fixed-`k` kNN (or `radius_search(max_points=...)`).
3. **Replace METIS with GPU recursive coordinate bisection + on-device halo
   growth** — this is the critical enabler, not the neighbor search.
4. **Bake fields once into a dense Zarr source cloud**; only subsample/gather at
   runtime (never re-interpolate from `.vtp`).
5. **Overlap construction on a side CUDA stream** and prefetch with pinned
   memory so build time hides behind forward/backward.
6. **Fix the RNG seed for validation** to keep metrics comparable.

Expected outcome: ~10× smaller dataset (≈TB → tens of GB), removal of the
per-epoch edge-I/O bottleneck, and free per-epoch resampling/jitter augmentation
— with graph construction cost hidden behind the (much larger) model step.

### Suggested incremental rollout

1. Land `DynamicGraphBuilder` + spatial partition (kNN build, edge features and
   on-device halo growth are implemented in `dynamic_graph_datapipe.py`); unit-
   test that a *fixed-seed* dynamic graph matches an offline `.bin` graph within
   tolerance (parity gate).
2. Add the Zarr writer to `preprocessor.py` behind a config flag; keep the
   `.bin` path until parity is proven.
3. Switch `train3d.py` to the builder behind `cfg.use_dynamic_graph`; benchmark
   step time and storage vs. the baseline before removing the legacy path.
