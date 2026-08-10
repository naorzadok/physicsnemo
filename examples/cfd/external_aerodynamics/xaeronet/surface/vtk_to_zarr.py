#!/usr/bin/env python3
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
"""Lightweight VTU/VTP/VTK + STL -> DoMINO Zarr converter.

Converts CFD mesh files into per-sample ``.zarr`` stores using the DoMINO
schema consumed by ``physicsnemo`` (``DoMINODataPipe``).  Designed to run
inside the PhysicsNeMo container with only PyVista, NumPy, Zarr (>=3.0) and
tqdm installed -- it has *no* dependency on ``physicsnemo`` or
``physicsnemo-curator``.

Per-sample output keys (only those derivable from the given inputs are
written):

Geometry (from STL, or synthesized from the surface mesh -- falling back to
the volume's external surface only when no surface mesh is given):
    stl_coordinates  (N_stl, 3)  float32   vertices
    stl_faces        (3 * M,)     int32     flattened triangle vertex indices
    stl_centers      (M, 3)       float32   triangle centroids
    stl_areas        (M,)         float32   triangle areas

Surface (from a .vtp/.vtk surface):
    surface_mesh_centers (N_s, 3) float32   cell centers
    surface_normals      (N_s, 3) float32   unit cell normals
    surface_areas        (N_s,)   float32   cell areas
    surface_fields       (N_s, F) float32   concatenated surface variables

Volume (from a .vtu/.vtk volume):
    volume_mesh_centers  (N_v, 3) float32   point / cell-center coordinates
    volume_fields        (N_v, G) float32   concatenated volume variables

Global parameters (per-sample scalars such as ``stream_velocity`` and
``air_density``) are written in *both* conventions so a single store loads
cleanly in either training pipeline:
    <name>                 scalar   float     zarr group attribute (GeoTransolver)
    global_params_values   (K, 1)   float32   per-sample values      (DoMINO)
    global_params_reference (K, 1)  float32   reference constants    (DoMINO)
Each datapipe reads only the keys it needs, so the extra keys are harmless.

The script is agile across solvers: OpenFOAM/DrivAerML store cell-centroid
data while SU2 typically stores nodal (point) data under different names.
Field names are resolved through an alias table and data located on points
is transferred to cells when the DoMINO cell-centered convention requires it.
Values are stored raw (no non-dimensionalization); normalization is left to
the training datapipe.

By default the converted stores are partitioned into ``train``/``val``/``test``
subfolders of the output directory (an 80/10/10 split), so each split is its own
directory of ``.zarr`` stores ready for the DoMINO ``input_dir`` /
``input_dir_val`` / ``eval.test_path`` config keys.  The ratios, folder names and
shuffle seed are configurable (``--split`` / ``--split-names`` / ``--split-seed``);
a fixed seed makes the division reproducible.  Pass ``--no-split`` to write all
stores flat into the output directory instead.

Example (batch conversion exercising every flag)::

    python3 vtk_to_zarr.py \\
        --input-dir /data/drivaer_raw \\
        --pattern "**/*" \\
        --group --group-regex "run_(?P<id>\\d+)" \\
        --limit 100 \\
        --output-dir /data/drivaer_zarr \\
        --surface-fields default --volume-fields all \\
        --surface-location auto --volume-location cell \\
        --no-synth-stl \\
        --global-attrs /data/globals.csv \\
        --global-param-order stream_velocity,air_density \\
        --global-ref "stream_velocity=30.0,air_density=1.205" \\
        --no-global-params \\
        --split 80 10 10 --split-names train val test --split-seed 42 \\
        --compression-level 3 --chunk-mb 8.0 --overwrite \\
        --sampling-weights --sampling-config conf/config_dynamic.yaml \\
        --jobs 8 --timings --verbose

    # Single-sample conversion (explicit inputs, flat output):
    python3 vtk_to_zarr.py \\
        --stl geom.stl --surface boundary.vtp --volume internal.vtu \\
        --name case_001 --output-dir /data/out --no-split

Importance-sampling weights
---------------------------
The dynamic-graph training pipeline can bias its per-step node subsampling toward
high-curvature / feature-edge regions (where CFD gradients are strongest).  This
is driven by an optional per-surface-point ``sampling_weight`` array precomputed
here -- once, at conversion -- so the training step only does a cheap weighted
draw and stays fully GPU-accelerated.  The array is aligned 1:1 with
``surface_mesh_centers`` and is written together with a ``sampling_weight_cfg``
attribute recording the parameters used (provenance).  It is optional: pipelines
that do not find the array fall back to uniform sampling.

Enable / produce the weights::

    --sampling-weights                  Compute ``sampling_weight`` during
                                        conversion (writes it into each store).
    --backfill-sampling-weights PATH    Add ``sampling_weight`` to already-
                                        converted store(s) at PATH (a single
                                        .zarr or a directory tree) and exit;
                                        rebuilds the STL from the stored
                                        ``stl_*`` arrays.  ``--output-dir`` is
                                        not required in this mode.  Combine with
                                        ``--overwrite`` to replace existing
                                        weights.

Configure the weighting (precedence: built-in defaults < ``--sampling-config`` <
explicit flags)::

    --sampling-config YAML              Read the parameters from the ``sampling:``
                                        block of a training config (e.g.
                                        conf/config_dynamic.yaml), so the Zarr weights
                                        track the config without copying numbers.
                                        Unknown keys (``strategy``,
                                        ``debug_area_check``) are ignored.
    --curvature-weight FLOAT            lambda: strength of the curvature bias
                                        (higher -> denser at features).
    --curvature-exponent FLOAT          Exponent applied to normalized curvature.
    --curvature-clip-percentile FLOAT   Clip curvature at this percentile so a
                                        few spikes do not dominate.
    --curvature-smoothing-iterations N  Neighbor-averaging passes that denoise
                                        per-vertex curvature (0 disables).
    --feature-edge-weight FLOAT         Strength of the feature-edge proximity
                                        bias (0 disables).
    --feature-edge-angle FLOAT          Dihedral angle (deg) above which an edge
                                        counts as a sharp feature.
    --feature-edge-falloff FLOAT        Proximity falloff length as a fraction of
                                        the mesh bounding-box diagonal.
    --density-exponent FLOAT            Size-field contrast: density scales as
                                        weight**p (>1 sharpens, <1 flattens).
    --min-weight FLOAT                  Floor on the per-cell weight so smooth
                                        regions are not starved.

Examples::

    # Backfill existing stores using the exact values from a training config:
    python3 vtk_to_zarr.py \\
        --backfill-sampling-weights /data/drivaer_zarr \\
        --sampling-config conf/config_dynamic.yaml

    # Same, but override a single knob:
    python3 vtk_to_zarr.py \\
        --backfill-sampling-weights /data/drivaer_zarr \\
        --sampling-config conf/config_dynamic.yaml --min-weight 0.5 --overwrite
"""

from __future__ import annotations

import argparse
import logging
import random
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import pyvista as pv
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit("pyvista is required: pip install pyvista") from exc

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    def tqdm(iterable, **_kwargs):  # type: ignore
        return iterable


logger = logging.getLogger("vtk_to_zarr")

# --------------------------------------------------------------------------- #
# File classification
# --------------------------------------------------------------------------- #
_STL_EXT = {".stl"}
_SURFACE_EXT = {".vtp"}
_VOLUME_EXT = {".vtu"}
_AMBIGUOUS_EXT = {".vtk"}  # resolved via filename keywords

_VOLUME_KEYWORDS = ("volume", "internal", "vol")
_SURFACE_KEYWORDS = ("boundary", "surface", "surf", "wall", "aero")


class Role:
    """Roles a source file can play in a sample."""

    STL = "stl"
    SURFACE = "surface"
    VOLUME = "volume"


def classify_file(path: Path) -> str | None:
    """Classify a file as ``stl``, ``surface`` or ``volume`` (or ``None``)."""
    ext = path.suffix.lower()
    stem = path.stem.lower()
    if ext in _STL_EXT:
        return Role.STL
    if ext in _SURFACE_EXT:
        return Role.SURFACE
    if ext in _VOLUME_EXT:
        # A .vtu is usually a volume, but some datasets store the surface as a
        # .vtu too -- respect an explicit "boundary"/"surface" keyword.
        if any(k in stem for k in _SURFACE_KEYWORDS) and not any(
            k in stem for k in _VOLUME_KEYWORDS
        ):
            return Role.SURFACE
        return Role.VOLUME
    if ext in _AMBIGUOUS_EXT:
        if any(k in stem for k in _VOLUME_KEYWORDS):
            return Role.VOLUME
        if any(k in stem for k in _SURFACE_KEYWORDS):
            return Role.SURFACE
        # Fall back to inspecting the mesh contents later; default to surface.
        return Role.SURFACE
    return None


# --------------------------------------------------------------------------- #
# Field resolution
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FieldSpec:
    """A semantic field with solver-specific aliases."""

    canonical: str
    aliases: tuple[str, ...]
    components: int  # expected component count (1 = scalar, 3 = vector)


# Default surface variables, in output column order: pressure, wall shear stress.
DEFAULT_SURFACE_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("pressure", ("pMean", "pMeanTrim", "p", "Pressure", "pmean", "Pressure_Coefficient"), 1),
    FieldSpec(
        "wallShearStress",
        ("wallShearStress", "wallShearStressMean", "wallShearStressMeanTrim", "WallShearStress", "Skin_Friction_Coefficient"),
        3,
    ),
)

# Default volume variables, in output column order: velocity, pressure, turb visc.
DEFAULT_VOLUME_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("velocity", ("UMean", "U", "Velocity", "Momentum"), 3),
    FieldSpec("pressure", ("pMean", "p", "Pressure", "pmean"), 1),
    FieldSpec("turbulentViscosity", ("nutMean", "nut", "Eddy_Viscosity", "muT"), 1),
)

# Arrays that are geometry-derived rather than solution fields; excluded from
# the "all" selection so they are not duplicated into *_fields.
_RESERVED_ARRAY_NAMES = {"normals", "area", "cellsize", "length", "volume", "vtkoriginalpointids", "vtkoriginalcellids"}


# Global (parametric) scalars shared by a whole sample -------------------------
# Column order for the DoMINO ``global_params_values`` / ``global_params_reference``
# arrays.  Velocity precedes density to match the default DoMINO config
# (``variables.global_parameters``: inlet_velocity then air_density).
GLOBAL_PARAM_ORDER: tuple[str, ...] = ("stream_velocity", "air_density")

# Reference constants used for non-dimensionalization by DoMINO.  These are the
# defaults from the DoMINO drivaer config; override via ``--global-ref``.
DEFAULT_GLOBAL_PARAM_REFERENCES: dict[str, float] = {
    "stream_velocity": 30.0,
    "air_density": 1.205,
}

# Dataset split -------------------------------------------------------------- #
# Converted samples are partitioned into train / validation / test subfolders of
# the output directory (each split becomes its own directory of ``.zarr`` stores,
# matching the DoMINO ``input_dir`` / ``input_dir_val`` / ``eval.test_path``
# convention).  Edit these defaults in code, or override them on the CLI via
# ``--split`` / ``--split-names`` / ``--split-seed``.  Ratios need not sum to 1;
# they are normalized by their total (so ``80 10 10`` and ``0.8 0.1 0.1`` are
# equivalent).  A fixed seed makes the division reproducible across runs.
DEFAULT_SPLIT_RATIOS: tuple[float, ...] = (0.8, 0.1, 0.1)
DEFAULT_SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")
DEFAULT_SPLIT_SEED: int = 42


def _lower_key_map(attrs) -> dict[str, str]:
    """Map lower-cased array names to their actual names for a PyVista attrs."""
    return {str(name).lower(): str(name) for name in attrs.keys()}


def _as_2d(arr: np.ndarray) -> np.ndarray:
    """Return a 2-D (N, C) float32 view of a field array."""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 1:
        a = a[:, np.newaxis]
    return a


def resolve_fields(
    attrs,
    specs: Sequence[FieldSpec] | None,
    *,
    location_label: str,
) -> tuple[np.ndarray | None, list[dict]]:
    """Resolve and concatenate requested fields from a PyVista attribute set.

    Parameters
    ----------
    attrs:
        A PyVista ``point_data`` or ``cell_data`` mapping.
    specs:
        Field specs to extract, or ``None`` to take *all* solution arrays.
    location_label:
        ``"point"`` or ``"cell"``, used only for logging.

    Returns
    -------
    (data, meta):
        ``data`` is the concatenated ``(N, F)`` float32 array (or ``None`` if
        nothing resolved) and ``meta`` describes each contributing field.
    """
    lower_map = _lower_key_map(attrs)
    columns: list[np.ndarray] = []
    meta: list[dict] = []

    if specs is None:
        # "all": take every solution array in a stable, name-sorted order.
        for name in sorted(attrs.keys(), key=lambda n: str(n).lower()):
            if str(name).lower() in _RESERVED_ARRAY_NAMES:
                continue
            block = _as_2d(attrs[name])
            columns.append(block)
            meta.append({"name": str(name), "components": int(block.shape[1])})
        if not columns:
            return None, []
        return np.concatenate(columns, axis=1), meta

    for spec in specs:
        actual = None
        for alias in spec.aliases:
            if alias.lower() in lower_map:
                actual = lower_map[alias.lower()]
                break
        if actual is None:
            logger.warning(
                "Field '%s' not found in %s data (tried %s) -- skipping",
                spec.canonical,
                location_label,
                ", ".join(spec.aliases),
            )
            continue
        block = _as_2d(attrs[actual])
        if block.shape[1] != spec.components:
            logger.warning(
                "Field '%s' resolved to '%s' with %d components (expected %d) -- using as-is",
                spec.canonical,
                actual,
                block.shape[1],
                spec.components,
            )
        columns.append(block)
        meta.append({"name": spec.canonical, "source": actual, "components": int(block.shape[1])})

    if not columns:
        return None, []
    return np.concatenate(columns, axis=1), meta


def parse_field_selection(value: str | None, defaults: Sequence[FieldSpec]) -> Sequence[FieldSpec] | None:
    """Parse a ``--*-fields`` CLI value into field specs (or ``None`` for all)."""
    if value is None or value.strip() == "":
        return defaults
    token = value.strip()
    if token.lower() == "all":
        return None
    if token.lower() == "default":
        return defaults
    names = [n.strip() for n in token.split(",") if n.strip()]
    # Explicit names: match against known specs by canonical/alias, else treat
    # the literal name as a scalar field (component count validated at read).
    known = {s.canonical.lower(): s for s in (*DEFAULT_SURFACE_FIELDS, *DEFAULT_VOLUME_FIELDS)}
    for s in (*DEFAULT_SURFACE_FIELDS, *DEFAULT_VOLUME_FIELDS):
        for a in s.aliases:
            known.setdefault(a.lower(), s)
    specs: list[FieldSpec] = []
    for name in names:
        spec = known.get(name.lower())
        if spec is not None:
            specs.append(spec)
        else:
            specs.append(FieldSpec(name, (name,), 1))
    return specs


# --------------------------------------------------------------------------- #
# Global (parametric) scalar resolution
# --------------------------------------------------------------------------- #
# Global parameters (e.g. ``stream_velocity``, ``air_density``) are per-sample
# scalars that condition the model.  They are not derivable from mesh geometry,
# so they come from one of three sources:
#   1. A ``--global-attrs`` CSV (implemented; see load_global_attrs_csv).
#   2. Field/metadata arrays embedded in the VTP file (scaffold below).
#   3. Parsed from the sample filename (scaffold below).
# Enable (2) or (3) by uncommenting the helper and its call site in
# ``_process_sample`` (and the matching CLI options in ``build_parser``).

# def extract_globals_from_mesh(
#     mesh: "pv.DataSet",
#     names: Sequence[str] = GLOBAL_PARAM_ORDER,
# ) -> dict[str, float]:
#     """Extract per-sample global scalars from a PyVista mesh's metadata.
#
#     Looks for each name in ``field_data`` first (the natural home for
#     mesh-global values), then falls back to a constant-valued ``cell_data`` /
#     ``point_data`` array (taking the first element).  Missing names are simply
#     omitted so the caller can fall back to other sources.
#     """
#     found: dict[str, float] = {}
#     lower_field = {str(n).lower(): str(n) for n in mesh.field_data.keys()}
#     lower_cell = {str(n).lower(): str(n) for n in mesh.cell_data.keys()}
#     lower_point = {str(n).lower(): str(n) for n in mesh.point_data.keys()}
#     for name in names:
#         key = name.lower()
#         if key in lower_field:
#             found[name] = float(np.asarray(mesh.field_data[lower_field[key]]).flat[0])
#         elif key in lower_cell:
#             found[name] = float(np.asarray(mesh.cell_data[lower_cell[key]]).flat[0])
#         elif key in lower_point:
#             found[name] = float(np.asarray(mesh.point_data[lower_point[key]]).flat[0])
#     return found


# def extract_globals_from_name(
#     sample_name: str,
#     pattern: str,
# ) -> dict[str, float]:
#     """Parse per-sample global scalars from the sample name via a regex.
#
#     ``pattern`` must contain named groups matching the global parameter names,
#     e.g. r"v(?P<stream_velocity>[0-9.]+)_rho(?P<air_density>[0-9.]+)".
#     Matched groups are converted to float; unmatched names are omitted.
#     """
#     m = re.search(pattern, sample_name)
#     if not m:
#         return {}
#     return {k: float(v) for k, v in m.groupdict().items() if v is not None}


def extract_global_params_array(
    sample: "Sample",
    param_order: Sequence[str],
) -> np.ndarray | None:
    """PLACEHOLDER: extract per-sample global parameters as a numpy array.

    Plug your own extractor here.  Return a 1-D (or ``(K, 1)``) float array of
    per-sample global-parameter values ordered to match ``param_order``
    (length ``K``), or ``None`` to fall back to the other sources
    (``--global-attrs`` CSV / reference constants).

    Example implementation::

        values = my_extractor(sample.surface)  # -> np.ndarray, shape (K,)
        return values
    """
    return None


def build_global_params_arrays(
    sample_globals: dict[str, float],
    param_order: Sequence[str],
    references: dict[str, float],
) -> dict[str, np.ndarray]:
    """Assemble DoMINO ``global_params_*`` arrays from per-sample scalars.

    Produces two ``(K, 1)`` float32 arrays -- ``global_params_values`` (the
    per-sample values) and ``global_params_reference`` (the reference constants
    used for non-dimensionalization) -- in the fixed ``param_order`` column
    order.  Parameters absent from ``sample_globals`` fall back to their
    reference value so the arrays are always fully populated (DoMINO requires
    these keys unconditionally).
    """
    values: list[float] = []
    refs: list[float] = []
    for name in param_order:
        ref = float(references.get(name, 0.0))
        values.append(float(sample_globals.get(name, ref)))
        refs.append(ref)
    return {
        "global_params_values": np.asarray(values, dtype=np.float32).reshape(-1, 1),
        "global_params_reference": np.asarray(refs, dtype=np.float32).reshape(-1, 1),
    }


# --------------------------------------------------------------------------- #
# Geometry / mesh extraction
# --------------------------------------------------------------------------- #
def _triangle_geometry(surface: "pv.PolyData") -> dict[str, np.ndarray]:
    """Extract DoMINO ``stl_*`` arrays from a triangulated PolyData surface."""
    # Avoid a full-mesh copy when the surface is already triangulated (always
    # true for STL, and common for extracted surfaces).
    tri = surface if getattr(surface, "is_all_triangles", False) else surface.triangulate()
    points = np.asarray(tri.points, dtype=np.float32)
    faces = np.asarray(tri.faces).reshape(-1, 4)[:, 1:].astype(np.int32)
    centers = np.asarray(tri.cell_centers().points, dtype=np.float32)
    sized = tri.compute_cell_sizes(length=False, area=True, volume=False)
    areas = np.asarray(sized.cell_data["Area"], dtype=np.float32)
    return {
        "stl_coordinates": points,
        "stl_faces": faces.reshape(-1),
        "stl_centers": centers,
        "stl_areas": areas,
    }


def read_stl(path: Path) -> dict[str, np.ndarray]:
    """Read an STL file into DoMINO geometry arrays."""
    mesh = pv.read(str(path))
    surface = mesh if isinstance(mesh, pv.PolyData) else mesh.extract_surface()
    return _triangle_geometry(surface)


def synthesize_stl_from_volume(volume: "pv.DataSet") -> dict[str, np.ndarray]:
    """Derive geometry arrays from the external surface of a volume mesh."""
    surface = volume.extract_surface()
    return _triangle_geometry(surface)


def _select_location(mesh: "pv.DataSet", specs, mode: str) -> str:
    """Decide whether to read fields from point or cell data.

    ``mode`` is ``auto``, ``point`` or ``cell``.  In ``auto`` mode the location
    that actually contains the requested arrays wins, preferring cell data on
    ties (and when selecting "all").
    """
    if mode in ("point", "cell"):
        return mode

    def _has_any(attrs) -> bool:
        if specs is None:
            return len(attrs.keys()) > 0
        lower = {str(n).lower() for n in attrs.keys()}
        return any(any(a.lower() in lower for a in s.aliases) for s in specs)

    cell_ok = _has_any(mesh.cell_data)
    point_ok = _has_any(mesh.point_data)
    if cell_ok:
        return "cell"
    if point_ok:
        return "point"
    return "cell"


def read_surface(source, specs, location_mode: str) -> dict[str, np.ndarray]:
    """Read a surface mesh into DoMINO ``surface_*`` arrays (cell-centered).

    ``source`` may be a filesystem path or an already-parsed PyVista mesh, so
    callers that have loaded the file can reuse it instead of parsing it twice.
    """
    mesh = source if isinstance(source, pv.DataSet) else pv.read(str(source))
    surface = mesh if isinstance(mesh, pv.PolyData) else mesh.extract_surface()

    location = _select_location(surface, specs, location_mode)
    if location == "point":
        # DoMINO surfaces are cell-centered: move nodal data onto cells.
        surface = surface.point_data_to_cell_data(pass_point_data=False)

    centers = np.asarray(surface.cell_centers().points, dtype=np.float32)
    normals = np.asarray(surface.cell_normals, dtype=np.float32)
    # Normalize to unit length, guarding against zero-area cells.
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    norm[norm == 0.0] = 1.0
    normals = normals / norm
    sized = surface.compute_cell_sizes(length=False, area=True, volume=False)
    areas = np.asarray(sized.cell_data["Area"], dtype=np.float32)

    fields, meta = resolve_fields(surface.cell_data, specs, location_label="cell")

    out: dict[str, np.ndarray] = {
        "surface_mesh_centers": centers,
        "surface_normals": normals,
        "surface_areas": areas,
    }
    if fields is not None:
        out["surface_fields"] = fields
        out["_surface_field_meta"] = meta  # type: ignore[assignment]
    else:
        logger.warning("No surface fields resolved for %s", getattr(source, "name", "surface mesh"))
    return out


def read_volume(source, specs, location_mode: str) -> dict[str, np.ndarray]:
    """Read a volume mesh into DoMINO ``volume_*`` arrays.

    ``source`` may be a filesystem path or an already-parsed PyVista mesh.
    """
    mesh = source if isinstance(source, pv.DataSet) else pv.read(str(source))
    location = _select_location(mesh, specs, location_mode)

    if location == "cell":
        coords = np.asarray(mesh.cell_centers().points, dtype=np.float32)
        fields, meta = resolve_fields(mesh.cell_data, specs, location_label="cell")
    else:
        coords = np.asarray(mesh.points, dtype=np.float32)
        fields, meta = resolve_fields(mesh.point_data, specs, location_label="point")

    out: dict[str, np.ndarray] = {"volume_mesh_centers": coords}
    if fields is not None:
        out["volume_fields"] = fields
        out["_volume_field_meta"] = meta  # type: ignore[assignment]
    else:
        logger.warning("No volume fields resolved for %s", getattr(source, "name", "volume mesh"))
    return out


# --------------------------------------------------------------------------- #
# Importance-sampling weights (curvature + feature edges)
# --------------------------------------------------------------------------- #
# Precomputed per-surface-point weights that bias the dynamic-graph node
# subsampling toward high-curvature / feature-edge regions (where CFD gradients
# are strongest), so the GNN can resolve hard physics. Computing them here (once,
# at conversion) keeps training fully GPU-accelerated: the training step only
# does a cheap weighted draw using this array. The logic is
# self-contained (PyVista + NumPy + SciPy only).

SAMPLING_WEIGHT_DEFAULTS: dict[str, float] = {
    "curvature_weight": 4.0,
    "curvature_exponent": 1.0,
    "curvature_clip_percentile": 95.0,
    "curvature_smoothing_iterations": 5.0,
    "feature_edge_weight": 2.0,
    "feature_edge_angle": 30.0,
    "feature_edge_falloff": 0.02,
    "density_exponent": 1.0,
    "min_weight": 0.3,
}


def _polydata_from_stl_arrays(
    coords: np.ndarray, faces_flat: np.ndarray
) -> "pv.PolyData":
    """Rebuild a triangular ``pv.PolyData`` from stored ``stl_*`` arrays."""
    faces = np.asarray(faces_flat, dtype=np.int64).reshape(-1, 3)
    vtk_faces = np.hstack(
        [np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]
    ).reshape(-1)
    return pv.PolyData(np.asarray(coords, dtype=np.float32), vtk_faces)


def _smooth_point_scalar(
    mesh: "pv.PolyData", values: np.ndarray, iterations: int
) -> np.ndarray:
    """Umbrella (neighbor-average) smoothing of a per-vertex scalar field."""
    if iterations <= 0:
        return values
    faces = mesh.regular_faces
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    n_points = mesh.n_points
    smoothed = values.astype(np.float64, copy=True)
    for _ in range(iterations):
        summed = np.zeros(n_points, dtype=np.float64)
        counts = np.zeros(n_points, dtype=np.float64)
        np.add.at(summed, edges[:, 0], smoothed[edges[:, 1]])
        np.add.at(summed, edges[:, 1], smoothed[edges[:, 0]])
        np.add.at(counts, edges[:, 0], 1.0)
        np.add.at(counts, edges[:, 1], 1.0)
        nz = counts > 0
        smoothed[nz] = summed[nz] / counts[nz]
    return smoothed


def _cell_curvature_weights(mesh: "pv.PolyData", cfg: dict) -> np.ndarray:
    """Normalized ``[0, 1]`` per-cell maximum-curvature importance."""
    clip_percentile = float(cfg.get("curvature_clip_percentile", 95.0))
    smoothing_iterations = int(cfg.get("curvature_smoothing_iterations", 0))

    # vtkCurvatures spams per-point warnings on faceted STL triangles; silence.
    try:
        import vtk

        prev = vtk.vtkObject.GetGlobalWarningDisplay()
        vtk.vtkObject.GlobalWarningDisplayOff()
    except Exception:
        vtk = None
        prev = None
    try:
        point_curv = np.abs(mesh.curvature(curv_type="maximum"))
    finally:
        if vtk is not None and prev is not None:
            vtk.vtkObject.SetGlobalWarningDisplay(prev)

    point_curv = np.nan_to_num(point_curv, nan=0.0, posinf=0.0, neginf=0.0)
    point_curv = _smooth_point_scalar(mesh, point_curv, smoothing_iterations)

    cell_curv = point_curv[mesh.regular_faces].mean(axis=1)
    clip_value = np.percentile(cell_curv, clip_percentile)
    if clip_value > 0:
        cell_curv = np.clip(cell_curv, 0.0, clip_value)
    max_curv = cell_curv.max()
    return cell_curv / max_curv if max_curv > 0 else np.zeros_like(cell_curv)


def _cell_edge_weights(mesh: "pv.PolyData", cfg: dict) -> np.ndarray:
    """Normalized ``[0, 1]`` per-cell proximity to sharp feature edges."""
    from scipy.spatial import cKDTree

    feature_angle = float(cfg.get("feature_edge_angle", 30.0))
    falloff_fraction = float(cfg.get("feature_edge_falloff", 0.02))

    edges = mesh.extract_feature_edges(
        feature_angle=feature_angle,
        boundary_edges=True,
        non_manifold_edges=True,
        feature_edges=True,
        manifold_edges=False,
    )
    if edges.n_points == 0:
        return np.zeros(mesh.n_cells)

    bbox = np.asarray(mesh.bounds).reshape(3, 2)
    diag = np.linalg.norm(bbox[:, 1] - bbox[:, 0])
    falloff_length = max(falloff_fraction * diag, 1e-12)

    centroids = np.asarray(mesh.cell_centers().points)
    tree = cKDTree(np.asarray(edges.points))
    distances, _ = tree.query(centroids, k=1)
    return np.exp(-distances / falloff_length)


def _cell_sampling_weights(mesh: "pv.PolyData", cfg: dict) -> np.ndarray:
    """Combine curvature + feature-edge importance into a per-cell weight."""
    lam_curv = float(cfg.get("curvature_weight", 4.0))
    exponent = float(cfg.get("curvature_exponent", 1.0))
    lam_edge = float(cfg.get("feature_edge_weight", 0.0))
    density_exponent = float(cfg.get("density_exponent", 1.0))
    min_weight = float(cfg.get("min_weight", 0.3))

    weights = np.ones(mesh.n_cells)
    if lam_curv > 0:
        weights = weights + lam_curv * (_cell_curvature_weights(mesh, cfg) ** exponent)
    if lam_edge > 0:
        weights = weights + lam_edge * _cell_edge_weights(mesh, cfg)
    if density_exponent != 1.0:
        weights = weights ** density_exponent
    return np.maximum(min_weight, weights)


def compute_sampling_weights(
    stl_coordinates: np.ndarray,
    stl_faces: np.ndarray,
    surface_centers: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    """Per-surface-point importance weights from STL curvature / feature edges.

    Per-cell weights are computed on the STL geometry, then transferred to each
    surface point by nearest STL-cell centroid. Returns a ``(N_surface,)``
    float32 array aligned with ``surface_mesh_centers``.
    """
    from scipy.spatial import cKDTree

    mesh = _polydata_from_stl_arrays(stl_coordinates, stl_faces)
    cell_weights = _cell_sampling_weights(mesh, cfg)
    cell_centers = np.asarray(mesh.cell_centers().points)
    tree = cKDTree(cell_centers)
    _, nearest = tree.query(np.asarray(surface_centers, dtype=np.float32), k=1)
    return cell_weights[nearest].astype(np.float32)


# --------------------------------------------------------------------------- #
# Zarr writing
# --------------------------------------------------------------------------- #
def _compute_chunks(shape: tuple[int, ...], itemsize: int, target_mb: float) -> tuple[int, ...]:
    """Chunk sizing targeting ``target_mb`` per chunk, chunking axis 0."""
    target_bytes = max(1, int(target_mb * 1024 * 1024))
    if len(shape) == 1:
        return (max(1, min(shape[0], target_bytes // itemsize)),)
    row_bytes = int(np.prod(shape[1:])) * itemsize
    chunk_rows = max(1, min(shape[0], target_bytes // max(1, row_bytes)))
    return (chunk_rows, *shape[1:])


def write_zarr(
    sample_name: str,
    arrays: dict[str, np.ndarray],
    attrs: dict,
    output_dir: Path,
    *,
    compression_level: int,
    chunk_mb: float,
    overwrite: bool,
    split: str | None = None,
) -> Path:
    """Write a sample's arrays to a ``<sample_name>.zarr`` store (zarr v3).

    When ``split`` is given the store is written into that subfolder of
    ``output_dir`` (e.g. ``output_dir/train/<sample_name>.zarr``); otherwise it
    is written directly under ``output_dir``.
    """
    import zarr
    from zarr.codecs import BloscCodec

    target_dir = output_dir / split if split else output_dir
    output_path = target_dir / f"{sample_name}.zarr"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists (use --overwrite to replace)")
    target_dir.mkdir(parents=True, exist_ok=True)

    compressors = [BloscCodec(cname="zstd", clevel=compression_level, shuffle="shuffle")]
    store = zarr.open_group(str(output_path), mode="w")

    for name, data in arrays.items():
        data = np.ascontiguousarray(data)
        chunks = _compute_chunks(data.shape, data.dtype.itemsize, chunk_mb)
        store.create_array(name, data=data, chunks=chunks, compressors=compressors)

    for key, value in attrs.items():
        store.attrs[key] = value

    return output_path


# --------------------------------------------------------------------------- #
# Sample assembly
# --------------------------------------------------------------------------- #
@dataclass
class Sample:
    """A group of files that together form one output store."""

    name: str
    stl: Path | None = None
    surface: Path | None = None
    volume: Path | None = None
    # Target split subfolder (e.g. "train"/"val"/"test") or ``None`` to write
    # directly into the output directory (flat, no split).
    split: str | None = None

    def sources(self) -> dict[str, str]:
        srcs = {}
        if self.stl:
            srcs["stl"] = self.stl.name
        if self.surface:
            srcs["surface"] = self.surface.name
        if self.volume:
            srcs["volume"] = self.volume.name
        return srcs


def build_sample(
    sample: Sample,
    *,
    surface_specs,
    volume_specs,
    surface_location: str,
    volume_location: str,
    synth_stl: bool,
    timings: bool = False,
    sampling_cfg: dict | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Read all inputs of a sample into DoMINO arrays + attributes."""
    arrays: dict[str, np.ndarray] = {}
    attrs: dict = {"sample_name": sample.name, "sources": sample.sources()}

    def _log_stage(stage: str, start: float) -> None:
        if timings:
            logger.info("[%s] %s: %.3fs", sample.name, stage, time.perf_counter() - start)

    # Parse each source file exactly once and reuse the mesh across geometry
    # synthesis and field extraction (previously the surface/volume file was
    # read twice when synthesizing geometry).
    surface_mesh = None
    volume_mesh = None
    if sample.surface is not None:
        t = time.perf_counter()
        surface_mesh = pv.read(str(sample.surface))
        _log_stage("read surface", t)
    if sample.volume is not None:
        t = time.perf_counter()
        volume_mesh = pv.read(str(sample.volume))
        _log_stage("read volume", t)

    # Geometry -------------------------------------------------------------- #
    # Geometry comes from the STL when available; otherwise it is synthesized
    # from the surface mesh (preferred), falling back to the volume's external
    # surface only when no surface mesh is provided.
    t = time.perf_counter()
    if sample.stl is not None:
        arrays.update(read_stl(sample.stl))
        attrs["stl_origin"] = "file"
    elif synth_stl and surface_mesh is not None:
        logger.info("[%s] no STL provided -- synthesizing geometry from surface mesh", sample.name)
        surf = surface_mesh if isinstance(surface_mesh, pv.PolyData) else surface_mesh.extract_surface()
        arrays.update(_triangle_geometry(surf))
        attrs["stl_origin"] = "synthesized_from_surface"
    elif synth_stl and volume_mesh is not None:
        logger.info("[%s] no STL provided -- synthesizing geometry from volume surface", sample.name)
        arrays.update(synthesize_stl_from_volume(volume_mesh))
        attrs["stl_origin"] = "synthesized_from_volume"
    _log_stage("geometry", t)

    # Surface --------------------------------------------------------------- #
    if surface_mesh is not None:
        t = time.perf_counter()
        surf = read_surface(surface_mesh, surface_specs, surface_location)
        meta = surf.pop("_surface_field_meta", None)
        arrays.update(surf)
        if meta is not None:
            attrs["surface_fields_meta"] = meta
        _log_stage("surface fields", t)

    # Volume ---------------------------------------------------------------- #
    if volume_mesh is not None:
        t = time.perf_counter()
        vol = read_volume(volume_mesh, volume_specs, volume_location)
        meta = vol.pop("_volume_field_meta", None)
        arrays.update(vol)
        if meta is not None:
            attrs["volume_fields_meta"] = meta
        _log_stage("volume fields", t)

    # Importance-sampling weights ------------------------------------------- #
    # Precompute per-surface-point weights from STL curvature / feature edges so
    # the dynamic-graph pipeline can bias node subsampling toward hard-physics
    # regions without any per-step mesh work.
    if (
        sampling_cfg is not None
        and "stl_coordinates" in arrays
        and "stl_faces" in arrays
        and "surface_mesh_centers" in arrays
    ):
        t = time.perf_counter()
        try:
            arrays["sampling_weight"] = compute_sampling_weights(
                arrays["stl_coordinates"],
                arrays["stl_faces"],
                arrays["surface_mesh_centers"],
                sampling_cfg,
            )
            attrs["sampling_weight_cfg"] = {k: float(v) for k, v in sampling_cfg.items()}
        except Exception as exc:  # keep conversion robust to odd geometry
            logger.warning("[%s] sampling weights failed: %s", sample.name, exc)
        _log_stage("sampling weights", t)

    return arrays, attrs


def backfill_sampling_weights(
    path: Path, sampling_cfg: dict, *, overwrite: bool = False
) -> int:
    """Add a ``sampling_weight`` array to already-converted ``.zarr`` stores.

    ``path`` may be a single ``.zarr`` store or a directory tree containing
    them. Each store's weights are computed from its stored ``stl_*`` arrays and
    written in place. Returns the number of stores updated.
    """
    import zarr

    if path.suffix == ".zarr" and path.is_dir():
        stores = [path]
    else:
        stores = sorted(p for p in path.rglob("*.zarr") if p.is_dir())
    if not stores:
        logger.warning("No .zarr stores found under %s", path)
        return 0

    updated = 0
    for store_path in tqdm(stores, desc="Backfilling weights", unit="store"):
        group = zarr.open_group(str(store_path), mode="a")
        keys = set(group.array_keys())
        if "sampling_weight" in keys and not overwrite:
            logger.info("%s already has sampling_weight; skipping", store_path.name)
            continue
        missing = {"stl_coordinates", "stl_faces", "surface_mesh_centers"} - keys
        if missing:
            logger.warning("%s missing %s; skipping", store_path.name, sorted(missing))
            continue

        weights = compute_sampling_weights(
            np.asarray(group["stl_coordinates"][:]),
            np.asarray(group["stl_faces"][:]),
            np.asarray(group["surface_mesh_centers"][:]),
            sampling_cfg,
        )
        chunks = _compute_chunks(weights.shape, weights.dtype.itemsize, 8.0)
        if "sampling_weight" in keys:
            del group["sampling_weight"]
        group.create_array("sampling_weight", data=weights, chunks=chunks)
        group.attrs["sampling_weight_cfg"] = {
            k: float(v) for k, v in sampling_cfg.items()
        }
        updated += 1
    logger.info("Backfilled sampling_weight into %d store(s)", updated)
    return updated


# --------------------------------------------------------------------------- #
# Batch driver
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ConvertOptions:
    """Immutable per-run settings shared across samples (picklable for pools)."""

    surface_specs: Sequence[FieldSpec] | None
    volume_specs: Sequence[FieldSpec] | None
    surface_location: str
    volume_location: str
    synth_stl: bool
    output_dir: Path
    compression_level: int
    chunk_mb: float
    overwrite: bool
    global_attrs: dict[str, dict[str, float]] | None = None  # sample_name -> {key: value}
    timings: bool = False
    verbose: bool = False
    # Precompute per-point importance weights from STL curvature / feature edges
    # and store them as ``sampling_weight``. ``None`` disables the feature.
    sampling_cfg: dict | None = None

    # Global (parametric) scalar handling:
    # Emit the DoMINO-style ``global_params_values`` / ``global_params_reference``
    # arrays (in addition to the GeoTransolver-style scalar attrs).
    emit_global_params: bool = True
    global_param_order: Sequence[str] = field(
        default_factory=lambda: list(GLOBAL_PARAM_ORDER)
    )
    global_param_references: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_GLOBAL_PARAM_REFERENCES)
    )
    # Sources for global scalars (scaffolds; see _process_sample):
    # global_from_fields: bool = False
    # global_name_regex: str | None = None


def load_global_attrs_csv(path: Path) -> dict[str, dict[str, float]]:
    """Load a CSV of per-sample global scalar attributes.

    The CSV must have a header row. The first column is treated as the sample
    name (matched against the ``Sample.name`` used for zarr output); all
    remaining columns become floating-point zarr attributes.

    Example CSV::

        sample_name,air_density,stream_velocity
        1,1.205,30.0
        2,1.205,35.0

    Returns a dict mapping sample name -> {attr_name: float_value}.
    """
    import csv

    result: dict[str, dict[str, float]] = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or len(reader.fieldnames) < 2:
            raise ValueError(f"--global-attrs CSV must have at least 2 columns, got: {path}")
        name_col = reader.fieldnames[0]
        value_cols = reader.fieldnames[1:]
        for row in reader:
            sample_name = row[name_col].strip()
            result[sample_name] = {col: float(row[col]) for col in value_cols}
    return result


def _process_sample(sample: Sample, opts: ConvertOptions) -> tuple[str, str, str | None]:
    """Convert a single sample end-to-end.

    Returns ``(name, status, error)`` where status is ``"ok"``, ``"skipped"``
    or ``"failed"``.  Never raises, so it is safe to run inside a worker pool
    with per-sample error isolation.
    """
    try:
        arrays, attrs = build_sample(
            sample,
            surface_specs=opts.surface_specs,
            volume_specs=opts.volume_specs,
            surface_location=opts.surface_location,
            volume_location=opts.volume_location,
            synth_stl=opts.synth_stl,
            timings=opts.timings,
            sampling_cfg=opts.sampling_cfg,
        )
        # Resolve per-sample global scalars.  Source precedence (only the CSV
        # path is enabled by default):
        #   1. --global-attrs CSV
        #   2. VTP field/metadata  (uncomment extract_globals_from_mesh below)
        #   3. filename parse       (uncomment extract_globals_from_name below)
        sample_globals: dict[str, float] = {}
        # if opts.global_from_fields and sample.surface is not None:
        #     sample_globals.update(extract_globals_from_mesh(pv.read(str(sample.surface))))
        # if opts.global_name_regex:
        #     sample_globals.update(
        #         extract_globals_from_name(sample.name, opts.global_name_regex)
        #     )
        if opts.global_attrs and sample.name in opts.global_attrs:
            sample_globals.update(opts.global_attrs[sample.name])

        if sample_globals:
            # (a) GeoTransolver reads these as individual scalar attrs.
            attrs.update(sample_globals)

        if opts.emit_global_params:
            # (b) DoMINO reads combined (K, 1) arrays.  Written unconditionally
            # (falling back to reference constants) so DoMINO's required-key
            # check always passes when switching models.
            #
            # To source the per-sample values from a numpy-array extractor
            # instead of the dict-based sources above, extract_global_params_array
            # takes precedence when it returns a non-None array and otherwise
            # falls through to build_global_params_arrays.  Like every other
            # source it feeds BOTH conventions: the (K, 1) arrays for DoMINO and
            # the per-name scalar attrs for GeoTransolver.
            custom_values = extract_global_params_array(
                sample, opts.global_param_order
            )
            if custom_values is not None:
                custom_values = np.asarray(
                    custom_values, dtype=np.float32
                ).reshape(-1)
                refs = np.asarray(
                    [
                        float(opts.global_param_references.get(name, 0.0))
                        for name in opts.global_param_order
                    ],
                    dtype=np.float32,
                ).reshape(-1, 1)
                # (DoMINO) combined (K, 1) arrays.
                arrays["global_params_values"] = custom_values.reshape(-1, 1)
                arrays["global_params_reference"] = refs
                # (GeoTransolver) per-name scalar attrs, keyed by param order.
                attrs.update(
                    {
                        name: float(value)
                        for name, value in zip(
                            opts.global_param_order, custom_values
                        )
                    }
                )
            else:
                arrays.update(
                    build_global_params_arrays(
                        sample_globals,
                        opts.global_param_order,
                        opts.global_param_references,
                    )
                )

        if not arrays:
            return (sample.name, "skipped", None)
        write_zarr(
            sample.name,
            arrays,
            attrs,
            opts.output_dir,
            compression_level=opts.compression_level,
            chunk_mb=opts.chunk_mb,
            overwrite=opts.overwrite,
            split=sample.split,
        )
        return (sample.name, "ok", None)
    except Exception as exc:  # noqa: BLE001 - keep batch robust
        error = traceback.format_exc() if opts.verbose else str(exc)
        return (sample.name, "failed", error)


# --------------------------------------------------------------------------- #
# Discovery / grouping
# --------------------------------------------------------------------------- #
def _extract_id(stem: str, group_regex: re.Pattern | None) -> str | None:
    """Extract a sample id from a filename stem."""
    if group_regex is not None:
        m = group_regex.search(stem)
        if m:
            return m.group("id") if "id" in m.groupdict() else m.group(0)
        return None
    # Default: the last run of digits in the stem (drivaer_12 -> "12").
    matches = re.findall(r"\d+", stem)
    return matches[-1] if matches else None


def discover_samples(
    input_dir: Path,
    pattern: str,
    group_regex: re.Pattern | None,
    group: bool,
) -> list[Sample]:
    """Discover and (optionally) group files under ``input_dir`` into samples."""
    paths = sorted(
        p
        for p in input_dir.glob(pattern)
        if p.is_file() and classify_file(p) is not None
    )
    if not paths:
        return []

    if not group:
        # Each file becomes its own sample.
        samples = []
        for p in paths:
            role = classify_file(p)
            s = Sample(name=p.stem)
            setattr(s, role, p)
            samples.append(s)
        return samples

    grouped: dict[str, Sample] = {}
    for p in paths:
        role = classify_file(p)
        sid = _extract_id(p.stem, group_regex)
        key = sid if sid is not None else p.stem
        sample = grouped.setdefault(key, Sample(name=key))
        # Prefer the first file seen for a given role; warn on collisions.
        if getattr(sample, role) is None:
            setattr(sample, role, p)
        else:
            logger.warning(
                "Sample '%s' already has a %s file (%s); ignoring %s",
                key,
                role,
                getattr(sample, role).name,
                p.name,
            )
    return [grouped[k] for k in sorted(grouped)]


# --------------------------------------------------------------------------- #
# Dataset split
# --------------------------------------------------------------------------- #
def _split_counts(total: int, ratios: Sequence[float]) -> list[int]:
    """Split ``total`` items across ``ratios`` using largest-remainder rounding.

    The returned counts always sum exactly to ``total`` regardless of rounding,
    with any leftover items assigned to the splits whose fractional part is
    largest (ties broken by split order).  ``ratios`` need not sum to 1 -- they
    are normalized by their total first.
    """
    weight = float(sum(ratios))
    if weight <= 0.0:
        raise ValueError("Split ratios must sum to a positive value")
    exact = [total * (r / weight) for r in ratios]
    floors = [int(x) for x in exact]
    remainder = total - sum(floors)
    # Distribute the leftover to the largest fractional parts.
    order = sorted(range(len(ratios)), key=lambda i: exact[i] - floors[i], reverse=True)
    for i in order[:remainder]:
        floors[i] += 1
    return floors


def assign_splits(
    samples: list[Sample],
    ratios: Sequence[float],
    names: Sequence[str],
    seed: int,
) -> list[Sample]:
    """Assign each sample to a train/val/test split, in place.

    Samples are first sorted by name (a stable, discovery-order-independent
    canonical order) then shuffled with ``random.Random(seed)`` before being
    partitioned according to ``ratios``.  This makes the division fully
    reproducible: the same ``seed`` + same sample set + same ratios always
    yields the identical assignment, independent of file discovery order or the
    number of parallel workers.  Splits receiving zero samples produce no
    output folder.  Returns the same ``samples`` list (mutated).
    """
    if len(ratios) != len(names):
        raise ValueError(
            f"Number of split ratios ({len(ratios)}) must match names ({len(names)})"
        )
    counts = _split_counts(len(samples), ratios)
    order = sorted(samples, key=lambda s: s.name)
    random.Random(seed).shuffle(order)

    idx = 0
    for name, count in zip(names, counts):
        for sample in order[idx : idx + count]:
            sample.split = name
        idx += count
    return samples


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert VTU/VTP/VTK + STL CFD meshes to DoMINO-schema Zarr stores.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    single = parser.add_argument_group("single-sample inputs (explicit)")
    single.add_argument("--stl", type=Path, help="STL geometry file")
    single.add_argument("--surface", type=Path, help="Surface mesh (.vtp/.vtk)")
    single.add_argument("--volume", type=Path, help="Volume mesh (.vtu/.vtk)")
    single.add_argument("--name", type=str, help="Output sample name (default: derived from inputs)")

    batch = parser.add_argument_group("batch inputs (directory)")
    batch.add_argument("--input-dir", type=Path, help="Directory to scan for mesh files")
    batch.add_argument("--pattern", type=str, default="**/*", help="Glob pattern under --input-dir")
    batch.add_argument(
        "--group",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Group discovered files by id into one sample each (--no-group: one file per sample)",
    )
    batch.add_argument(
        "--group-regex",
        type=str,
        default=None,
        help="Regex with an 'id' group to key samples (default: trailing integer in the stem)",
    )
    batch.add_argument("--limit", type=int, default=None, help="Process at most N samples")

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for .zarr output (not required with --backfill-sampling-weights)",
    )

    fields = parser.add_argument_group("field selection")
    fields.add_argument(
        "--surface-fields",
        type=str,
        default=None,
        help="'default', 'all', or comma-separated names for surface_fields",
    )
    fields.add_argument(
        "--volume-fields",
        type=str,
        default=None,
        help="'default', 'all', or comma-separated names for volume_fields",
    )
    fields.add_argument(
        "--surface-location",
        choices=("auto", "point", "cell"),
        default="auto",
        help="Where surface field arrays live in the source",
    )
    fields.add_argument(
        "--volume-location",
        choices=("auto", "point", "cell"),
        default="auto",
        help="Where volume field arrays live in the source",
    )
    fields.add_argument(
        "--no-synth-stl",
        dest="synth_stl",
        action="store_false",
        help="Do not synthesize geometry from the volume/surface when STL is missing",
    )
    parser.set_defaults(synth_stl=True)

    fields.add_argument(
        "--global-attrs",
        type=Path,
        default=None,
        metavar="CSV",
        help=(
            "CSV file with per-sample global scalar attributes to embed in zarr attrs. "
            "First column = sample name, remaining columns = attribute name/value pairs. "
            "Example: sample_name,air_density,stream_velocity"
        ),
    )

    fields.add_argument(
        "--no-global-params",
        dest="emit_global_params",
        action="store_false",
        help=(
            "Do not emit the DoMINO-style global_params_values / "
            "global_params_reference arrays (still writes scalar attrs)"
        ),
    )
    parser.set_defaults(emit_global_params=True)
    fields.add_argument(
        "--global-param-order",
        type=str,
        default=",".join(GLOBAL_PARAM_ORDER),
        help="Comma-separated column order for the DoMINO global_params_* arrays",
    )
    fields.add_argument(
        "--global-ref",
        type=str,
        default=None,
        metavar="NAME=VALUE,...",
        help=(
            "Override reference constants for global_params_reference, e.g. "
            "'stream_velocity=30.0,air_density=1.205'"
        ),
    )
    # Scaffolds for deriving global scalars without a CSV (see _process_sample):
    # fields.add_argument(
    #     "--global-from-fields",
    #     action="store_true",
    #     help="Extract global scalars from VTP field/metadata arrays",
    # )
    # fields.add_argument(
    #     "--global-name-regex",
    #     type=str,
    #     default=None,
    #     help=(
    #         "Regex with named groups (e.g. stream_velocity, air_density) to "
    #         "parse globals from the sample name"
    #     ),
    # )

    out = parser.add_argument_group("output options")
    out.add_argument("--compression-level", type=int, default=3, help="Blosc/zstd level (0-9)")
    out.add_argument("--chunk-mb", type=float, default=8.0, help="Target chunk size (MB)")
    out.add_argument("--overwrite", action="store_true", help="Overwrite existing .zarr stores")
    out.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    split = parser.add_argument_group("dataset split")
    split.add_argument(
        "--split",
        type=float,
        nargs=3,
        default=list(DEFAULT_SPLIT_RATIOS),
        metavar=("TRAIN", "VAL", "TEST"),
        help=(
            "Train/val/test split ratios (need not sum to 1; normalized by their "
            "total, so '80 10 10' and '0.8 0.1 0.1' are equivalent)"
        ),
    )
    split.add_argument(
        "--split-names",
        type=str,
        nargs=3,
        default=list(DEFAULT_SPLIT_NAMES),
        metavar=("TRAIN", "VAL", "TEST"),
        help="Subfolder names for the three splits",
    )
    split.add_argument(
        "--split-seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help="Random seed for the split shuffle (fix it to reproduce the exact division)",
    )
    split.add_argument(
        "--no-split",
        dest="split_enabled",
        action="store_false",
        help="Disable splitting; write all stores flat into --output-dir",
    )
    parser.set_defaults(split_enabled=True)

    sampling = parser.add_argument_group("importance-sampling weights")
    sampling.add_argument(
        "--sampling-weights",
        action="store_true",
        help=(
            "Precompute a per-surface-point 'sampling_weight' array from STL "
            "curvature / feature edges (for biased dynamic-graph subsampling)"
        ),
    )
    sampling.add_argument(
        "--backfill-sampling-weights",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Add 'sampling_weight' to already-converted .zarr store(s) at PATH "
            "(a single store or a directory tree) and exit. Uses --overwrite to "
            "replace existing weights."
        ),
    )
    sampling.add_argument(
        "--sampling-config",
        type=Path,
        default=None,
        metavar="YAML",
        help=(
            "Read weight parameters from the 'sampling:' block of a YAML config "
            "(e.g. conf/config_dynamic.yaml). Values seed the defaults; any explicit "
            "--curvature-*/--feature-edge-*/etc. flag still overrides them."
        ),
    )
    # Numeric knobs default to SUPPRESS so we can tell an explicit override from
    # an unset flag (unset -> value comes from --sampling-config or the defaults).
    _sd = SAMPLING_WEIGHT_DEFAULTS
    sampling.add_argument("--curvature-weight", type=float, default=argparse.SUPPRESS, help=f"(default: {_sd['curvature_weight']})")
    sampling.add_argument("--curvature-exponent", type=float, default=argparse.SUPPRESS, help=f"(default: {_sd['curvature_exponent']})")
    sampling.add_argument("--curvature-clip-percentile", type=float, default=argparse.SUPPRESS, help=f"(default: {_sd['curvature_clip_percentile']})")
    sampling.add_argument("--curvature-smoothing-iterations", type=int, default=argparse.SUPPRESS, help=f"(default: {int(_sd['curvature_smoothing_iterations'])})")
    sampling.add_argument("--feature-edge-weight", type=float, default=argparse.SUPPRESS, help=f"(default: {_sd['feature_edge_weight']})")
    sampling.add_argument("--feature-edge-angle", type=float, default=argparse.SUPPRESS, help=f"(default: {_sd['feature_edge_angle']})")
    sampling.add_argument("--feature-edge-falloff", type=float, default=argparse.SUPPRESS, help=f"(default: {_sd['feature_edge_falloff']})")
    sampling.add_argument("--density-exponent", type=float, default=argparse.SUPPRESS, help=f"(default: {_sd['density_exponent']})")
    sampling.add_argument("--min-weight", type=float, default=argparse.SUPPRESS, help=f"(default: {_sd['min_weight']})")

    perf = parser.add_argument_group("performance")
    perf.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Parallel worker processes for batch conversion (1 = sequential)",
    )
    perf.add_argument(
        "--timings",
        action="store_true",
        help="Log per-stage timing (read/geometry/fields) for each sample",
    )

    return parser


def _derive_name(args: argparse.Namespace) -> str:
    if args.name:
        return args.name
    for p in (args.volume, args.surface, args.stl):
        if p is not None:
            return Path(p).stem
    return "sample"


def _resolve_samples(args: argparse.Namespace) -> list[Sample]:
    explicit = any(x is not None for x in (args.stl, args.surface, args.volume))
    if explicit and args.input_dir is not None:
        raise SystemExit("Use either explicit --stl/--surface/--volume or --input-dir, not both.")

    if explicit:
        return [
            Sample(
                name=_derive_name(args),
                stl=args.stl,
                surface=args.surface,
                volume=args.volume,
            )
        ]

    if args.input_dir is None:
        raise SystemExit("Provide inputs via --stl/--surface/--volume or --input-dir.")

    group_regex = re.compile(args.group_regex) if args.group_regex else None
    samples = discover_samples(args.input_dir, args.pattern, group_regex, args.group)
    if args.limit is not None:
        samples = samples[: args.limit]
    return samples


def _load_sampling_config(path: Path) -> dict:
    """Read weight parameters from the ``sampling:`` block of a YAML config.

    Accepts either a full training config (reads its ``sampling`` mapping) or a
    file that is itself the sampling mapping. Only the keys known to
    ``SAMPLING_WEIGHT_DEFAULTS`` are picked up; extras (e.g. ``strategy``,
    ``debug_area_check``) are ignored.
    """
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f) or {}
    block = data.get("sampling", data) if isinstance(data, dict) else None
    if not isinstance(block, dict):
        raise SystemExit(f"No 'sampling' mapping found in {path}")
    cfg: dict = {}
    for key in SAMPLING_WEIGHT_DEFAULTS:
        if block.get(key) is not None:
            cfg[key] = float(block[key])
    return cfg


def _sampling_cfg_from_args(args: argparse.Namespace) -> dict:
    """Assemble the weight config: defaults < --sampling-config < explicit flags."""
    cfg = dict(SAMPLING_WEIGHT_DEFAULTS)
    if getattr(args, "sampling_config", None) is not None:
        cfg.update(_load_sampling_config(args.sampling_config))
    # SUPPRESS defaults mean an attribute exists only when the flag was passed.
    for key in SAMPLING_WEIGHT_DEFAULTS:
        if hasattr(args, key):
            cfg[key] = getattr(args, key)
    return cfg


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    # Backfill mode: add sampling_weight to existing stores and exit.
    if args.backfill_sampling_weights is not None:
        updated = backfill_sampling_weights(
            args.backfill_sampling_weights,
            _sampling_cfg_from_args(args),
            overwrite=args.overwrite,
        )
        return 0 if updated > 0 else 1

    if args.output_dir is None:
        raise SystemExit("--output-dir is required (except with --backfill-sampling-weights).")

    surface_specs = parse_field_selection(args.surface_fields, DEFAULT_SURFACE_FIELDS)
    volume_specs = parse_field_selection(args.volume_fields, DEFAULT_VOLUME_FIELDS)

    global_param_order = [
        n.strip() for n in args.global_param_order.split(",") if n.strip()
    ]
    global_param_references = dict(DEFAULT_GLOBAL_PARAM_REFERENCES)
    if args.global_ref:
        for token in args.global_ref.split(","):
            token = token.strip()
            if not token:
                continue
            name, _, raw = token.partition("=")
            global_param_references[name.strip()] = float(raw)

    samples = _resolve_samples(args)
    if not samples:
        logger.error("No samples found.")
        return 1

    logger.info("Discovered %d sample(s) -> %s", len(samples), args.output_dir)

    if args.split_enabled:
        assign_splits(samples, args.split, args.split_names, args.split_seed)
        counts: dict[str, int] = {name: 0 for name in args.split_names}
        for s in samples:
            if s.split is not None:
                counts[s.split] += 1
        summary = ", ".join(f"{name}={counts[name]}" for name in args.split_names)
        logger.info("Split (seed=%d): %s", args.split_seed, summary)

    global_attrs = None
    if args.global_attrs is not None:
        global_attrs = load_global_attrs_csv(args.global_attrs)
        logger.info("Loaded global attrs for %d sample(s) from %s", len(global_attrs), args.global_attrs)

    opts = ConvertOptions(
        surface_specs=surface_specs,
        volume_specs=volume_specs,
        surface_location=args.surface_location,
        volume_location=args.volume_location,
        synth_stl=args.synth_stl,
        output_dir=args.output_dir,
        compression_level=args.compression_level,
        chunk_mb=args.chunk_mb,
        overwrite=args.overwrite,
        global_attrs=global_attrs,
        timings=args.timings,
        verbose=args.verbose,
        emit_global_params=args.emit_global_params,
        global_param_order=global_param_order,
        global_param_references=global_param_references,
        sampling_cfg=_sampling_cfg_from_args(args) if args.sampling_weights else None,
    )

    jobs = max(1, args.jobs) if args.jobs else 1
    jobs = min(jobs, len(samples))

    written = skipped = failed = 0
    split_by_name = {s.name: s.split for s in samples}
    written_per_split: dict[str, int] = {}

    def _tally(result: tuple[str, str, str | None]) -> None:
        nonlocal written, skipped, failed
        name, status, error = result
        if status == "ok":
            written += 1
            split = split_by_name.get(name)
            if split is not None:
                written_per_split[split] = written_per_split.get(split, 0) + 1
            logger.debug("[%s] wrote", name)
        elif status == "skipped":
            skipped += 1
            logger.warning("[%s] produced no arrays -- skipping", name)
        else:
            failed += 1
            logger.error("[%s] failed: %s", name, error)

    if jobs == 1:
        for sample in tqdm(samples, desc="converting", unit="sample"):
            _tally(_process_sample(sample, opts))
    else:
        logger.info("Converting with %d parallel worker(s)", jobs)
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(_process_sample, s, opts) for s in samples]
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="converting",
                unit="sample",
            ):
                _tally(fut.result())

    logger.info("Done: %d written, %d skipped, %d failed", written, skipped, failed)
    if written_per_split:
        per_split = ", ".join(
            f"{name}={written_per_split.get(name, 0)}" for name in args.split_names
        )
        logger.info("Written per split: %s", per_split)
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
