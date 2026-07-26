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
Standalone converter from Tecplot and CGNS surface meshes to VTP (``.vtp``).

This script is intentionally self-contained: it has no dependency on the rest
of this repository and can be copied out and run anywhere. It supports:

* Tecplot ASCII files (``.dat``) via VTK's ``vtkTecplotReader`` (no license
  required).
* Tecplot binary files (``.plt``) via ``pytecplot`` (imported lazily, only when
  a ``.plt`` file is encountered; requires a Tecplot 360 installation/license).
* CGNS files (``.cgns``) via PyVista's CGNS reader; all blocks/zones are merged
  into a single surface.

Field data (e.g. pressure, wall shear stress) is carried over automatically. By
default every point/cell array is preserved; pass ``--fields`` to keep only a
chosen subset.

Examples
--------
    python convert_to_vtp.py INPUT [-o OUTPUT] [--fields ...] [-j WORKERS] [-r] [--overwrite]
    
Convert a single file::

    python convert_to_vtp.py mesh.cgns --output mesh.vtp

Batch-convert every supported file in a directory tree, in parallel::

    python convert_to_vtp.py ./inputs --output ./outputs --recursive --workers 4

Keep only selected fields::

    python convert_to_vtp.py mesh.dat --fields Pressure WallShearStress
    
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pyvista as pv
import vtk

# Extensions recognized by this converter, mapped to a human-readable format id.
SUPPORTED_EXTENSIONS = {
    ".plt": "tecplot_binary",
    ".dat": "tecplot_ascii",
    ".cgns": "cgns",
}


def detect_format(path: Path) -> str:
    """Return the converter format id for ``path`` based on its extension.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    """
    ext = path.suffix.lower()
    try:
        return SUPPORTED_EXTENSIONS[ext]
    except KeyError as exc:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"Unsupported file extension '{ext}' for '{path}'. "
            f"Supported extensions: {supported}."
        ) from exc


def read_cgns(path: Path) -> pv.DataSet:
    """Read a CGNS file and merge all of its blocks into a single mesh."""
    dataset = pv.read(str(path))
    if isinstance(dataset, pv.MultiBlock):
        merged = dataset.combine(merge_points=False)
        return merged
    return dataset


def read_tecplot_ascii(path: Path) -> pv.DataSet:
    """Read an ASCII Tecplot file using VTK's ``vtkTecplotReader``.

    The reader returns a multiblock dataset; all blocks are combined into a
    single mesh so downstream handling matches the other readers.
    """
    reader = vtk.vtkTecplotReader()
    reader.SetFileName(str(path))
    reader.Update()

    output = pv.wrap(reader.GetOutput())
    if isinstance(output, pv.MultiBlock):
        return output.combine(merge_points=False)
    return output


def read_tecplot_binary(path: Path) -> pv.DataSet:
    """Read a binary Tecplot file (``.plt``) via ``pytecplot``.

    ``pytecplot`` is imported lazily so that this script keeps working for CGNS
    and ASCII Tecplot inputs even when Tecplot is not installed or licensed.

    Only finite-element and ordered surface zones are supported. Every Tecplot
    variable other than the spatial coordinates is attached as a point array.
    """
    try:
        import tecplot
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Reading binary Tecplot (.plt) files requires the 'pytecplot' "
            "package and a Tecplot 360 installation/license. Install it with "
            "'pip install pytecplot', or pre-convert the file to ASCII (.dat)."
        ) from exc

    dataset = tecplot.data.load_tecplot(str(path))
    return _tecplot_dataset_to_pyvista(dataset)


def _tecplot_dataset_to_pyvista(dataset) -> pv.DataSet:
    """Convert a loaded ``pytecplot`` dataset into a merged PyVista mesh."""
    from tecplot.constant import ZoneType

    variable_names = [v.name for v in dataset.variables()]
    coord_names = _identify_coordinate_variables(variable_names)
    field_names = [n for n in variable_names if n not in coord_names]

    blocks = pv.MultiBlock()
    for zone in dataset.zones():
        points = np.column_stack(
            [np.asarray(zone.values(name)[:], dtype=np.float64) for name in coord_names]
        )
        if points.shape[1] == 2:
            points = np.column_stack([points, np.zeros(len(points))])

        if zone.zone_type == ZoneType.Ordered:
            mesh = _build_ordered_surface(zone, points)
        else:
            mesh = _build_fe_surface(zone, points)

        for name in field_names:
            values = np.asarray(zone.values(name)[:], dtype=np.float64)
            if len(values) == mesh.n_points:
                mesh.point_data[name] = values
            elif len(values) == mesh.n_cells:
                mesh.cell_data[name] = values

        blocks.append(mesh)

    if len(blocks) == 1:
        return blocks[0]
    return blocks.combine(merge_points=False)


def _identify_coordinate_variables(variable_names: Sequence[str]) -> list[str]:
    """Pick the spatial-coordinate variables from a Tecplot variable list.

    Tecplot does not tag coordinates explicitly, so we match common names and
    otherwise fall back to the first three variables.
    """
    lookup = {n.lower(): n for n in variable_names}
    coords: list[str] = []
    for axis in ("x", "y", "z"):
        for candidate in (axis, f"coordinate{axis}", f"{axis}coordinate"):
            if candidate in lookup:
                coords.append(lookup[candidate])
                break
    if len(coords) >= 2:
        return coords
    # Fallback: assume the first variables are the coordinates.
    return list(variable_names[: min(3, len(variable_names))])


def _build_fe_surface(zone, points: np.ndarray) -> pv.PolyData:
    """Build a PolyData surface from a finite-element Tecplot zone."""
    from tecplot.constant import ZoneType

    nodemap = np.asarray(zone.nodemap[:], dtype=np.int64)
    if zone.zone_type == ZoneType.FETriangle:
        nodes_per_cell = 3
    elif zone.zone_type == ZoneType.FEQuad:
        nodes_per_cell = 4
    else:
        raise ValueError(
            f"Unsupported Tecplot surface zone type '{zone.zone_type}'. "
            "Only FETriangle and FEQuad zones are supported."
        )

    connectivity = nodemap.reshape(-1, nodes_per_cell)
    counts = np.full((len(connectivity), 1), nodes_per_cell, dtype=np.int64)
    faces = np.hstack([counts, connectivity]).ravel()
    return pv.PolyData(points, faces)


def _build_ordered_surface(zone, points: np.ndarray) -> pv.DataSet:
    """Build a structured surface from an ordered (I/J/K) Tecplot zone."""
    imax, jmax, kmax = zone.dimensions
    grid = pv.StructuredGrid()
    grid.points = points
    grid.dimensions = [max(imax, 1), max(jmax, 1), max(kmax, 1)]
    return grid.extract_surface()


def to_polydata(mesh: pv.DataSet, fields: Optional[Sequence[str]] = None) -> pv.PolyData:
    """Normalize any mesh to surface ``PolyData`` and select field arrays.

    Parameters
    ----------
    mesh:
        The mesh returned by one of the readers.
    fields:
        If given, only these point/cell arrays are kept. When ``None`` (the
        default) every array is preserved.
    """
    if isinstance(mesh, pv.PolyData):
        surface = mesh
    else:
        surface = _extract_surface(mesh)

    if fields is not None:
        _filter_fields(surface, fields)

    return surface


def _extract_surface(mesh: pv.DataSet) -> pv.PolyData:
    """Extract the external surface as PolyData without adding id arrays.

    ``pass_pointid``/``pass_cellid`` are disabled so the output only carries the
    original field data, and ``algorithm`` is set explicitly to silence a
    PyVista ``FutureWarning`` when available.
    """
    try:
        return mesh.extract_surface(
            pass_pointid=False, pass_cellid=False, algorithm="dataset_surface"
        )
    except TypeError:
        # Older PyVista versions without the ``algorithm`` keyword.
        return mesh.extract_surface(pass_pointid=False, pass_cellid=False)


def _filter_fields(surface: pv.PolyData, fields: Sequence[str]) -> None:
    """Drop every point/cell array on ``surface`` not present in ``fields``."""
    keep = set(fields)
    for name in list(surface.point_data.keys()):
        if name not in keep:
            del surface.point_data[name]
    for name in list(surface.cell_data.keys()):
        if name not in keep:
            del surface.cell_data[name]

    missing = keep - set(surface.point_data.keys()) - set(surface.cell_data.keys())
    if missing:
        print(
            f"  Warning: requested fields not found and skipped: "
            f"{', '.join(sorted(missing))}",
            file=sys.stderr,
        )


def write_vtp(surface: pv.PolyData, out_path: Path) -> None:
    """Write ``surface`` to ``out_path`` as a binary VTP file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    surface.save(str(out_path), binary=True)


_READERS = {
    "tecplot_binary": read_tecplot_binary,
    "tecplot_ascii": read_tecplot_ascii,
    "cgns": read_cgns,
}


def convert_file(
    in_path: Path,
    out_path: Path,
    fields: Optional[Sequence[str]] = None,
    overwrite: bool = False,
) -> str:
    """Convert a single input file to VTP.

    Returns a short human-readable status string describing the outcome.
    """
    if out_path.exists() and not overwrite:
        return f"Skipped (exists): {out_path}"

    fmt = detect_format(in_path)
    mesh = _READERS[fmt](in_path)
    surface = to_polydata(mesh, fields)
    write_vtp(surface, out_path)
    return (
        f"Converted: {in_path} -> {out_path} "
        f"({surface.n_points} points, {surface.n_cells} cells)"
    )


def _output_path_for(
    in_path: Path, input_root: Path, output_root: Path, recursive: bool
) -> Path:
    """Compute the ``.vtp`` output path for a given input file."""
    if recursive:
        relative = in_path.relative_to(input_root).with_suffix(".vtp")
        return output_root / relative
    return output_root / in_path.with_suffix(".vtp").name


def _iter_input_files(input_root: Path, recursive: bool) -> Iterable[Path]:
    """Yield supported input files under ``input_root``."""
    pattern = "**/*" if recursive else "*"
    for path in sorted(input_root.glob(pattern)):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def _convert_file_task(args: tuple) -> str:
    """Worker wrapper for :func:`convert_file` usable with process pools."""
    in_path, out_path, fields, overwrite = args
    try:
        return convert_file(in_path, out_path, fields, overwrite)
    except Exception as exc:  # noqa: BLE001 - report per-file failures, keep going
        return f"Failed: {in_path} ({exc})"


def _run_batch(
    tasks: list[tuple], workers: int
) -> list[str]:
    """Execute conversion ``tasks`` either serially or across a process pool."""
    if workers <= 1 or len(tasks) <= 1:
        return [_convert_file_task(task) for task in _with_progress(tasks)]

    results: list[str] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_convert_file_task, task) for task in tasks]
        for future in _with_progress(as_completed(futures), total=len(futures)):
            results.append(future.result())
    return results


def _with_progress(iterable, total: Optional[int] = None):
    """Wrap ``iterable`` with a tqdm progress bar when tqdm is available."""
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total)
    except ImportError:
        return iterable


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert Tecplot (.plt/.dat) and CGNS (.cgns) surface "
        "meshes to VTP (.vtp).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input file or a directory to batch-convert.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output .vtp path (for a single file) or output directory (for a "
        "directory input). Defaults next to the input file/directory.",
    )
    parser.add_argument(
        "--fields",
        nargs="*",
        default=None,
        help="Only keep these point/cell field arrays. Default: keep all.",
    )
    parser.add_argument(
        "--workers",
        "-j",
        type=int,
        default=1,
        help="Number of parallel worker processes for directory batches.",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Recurse into subdirectories when the input is a directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .vtp outputs instead of skipping them.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    input_path: Path = args.input

    if not input_path.exists():
        print(f"Error: input path does not exist: {input_path}", file=sys.stderr)
        return 1

    if input_path.is_file():
        out_path = args.output or input_path.with_suffix(".vtp")
        if out_path.is_dir():
            out_path = out_path / input_path.with_suffix(".vtp").name
        print(_convert_file_task((input_path, out_path, args.fields, args.overwrite)))
        return 0

    # Directory batch.
    output_root = args.output or input_path
    tasks = [
        (
            in_path,
            _output_path_for(in_path, input_path, output_root, args.recursive),
            args.fields,
            args.overwrite,
        )
        for in_path in _iter_input_files(input_path, args.recursive)
    ]

    if not tasks:
        print(f"No supported input files found in {input_path}.", file=sys.stderr)
        return 1

    results = _run_batch(tasks, args.workers)
    for line in results:
        print(line)

    failures = sum(1 for line in results if line.startswith("Failed:"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
