#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``vtk_to_zarr.py``.

Run inside the PhysicsNeMo container (where pyvista, numpy and zarr are
available)::

    python3 -m unittest test_vtk_to_zarr -v
    # or
    python3 test_vtk_to_zarr.py

Tests that need PyVista or Zarr are skipped automatically when those packages
are not importable, so the pure-Python logic tests still run anywhere numpy is
installed.
"""

from __future__ import annotations

import importlib.util
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# --------------------------------------------------------------------------- #
# Import the module under test, tolerating a missing optional dependency.
# --------------------------------------------------------------------------- #
_HERE = Path(__file__).resolve().parent
_MODULE_PATH = _HERE / "vtk_to_zarr.py"

try:
    import numpy as np  # noqa: F401

    HAVE_NUMPY = True
except ImportError:  # pragma: no cover
    HAVE_NUMPY = False

try:
    import pyvista as pv  # noqa: F401

    HAVE_PYVISTA = True
except Exception:  # noqa: BLE001 - pyvista import can raise beyond ImportError
    HAVE_PYVISTA = False

try:
    import zarr  # noqa: F401

    HAVE_ZARR = True
except ImportError:  # pragma: no cover
    HAVE_ZARR = False


def _load_module():
    """Load vtk_to_zarr.py, stubbing pyvista if it is unavailable.

    The module raises ``SystemExit`` at import time when pyvista is missing.
    For pure-logic tests we inject a minimal stub so the module imports; the
    pyvista-dependent tests are skipped separately.
    """
    import sys
    import types

    if not HAVE_PYVISTA:
        stub = types.ModuleType("pyvista")
        stub.PolyData = type("PolyData", (), {})
        stub.DataSet = type("DataSet", (), {})
        sys.modules.setdefault("pyvista", stub)

    spec = importlib.util.spec_from_file_location("vtk_to_zarr", str(_MODULE_PATH))
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass annotation resolution can find the
    # module via sys.modules[cls.__module__] (required on CPython 3.12+).
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


vz = _load_module()


# --------------------------------------------------------------------------- #
# Pure-Python logic (no pyvista / zarr required)
# --------------------------------------------------------------------------- #
class TestClassifyFile(unittest.TestCase):
    def test_by_extension(self):
        self.assertEqual(vz.classify_file(Path("drivaer_1.stl")), vz.Role.STL)
        self.assertEqual(vz.classify_file(Path("boundary_1.vtp")), vz.Role.SURFACE)
        self.assertEqual(vz.classify_file(Path("volume_1.vtu")), vz.Role.VOLUME)

    def test_vtu_surface_keyword_override(self):
        self.assertEqual(vz.classify_file(Path("boundary_1.vtu")), vz.Role.SURFACE)
        self.assertEqual(vz.classify_file(Path("surface.vtu")), vz.Role.SURFACE)

    def test_vtk_keyword_resolution(self):
        self.assertEqual(vz.classify_file(Path("internal.vtk")), vz.Role.VOLUME)
        self.assertEqual(vz.classify_file(Path("wall.vtk")), vz.Role.SURFACE)
        # Unknown .vtk defaults to surface.
        self.assertEqual(vz.classify_file(Path("mesh.vtk")), vz.Role.SURFACE)

    def test_unknown_extension(self):
        self.assertIsNone(vz.classify_file(Path("notes.txt")))
        self.assertIsNone(vz.classify_file(Path("data.csv")))


class TestExtractId(unittest.TestCase):
    def test_default_trailing_integer(self):
        self.assertEqual(vz._extract_id("drivaer_12", None), "12")
        self.assertEqual(vz._extract_id("run_7_volume_3", None), "3")
        self.assertIsNone(vz._extract_id("mesh", None))

    def test_named_group_regex(self):
        rx = re.compile(r"run_(?P<id>\d+)")
        self.assertEqual(vz._extract_id("run_5_volume", rx), "5")

    def test_regex_no_named_group_uses_full_match(self):
        rx = re.compile(r"case\d+")
        self.assertEqual(vz._extract_id("case42_surface", rx), "case42")


class TestParseFieldSelection(unittest.TestCase):
    def test_default_and_none(self):
        self.assertIs(vz.parse_field_selection(None, vz.DEFAULT_SURFACE_FIELDS), vz.DEFAULT_SURFACE_FIELDS)
        self.assertIs(vz.parse_field_selection("default", vz.DEFAULT_SURFACE_FIELDS), vz.DEFAULT_SURFACE_FIELDS)

    def test_all_returns_none(self):
        self.assertIsNone(vz.parse_field_selection("all", vz.DEFAULT_SURFACE_FIELDS))

    def test_named_fields_resolve_to_specs(self):
        specs = vz.parse_field_selection("pMean,U", vz.DEFAULT_VOLUME_FIELDS)
        self.assertEqual([s.canonical for s in specs], ["pressure", "velocity"])

    def test_unknown_name_becomes_scalar_spec(self):
        specs = vz.parse_field_selection("myCustomVar", vz.DEFAULT_VOLUME_FIELDS)
        self.assertEqual(specs[0].canonical, "myCustomVar")
        self.assertEqual(specs[0].components, 1)


class TestSplitCounts(unittest.TestCase):
    def test_counts_sum_to_total(self):
        for total in (0, 1, 3, 7, 10, 101, 999):
            counts = vz._split_counts(total, (0.8, 0.1, 0.1))
            self.assertEqual(sum(counts), total)
            self.assertTrue(all(c >= 0 for c in counts))

    def test_percentages_and_fractions_equivalent(self):
        self.assertEqual(
            vz._split_counts(100, (80, 10, 10)),
            vz._split_counts(100, (0.8, 0.1, 0.1)),
        )

    def test_default_80_10_10_on_100(self):
        self.assertEqual(vz._split_counts(100, (0.8, 0.1, 0.1)), [80, 10, 10])

    def test_largest_remainder_assignment(self):
        # total=10 with equal thirds -> 4/3/3 (leftover to first by order).
        self.assertEqual(vz._split_counts(10, (1, 1, 1)), [4, 3, 3])

    def test_zero_ratio_gets_no_samples(self):
        counts = vz._split_counts(10, (0.5, 0.5, 0.0))
        self.assertEqual(counts, [5, 5, 0])

    def test_nonpositive_weight_raises(self):
        with self.assertRaises(ValueError):
            vz._split_counts(10, (0.0, 0.0, 0.0))


class TestAssignSplits(unittest.TestCase):
    def _samples(self, n):
        return [vz.Sample(name=str(i)) for i in range(n)]

    def test_assigns_all_and_matches_counts(self):
        samples = self._samples(10)
        vz.assign_splits(samples, (0.8, 0.1, 0.1), ("train", "val", "test"), seed=42)
        counts = {"train": 0, "val": 0, "test": 0}
        for s in samples:
            self.assertIsNotNone(s.split)
            counts[s.split] += 1
        self.assertEqual(counts, {"train": 8, "val": 1, "test": 1})

    def test_reproducible_with_same_seed(self):
        a = self._samples(50)
        b = self._samples(50)
        vz.assign_splits(a, (0.8, 0.1, 0.1), ("train", "val", "test"), seed=123)
        vz.assign_splits(b, (0.8, 0.1, 0.1), ("train", "val", "test"), seed=123)
        self.assertEqual(
            {s.name: s.split for s in a},
            {s.name: s.split for s in b},
        )

    def test_different_seed_changes_assignment(self):
        a = self._samples(50)
        b = self._samples(50)
        vz.assign_splits(a, (0.8, 0.1, 0.1), ("train", "val", "test"), seed=1)
        vz.assign_splits(b, (0.8, 0.1, 0.1), ("train", "val", "test"), seed=2)
        self.assertNotEqual(
            {s.name: s.split for s in a},
            {s.name: s.split for s in b},
        )

    def test_order_independent(self):
        a = self._samples(20)
        b = list(reversed(self._samples(20)))
        vz.assign_splits(a, (0.7, 0.2, 0.1), ("train", "val", "test"), seed=7)
        vz.assign_splits(b, (0.7, 0.2, 0.1), ("train", "val", "test"), seed=7)
        self.assertEqual(
            {s.name: s.split for s in a},
            {s.name: s.split for s in b},
        )

    def test_ratio_name_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            vz.assign_splits(self._samples(3), (0.5, 0.5), ("train", "val", "test"), seed=0)


@unittest.skipUnless(HAVE_NUMPY, "numpy required")
class TestComputeChunks(unittest.TestCase):
    def test_1d(self):
        self.assertEqual(vz._compute_chunks((1000,), 4, 8.0), (1000,))

    def test_2d_preserves_trailing_dims(self):
        chunks = vz._compute_chunks((10_000_000, 3), 4, 1.0)
        self.assertEqual(chunks[1], 3)
        self.assertGreaterEqual(chunks[0], 1)
        self.assertLessEqual(chunks[0], 10_000_000)


# --------------------------------------------------------------------------- #
# PyVista-backed mesh reading
# --------------------------------------------------------------------------- #
@unittest.skipUnless(HAVE_PYVISTA and HAVE_NUMPY, "pyvista + numpy required")
class TestMeshReading(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _surface_with_cell_fields(self):
        sphere = pv.Sphere(theta_resolution=12, phi_resolution=12).triangulate()
        n = sphere.n_cells
        sphere.cell_data["pMean"] = np.random.rand(n).astype(np.float32)
        sphere.cell_data["wallShearStress"] = np.random.rand(n, 3).astype(np.float32)
        return sphere

    def _surface_with_point_fields(self):
        sphere = pv.Sphere(theta_resolution=12, phi_resolution=12).triangulate()
        n = sphere.n_points
        sphere.point_data["Pressure"] = np.random.rand(n).astype(np.float32)
        return sphere

    def test_read_stl_geometry(self):
        path = self.dir / "geom.stl"
        pv.Sphere(theta_resolution=12, phi_resolution=12).triangulate().save(str(path))
        out = vz.read_stl(path)
        m = out["stl_coordinates"].shape[0]
        f = out["stl_faces"].shape[0]
        c = out["stl_centers"].shape[0]
        a = out["stl_areas"].shape[0]
        self.assertEqual(out["stl_coordinates"].shape[1], 3)
        self.assertEqual(out["stl_centers"].shape[1], 3)
        self.assertEqual(f, 3 * c)  # flattened triangle indices
        self.assertEqual(a, c)
        self.assertEqual(out["stl_faces"].dtype, np.int32)
        self.assertTrue((out["stl_faces"] < m).all())

    def test_read_surface_cell_fields(self):
        path = self.dir / "boundary_1.vtp"
        self._surface_with_cell_fields().save(str(path))
        out = vz.read_surface(path, vz.DEFAULT_SURFACE_FIELDS, "auto")
        ns = out["surface_mesh_centers"].shape[0]
        self.assertEqual(out["surface_mesh_centers"].shape[1], 3)
        self.assertEqual(out["surface_normals"].shape, (ns, 3))
        self.assertEqual(out["surface_areas"].shape, (ns,))
        # pressure (1) + wallShearStress (3) = 4
        self.assertEqual(out["surface_fields"].shape, (ns, 4))
        # normals are unit length
        norms = np.linalg.norm(out["surface_normals"], axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-4))

    def test_read_surface_point_fields_su2_style(self):
        path = self.dir / "surface_su2.vtp"
        self._surface_with_point_fields().save(str(path))
        # Request pressure only; data lives on points -> auto must convert to cell.
        specs = vz.parse_field_selection("Pressure", vz.DEFAULT_SURFACE_FIELDS)
        out = vz.read_surface(path, specs, "auto")
        ns = out["surface_mesh_centers"].shape[0]
        self.assertIn("surface_fields", out)
        self.assertEqual(out["surface_fields"].shape, (ns, 1))

    def test_read_volume_point_fields(self):
        grid = pv.ImageData(dimensions=(6, 6, 6))
        n = grid.n_points
        grid.point_data["UMean"] = np.random.rand(n, 3).astype(np.float32)
        grid.point_data["pMean"] = np.random.rand(n).astype(np.float32)
        grid.point_data["nutMean"] = np.random.rand(n).astype(np.float32)
        path = self.dir / "volume_1.vtu"
        grid.cast_to_unstructured_grid().save(str(path))

        out = vz.read_volume(path, vz.DEFAULT_VOLUME_FIELDS, "point")
        nv = out["volume_mesh_centers"].shape[0]
        self.assertEqual(out["volume_mesh_centers"].shape[1], 3)
        # velocity (3) + pressure (1) + nut (1) = 5
        self.assertEqual(out["volume_fields"].shape, (nv, 5))

    def test_all_fields_selection(self):
        path = self.dir / "boundary_all.vtp"
        self._surface_with_cell_fields().save(str(path))
        out = vz.read_surface(path, None, "cell")  # None == "all"
        ns = out["surface_mesh_centers"].shape[0]
        # pMean (1) + wallShearStress (3) = 4 columns
        self.assertEqual(out["surface_fields"].shape, (ns, 4))


# --------------------------------------------------------------------------- #
# Geometry synthesis priority: surface preferred over volume
# --------------------------------------------------------------------------- #
@unittest.skipUnless(HAVE_PYVISTA and HAVE_NUMPY, "pyvista + numpy required")
class TestGeometrySynthesis(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _write_surface(self):
        sphere = pv.Sphere(theta_resolution=12, phi_resolution=12).triangulate()
        sphere.cell_data["pMean"] = np.random.rand(sphere.n_cells).astype(np.float32)
        path = self.dir / "boundary_9.vtp"
        sphere.save(str(path))
        return path

    def _write_volume(self):
        grid = pv.ImageData(dimensions=(6, 6, 6))
        grid.point_data["pMean"] = np.random.rand(grid.n_points).astype(np.float32)
        path = self.dir / "volume_9.vtu"
        grid.cast_to_unstructured_grid().save(str(path))
        return path

    def test_prefers_surface_over_volume_for_synthesis(self):
        sample = vz.Sample(
            name="9",
            stl=None,
            surface=self._write_surface(),
            volume=self._write_volume(),
        )
        arrays, attrs = vz.build_sample(
            sample,
            surface_specs=vz.DEFAULT_SURFACE_FIELDS,
            volume_specs=vz.DEFAULT_VOLUME_FIELDS,
            surface_location="auto",
            volume_location="auto",
            synth_stl=True,
        )
        self.assertEqual(attrs["stl_origin"], "synthesized_from_surface")
        self.assertIn("stl_coordinates", arrays)
        self.assertIn("surface_mesh_centers", arrays)

    def test_falls_back_to_volume_when_no_surface(self):
        sample = vz.Sample(name="9", stl=None, surface=None, volume=self._write_volume())
        _arrays, attrs = vz.build_sample(
            sample,
            surface_specs=vz.DEFAULT_SURFACE_FIELDS,
            volume_specs=vz.DEFAULT_VOLUME_FIELDS,
            surface_location="auto",
            volume_location="auto",
            synth_stl=True,
        )
        self.assertEqual(attrs["stl_origin"], "synthesized_from_volume")

    def test_no_synth_when_disabled(self):
        sample = vz.Sample(name="9", stl=None, surface=None, volume=self._write_volume())
        arrays, attrs = vz.build_sample(
            sample,
            surface_specs=vz.DEFAULT_SURFACE_FIELDS,
            volume_specs=vz.DEFAULT_VOLUME_FIELDS,
            surface_location="auto",
            volume_location="auto",
            synth_stl=False,
        )
        self.assertNotIn("stl_origin", attrs)
        self.assertNotIn("stl_coordinates", arrays)


# --------------------------------------------------------------------------- #
# Zarr round-trip
# --------------------------------------------------------------------------- #
@unittest.skipUnless(HAVE_PYVISTA and HAVE_NUMPY and HAVE_ZARR, "pyvista + numpy + zarr required")
class TestZarrRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_write_and_reload(self):
        # Build a full sample: stl + surface + volume.
        sphere = pv.Sphere(theta_resolution=12, phi_resolution=12).triangulate()
        sphere.save(str(self.dir / "drivaer_3.stl"))

        surf = pv.Sphere(theta_resolution=12, phi_resolution=12).triangulate()
        surf.cell_data["pMean"] = np.random.rand(surf.n_cells).astype(np.float32)
        surf.cell_data["wallShearStress"] = np.random.rand(surf.n_cells, 3).astype(np.float32)
        surf.save(str(self.dir / "boundary_3.vtp"))

        grid = pv.ImageData(dimensions=(6, 6, 6))
        grid.point_data["UMean"] = np.random.rand(grid.n_points, 3).astype(np.float32)
        grid.point_data["pMean"] = np.random.rand(grid.n_points).astype(np.float32)
        grid.point_data["nutMean"] = np.random.rand(grid.n_points).astype(np.float32)
        grid.cast_to_unstructured_grid().save(str(self.dir / "volume_3.vtu"))

        samples = vz.discover_samples(self.dir, "*", None, group=True)
        self.assertEqual(len(samples), 1)
        sample = samples[0]

        arrays, attrs = vz.build_sample(
            sample,
            surface_specs=vz.DEFAULT_SURFACE_FIELDS,
            volume_specs=vz.DEFAULT_VOLUME_FIELDS,
            surface_location="auto",
            volume_location="auto",
            synth_stl=True,
        )
        out_dir = self.dir / "out"
        path = vz.write_zarr(
            sample.name, arrays, attrs, out_dir,
            compression_level=3, chunk_mb=8.0, overwrite=True,
        )
        self.assertTrue(path.exists())

        g = zarr.open_group(str(path), mode="r")
        keys = set(g.array_keys())
        for expected in (
            "stl_coordinates", "stl_faces", "stl_centers", "stl_areas",
            "surface_mesh_centers", "surface_normals", "surface_areas", "surface_fields",
            "volume_mesh_centers", "volume_fields",
        ):
            self.assertIn(expected, keys)

        # Axis-0 alignment invariants.
        ns = g["surface_mesh_centers"].shape[0]
        self.assertEqual(g["surface_normals"].shape[0], ns)
        self.assertEqual(g["surface_areas"].shape[0], ns)
        self.assertEqual(g["surface_fields"].shape, (ns, 4))

        nv = g["volume_mesh_centers"].shape[0]
        self.assertEqual(g["volume_fields"].shape, (nv, 5))

        m = g["stl_centers"].shape[0]
        self.assertEqual(g["stl_areas"].shape[0], m)
        self.assertEqual(g["stl_faces"].shape[0], 3 * m)

        # Attributes round-trip.
        self.assertEqual(g.attrs["sample_name"], "3")
        self.assertEqual(g.attrs["stl_origin"], "file")

    def test_overwrite_guard(self):
        arrays = {"volume_mesh_centers": np.zeros((4, 3), dtype=np.float32)}
        out_dir = self.dir / "out2"
        vz.write_zarr("s", arrays, {}, out_dir, compression_level=3, chunk_mb=1.0, overwrite=False)
        with self.assertRaises(FileExistsError):
            vz.write_zarr("s", arrays, {}, out_dir, compression_level=3, chunk_mb=1.0, overwrite=False)
        # Overwrite succeeds.
        vz.write_zarr("s", arrays, {}, out_dir, compression_level=3, chunk_mb=1.0, overwrite=True)


# --------------------------------------------------------------------------- #
# Performance-oriented behaviour: read de-duplication and parallel batch
# --------------------------------------------------------------------------- #
@unittest.skipUnless(HAVE_PYVISTA and HAVE_NUMPY, "pyvista + numpy required")
class TestReadDeduplication(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_surface_parsed_once_during_synthesis(self):
        sphere = pv.Sphere(theta_resolution=12, phi_resolution=12).triangulate()
        sphere.cell_data["pMean"] = np.random.rand(sphere.n_cells).astype(np.float32)
        path = self.dir / "boundary_1.vtp"
        sphere.save(str(path))

        sample = vz.Sample(name="1", surface=path)
        real_read = vz.pv.read
        calls: list[str] = []

        def counting_read(arg, *a, **k):
            calls.append(str(arg))
            return real_read(arg, *a, **k)

        with mock.patch.object(vz.pv, "read", side_effect=counting_read):
            arrays, attrs = vz.build_sample(
                sample,
                surface_specs=vz.DEFAULT_SURFACE_FIELDS,
                volume_specs=vz.DEFAULT_VOLUME_FIELDS,
                surface_location="auto",
                volume_location="auto",
                synth_stl=True,
            )

        # The surface file feeds both geometry synthesis and field extraction,
        # but must be parsed exactly once.
        self.assertEqual(sum(str(path) in c for c in calls), 1)
        self.assertEqual(attrs["stl_origin"], "synthesized_from_surface")
        self.assertIn("stl_coordinates", arrays)
        self.assertIn("surface_fields", arrays)


@unittest.skipUnless(HAVE_PYVISTA and HAVE_NUMPY and HAVE_ZARR, "pyvista + numpy + zarr required")
class TestParallelBatch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.out = self.dir / "out"

    def tearDown(self):
        self.tmp.cleanup()

    def _write_surface_sample(self, idx: int):
        surf = pv.Sphere(theta_resolution=12, phi_resolution=12).triangulate()
        surf.cell_data["pMean"] = np.random.rand(surf.n_cells).astype(np.float32)
        surf.save(str(self.dir / f"boundary_{idx}.vtp"))

    def test_parallel_batch_writes_all_samples(self):
        for i in (1, 2, 3):
            self._write_surface_sample(i)
        rc = vz.main(
            [
                "--input-dir", str(self.dir),
                "--output-dir", str(self.out),
                "--jobs", "2",
                "--no-split",
            ]
        )
        self.assertEqual(rc, 0)
        for i in (1, 2, 3):
            self.assertTrue((self.out / f"{i}.zarr").exists())

    def test_batch_writes_split_subfolders(self):
        for i in range(1, 11):
            self._write_surface_sample(i)
        rc = vz.main(
            [
                "--input-dir", str(self.dir),
                "--output-dir", str(self.out),
                "--split", "80", "10", "10",
                "--split-seed", "42",
            ]
        )
        self.assertEqual(rc, 0)
        train = list((self.out / "train").glob("*.zarr"))
        val = list((self.out / "val").glob("*.zarr"))
        test = list((self.out / "test").glob("*.zarr"))
        self.assertEqual(len(train), 8)
        self.assertEqual(len(val), 1)
        self.assertEqual(len(test), 1)
        # No flat stores written when splitting is enabled.
        self.assertEqual(list(self.out.glob("*.zarr")), [])

    def test_process_sample_isolates_errors(self):
        opts = vz.ConvertOptions(
            surface_specs=vz.DEFAULT_SURFACE_FIELDS,
            volume_specs=vz.DEFAULT_VOLUME_FIELDS,
            surface_location="auto",
            volume_location="auto",
            synth_stl=True,
            output_dir=self.out,
            compression_level=3,
            chunk_mb=1.0,
            overwrite=True,
        )
        # A sample pointing at a nonexistent file fails without raising.
        bad = vz.Sample(name="bad", surface=self.dir / "does_not_exist.vtp")
        name, status, error = vz._process_sample(bad, opts)
        self.assertEqual(name, "bad")
        self.assertEqual(status, "failed")
        self.assertIsNotNone(error)


if __name__ == "__main__":
    unittest.main(verbosity=2)
