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
Aerodynamic coefficient integration for pre-normalized surface fields.

This module computes total aerodynamic force/moment vectors and the resulting
non-dimensional aerodynamic coefficients (lift, drag, side force, and the three
moments) by integrating surface pressure and wall-shear-stress fields over a
point cloud / surface mesh.

The physical fields are assumed to be *already non-dimensionalized* (the pressure
field is treated as a pressure coefficient ``Cp`` and the wall shear stress is
treated as a normalized skin-friction vector), so the engine only performs the
geometric integration and coordinate transformations.

Two entry points are exposed by :class:`AerodynamicCoefficients`:

* :meth:`AerodynamicCoefficients.calculate_from_vtp` -- read a VTK PolyData
  (``.vtp``) file with ``pyvista`` and integrate its point-data arrays.
* :meth:`AerodynamicCoefficients.calculate_from_dict` -- integrate pre-extracted
  NumPy arrays supplied in a dictionary (no file I/O).

Coordinate convention (automotive / DrivAer):

* ``x`` -- streamwise (drag) direction,
* ``y`` -- lateral (side-force) direction,
* ``z`` -- vertical (lift) direction, positive up.

The angle of attack ``alpha`` is a rotation about the ``y`` axis and the sideslip
``beta`` is a rotation about the ``z`` axis. Two orientation conventions are
supported (see ``convention``):

* ``"alpha_beta"`` -- the user supplies the body-axis pitch angle of attack
  ``alpha`` and the sideslip angle ``beta`` directly.
* ``"alpha_phi"``  -- aeroballistic form where the user supplies the *total*
  angle of attack ``alpha_total`` and the aerodynamic roll angle
  ``phi_aerodynamic``. The body-axis pitch/sideslip angles are recovered from
  ``tan(alpha_pitch) = tan(alpha_total) * cos(phi)`` and
  ``tan(beta) = tan(alpha_total) * sin(phi)``.

``alpha`` (a body-axis pitch angle) and ``alpha_total`` (the angle between the
freestream and the body ``x`` axis) are different physical quantities, so they
are kept as separate parameters; supply the one that matches your ``convention``.

A Hydra entry point (``main``) is provided so the same computation can be driven
from ``conf/config3d.yaml`` and overridden from the command line, e.g.::

    python aero_coefficients.py aero_coefficients.input_file=point_clouds/pc.vtp \
        aero_coefficients.alpha=5 aero_coefficients.convention=alpha_phi

Usage
-----
The class works the same whether the fields are NumPy arrays or torch tensors
(tensors on the GPU or with ``requires_grad`` are detached and moved to the host
automatically).

1. From a ``.vtp`` file (configure the point-data array names if they differ
   from the defaults, e.g. ``pressure_key="pressure_pred"`` for model output)::

    from aero_coefficients import AerodynamicCoefficients

    calc = AerodynamicCoefficients(
        mrc=(1.0, 0.0, 0.0),        # moment reference center
        alpha=5.0,                  # body-axis angle of attack (deg)
        beta=2.0,                   # sideslip (deg)
        convention="alpha_beta",
        output_frame="both",        # "body", "wind" or "both"
        ref_area=2.17,              # reference area (1.0 if fields pre-normalized)
        ref_length=2.79,            # reference length for moment coefficients
    )
    results = calc.calculate_from_vtp("point_clouds/point_cloud_1/pc_0.vtp")
    print(results["wind_frame"]["coefficients"])   # {'CD':..., 'CL':..., ...}

2. From in-memory arrays or torch tensors (no file I/O)::

    import torch
    data = {
        "coordinates": coords,      # (N, 3) numpy array or torch tensor
        "normals": normals,         # (N, 3)
        "area": area,               # (N,) or (N, 1)
        "pressure": cp,             # (N,) or (N, 1) -- treated as Cp
        "shear_stress": tau,        # (N, 3) -- normalized skin friction
    }
    # Any parameter can also be overridden per call:
    results = calc.calculate_from_dict(
        data, convention="alpha_phi", alpha_total=6.0, phi_aerodynamic=30.0
    )

The returned dictionary has a ``"body_frame"`` and/or ``"wind_frame"`` block
(depending on ``output_frame``); each block exposes ``forces`` and ``moments``
(with ``"pressure"``/``"friction"``/``"total"`` vectors) and the named
``coefficients`` (plus ``coefficients_pressure`` and ``coefficients_friction``
for the split contributions). A ``"config"`` block echoes the settings used.
Every call also prints a summary table separating the pressure and friction
contributions.
"""

import glob
import os

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


# Valid selectors, kept as module-level constants so both the class and the
# Hydra entry point validate against the same set.
_CONVENTIONS = ("alpha_beta", "alpha_phi")
_OUTPUT_FRAMES = ("body", "wind", "both")


class AerodynamicCoefficients:
    """Integrate pre-normalized surface fields into aerodynamic coefficients.

    Parameters
    ----------
    mrc : sequence of float, optional
        Moment reference center ``(x, y, z)`` about which moments are taken.
        Defaults to the origin.
    alpha : float, optional
        Body-axis pitch angle of attack in degrees (used by the
        ``"alpha_beta"`` convention). Defaults to ``0.0``.
    beta : float, optional
        Sideslip angle in degrees (used by the ``"alpha_beta"`` convention).
        Defaults to ``0.0``.
    alpha_total : float, optional
        Total angle of attack in degrees, i.e. the angle between the freestream
        and the body ``x`` axis (used by the ``"alpha_phi"`` convention).
        Defaults to ``0.0``.
    phi_aerodynamic : float, optional
        Aerodynamic roll angle in degrees (used by the ``"alpha_phi"``
        convention). Defaults to ``0.0``.
    convention : str, optional
        Either ``"alpha_beta"`` or ``"alpha_phi"``. Defaults to ``"alpha_beta"``.
    output_frame : str, optional
        Which coefficient frame(s) to return: ``"body"``, ``"wind"`` or
        ``"both"``. Defaults to ``"both"``.
    ref_area : float, optional
        Reference area used to non-dimensionalize forces. Defaults to ``1.0``
        because the input fields are already normalized.
    ref_length : float, optional
        Reference length used (together with ``ref_area``) to non-dimensionalize
        moments. Defaults to ``1.0``.
    """

    def __init__(
        self,
        mrc=(0.0, 0.0, 0.0),
        alpha=0.0,
        beta=0.0,
        alpha_total=0.0,
        phi_aerodynamic=0.0,
        convention="alpha_beta",
        output_frame="both",
        ref_area=1.0,
        ref_length=1.0,
    ):
        self.mrc = np.asarray(mrc, dtype=np.float64).reshape(3)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.alpha_total = float(alpha_total)
        self.phi_aerodynamic = float(phi_aerodynamic)
        self.convention = self._validate_choice(
            convention, _CONVENTIONS, "convention"
        )
        self.output_frame = self._validate_choice(
            output_frame, _OUTPUT_FRAMES, "output_frame"
        )
        self.ref_area = float(ref_area)
        self.ref_length = float(ref_length)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def calculate_from_dict(self, data_dict, **overrides):
        """Compute coefficients from a dictionary of NumPy arrays.

        Parameters
        ----------
        data_dict : dict
            Mapping with the keys ``"coordinates"``, ``"normals"``, ``"area"``,
            ``"pressure"`` and ``"shear_stress"``. ``area`` and ``pressure`` may
            be provided as ``(N,)`` or ``(N, 1)``; the vector fields must be
            ``(N, 3)``.
        **overrides
            Optional per-call overrides for any of ``mrc``, ``alpha``, ``beta``,
            ``alpha_total``, ``phi_aerodynamic``, ``convention``,
            ``output_frame``, ``ref_area`` and ``ref_length``.

        Returns
        -------
        dict
            Structured results (see :meth:`_assemble_results`).
        """
        config = self._resolve_config(overrides)

        coordinates = self._as_vector(data_dict["coordinates"], "coordinates")
        normals = self._as_vector(data_dict["normals"], "normals")
        shear_stress = self._as_vector(data_dict["shear_stress"], "shear_stress")
        area = self._as_column(data_dict["area"], "area")
        pressure = self._as_column(data_dict["pressure"], "pressure")

        n = coordinates.shape[0]
        for name, arr in (
            ("normals", normals),
            ("shear_stress", shear_stress),
            ("area", area),
            ("pressure", pressure),
        ):
            if arr.shape[0] != n:
                raise ValueError(
                    f"Field '{name}' has {arr.shape[0]} points but 'coordinates' "
                    f"has {n}."
                )

        integrals = self._integrate(
            coordinates, normals, area, pressure, shear_stress, config
        )
        results = self._assemble_results(integrals, config)
        self._print_summary(results, config)
        return results

    def calculate_from_vtp(
        self,
        file_path,
        coordinates_key="coordinates",
        normals_key="normals",
        area_key="area",
        pressure_key="pressure",
        shear_stress_key="shear_stress",
        **overrides,
    ):
        """Compute coefficients from a ``.vtp`` PolyData file.

        Parameters
        ----------
        file_path : str
            Path to the ``.vtp`` file.
        coordinates_key : str, optional
            Point-data array name for coordinates. If the array is missing the
            mesh point coordinates (``mesh.points``) are used instead.
        normals_key, area_key, pressure_key, shear_stress_key : str, optional
            Point-data array names for the respective fields.
        **overrides
            Optional per-call configuration overrides (see
            :meth:`calculate_from_dict`).

        Returns
        -------
        dict
            Structured results (see :meth:`_assemble_results`).
        """
        # Imported lazily so importing this module does not require pyvista.
        import pyvista as pv

        mesh = pv.read(file_path)
        point_data = mesh.point_data

        if coordinates_key in point_data:
            coordinates = np.asarray(point_data[coordinates_key])
        else:
            coordinates = np.asarray(mesh.points)

        try:
            data_dict = {
                "coordinates": coordinates,
                "normals": np.asarray(point_data[normals_key]),
                "area": np.asarray(point_data[area_key]),
                "pressure": np.asarray(point_data[pressure_key]),
                "shear_stress": np.asarray(point_data[shear_stress_key]),
            }
        except KeyError as exc:
            available = list(point_data.keys())
            raise KeyError(
                f"Point-data array {exc} not found in '{file_path}'. "
                f"Available arrays: {available}."
            ) from exc

        return self.calculate_from_dict(data_dict, **overrides)

    # ------------------------------------------------------------------ #
    # Core integration engine
    # ------------------------------------------------------------------ #
    def _integrate(self, coordinates, normals, area, pressure, shear_stress, config):
        """Integrate forces and moments in the body/simulation axes.

        The pressure force on point ``i`` is ``-Cp_i * A_i * n_i`` and the
        friction force is ``tau_i * A_i``. Moments are taken about the moment
        reference center ``r_mrc``:
        ``M = sum_i (r_i - r_mrc) x f_i``.
        """
        # Per-point force contributions (N, 3).
        f_press = -pressure * area * normals
        f_fric = shear_stress * area

        # Total force vectors (body axes).
        force_press = f_press.sum(axis=0)
        force_fric = f_fric.sum(axis=0)
        force_total = force_press + force_fric

        # Moment arms relative to the moment reference center.
        lever = coordinates - config["mrc"]
        moment_press = np.cross(lever, f_press).sum(axis=0)
        moment_fric = np.cross(lever, f_fric).sum(axis=0)
        moment_total = moment_press + moment_fric

        return {
            "force": {
                "pressure": force_press,
                "friction": force_fric,
                "total": force_total,
            },
            "moment": {
                "pressure": moment_press,
                "friction": moment_fric,
                "total": moment_total,
            },
        }

    # ------------------------------------------------------------------ #
    # Frame transforms and coefficients
    # ------------------------------------------------------------------ #
    def _resolve_angles(self, config):
        """Resolve the body-axis pitch/sideslip angles (in radians).

        Returns ``(alpha_pitch, beta)`` in radians based on the selected
        convention.
        """
        if config["convention"] == "alpha_beta":
            return np.radians(config["alpha"]), np.radians(config["beta"])

        # Aeroballistic "alpha_phi": alpha_total is the angle between the
        # freestream and the body x axis, and phi is the aerodynamic roll angle.
        alpha_total = np.radians(config["alpha_total"])
        phi = np.radians(config["phi_aerodynamic"])
        tan_total = np.tan(alpha_total)
        alpha_pitch = np.arctan(tan_total * np.cos(phi))
        beta = np.arctan(tan_total * np.sin(phi))
        return alpha_pitch, beta

    def _wind_rotation_matrix(self, config):
        """Body -> wind rotation matrix for the x=streamwise, z=up convention.

        With ``a = alpha_pitch`` (rotation about ``y``) and ``b = beta``
        (rotation about ``z``), the freestream direction in body axes is
        ``(cos a cos b, sin b, sin a cos b)``. The returned matrix ``R`` maps a
        body-axis vector into the wind frame such that the first row is aligned
        with the drag (freestream) direction, the third row with lift (up), and
        the second row with side force.

        NOTE: this uses a z-up (automotive) convention. If your solver uses a
        z-down (classical aerospace) convention, negate the ``z``-coupled terms.
        """
        a, b = self._resolve_angles(config)
        ca, sa = np.cos(a), np.sin(a)
        cb, sb = np.cos(b), np.sin(b)
        return np.array(
            [
                [ca * cb, sb, sa * cb],
                [-ca * sb, cb, -sa * sb],
                [-sa, 0.0, ca],
            ],
            dtype=np.float64,
        )

    def _coefficients(self, force, moment, config, frame):
        """Non-dimensionalize a force/moment triple into named coefficients."""
        area = config["ref_area"]
        moment_denom = config["ref_area"] * config["ref_length"]
        fx, fy, fz = force / area
        mx, my, mz = moment / moment_denom

        if frame == "wind":
            return {
                "CD": float(fx),
                "CY": float(fy),
                "CL": float(fz),
                "Cl": float(mx),
                "Cm": float(my),
                "Cn": float(mz),
            }
        # Body-axis force/moment coefficients.
        return {
            "CFx": float(fx),
            "CFy": float(fy),
            "CFz": float(fz),
            "CMx": float(mx),
            "CMy": float(my),
            "CMz": float(mz),
        }

    def _frame_block(self, integrals, config, frame):
        """Build the results block (forces, moments, coefficients) for a frame."""
        if frame == "body":
            force = integrals["force"]
            moment = integrals["moment"]
        else:
            rot = self._wind_rotation_matrix(config)
            force = {k: rot @ v for k, v in integrals["force"].items()}
            moment = {k: rot @ v for k, v in integrals["moment"].items()}

        return {
            "forces": {k: v.copy() for k, v in force.items()},
            "moments": {k: v.copy() for k, v in moment.items()},
            "coefficients": self._coefficients(
                force["total"], moment["total"], config, frame
            ),
            "coefficients_pressure": self._coefficients(
                force["pressure"], moment["pressure"], config, frame
            ),
            "coefficients_friction": self._coefficients(
                force["friction"], moment["friction"], config, frame
            ),
        }

    def _assemble_results(self, integrals, config):
        """Assemble the final structured results dictionary."""
        results = {
            "config": {
                "mrc": config["mrc"].tolist(),
                "alpha": config["alpha"],
                "beta": config["beta"],
                "alpha_total": config["alpha_total"],
                "phi_aerodynamic": config["phi_aerodynamic"],
                "convention": config["convention"],
                "output_frame": config["output_frame"],
                "ref_area": config["ref_area"],
                "ref_length": config["ref_length"],
            }
        }

        frames = (
            ("body", "wind")
            if config["output_frame"] == "both"
            else (config["output_frame"],)
        )
        for frame in frames:
            key = f"{frame}_frame"
            results[key] = self._frame_block(integrals, config, frame)
        return results

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def _print_summary(self, results, config):
        """Print a clean summary separating pressure and friction contributions."""
        cfg = results["config"]
        print("=" * 68)
        print("Aerodynamic coefficient integration")
        print("-" * 68)
        if cfg["convention"] == "alpha_beta":
            print(
                f"convention={cfg['convention']}  alpha={cfg['alpha']}  "
                f"beta={cfg['beta']}"
            )
        else:
            print(
                f"convention={cfg['convention']}  "
                f"alpha_total={cfg['alpha_total']}  "
                f"phi_aerodynamic={cfg['phi_aerodynamic']}"
            )
        print(
            f"MRC={cfg['mrc']}  ref_area={cfg['ref_area']}  "
            f"ref_length={cfg['ref_length']}"
        )

        for frame in ("body", "wind"):
            key = f"{frame}_frame"
            if key not in results:
                continue
            block = results[key]
            print("-" * 68)
            print(f"{frame.upper()} frame coefficients")
            labels = list(block["coefficients"].keys())
            header = "  ".join(f"{lab:>10}" for lab in labels)
            print(f"{'contribution':>14}  {header}")
            for name, ckey in (
                ("pressure", "coefficients_pressure"),
                ("friction", "coefficients_friction"),
                ("total", "coefficients"),
            ):
                row = "  ".join(
                    f"{block[ckey][lab]:>10.6f}" for lab in labels
                )
                print(f"{name:>14}  {row}")
        print("=" * 68)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _resolve_config(self, overrides):
        """Merge instance defaults with per-call overrides and validate them."""
        config = {
            "mrc": self.mrc,
            "alpha": self.alpha,
            "beta": self.beta,
            "alpha_total": self.alpha_total,
            "phi_aerodynamic": self.phi_aerodynamic,
            "convention": self.convention,
            "output_frame": self.output_frame,
            "ref_area": self.ref_area,
            "ref_length": self.ref_length,
        }

        for name, value in overrides.items():
            if name not in config:
                raise TypeError(f"Unknown override '{name}'.")
            if name == "mrc":
                config[name] = np.asarray(value, dtype=np.float64).reshape(3)
            elif name in ("convention", "output_frame"):
                choices = _CONVENTIONS if name == "convention" else _OUTPUT_FRAMES
                config[name] = self._validate_choice(value, choices, name)
            else:
                config[name] = float(value)

        return config

    @staticmethod
    def _validate_choice(value, choices, name):
        """Validate that ``value`` is one of ``choices``."""
        if value not in choices:
            raise ValueError(
                f"Invalid {name} '{value}'. Expected one of {list(choices)}."
            )
        return value

    @staticmethod
    def _to_numpy(array):
        """Convert a torch tensor or array-like into a NumPy ``float64`` array.

        Torch tensors (including tensors on the GPU or with ``requires_grad``)
        are detached and moved to the host before conversion, so both NumPy
        arrays and torch tensors can be supplied interchangeably.
        """
        # Duck-typed torch detection avoids importing torch in this module.
        if hasattr(array, "detach") and hasattr(array, "cpu"):
            array = array.detach().cpu().numpy()
        return np.asarray(array, dtype=np.float64)

    @staticmethod
    def _as_vector(array, name):
        """Coerce an input array to a contiguous float ``(N, 3)`` array."""
        arr = AerodynamicCoefficients._to_numpy(array)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(
                f"Field '{name}' must have shape (N, 3), got {arr.shape}."
            )
        return arr

    @staticmethod
    def _as_column(array, name):
        """Coerce an input array to a float ``(N, 1)`` column array."""
        arr = AerodynamicCoefficients._to_numpy(array)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim == 2 and arr.shape[1] == 1:
            pass
        else:
            raise ValueError(
                f"Field '{name}' must have shape (N,) or (N, 1), got {arr.shape}."
            )
        return arr


def _iter_input_files(cfg):
    """Yield the list of ``.vtp`` files selected by the config block."""
    files = []
    input_file = cfg.get("input_file", None)
    input_dir = cfg.get("input_dir", None)

    if input_file:
        files.append(to_absolute_path(input_file))
    if input_dir:
        pattern = os.path.join(to_absolute_path(input_dir), "*.vtp")
        files.extend(sorted(glob.glob(pattern)))

    if not files:
        raise ValueError(
            "No input files found. Set 'aero_coefficients.input_file' or "
            "'aero_coefficients.input_dir'."
        )
    return files


@hydra.main(version_base="1.3", config_path="conf", config_name="config3d")
def main(cfg: DictConfig) -> None:
    """Hydra entry point driven by the ``aero_coefficients`` config block."""
    aero_cfg = cfg.aero_coefficients

    calculator = AerodynamicCoefficients(
        mrc=list(aero_cfg.mrc),
        alpha=aero_cfg.alpha,
        beta=aero_cfg.beta,
        alpha_total=aero_cfg.alpha_total,
        phi_aerodynamic=aero_cfg.phi_aerodynamic,
        convention=aero_cfg.convention,
        output_frame=aero_cfg.output_frame,
        ref_area=aero_cfg.ref_area,
        ref_length=aero_cfg.ref_length,
    )

    for file_path in _iter_input_files(aero_cfg):
        print(f"\nFile: {file_path}")
        calculator.calculate_from_vtp(
            file_path,
            coordinates_key=aero_cfg.coordinates_key,
            normals_key=aero_cfg.normals_key,
            area_key=aero_cfg.area_key,
            pressure_key=aero_cfg.pressure_key,
            shear_stress_key=aero_cfg.shear_stress_key,
        )


if __name__ == "__main__":
    main()
