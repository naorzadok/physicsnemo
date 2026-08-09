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
Post-processing utilities for XAeroNet-S dynamic-graph inference.

This module turns the aerodynamic-coefficient predictions produced by
``inference_dynGraph.py`` (integrated with ``aero_coefficients.py``) into a
metric report and a couple of diagnostic plots. It is used two ways:

* **Imported** by ``inference_dynGraph.py``, which calls :func:`generate_report`
  at the end of a run (metric report + true-vs-predicted scatter) and
  :func:`plot_pressure_error_heatmap` per sample.
* **Standalone**, re-generating the same report and plots from the saved
  artifacts (``inference_dynGraph_errors.json`` + ``inference_cloud_*.vtp``)::

    python util_inference.py \
        --errors-json outputs/XAeroNetS/inference_dynGraph_errors.json \
        --clouds-dir outputs/XAeroNetS \
        --output-dir report \
        --color-by alpha \
        --global-feature-names alpha beta mach reynolds

The metric report scores each aerodynamic coefficient (CD, CL, CM, ...) across
all samples with the normalized fit score (``1 - RMSE/std``), the relative L2
error, the coefficient of determination ``R^2``, the mean and maximum absolute
errors, and the Pearson correlation coefficient.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _relative_l2(pred: np.ndarray, true: np.ndarray) -> float:
    """Relative L2 error ``||pred - true||_2 / ||true||_2`` across samples."""
    denom = float(np.linalg.norm(true))
    if denom == 0.0:
        return float("nan")
    return float(np.linalg.norm(pred - true) / denom)


def _r2(pred: np.ndarray, true: np.ndarray) -> float:
    """Coefficient of determination (nan if < 2 samples or flat truth)."""
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    if pred.size < 2 or ss_tot == 0.0:
        return float("nan")
    ss_res = float(np.sum((true - pred) ** 2))
    return 1.0 - ss_res / ss_tot


def _normalized_fit(pred: np.ndarray, true: np.ndarray) -> float:
    """1 - RMSE / std(true) across samples (nan if < 2 samples or flat truth)."""
    sd = float(np.std(true))
    if pred.size < 2 or sd == 0.0:
        return float("nan")
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    return 1.0 - rmse / sd


def _pearson_r(pred: np.ndarray, true: np.ndarray) -> float:
    """Pearson correlation coefficient (nan if < 2 samples or flat input)."""
    if pred.size < 2 or np.std(pred) == 0.0 or np.std(true) == 0.0:
        return float("nan")
    return float(np.corrcoef(pred, true)[0, 1])


# Ordered metric columns reported for every coefficient.
_METRIC_COLUMNS = (
    "Normalized_fit_score",
    "relative_L2",
    "R2",
    "MAE",
    "max_abs_error",
    "pearson_r",
)


def coefficient_metrics(
    coeff_rows: list[dict], coeff_names: list[str]
) -> dict[str, dict[str, float]]:
    """Score each coefficient across all samples.

    Parameters
    ----------
    coeff_rows : list of dict
        Per-sample records with ``"pred"`` and ``"true"`` dictionaries keyed by
        coefficient name (as written by ``inference_dynGraph.py``).
    coeff_names : list of str
        Coefficients to score (e.g. ``["CD", "CL", "CM"]``).

    Returns
    -------
    dict
        ``{coeff_name: {metric_name: value}}``.
    """
    metrics: dict[str, dict[str, float]] = {}
    for name in coeff_names:
        pred = np.array([r["pred"][name] for r in coeff_rows], dtype=np.float64)
        true = np.array([r["true"][name] for r in coeff_rows], dtype=np.float64)
        diff = np.abs(pred - true)
        metrics[name] = {
            "Normalized_fit_score": _normalized_fit(pred, true),
            "relative_L2": _relative_l2(pred, true),
            "R2": _r2(pred, true),
            "MAE": float(np.mean(diff)) if diff.size else float("nan"),
            "max_abs_error": float(np.max(diff)) if diff.size else float("nan"),
            "pearson_r": _pearson_r(pred, true),
        }
    return metrics


def _print_metrics_table(metrics: dict[str, dict[str, float]]) -> None:
    """Print the coefficient metrics as a GitHub-flavored table."""
    from tabulate import tabulate

    rows = [
        [name] + [f"{values[col]:.5f}" for col in _METRIC_COLUMNS]
        for name, values in metrics.items()
    ]
    print("\nAerodynamic-coefficient metric report:\n")
    print(tabulate(rows, headers=["Coeff", *_METRIC_COLUMNS], tablefmt="github"))


def write_report(metrics: dict[str, dict[str, float]], out_path: str) -> str:
    """Write the metric report to JSON and print it as a table."""
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    _print_metrics_table(metrics)
    return out_path


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _resolve_color_by(
    color_by: int | str, names: list[str] | None
) -> tuple[int, str]:
    """Resolve the color-by selector into a ``(index, label)`` pair.

    ``color_by`` may be a global-feature name, a numeric string, or an int.
    """
    names = names or []
    if isinstance(color_by, str):
        if names and color_by in names:
            idx = names.index(color_by)
            return idx, color_by
        try:
            idx = int(color_by)
        except ValueError:
            idx = 0
    else:
        idx = int(color_by)
    label = names[idx] if names and 0 <= idx < len(names) else f"global_feature_{idx}"
    return idx, label


def _extract_color_values(
    coeff_rows: list[dict], index: int
) -> np.ndarray | None:
    """Collect the selected global feature for every sample, or ``None``."""
    values = []
    for row in coeff_rows:
        globals_ = row.get("global_features")
        if globals_ is None or index >= len(globals_):
            return None
        values.append(globals_[index])
    return np.array(values, dtype=np.float64)


def plot_true_vs_pred(
    coeff_rows: list[dict],
    coeff_names: list[str],
    global_feature_names: list[str] | None = None,
    color_by: int | str = 0,
    out_path: str = "inference_coeff_scatter.png",
    dpi: int = 150,
) -> str:
    """Scatter true vs predicted coefficients as one multi-panel figure.

    One subplot per coefficient (e.g. CD, CL, CM), each with the identity line
    and points colored by a chosen global feature (default: the first one).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    color_index, color_label = _resolve_color_by(color_by, global_feature_names)
    colors = _extract_color_values(coeff_rows, color_index)
    metrics = coefficient_metrics(coeff_rows, coeff_names)

    n = len(coeff_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    axes = axes[0]

    scatter = None
    for ax, name in zip(axes, coeff_names):
        true = np.array([r["true"][name] for r in coeff_rows], dtype=np.float64)
        pred = np.array([r["pred"][name] for r in coeff_rows], dtype=np.float64)

        lo = float(min(true.min(), pred.min()))
        hi = float(max(true.max(), pred.max()))
        pad = 0.05 * (hi - lo if hi > lo else 1.0)
        lo, hi = lo - pad, hi + pad

        ax.plot([lo, hi], [lo, hi], "k--", lw=1, zorder=1, label="ideal")
        if colors is not None:
            scatter = ax.scatter(
                true,
                pred,
                c=colors,
                cmap="viridis",
                s=45,
                zorder=2,
                edgecolors="k",
                linewidths=0.3,
            )
        else:
            ax.scatter(
                true, pred, s=45, zorder=2, edgecolors="k", linewidths=0.3
            )

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", "box")
        ax.set_xlabel(f"{name} true")
        ax.set_ylabel(f"{name} predicted")
        ax.set_title(
            f"{name}  (R\u00b2={metrics[name]['R2']:.3f}, "
            f"NFS={metrics[name]['Normalized_fit_score']:.3f})"
        )
        ax.grid(True, ls=":", alpha=0.5)

    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=list(axes), fraction=0.046, pad=0.04)
        cbar.set_label(color_label)

    fig.suptitle("Aerodynamic coefficients: true vs predicted")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_pressure_error_heatmap(
    coordinates: np.ndarray,
    pressure_pred: np.ndarray,
    pressure_true: np.ndarray,
    out_path: str,
    run_id: str = "",
    plane: str = "xz",
    nbins: int = 200,
    smooth_sigma: float = 2.0,
    cmap: str = "turbo",
    dpi: int = 150,
) -> str:
    """Render a smooth, continuous map of the pressure error over the geometry.

    The absolute pressure error ``|Cp_pred - Cp_true|`` is normalized by the
    true-pressure span (reported as a percentage of the ``Cp`` range) and
    averaged into a 2D grid over the requested projection plane (default: the
    X-Z side view). To avoid a pixelated look the per-cell means are then
    linearly interpolated (``scipy.interpolate.griddata``) into a continuous
    field, gaps between populated cells are filled, an edge-aware Gaussian blur
    (``smooth_sigma`` cells, set to 0 to disable) removes residual speckle, and
    the image is drawn with bilinear sampling. The field runs cool (blue, low
    error) to warm (high error); the colorbar upper bound is the maximum
    normalized error.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coordinates = np.asarray(coordinates, dtype=np.float64)
    p_pred = np.asarray(pressure_pred, dtype=np.float64).reshape(-1)
    p_true = np.asarray(pressure_true, dtype=np.float64).reshape(-1)

    axis_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    ai, aj = axis_map.get(plane, (0, 2))
    u = coordinates[:, ai]
    v = coordinates[:, aj]

    err = np.abs(p_pred - p_true)
    span = float(p_true.max() - p_true.min())
    err_norm = err / span * 100.0 if span > 0 else err  # % of true Cp span

    # Average the normalized error per 2D cell (collapses the many 3D points
    # that project onto the same pixel into a single stable value).
    sums, u_edges, v_edges = np.histogram2d(u, v, bins=nbins, weights=err_norm)
    counts, _, _ = np.histogram2d(u, v, bins=[u_edges, v_edges])
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_err = np.where(counts > 0, sums / counts, np.nan)

    # Interpolate the populated cell centers into a continuous field so the map
    # is smooth rather than blocky; cells outside the interpolation region stay
    # NaN and are masked (rendered transparent/white).
    u_cent = 0.5 * (u_edges[:-1] + u_edges[1:])
    v_cent = 0.5 * (v_edges[:-1] + v_edges[1:])
    grid_u, grid_v = np.meshgrid(u_cent, v_cent, indexing="ij")
    filled = np.isfinite(mean_err)
    dense = mean_err
    if filled.sum() >= 3:
        try:
            from scipy.interpolate import griddata

            points = np.column_stack([grid_u[filled], grid_v[filled]])
            values = mean_err[filled]
            # Linear interpolation fills the whole populated region continuously;
            # cells outside it remain NaN and are masked.
            dense = griddata(points, values, (grid_u, grid_v), method="linear")
        except Exception:
            dense = mean_err

    # Edge-aware Gaussian blur: smooth the field without letting the masked
    # exterior bleed in, so the map is continuous but the silhouette stays crisp.
    if smooth_sigma and smooth_sigma > 0 and np.isfinite(dense).any():
        try:
            from scipy.ndimage import gaussian_filter

            support = np.isfinite(dense)
            values0 = np.where(support, dense, 0.0)
            weight = support.astype(np.float64)
            blurred = gaussian_filter(values0, smooth_sigma, mode="constant")
            norm = gaussian_filter(weight, smooth_sigma, mode="constant")
            with np.errstate(invalid="ignore", divide="ignore"):
                smoothed = np.where(norm > 0, blurred / norm, np.nan)
            dense = np.where(support, smoothed, np.nan)
        except Exception:
            pass

    grid = np.ma.masked_invalid(dense.T)  # transpose so rows=v, cols=u
    finite = dense[np.isfinite(dense)]
    vmax = float(finite.max()) if finite.size else 1.0
    if vmax <= 0.0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(9, 4))
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("white")
    im = ax.imshow(
        grid,
        origin="lower",
        extent=[u_edges[0], u_edges[-1], v_edges[0], v_edges[-1]],
        aspect="equal",
        cmap=cmap_obj,
        vmin=0.0,
        vmax=vmax,
        interpolation="bilinear",
    )
    axis_labels = {0: "x", 1: "y", 2: "z"}
    ax.set_xlabel(axis_labels[ai])
    ax.set_ylabel(axis_labels[aj])
    title = "Pressure error"
    if run_id:
        title += f" \u2014 {run_id}"
    title += f"  (max {vmax:.1f}% of Cp span)"
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("|Cp error| (% of true Cp span)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def generate_report(
    coeff_rows: list[dict],
    coeff_names: list[str],
    global_feature_names: list[str] | None = None,
    output_dir: str = ".",
    color_by: int | str = 0,
) -> dict:
    """Write the coefficient metric report and the true-vs-predicted scatter.

    Returns a dict with the written ``report`` / ``scatter`` paths and the
    computed ``metrics``.
    """
    os.makedirs(output_dir, exist_ok=True)

    metrics = coefficient_metrics(coeff_rows, coeff_names)
    report_path = os.path.join(output_dir, "coefficient_metric_report.json")
    write_report(metrics, report_path)
    print(f"Wrote {report_path}")

    scatter_path = os.path.join(output_dir, "inference_coeff_scatter.png")
    plot_true_vs_pred(
        coeff_rows,
        coeff_names,
        global_feature_names,
        color_by=color_by,
        out_path=scatter_path,
    )
    print(f"Wrote {scatter_path}")

    return {"report": report_path, "scatter": scatter_path, "metrics": metrics}


# --------------------------------------------------------------------------- #
# Standalone entry point
# --------------------------------------------------------------------------- #
def _load_from_errors_json(
    errors_json: str,
) -> tuple[list[dict], list[dict], list[str]]:
    """Load metric rows, coefficient rows and names from the errors JSON."""
    with open(errors_json, "r") as f:
        data = json.load(f)
    coeff = data.get("coefficients", {})
    coeff_rows = coeff.get("per_sample", [])
    coeff_names = list(coeff_rows[0]["pred"].keys()) if coeff_rows else []
    metric_rows = data.get("per_sample", [])
    return metric_rows, coeff_rows, coeff_names


def _heatmaps_from_clouds(clouds_dir: str, output_dir: str) -> list[str]:
    """Regenerate the per-sample pressure-error heatmaps from saved VTPs."""
    import pyvista as pv

    os.makedirs(output_dir, exist_ok=True)
    written: list[str] = []
    paths = sorted(glob.glob(os.path.join(clouds_dir, "inference_cloud_*.vtp")))
    for path in paths:
        mesh = pv.read(path)
        point_data = mesh.point_data
        if "pressure_pred" not in point_data or "pressure_true" not in point_data:
            continue
        coords = (
            np.asarray(point_data["coordinates"])
            if "coordinates" in point_data
            else np.asarray(mesh.points)
        )
        base = os.path.basename(path)
        run_id = base[len("inference_cloud_") : -len(".vtp")]
        out_path = os.path.join(output_dir, f"error_heatmap_{run_id}.png")
        plot_pressure_error_heatmap(
            coords,
            np.asarray(point_data["pressure_pred"]),
            np.asarray(point_data["pressure_true"]),
            out_path=out_path,
            run_id=run_id,
        )
        written.append(out_path)
    return written


def main() -> None:
    """Regenerate the report and plots from saved inference artifacts."""
    parser = argparse.ArgumentParser(
        description="XAeroNet-S inference report and diagnostic plots."
    )
    parser.add_argument(
        "--errors-json",
        required=True,
        help="Path to inference_dynGraph_errors.json.",
    )
    parser.add_argument(
        "--clouds-dir",
        default=None,
        help="Directory with inference_cloud_*.vtp for the error heatmaps.",
    )
    parser.add_argument(
        "--output-dir", default="report", help="Directory to write outputs to."
    )
    parser.add_argument(
        "--color-by",
        default="0",
        help="Global feature name or index to color the scatter by.",
    )
    parser.add_argument(
        "--global-feature-names",
        nargs="*",
        default=None,
        help="Names of the global features (e.g. alpha beta mach reynolds).",
    )
    args = parser.parse_args()

    _, coeff_rows, coeff_names = _load_from_errors_json(args.errors_json)
    if not coeff_rows:
        raise SystemExit(
            f"No coefficient data found in {args.errors_json}; "
            "run inference with aero_coefficients.compute=true."
        )

    generate_report(
        coeff_rows,
        coeff_names,
        args.global_feature_names,
        output_dir=args.output_dir,
        color_by=args.color_by,
    )

    if args.clouds_dir:
        written = _heatmaps_from_clouds(args.clouds_dir, args.output_dir)
        for path in written:
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
