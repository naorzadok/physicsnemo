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
This code processes mesh data from .stl and .vtp files to create partitioned
graphs for large scale training. It first converts meshes to triangular format
and extracts surface triangles, vertices, and relevant attributes such as pressure
and shear stress. Using nearest neighbors, the code interpolates these attributes
for a sampled boundary of points, and constructs a graph based on these points, with
node features like coordinates, normals, pressure, and shear stress, as well as edge
features representing relative displacement. The graph is partitioned into subgraphs,
and the partitions are saved. The code supports parallel processing to handle multiple
samples simultaneously, improving efficiency. Additionally, it provides an option to
save the point cloud of each graph for visualization purposes.
"""

import os
import vtk
import pyvista as pv
import numpy as np
import torch
import hydra

import random
import torch_geometric as pyg
from Tessellation2D import Tessellation2D

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from sklearn.neighbors import NearestNeighbors
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from physicsnemo.datapipes.cae.readers import read_vtp
from physicsnemo.sym.geometry.tessellation import Tessellation

from dataloader_2d import PartitionedGraph

def calculate_area(points):
    nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    # distances[:, 1] and distances[:, 2] are the distances to the two nearest neighbors
    point_areas = 0.5 * (distances[:, 1] + distances[:, 2])
    
    return point_areas

def convert_to_triangular_mesh(
    polydata, write=False, output_filename="surface_mesh_triangular.vtu"
):
    """Converts a vtkPolyData object to a triangular mesh."""
    tet_filter = vtk.vtkDataSetTriangleFilter()
    tet_filter.SetInputData(polydata)
    tet_filter.Update()

    tet_mesh = pv.wrap(tet_filter.GetOutput())

    if write:
        tet_mesh.save(output_filename)

    return tet_mesh


def extract_surface_triangles(tet_mesh):
    """Extracts the surface triangles from a triangular mesh."""
    surface_filter = vtk.vtkDataSetSurfaceFilter()
    surface_filter.SetInputData(tet_mesh)
    surface_filter.Update()

    surface_mesh = pv.wrap(surface_filter.GetOutput())
    triangle_indices = []
    faces = surface_mesh.faces.reshape((-1, 4))
    for face in faces:
        if face[0] == 3:
            triangle_indices.extend([face[1], face[2], face[3]])
        else:
            raise ValueError("Face is not a triangle")

    return triangle_indices


def fetch_mesh_vertices(mesh):
    """Fetches the vertices of a mesh."""
    points = mesh.GetPoints()
    num_points = points.GetNumberOfPoints()
    vertices = [points.GetPoint(i) for i in range(num_points)]
    return vertices

def read_vtk(file_path):
    """Reads a .vtk file and returns a PyVista mesh."""
    return pv.read(file_path)

def add_edge_features(graph: pyg.data.Data) -> pyg.data.Data:
    """
    Add relative displacement and displacement norm as edge features to the graph.
    The calculations are done using the 'pos' attribute in the
    node data of each graph. The resulting edge features are stored in the 'x' attribute
    in the edge data of each graph.

    This method will modify the graph in-place.

    Returns
    -------
    pyg.data.Data
        Graph with updated edge features.
    """

    pos = graph.coordinates
    row, col = graph.edge_index

    disp = pos[row] - pos[col]
    disp_norm = torch.linalg.norm(disp, dim=-1, keepdim=True)
    graph.edge_attr = torch.cat((disp, disp_norm), dim=-1)

    return graph

def extract_global_context(file_name):
    """Extracts global context parameters from the file name."""
    parts = file_name.split('_')
    M = float(parts[1])
    ReL = float(parts[3])
    AOA = float(parts[5].replace('.vtk', ''))
    
    unnorm_vec = [M, ReL, AOA]
    return unnorm_vec

# Define this function outside of any local scope so it can be pickled
def run_task(params):
    """Wrapper function to unpack arguments for process_run."""
    return process_run(*params)


def process_partition(graph, num_partitions, halo_hops):
    """
    Helper function to partition a single graph and include node and edge features.
    """
    # Perform the partitioning
    return PartitionedGraph(graph, num_partitions, halo_hops)


def calculate_2d_centroid(points):
    """Calculate the centroid of a set of 2D points."""
    return np.mean(points, axis=0)

def ensure_normals_point_outward(points, normals):
    """Ensure normals point outward for a closed 2D shape."""
    centroid = calculate_2d_centroid(points)
    for i, point in enumerate(points):
        normal = normals[i]
        direction_to_centroid = centroid - point
        if np.dot(normal, direction_to_centroid) > 0:
            normals[i] = -normal  # Reverse the normal
    return normals

def calculate_3d_normals(points, n_neighbors=2, epsilon=1e-6):
    """Calculate normals for a 3D point cloud with weighted averaging and normalization."""
    # Ensure points are 3D
    if points.shape[1] == 2:
        points = np.hstack((points, np.zeros((points.shape[0], 1), dtype=np.float32)))
    
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(points)
    _, indices = nbrs.kneighbors(points)
    
    normals = np.zeros(points.shape, dtype=np.float32)
    
    for i, neighbors in enumerate(indices):
        p0 = points[i]
        normal_sum = np.zeros(3, dtype=np.float32)
        weight_sum = 0.0
        
        for j in range(1, n_neighbors):
            p1 = points[neighbors[j]]
            vector = p1 - p0
            distance = np.linalg.norm(vector)
            if distance > epsilon:
                normal = np.array([-vector[1], vector[0], 0], dtype=np.float32)  # Perpendicular vector in XY plane
                normal /= distance  # Normalize
                weight = 1.0 / distance  # Weight inversely proportional to distance
                normal_sum += weight * normal
                weight_sum += weight
        
        if weight_sum > 0:
            normal_avg = normal_sum / weight_sum  # Weighted average
            norm = np.linalg.norm(normal_avg)
            if norm > epsilon:
                normals[i] = normal_avg / norm  # Normalize the final normal
    
    normals = ensure_normals_point_outward(points, normals)
    # visualize_normals(points, normals)
    return normals

def split_data(run_dirs, train_ratio=0.85, val_ratio=0.02, test_ratio=0.13):
    """Split the run directories into training, validation, and test sets."""
    random.shuffle(run_dirs)
    total_runs = len(run_dirs)
    train_end = int(total_runs * train_ratio)
    val_end = train_end + int(total_runs * val_ratio)
    
    train_dirs = run_dirs[:train_end]
    val_dirs = run_dirs[train_end:val_end]
    test_dirs = run_dirs[val_end:]
    
    return train_dirs, val_dirs, test_dirs

def process_run(
    run_path, partition_path, point_list, node_degree, num_partitions, halo_hops, save_point_cloud=False, data_type="train", file_index=1, last_lvl_OG=True
):
    """Process a single run directory to generate a multi-level graph and apply partitioning."""
    run_id = os.path.basename(run_path).split("_")[-1]

    stl_file = [f for f in os.listdir(run_path) if f.endswith(".stl")]
    vtp_file = [f for f in os.listdir(run_path) if f.endswith(".vtp")]
    if len(vtp_file) == 0:
        print(f"Warning: No VTP file found for run {run_id}. trying .vtk extension...")
        vtp_file = [f for f in os.listdir(run_path) if f.endswith(".vtk")]
        Flag_sample_from_vtk = True
        if len(vtp_file) == 0:
            print(f"Warning: No VTK file found for run {run_id}. Skipping...")
            return
    
    # Path to save the list of partitions
    partition_file_path = to_absolute_path(f"{partition_path}/{data_type}_partitions/graph_partitions_{file_index}.bin")
    # Extract global context from file name
    global_context_unnorm = extract_global_context(vtp_file[0])
    Mach, ReL, AOA = global_context_unnorm

    vtp_file = os.path.join(run_path, vtp_file[0])
    
    if os.path.exists(partition_file_path):
        print(f"Partitions for run {run_id} already exist. Skipping...")
        return

    if not os.path.exists(vtp_file):
        print(f"Warning: Missing files for run {run_id}. Skipping...")
        return
    
    if len(stl_file) == 0:
        print(f"Warning: Missing stl file for run {run_id}, shifting to vtp sampling, currently only set to 2d")
        Flag_sample_from_vtp = True
    else:
        stl_file = os.path.join(run_path, stl_file[0])
        if not os.path.exists(stl_file):
            print(f"Warning: Missing files for run {run_id}. Skipping...")
            return
        
    try:
        # Load the STL and VTP files
        if Flag_sample_from_vtk:
            surface_mesh = read_vtk(vtp_file)
        else:
            surface_mesh = read_vtp(vtp_file)
        surface_mesh = convert_to_triangular_mesh(surface_mesh)
        surface_vertices = fetch_mesh_vertices(surface_mesh)
        surface_mesh = surface_mesh.cell_data_to_point_data()
        node_attributes = surface_mesh.point_data
        pressure_ref = node_attributes["Pressure_Coefficient"]
        shear_stress_ref = node_attributes["Skin_Friction_Coefficient"]

        if not Flag_sample_from_vtp:
            obj = Tessellation.from_stl(stl_file, airtight=False)
        else:
            obj = Tessellation2D(vtp_file)
        # Sort the list of points in ascending order
        sorted_points = sorted(point_list)

        # Initialize arrays to store all points, normals, and areas
        all_points = np.empty((0, 3))
        all_normals = np.empty((0, 3))
        all_areas = np.empty((0, 1))
        edge_sources = []
        edge_destinations = []

        # Precompute the nearest neighbors for surface vertices
        nbrs_surface = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(
            surface_vertices
        )

        for num_points in sorted_points:
            # Sample the boundary points for the current level
            boundary = obj.sample_boundary(num_points, quasirandom=True)
            points = np.concatenate(
                [boundary["x"], boundary["y"], boundary["z"]], axis=1
            )
            normals = np.concatenate(
                [boundary["normal_x"], boundary["normal_y"], boundary["normal_z"]],
                axis=1,
            )
            area = boundary["area"]

            # Concatenate new points with the previous ones
            all_points = np.vstack([all_points, points])
            all_normals = np.vstack([all_normals, normals])
            all_areas = np.vstack([all_areas, area])

            # Construct edges for the combined point cloud at this level
            nbrs_points = NearestNeighbors(
                n_neighbors=node_degree + 1, algorithm="ball_tree"
            ).fit(all_points)
            _, indices_within = nbrs_points.kneighbors(all_points)
            src_within = [i for i in range(len(all_points)) for _ in range(node_degree)]
            dst_within = indices_within[:, 1:].flatten()

            # Add the within-level edges
            edge_sources.extend(src_within)
            edge_destinations.extend(dst_within)

        if last_lvl_OG:
            # Concatenate new points with the previous ones
            points = np.array([obj.data['x'], obj.data['y'], obj.data['z']]).T
            all_points = np.vstack([all_points, points])
            all_normals = np.vstack([all_normals, np.array([obj.data['normal_x'], obj.data['normal_y'], obj.data['normal_z']]).T])
            all_areas = np.vstack([all_areas, np.array([obj.data['area']]).T])

            # Construct edges for the combined point cloud at this level
            nbrs_points = NearestNeighbors(
                n_neighbors=node_degree + 1, algorithm="ball_tree"
            ).fit(all_points)
            _, indices_within = nbrs_points.kneighbors(all_points)
            src_within = [i for i in range(len(all_points)) for _ in range(node_degree)]
            dst_within = indices_within[:, 1:].flatten()

            # Add the within-level edges
            edge_sources.extend(src_within)
            edge_destinations.extend(dst_within)
        # Now, compute pressure and shear stress for the final combined point cloud
        _, indices = nbrs_surface.kneighbors(all_points)
        indices = indices.flatten()

        pressure = pressure_ref[indices]
        shear_stress = shear_stress_ref[indices]

        num_last_lvl = points.shape[0]
        
    except Exception as e:
        print(f"Error processing run {run_id}: {e}. Skipping this run...")
        return

    try:
        # Create the final graph with multi-level edges
        edge_index = torch.stack(
            [
                torch.tensor(edge_sources, dtype=torch.long),
                torch.tensor(edge_destinations, dtype=torch.long),
            ],
            dim=0,
        )

        # Create a bidirectional graph object.
        edge_index = pyg.utils.coalesce(edge_index)
        edge_index = pyg.utils.to_undirected(edge_index)
        edge_index, _ = pyg.utils.add_self_loops(edge_index)

        graph = pyg.data.Data(
            edge_index=edge_index,
            coordinates=torch.tensor(all_points, dtype=torch.float32),
            normals=torch.tensor(all_normals, dtype=torch.float32),
            area=torch.tensor(all_areas, dtype=torch.float32),
            pressure=torch.tensor(pressure, dtype=torch.float32).unsqueeze(-1),
            shear_stress=torch.tensor(shear_stress, dtype=torch.float32),
            Mach=torch.tensor([Mach], dtype=torch.float32),
            ReL=torch.tensor([ReL], dtype=torch.float32),
            AOA=torch.tensor([AOA], dtype=torch.float32),
            run_id=torch.tensor([int(run_id)], dtype=torch.int32),
            num_last_lvl=torch.tensor([num_last_lvl], dtype=torch.int32)

        )

        graph = add_edge_features(graph)

        # PyG ClusterData uses `x` attribute of the source graph to set the number of nodes in each partition.
        # This is required to make ClusterData indexing work properly. The real value of `x` will
        # be set in a trainer, so set `x` to a NaN tensor to make sure it is not used.
        graph.x = torch.full((graph.coordinates.shape[0], 1), float("nan"))

        # Partition the graph
        partitioned_graphs = process_partition(graph, num_partitions, halo_hops)

        # Save the partitions
        os.makedirs(os.path.dirname(partition_file_path), exist_ok=True)
        torch.save(partitioned_graphs, partition_file_path)

        if save_point_cloud:
            if num_partitions == 1:
                point_cloud = pv.PolyData(partitioned_graphs['coordinates'].numpy())
                point_cloud["coordinates"] = partitioned_graphs['coordinates'].numpy()
                point_cloud["normals"] = partitioned_graphs['normals'].numpy()
                point_cloud["area"] = partitioned_graphs['area'].numpy()
                point_cloud["pressure"] = partitioned_graphs['pressure'].numpy()
                point_cloud["shear_stress"] = partitioned_graphs['shear_stress'].numpy()
                point_cloud.field_data["Mach"] = partitioned_graphs['Mach'].numpy()
                point_cloud.field_data["ReL"] = partitioned_graphs['ReL'].numpy()
                point_cloud.field_data["AOA"] = partitioned_graphs['AOA'].numpy()
                point_cloud.field_data["run_id"] = partitioned_graphs['run_id'].numpy()
                point_cloud.field_data["num_last_lvl"] = partitioned_graphs['num_last_lvl'].numpy()
                vtp_file_path = to_absolute_path(f"point_clouds/point_cloud_{run_id}.vtp")
                os.makedirs(os.path.dirname(vtp_file_path), exist_ok=True)
                point_cloud.save(vtp_file_path)
            else:
                parts = []
                for part in partitioned_graphs:
                    point_cloud = pv.PolyData(part.coordinates.numpy())
                    point_cloud["coordinates"] = part.coordinates.numpy()
                    point_cloud["normals"] = part.normals.numpy()
                    point_cloud["area"] = part.area.numpy()
                    point_cloud["pressure"] = part.pressure.numpy()
                    point_cloud["shear_stress"] = part.shear_stress.numpy()
                    point_cloud.field_data["Mach"] = part.Mach.numpy()
                    point_cloud.field_data["ReL"] = part.ReL.numpy()
                    point_cloud.field_data["AOA"] = part.AOA.numpy()
                    point_cloud.field_data["run_id"] = part.run_id.numpy()
                    point_cloud.field_data["num_last_lvl"] = part.num_last_lvl.numpy()
                    parts.append(point_cloud)

                multi_point_cloud = pv.MultiBlock(parts)
                for part_id in range(len(parts)):
                    multi_point_cloud.set_block_name(part_id, f"part_{part_id}")
                    # multi_point_cloud[part_id].name = part_id
                vtp_file_path = to_absolute_path(f"point_clouds/point_cloud_{run_id}.vtm")
                os.makedirs(os.path.dirname(vtp_file_path), exist_ok=True)
                multi_point_cloud.save(vtp_file_path)

    except Exception as e:
        print(
            f"Error while constructing graph or saving data for run {run_id}: {e}. Skipping this run..."
        )
        return

def process_all_runs(
    base_path,
    partition_path,
    num_points,
    node_degree,
    num_partitions,
    halo_hops,
    num_workers=8,
    save_point_cloud=False,
    last_lvl_OG = True
):
    """Process all runs in the base directory in parallel."""
        
    run_dirs = [
        os.path.join(base_path, d)
        for d in os.listdir(base_path)
        if d.startswith("run_") and os.path.isdir(os.path.join(base_path, d))
    ]

    train_dirs, val_dirs, test_dirs = split_data(run_dirs)

    tasks_train = [
        (run_dir, partition_path, num_points, node_degree, num_partitions, halo_hops, save_point_cloud, "train", i+1, last_lvl_OG)
        for i, run_dir in enumerate(train_dirs)
    ]

    tasks_val = [
        (run_dir, partition_path, num_points, node_degree, num_partitions, halo_hops, save_point_cloud, "validation", i+1, last_lvl_OG)
        for i, run_dir in enumerate(val_dirs)
    ]

    tasks_test = [
        (run_dir, partition_path, num_points, node_degree, num_partitions, halo_hops, save_point_cloud, "test", i+1, last_lvl_OG)
        for i, run_dir in enumerate(test_dirs)
    ]

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        for _ in tqdm(
            pool.map(run_task, tasks_train),
            total=len(tasks_train),
            desc="Processing Training Runs",
            unit="run",
        ):
            pass

        for _ in tqdm(
            pool.map(run_task, tasks_val),
            total=len(tasks_val),
            desc="Processing Validation Runs",
            unit="run",
        ):
            pass

        for _ in tqdm(
            pool.map(run_task, tasks_test),
            total=len(tasks_test),
            desc="Processing Test Runs",
            unit="run",
        ):
            pass

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    process_all_runs(
        base_path=to_absolute_path(cfg.data_path),
        partition_path=to_absolute_path(cfg.partitions_path),
        num_points=cfg.num_nodes,
        node_degree=cfg.node_degree,
        num_partitions=cfg.num_partitions,
        halo_hops=cfg.num_message_passing_layers,
        num_workers=cfg.num_preprocess_workers,
        save_point_cloud=cfg.save_point_clouds,
        last_lvl_OG=cfg.last_lvl_OG,
    )

if __name__ == "__main__":
    main()
