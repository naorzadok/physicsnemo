import os
import random
import numpy as np
import pyvista as pv
from pathlib import Path
from sklearn.neighbors import NearestNeighbors 
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Union, Optional, Literal

# --- Global Configuration for the Split ---
# Define the percentage split for Train, Validation, and Test sets
SPLIT_RATIOS = {
    'train': 0.80,  # 70% for training
    'val': 0.1,    # 15% for validation
    'test': 0.1    # 15% for testing
}
# Ensure the ratios sum up to 1.0 (100%)
assert np.isclose(sum(SPLIT_RATIOS.values()), 1.0), "Split ratios must sum to 1.0 (100%)"
# ------------------------------------------

# Define a small tolerance for floating-point comparisons
GEOMETRY_TOLERANCE = 1e-4

# ==========================================
# 1. GEOMETRY & GLOBAL CONTEXT HELPERS
# ==========================================

# --- Custom 2D Geometry Helpers (Used when Z-dim is effectively 0) ---
def fetch_mesh_vertices(mesh):
    """Fetches the vertices of a mesh."""
    points = mesh.GetPoints()
    num_points = points.GetNumberOfPoints()
    vertices = [points.GetPoint(i) for i in range(num_points)]  # Only X and Y
    return vertices

def calculate_area(points):
    """Approximates area for point clouds using nearest neighbors."""
    if points.shape[1] == 2:
        points = np.hstack((points, np.zeros((points.shape[0], 1), dtype=np.float32)))
    nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(points)
    distances, _ = nbrs.kneighbors(points)
    point_areas = 0.5 * (distances[:, 1] + distances[:, 2])
    return point_areas.astype(np.float32)

def calculate_2d_centroid(points):
    return np.mean(points, axis=0)

def ensure_normals_point_outward(points, normals):
    """Ensures normals point outward for a closed 2D shape (e.g., an airfoil)."""
    centroid = calculate_2d_centroid(points)
    for i, point in enumerate(points):
        normal = normals[i]
        direction_to_centroid = centroid - point
        if np.dot(normal, direction_to_centroid) > 0:
            normals[i] = -normal
    return normals

def calculate_3d_normals(points, n_neighbors=2, epsilon=1e-6):
    """Calculate normals using weighted averaging of neighbors, specifically for 2D geometry in 3D space."""
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
                # Normal must be perpendicular in the XY plane
                normal = np.array([-vector[1], vector[0], 0], dtype=np.float32) 
                normal /= distance 
                weight = 1.0 / distance 
                normal_sum += weight * normal
                weight_sum += weight
        
        if weight_sum > 0:
            normal_avg = normal_sum / weight_sum
            norm = np.linalg.norm(normal_avg)
            if norm > epsilon:
                normals[i] = normal_avg / norm
    
    return ensure_normals_point_outward(points, normals).astype(np.float32)

# --- Global Context Parser ---

def parse_globals_from_filename(filename):
    """
    Extracts global context parameters (Mach, AOA, ReL) from the file name.
    """
    try:
        parts = filename.split('_')
        M = float(parts[1])
        ReL = float(parts[3])
        AOA_str = parts[5].replace('.vtp', '').replace('.vtk', '')
        AOA = float(AOA_str)
    except (IndexError, ValueError) as e:
        print(f"Warning: Failed to parse globals from {filename}. Using defaults. Error: {e}")
        M, ReL, AOA = 0.5, 1e6, 0.0
    return np.array([M, AOA, ReL], dtype=np.float32)


# ==========================================
# 2. CUSTOM DATASET CLASS (Final Version)
# ==========================================

class CustomPaths:
    @staticmethod
    def get_filenames(path):
        return [f.name for f in Path(path).iterdir() if f.is_dir()]

    @staticmethod
    def surface_path(data_dir: Path) -> Path:
        files = list(data_dir.glob("*.vtp"))
        if not files:
            files = list(data_dir.glob("*.vtk"))
        if not files:
            raise FileNotFoundError(f"No .vtp or .vtk file found in {data_dir}")
        return files[0]


class CustomSurfaceProcessor(Dataset):
    """
    Custom Datapipe for converting surface-only CFD data (VTP/VTK) to DoMINO's NPY format,
    using a hybrid approach for 2D/3D geometry processing.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        surface_variables: Optional[list] = [
            "Pressure_Coefficient", 
            "Skin_Friction_Coefficient"
        ],
        volume_variables: Optional[list] = [], # Not used, but kept for compatibility
        global_params_types: Optional[dict] = {
            "Mach": "scalar",
            "AOA": "scalar",
            "ReL": "scalar",
        },
        global_params_reference: Optional[dict] = {
            "Mach": 1.0, # Using 1.0 ensures raw values are passed
            "AOA": 10, 
            "ReL": 6.3 
        },
        device: int = 0, # Not strictly needed for preprocessing, but kept for signature
        model_type: Literal["surface"] = "surface",
    ):
        if isinstance(data_path, str):
            data_path = Path(data_path)
            
        self.data_path = data_path.expanduser()
        assert self.data_path.exists(), f"Path {self.data_path} does not exist"
        
        self.filenames = CustomPaths.get_filenames(self.data_path)
        self.surface_variables = surface_variables
        self.volume_variables = volume_variables
        self.global_params_types = global_params_types
        self.global_params_reference = global_params_reference
        self.model_type = model_type
        
        # Mapping for actual data fields in the VTP/VTK file
        self.field_map = {
            "Pressure_Coefficient": ["Pressure_Coefficient"],
            "Skin_Friction_Coefficient": ["Skin_Friction_Coefficient"]
        }


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        cfd_filename = self.filenames[idx]
        data_dir = self.data_path / cfd_filename
        
        # 1. Find and Load File
        surface_filepath = CustomPaths.surface_path(data_dir)
        surface_mesh = pv.read(surface_filepath)
        
        surface_vertices = fetch_mesh_vertices(surface_mesh)
        surface_mesh = surface_mesh.cell_data_to_point_data()
        node_attributes = surface_mesh.point_data
            
        node_coordinates = np.array(surface_vertices)

        # Initialize placeholder for face indices
        mesh_indices_flattened = np.array([], dtype=np.float32)

        # 2. Check Z-Dimension Variance for Hybrid Logic
        z_coords = node_coordinates[:, 2]
        is_2d_geometry = np.all(np.abs(z_coords) < GEOMETRY_TOLERANCE)
        
        if is_2d_geometry:
            # 2D Logic: Use custom point-based geometry calculations
            print(f"-> 2D Geometry detected for {cfd_filename}. Using custom point cloud geometry.")
            cell_normals = calculate_3d_normals(node_coordinates)
            cell_areas = calculate_area(node_coordinates)
            cell_centers = node_coordinates
            surface_fields, surface_fields_name = self._extract_point_fields(surface_mesh)
            
            # 🚨 FIX FOR 2D CLOSED POLYLINE: 
            # Construct connectivity for a closed polyline (line segments)
            N_points = node_coordinates.shape[0]
            indices = np.arange(N_points, dtype=np.int32)
            
            # Generate pairs of indices: [0, 1, 1, 2, 2, 3, ..., N-2, N-1, N-1, 0]
            edges = np.empty((N_points * 2), dtype=np.int32)
            # Edge i -> i+1
            edges[::2] = indices 
            # Edge i+1 -> i+2, and the last index connects back to 0
            edges[1::2] = np.roll(indices, -1)
            
            mesh_indices_flattened = edges.flatten()
            
        else:
            # 3D Logic: Use PyVista mesh connectivity features
            print(f"-> 3D Geometry detected for {cfd_filename}. Using PyVista mesh geometry.")
            mesh_with_features = surface_mesh.compute_normals(cell_normals=True, point_normals=False)
            mesh_with_features = mesh_with_features.compute_cell_sizes(length=False, area=True, volume=False)
            
            cell_normals = np.array(mesh_with_features.cell_data["Normals"], dtype=np.float32)
            cell_areas = np.array(mesh_with_features.cell_data["Area"], dtype=np.float32)[:, np.newaxis]
            cell_centers = np.array(mesh_with_features.cell_centers().points, dtype=np.float32)
            surface_fields = self._extract_cell_fields(surface_mesh)
            
            # 🚀 FIX FOR 3D MESH: Calculate faces (connectivity) correctly
            # 1. Reshape from [3, v1, v2, v3, ...] to [[3, v1, v2, v3], ...]
            # 2. Slice off the first column (the '3' count) to get [[v1, v2, v3], ...]
            # 3. Flatten to get [v1, v2, v3, v4, v5, v6, ...]
            
            # Note: We use the original mesh before cell_data_to_point_data() for faces
            faces_raw = pv.read(surface_filepath).faces 
            if not pv.read(surface_filepath).is_all_triangles:
                faces_raw = pv.read(surface_filepath).triangulate().faces

            faces = faces_raw.reshape(-1, 4)[:, 1:]
            mesh_indices_flattened = faces.flatten().astype(np.int32)
            
        # 3. Extract Global Parameters
        global_vals = parse_globals_from_filename(surface_filepath.name)
        global_params_ref = np.array(list(self.global_params_reference.values()), dtype=np.float32)

        # 4. Construct Final Dictionary
        return {
            "stl_coordinates": np.float32(node_coordinates),
            "stl_centers": np.float32(node_coordinates),
            "stl_faces": np.float32(mesh_indices_flattened),             
            "stl_areas": cell_areas,              
            "surface_mesh_centers": np.float32(cell_centers), 
            "surface_normals": np.float32(cell_normals),      
            "surface_areas": cell_areas,
            "surface_fields": surface_fields, 
            "surface_fields_name": surface_fields_name,     
            "filename": str(cfd_filename),
            "global_params_values": global_vals,
            "global_params_reference": global_params_ref,
            "volume_fields": surface_fields,
            "volume_mesh_centers": np.float32(cell_centers),
        }

    def _extract_cell_fields(self, mesh):
        """Extracts fields assuming they are stored as CELL data (used for 3D geometry)."""
        field_data_list = []
        for target_name in self.surface_variables:
            found = False
            for source_name in self.field_map.get(target_name, [target_name]):
                if source_name in mesh.cell_data:
                    data = mesh.cell_data[source_name]
                    if len(data.shape) == 1:
                        data = data[:, np.newaxis]
                    field_data_list.append(data)
                    found = True
                    break
            if not found:
                raise KeyError(f"Missing cell field: Could not find any source for '{target_name}'")
        return np.concatenate(field_data_list, axis=-1).astype(np.float32)
    
    def _extract_point_fields(self, mesh):
        """Extracts fields assuming they are stored as POINT data (used for 2D geometry)."""
        field_data_list = []
        field_name_list = []
        # Ensure fields are point data for the point-based geometry
        if not mesh.point_data:
             mesh = mesh.cell_data_to_point_data()

        for target_name in self.surface_variables:
            found = False
            for source_name in self.field_map.get(target_name, [target_name]):
                if source_name in mesh.point_data:
                    data = mesh.point_data[source_name]
                    if len(data.shape) == 1:
                        data = data[:, np.newaxis]
                    field_data_list.append(data)
                    field_name_list.append(source_name)
                    found = True
                    break
            if not found:
                raise KeyError(f"Missing point field: Could not find any source for '{target_name}'")
        return np.concatenate(field_data_list, axis=-1).astype(np.float32), field_name_list


# ==========================================
# 3. STANDALONE RUNNER
# ==========================================

def run_preprocessing(input_dir, output_dir):
    """Initializes the dataset, splits data, and saves processed files to train/val/test folders."""
    
    input_path = Path(input_dir)
    output_base_path = Path(output_dir)
    
    # 1. Initialize Dataset and Get All File Indices
    try:
        dataset = CustomSurfaceProcessor(data_path=input_path)
    except NameError:
        print("CRITICAL ERROR: CustomSurfaceProcessor class not found. Please ensure it is imported.")
        return

    total_files = len(dataset)
    print(f"Found {total_files} simulation folders to process.")
    
    # Get a list of all indices and shuffle them
    all_indices = list(range(total_files))
    random.shuffle(all_indices)
    
    # 2. Calculate Split Points
    n_train = int(total_files * SPLIT_RATIOS['train'])
    n_val = int(total_files * SPLIT_RATIOS['val'])
    # Assign remaining files to test to ensure we use every file (handles rounding)
    n_test = total_files - n_train - n_val 
    
    # 3. Split Indices
    train_indices = all_indices[:n_train]
    val_indices = all_indices[n_train : n_train + n_val]
    test_indices = all_indices[n_train + n_val :]
    
    # 4. Define Index Mapping to Output Folders
    split_map = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }
    
    print("\n--- Data Split Summary ---")
    print(f"Total Files: {total_files}")
    print(f"Train: {len(train_indices)} files ({SPLIT_RATIOS['train'] * 100:.0f}%)")
    print(f"Validation: {len(val_indices)} files ({SPLIT_RATIOS['val'] * 100:.0f}%)")
    print(f"Test: {len(test_indices)} files ({SPLIT_RATIOS['test'] * 100:.0f}%)")
    print("--------------------------\n")
    
    # 5. Process and Save Files
    # We iterate over the original indices and save them to the correct subfolder
    
    for i in tqdm(range(total_files), desc="Processing and Saving"):
        try:
            processed_data = dataset[i]
            save_name = processed_data["filename"] + ".npy"
            del processed_data["filename"]
            
            # Determine the target split folder for the current index 'i'
            target_folder = None
            if i in train_indices:
                target_folder = 'train'
            elif i in val_indices:
                target_folder = 'val'
            elif i in test_indices:
                target_folder = 'test'
            
            if target_folder:
                # Create the specific output subdirectory (e.g., /workspace/naca0012_train/train)
                output_path = output_base_path / target_folder
                output_path.mkdir(parents=True, exist_ok=True)
                
                save_path = output_path / save_name
                np.save(save_path, processed_data)
            
        except Exception as e:
            print(f"\nSkipping run {dataset.filenames[i]} due to error: {e}")

if __name__ == "__main__":
    # 🚨 CONFIGURE THESE PATHS 🚨
    RAW_DATA_PATH = "/workspace/NACA0012_SurfaceFlow"
    PROCESSED_DATA_PATH = "/workspace/naca0012"
    
    # IMPORTANT: You can easily change the split here if needed:
    # SPLIT_RATIOS['train'] = 0.80 
    # SPLIT_RATIOS['val'] = 0.10
    # SPLIT_RATIOS['test'] = 0.10
    
    run_preprocessing(RAW_DATA_PATH, PROCESSED_DATA_PATH)
    print("Custom Preprocessing Complete. Randomly split NPY files saved.")