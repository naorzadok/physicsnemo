import time, random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, List, Literal, Mapping, Optional, Union, Callable

import numpy as np
import pandas as pd
import pyvista as pv
import vtk
from torch.utils.data import Dataset

# Helper function to mimic original utils
def get_filenames(path):
    return [f.name for f in Path(path).iterdir() if f.is_dir()]

class CustomPaths:
    """
    Custom path getter. Since we only use VTP, we point everything to that.
    Adjust the string formatting in surface_path to match your folder structure.
    """
    @staticmethod
    def surface_path(car_dir: Path) -> Path:
        # EXAMPLE: assumes file is named like 'surface_data.vtp' inside the directory
        # Modify this line to point to your specific .vtp file
        return list(car_dir.glob("*.vtk"))[0] 

class OpenFoamDataset(Dataset):
    def __init__(
        self,
        data_path: Union[str, Path],
        surface_variables: Optional[list] = [
            "PressureCoefficient", 
            "Wall_Shear_Stress"
        ],
        # We only use surface, so volume is empty
        volume_variables: Optional[list] = [],
        # Your new 3 global scalars
        global_params_types: Optional[dict] = {
            "Mach": "scalar",
            "AOA": "scalar",
            "ReL": "scalar",
        },
        # Reference values for normalization (if needed by the model for scaling)
        global_params_reference: Optional[dict] = {
            "Mach": 1.0, 
            "AOA": 0.0, 
            "ReL": 1e6
        },
        device: int = 0,
        model_type="surface", # Enforce surface mode
    ):
        if isinstance(data_path, str):
            data_path = Path(data_path)
        self.data_path = data_path.expanduser()
        
        assert self.data_path.exists(), f"Path {self.data_path} does not exist"
        
        self.filenames = get_filenames(self.data_path)
        self.path_getter = CustomPaths

        self.surface_variables = surface_variables
        self.volume_variables = volume_variables
        self.global_params_types = global_params_types
        self.global_params_reference = global_params_reference
        self.model_type = model_type

    def _parse_globals_from_filename(self, filename):
        """
        TODO: CUSTOMIZE THIS FUNCTION
        Extract Mach, AOA, ReL from the directory name or lookup table.
        Example: if folder is 'sim_mach0.5_aoa10_rel5e6'
        """
        # Placeholder values - YOU MUST IMPLEMENT YOUR PARSING LOGIC HERE
        # val_mach = float(filename.split('_')[1].replace('mach',''))
        val_mach = 0.5 
        val_aoa = 2.0
        val_rel = 1e6
        
        return {
            "Mach": val_mach, 
            "AOA": val_aoa, 
            "ReL": val_rel
        }

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        cfd_filename = self.filenames[idx]
        car_dir = self.data_path / cfd_filename

        # --- 1. Load VTP File (Physics + Geometry) ---
        surface_filepath = self.path_getter.surface_path(car_dir)
        
        # Use PyVista for easier handling of both mesh and fields
        mesh = pv.read(surface_filepath)
        
        # --- 2. Extract Geometry (Points & Faces) ---
        # We map the VTP geometry to 'stl_' keys because the model expects them
        # Vertices (Points)
        vtp_vertices = np.array(mesh.points)
        
        # Faces (need to ensure they are triangulated for standard processing)
        if not mesh.is_all_triangles:
            mesh = mesh.triangulate()
        
        # PyVista faces come as [3, v1, v2, v3, 3, v4, v5, v6...], we need to reshape
        faces = mesh.faces.reshape(-1, 4)[:, 1:] 
        mesh_indices_flattened = faces.flatten()
        
        # Cell Centers
        vtp_centers = np.array(mesh.cell_centers().points)
        
        # Areas
        sizes = mesh.compute_cell_sizes(length=False, area=True, volume=False)
        vtp_areas = np.array(sizes.cell_data["Area"])

        # Normals
        vtp_normals = np.array(mesh.cell_normals)
        # Normalize normals just in case
        norm_mag = np.linalg.norm(vtp_normals, axis=1, keepdims=True)
        # Avoid division by zero
        norm_mag[norm_mag == 0] = 1.0
        vtp_normals = vtp_normals / norm_mag

        # --- 3. Extract Fields (Pressure/Shear) ---
        # This checks both point_data and cell_data for your variables
        field_list = []
        for var_name in self.surface_variables:
            data = None
            if var_name in mesh.cell_data:
                data = mesh.cell_data[var_name]
            elif var_name in mesh.point_data:
                # Interpolate point data to cell centers if necessary
                # Or just grab it if your pipeline expects point data
                # Here we assume we want data at cell centers to match vtp_centers
                data = mesh.point_data_to_cell_data()[var_name]
            else:
                raise ValueError(f"Variable {var_name} not found in {surface_filepath}")
            
            # Ensure shape is (N, 1) for scalars
            if len(data.shape) == 1:
                data = data[:, np.newaxis]
            field_list.append(data)
            
        surface_fields = np.concatenate(field_list, axis=-1)

        # --- 4. Normalization Logic ---
        # WARNING: PressureCoefficient is likely already dimensionless. 
        # Wall_Shear_Stress is likely in Pascals (dimensional).
        # You may want to normalize Shear but keep Cp as is.
        
        # Current logic: Pass through (Identity). 
        # TODO: Add specific normalization here if needed (e.g. divide Shear by dynamic pressure).
        # For now, we return raw values as requested.

        # --- 5. Handle Global Parameters ---
        current_globals = self._parse_globals_from_filename(cfd_filename)
        
        # Construct the values array in the order defined in init
        global_params_values_list = []
        global_params_ref_list = []
        
        for name, ptype in self.global_params_types.items():
            val = current_globals[name]
            ref = self.global_params_reference[name]
            
            # Append Value
            if isinstance(val, (list, tuple, np.ndarray)):
                 global_params_values_list.extend(val)
            else:
                 global_params_values_list.append(val)
                 
            # Append Reference
            if isinstance(ref, (list, tuple, np.ndarray)):
                 global_params_ref_list.extend(ref)
            else:
                 global_params_ref_list.append(ref)

        global_params_values = np.array(global_params_values_list, dtype=np.float32)
        global_params_ref = np.array(global_params_ref_list, dtype=np.float32)

        return {
            # Geometry (mapped from VTP)
            "stl_coordinates": np.float32(vtp_vertices),
            "stl_centers": np.float32(vtp_centers),
            "stl_faces": np.float32(mesh_indices_flattened),
            "stl_areas": np.float32(vtp_areas),
            
            # Physics Surface
            "surface_mesh_centers": np.float32(vtp_centers), # Same as stl_centers in this workflow
            "surface_normals": np.float32(vtp_normals),
            "surface_areas": np.float32(vtp_areas),
            "surface_fields": np.float32(surface_fields),
            
            # Meta
            "filename": str(cfd_filename),
            "global_params_values": global_params_values,
            "global_params_reference": global_params_ref,
            
            # Placeholders for Volume (required to prevent errors in collate_fn)
            "volume_fields": np.array([]),
            "volume_mesh_centers": np.array([]),
        }