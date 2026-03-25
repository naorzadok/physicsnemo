import numpy as np
import pyvista as pv
from scipy.spatial import KDTree
from physicsnemo.datapipes.cae.readers import read_vtp
from scipy.stats.qmc import Sobol

class Tessellation2D:
    """
    A standalone 2D Tessellation reader for ordered point loops (e.g., airfoils).
    Calculates point-based area (length) and normals using segment-weighted averaging.
    """
    def __init__(self, filename, flip_normals=False):
        def fetch_mesh_vertices(mesh):
            """Fetches the vertices of a mesh."""
            points = mesh.GetPoints()
            num_points = points.GetNumberOfPoints()
            vertices = [points.GetPoint(i) for i in range(num_points)]
            return vertices
        
        # 1. Load Data
        try:
            self.raw_mesh = read_vtp(filename)
        except Exception as e:
            self.raw_mesh = pv.read(filename)
        self.points = np.array(fetch_mesh_vertices(self.raw_mesh))
        
        # 2. Compute Segment Data
        # We assume the points are ordered and form a closed loop.
        # Ensure the loop is closed for calculations
        if not np.allclose(self.points[0], self.points[-1]):
            calc_points = np.vstack([self.points, self.points[0]])
        else:
            calc_points = self.points

        # 3. Calculate Point Data (Area and Normals)
        self.data = self._compute_point_data(calc_points, flip_normals)
        
        # Store points back in the dictionary for consistency
        self.data['x'] = self.points[:, 0]
        self.data['y'] = self.points[:, 1]
        self.data['z'] = self.points[:, 2]
        self.data['t'] = np.cumsum(self.data['area'])/np.sum(self.data['area'])

    def _compute_point_data(self, points, flip_normals):
        """
        Calculates area (segment length) and normals for each point.
        Uses vectorized numpy operations for efficiency.
        """
        # vectors between consecutive points (dx, dy)
        diffs = np.diff(points, axis=0)
        segment_lengths = np.linalg.norm(diffs[:, :2], axis=1)
        
        # Segment normals using nx = -dy, ny = dx
        # These are normals for the 'edges'
        seg_nx = -diffs[:, 1]
        seg_ny = diffs[:, 0]
        
        # Normalize segment normals
        norms = np.sqrt(seg_nx**2 + seg_ny**2)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        seg_nx /= norms
        seg_ny /= norms

        # --- Point Area Calculation ---
        # Area at point i = 0.5 * (length of segment i-1 + length of segment i)
        # Using np.roll to get neighbors in the closed loop
        prev_lengths = np.roll(segment_lengths, 1)
        point_areas = 0.5 * (prev_lengths + segment_lengths)

        # --- Point Normal Calculation (Weighted Average) ---
        # We average the normal of the segment before and the segment after the point.
        # Using inverse length weighting as requested:
        # Higher weight to the normal of the shorter adjacent segment.
        w1 = 1.0 / (prev_lengths + 1e-8)
        w2 = 1.0 / (segment_lengths + 1e-8)
        
        prev_nx = np.roll(seg_nx, 1)
        prev_ny = np.roll(seg_ny, 1)
        
        avg_nx = (w1 * prev_nx + w2 * seg_nx) / (w1 + w2)
        avg_ny = (w1 * prev_ny + w2 * seg_ny) / (w1 + w2)
        
        # Re-normalize point normals
        final_norms = np.sqrt(avg_nx**2 + avg_ny**2)
        final_norms[final_norms == 0] = 1.0
        avg_nx /= final_norms
        avg_ny /= final_norms

        normals = np.column_stack((avg_nx, avg_ny))
        
        # Ensure they point outward
        normals = self._ensure_normals_outward(points[:-1, :2], normals)

        if flip_normals:
            normals *= -1

        return {
            'area': point_areas,
            'normal_x': normals[:, 0],
            'normal_y': normals[:, 1],
            'normal_z': np.zeros_like(normals[:, 1]),
        }

    def _ensure_normals_outward(self, points, normals):
        centroid = np.mean(points, axis=0)
        directions = points - centroid
        # Dot product: if positive, normal and direction-from-center align
        dot = np.sum(normals * directions, axis=1)
        normals[dot < 0] *= -1
        return normals

    def _sample_t(self, n, method="sobol", sigma=0.25, scramble=True):
            """Helper to generate t-distribution."""
            if method == "sobol":
                sampler = Sobol(d=1, scramble=scramble)
                return sampler.random(n).flatten()
            
            # Gaussian/Random spacing
            mu = 1.0 / n
            dt = np.random.normal(loc=mu, scale=sigma * mu, size=n)
            dt = np.clip(dt, 1e-8, None)
            t = np.cumsum(dt)
            t /= t[-1]
            return np.mod(t, 1.0)

    def _calculate_point_areas_from_coords(self, points_2d):
        """Helper for the resampling function to compute areas of new points."""
        # points_2d is (N, 2). Wrap it to (N+1, 2)
        wrapped = np.vstack([points_2d, points_2d[0]])
        diffs = np.diff(wrapped, axis=0)
        lens = np.linalg.norm(diffs, axis=1)
        prev_lens = np.roll(lens, 1)
        return 0.5 * (prev_lens + lens)

    def sample_boundary(self, nr_points, method="gaussian", sigma=0.25, quasirandom=False):
        """
        Main sampling function. Returns a dictionary of arrays (invar style).
        """
        # 1. Generate t samples
        sampling_method = "sobol" if quasirandom else method
        t_samples = self._sample_t(nr_points, method=sampling_method, sigma=sigma)
        t_samples = np.sort(t_samples)

        # 2. Interpolate Geometry
        # We use the raw point data (N+1) to interpolate new points at t_samples
        x_new = np.interp(t_samples, self.data['t'], self.data['x'], period=1.0)
        y_new = np.interp(t_samples, self.data['t'], self.data['y'], period=1.0)
        
        # Interpolate normals
        nx_new = np.interp(t_samples, self.data['t'], self.data['normal_x'], period=1.0)
        ny_new = np.interp(t_samples, self.data['t'], self.data['normal_y'], period=1.0)
        
        # Re-normalize interpolated normals
        mags = np.sqrt(nx_new**2 + ny_new**2)
        mags[mags == 0] = 1.0
        nx_new, ny_new = nx_new/mags, ny_new/mags

        # 3. Calculate New Point Areas (Integration weights)
        new_coords = np.stack([x_new, y_new], axis=1)
        new_areas = self._calculate_point_areas_from_coords(new_coords)

        # 4. Format as PhysicsNemo invar
        return {
            "x": x_new.reshape(-1, 1),
            "y": y_new.reshape(-1, 1),
            "z": np.zeros_like(x_new).reshape(-1, 1),
            "normal_x": nx_new.reshape(-1, 1),
            "normal_y": ny_new.reshape(-1, 1),
            "normal_z": np.zeros_like(nx_new).reshape(-1, 1),
            "area": new_areas.reshape(-1, 1),
            "t": t_samples.reshape(-1, 1)
        }

    def visualize_samples(self, invar, mag=0.05):
        """Visualize the sampled points and their normals."""
        pts = np.hstack([invar['x'], invar['y'], invar['z']])
        norms = np.hstack([invar['normal_x'], invar['normal_y'], invar['normal_z']])
        
        cloud = pv.PolyData(pts)
        plotter = pv.Plotter()
        plotter.add_mesh(cloud, color='blue', point_size=10, render_points_as_spheres=True)
        plotter.add_arrows(pts, norms, mag=mag, color='red')
        plotter.add_text(f"Samples: {len(pts)}", font_size=10)
        plotter.show()