"""
Data structures for transmission line and pylon extraction
Based on "Automatic Extraction of High-Voltage Power Transmission Objects from UAV Lidar Point Clouds"
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import hashlib
from collections import defaultdict

@dataclass
class Point3D:
    """3D point with coordinates and optional attributes"""
    x: float
    y: float
    z: float
    intensity: Optional[float] = None
    return_number: Optional[int] = None
    classification: Optional[int] = None
    
    def __hash__(self):
        return hash((self.x, self.y, self.z))
    
    def __eq__(self, other):
        if not isinstance(other, Point3D):
            return False
        return np.isclose(self.x, other.x) and np.isclose(self.y, other.y) and np.isclose(self.z, other.z)
    
    def distance_to(self, other: 'Point3D') -> float:
        """Calculate Euclidean distance to another point"""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)

@dataclass
class GridKey:
    """Key for 2D grid indexing"""
    m: int  # row
    n: int  # column
    
    def __hash__(self):
        return hash((self.m, self.n))
    
    def __eq__(self, other):
        return isinstance(other, GridKey) and self.m == other.m and self.n == other.n

@dataclass
class VoxelKey:
    """Key for 3D voxel indexing"""
    r: int  # row
    c: int  # column
    h: int  # height
    
    def __hash__(self):
        return hash((self.r, self.c, self.h))
    
    def __eq__(self, other):
        return isinstance(other, VoxelKey) and self.r == other.r and self.c == other.c and self.h == other.h

@dataclass
class Grid2D:
    """2D grid cell containing points and features"""
    key: GridKey
    points: List[Point3D]
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    
    # Grid features
    dem: Optional[float] = None  # Digital Elevation Model
    dsm: Optional[float] = None  # Digital Surface Model
    height_diff: Optional[float] = None  # DSM - DEM
    point_density: Optional[float] = None
    local_max_height: Optional[float] = None
    
    def __post_init__(self):
        if not self.points:
            return
        
        z_values = [p.z for p in self.points]
        self.dem = min(z_values)
        self.dsm = max(z_values)
        self.height_diff = self.dsm - self.dem
        self.point_density = len(self.points) / ((self.x_max - self.x_min) * (self.y_max - self.y_min))
        self.local_max_height = self.dsm
    
    def add_point(self, point: Point3D):
        """Add a point to the grid"""
        self.points.append(point)
        # Update features
        if self.dem is None or point.z < self.dem:
            self.dem = point.z
        if self.dsm is None or point.z > self.dsm:
            self.dsm = point.z
            self.local_max_height = point.z
        if self.dem is not None and self.dsm is not None:
            self.height_diff = self.dsm - self.dem
        
        grid_area = (self.x_max - self.x_min) * (self.y_max - self.y_min)
        self.point_density = len(self.points) / grid_area if grid_area > 0 else 0

@dataclass
class Voxel3D:
    """3D voxel containing points and features"""
    key: VoxelKey
    points: List[Point3D]
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    
    # Dimensional features based on eigenvalues
    eigenvalues: Optional[Tuple[float, float, float]] = None
    a1d: Optional[float] = None  # linearity
    a2d: Optional[float] = None  # planarity  
    a3d: Optional[float] = None  # sphericity
    is_linear: bool = False

class SpatialHashGrid:
    """
    Spatial hashing storage structure for efficient point cloud management
    Implements the paper's sparse grid method using spatial hashing matrix
    """
    
    def __init__(self, x_min=None, x_max=None, y_min=None, y_max=None, z_min=None, z_max=None, 
                 grid_size_2d: float = 5.0, voxel_size_3d: float = 0.5):
        """
        Initialize spatial hash grid
        
        Args:
            x_min, x_max, y_min, y_max, z_min, z_max: Bounding box (will be auto-computed if None)
            grid_size_2d: Size of 2D grid cells (default 5m as in paper)
            voxel_size_3d: Size of 3D voxel cells (default 0.5m as in paper)
        """
        self.grid_size_2d = grid_size_2d
        self.voxel_size_3d = voxel_size_3d
        
        # Hash tables for storage
        self.grid_2d_hash: Dict[GridKey, Grid2D] = {}
        self.voxel_3d_hash: Dict[VoxelKey, Voxel3D] = {}
        
        # Bounding box for normalization
        if all(v is not None for v in [x_min, x_max, y_min, y_max, z_min, z_max]):
            self.x_min_global = x_min
            self.x_max_global = x_max
            self.y_min_global = y_min
            self.y_max_global = y_max
            self.z_min_global = z_min
            self.z_max_global = z_max
            self.bbox_initialized = True
        else:
            self.x_min_global = float('inf')
            self.x_max_global = float('-inf')
            self.y_min_global = float('inf')
            self.y_max_global = float('-inf')
            self.z_min_global = float('inf')
            self.z_max_global = float('-inf')
            self.bbox_initialized = False
        
        # Grid-to-voxel mapping for hierarchical structure
        self.grid_to_voxels: Dict[GridKey, Set[VoxelKey]] = defaultdict(set)
    
    def update_bounding_box(self, points: List[Point3D]):
        """Update global bounding box with new points"""
        for point in points:
            self.x_min_global = min(self.x_min_global, point.x)
            self.x_max_global = max(self.x_max_global, point.x)
            self.y_min_global = min(self.y_min_global, point.y)
            self.y_max_global = max(self.y_max_global, point.y)
            self.z_min_global = min(self.z_min_global, point.z)
            self.z_max_global = max(self.z_max_global, point.z)
    
    def point_to_grid_key(self, point: Point3D) -> GridKey:
        """
        Convert point coordinates to 2D grid key
        Based on Equation (1) in the paper
        """
        m = int((point.y - self.y_min_global) / self.grid_size_2d)
        n = int((point.x - self.x_min_global) / self.grid_size_2d)
        return GridKey(m, n)
    
    def point_to_voxel_key(self, point: Point3D, grid_key: GridKey) -> VoxelKey:
        """
        Convert point coordinates to 3D voxel key within a grid
        Based on Equation (2) in the paper
        """
        # Local coordinates within the grid
        x_min_local = self.x_min_global + grid_key.n * self.grid_size_2d
        y_min_local = self.y_min_global + grid_key.m * self.grid_size_2d
        z_min_local = self.z_min_global
        
        r = int((point.y - y_min_local) / self.voxel_size_3d)
        c = int((point.x - x_min_local) / self.voxel_size_3d)
        h = int((point.z - z_min_local) / self.voxel_size_3d)
        
        return VoxelKey(r, c, h)
    
    def insert_point(self, point: Point3D):
        """Insert a point into the spatial hash structure"""
        # Initialize bounding box if needed
        if not self.bbox_initialized:
            self.update_bounding_box([point])
            self.bbox_initialized = True
        else:
            self.update_bounding_box([point])
        
        # Get 2D grid key
        grid_key = self.point_to_grid_key(point)
        
        # Create or get 2D grid
        if grid_key not in self.grid_2d_hash:
            x_min = self.x_min_global + grid_key.n * self.grid_size_2d
            x_max = x_min + self.grid_size_2d
            y_min = self.y_min_global + grid_key.m * self.grid_size_2d
            y_max = y_min + self.grid_size_2d
            
            self.grid_2d_hash[grid_key] = Grid2D(
                key=grid_key,
                points=[],
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max
            )
        
        # Add point to 2D grid
        self.grid_2d_hash[grid_key].add_point(point)
        
        # Get 3D voxel key
        voxel_key = self.point_to_voxel_key(point, grid_key)
        
        # Create or get 3D voxel
        if voxel_key not in self.voxel_3d_hash:
            grid = self.grid_2d_hash[grid_key]
            
            voxel_x_min = grid.x_min + voxel_key.c * self.voxel_size_3d
            voxel_x_max = voxel_x_min + self.voxel_size_3d
            voxel_y_min = grid.y_min + voxel_key.r * self.voxel_size_3d
            voxel_y_max = voxel_y_min + self.voxel_size_3d
            voxel_z_min = self.z_min_global + voxel_key.h * self.voxel_size_3d
            voxel_z_max = voxel_z_min + self.voxel_size_3d
            
            self.voxel_3d_hash[voxel_key] = Voxel3D(
                key=voxel_key,
                points=[],
                x_min=voxel_x_min,
                x_max=voxel_x_max,
                y_min=voxel_y_min,
                y_max=voxel_y_max,
                z_min=voxel_z_min,
                z_max=voxel_z_max
            )
        
        # Add point to 3D voxel
        self.voxel_3d_hash[voxel_key].points.append(point)
        
        # Update grid-to-voxel mapping
        self.grid_to_voxels[grid_key].add(voxel_key)
    
    def get_grid(self, grid_key: GridKey) -> Optional[Grid2D]:
        """Get 2D grid by key"""
        return self.grid_2d_hash.get(grid_key)
    
    def get_voxel(self, voxel_key: VoxelKey) -> Optional[Voxel3D]:
        """Get 3D voxel by key"""
        return self.voxel_3d_hash.get(voxel_key)
    
    def get_neighboring_grids(self, grid_key: GridKey, radius: int = 1) -> List[Grid2D]:
        """Get neighboring grids within specified radius"""
        neighbors = []
        for dm in range(-radius, radius + 1):
            for dn in range(-radius, radius + 1):
                neighbor_key = GridKey(grid_key.m + dm, grid_key.n + dn)
                if neighbor_key in self.grid_2d_hash:
                    neighbors.append(self.grid_2d_hash[neighbor_key])
        return neighbors
    
    def get_neighboring_voxels(self, voxel_key: VoxelKey, radius: int = 1) -> List[Voxel3D]:
        """Get neighboring voxels within specified radius"""
        neighbors = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                for dh in range(-radius, radius + 1):
                    neighbor_key = VoxelKey(
                        voxel_key.r + dr,
                        voxel_key.c + dc,
                        voxel_key.h + dh
                    )
                    if neighbor_key in self.voxel_3d_hash:
                        neighbors.append(self.voxel_3d_hash[neighbor_key])
        return neighbors
    
    def get_voxels_in_grid(self, grid_key: GridKey) -> List[Voxel3D]:
        """Get all voxels belonging to a specific grid"""
        voxels = []
        if grid_key in self.grid_to_voxels:
            for voxel_key in self.grid_to_voxels[grid_key]:
                if voxel_key in self.voxel_3d_hash:
                    voxels.append(self.voxel_3d_hash[voxel_key])
        return voxels
    
    def get_all_grids(self) -> List[Grid2D]:
        """Get all 2D grids"""
        return list(self.grid_2d_hash.values())
    
    def get_all_voxels(self) -> List[Voxel3D]:
        """Get all 3D voxels"""
        return list(self.voxel_3d_hash.values())
    
    def get_all_2d_grids(self) -> Dict[GridKey, Grid2D]:
        """Get all 2D grids as dictionary"""
        return self.grid_2d_hash.copy()
    
    def get_all_3d_voxels(self) -> Dict[VoxelKey, Voxel3D]:
        """Get all 3D voxels as dictionary"""
        return self.voxel_3d_hash.copy()
    
    def get_grid_count(self) -> int:
        """Get total number of grids"""
        return len(self.grid_2d_hash)
    
    def get_voxel_count(self) -> int:
        """Get total number of voxels"""
        return len(self.voxel_3d_hash)
    
    def add_point(self, point: Point3D):
        """Alias for insert_point for compatibility"""
        self.insert_point(point)
    
    def clear(self):
        """Clear all data structures"""
        self.grid_2d_hash.clear()
        self.voxel_3d_hash.clear()
        self.grid_to_voxels.clear()
        
        # Reset bounding box
        self.x_min_global = float('inf')
        self.x_max_global = float('-inf')
        self.y_min_global = float('inf')
        self.y_max_global = float('-inf')
        self.z_min_global = float('inf')
        self.z_max_global = float('-inf')
        self.bbox_initialized = False

@dataclass
class PowerLineSegment:
    """Power line segment with geometric properties"""
    points: List[Point3D]
    start_point: Optional[Point3D] = None
    end_point: Optional[Point3D] = None
    direction: Optional[np.ndarray] = None
    length: float = 0.0
    grid_keys: Optional[List[GridKey]] = None
    voxel_keys: Optional[List[VoxelKey]] = None
    is_candidate: bool = True
    line_id: Optional[int] = None
    confidence: float = 0.0
    catenary_parameters: Optional['CatenaryParameters'] = None
    
    def __post_init__(self):
        if self.grid_keys is None:
            self.grid_keys = []
        if self.voxel_keys is None:
            self.voxel_keys = []
        
        if self.points and len(self.points) >= 2:
            self.start_point = self.points[0]
            self.end_point = self.points[-1]
            
            # Calculate direction vector
            start_coords = np.array([self.start_point.x, self.start_point.y, self.start_point.z])
            end_coords = np.array([self.end_point.x, self.end_point.y, self.end_point.z])
            direction_vec = end_coords - start_coords
            self.length = np.linalg.norm(direction_vec)
            self.direction = direction_vec / self.length if self.length > 0 else np.zeros(3)

@dataclass
class CatenaryParameters:
    """Parameters for catenary curve fitting as described in paper Equation (5)"""
    a: float  # catenary parameter
    h: float  # horizontal offset
    k: float  # vertical offset
    theta: float  # line orientation angle
    rho: float   # line distance parameter
    fitting_error: float = 0.0

@dataclass
class TowerModel:
    """3D model parameters for tower reconstruction"""
    tower_type: str  # drum-like, goblet-like, zigzag, cat-head-like
    height: float
    base_width: float
    top_width: float
    model_parameters: Dict[str, float]
    gibbs_energy: float = 0.0  # Energy model fitness

@dataclass 
class TransmissionTower:
    """Transmission tower with properties and location"""
    center_point: Optional[Point3D] = None
    height: float = 0.0
    points: Optional[List[Point3D]] = None
    grid_keys: Optional[List[GridKey]] = None
    wing_length: float = 0.0
    tower_type: Optional[str] = None  # drum-like, goblet-like, zigzag, cat-head-like
    insulators: Optional[List['Insulator']] = None
    confidence: float = 0.0
    tower_model: Optional[TowerModel] = None
    
    def __post_init__(self):
        if self.points is None:
            self.points = []
        if self.grid_keys is None:
            self.grid_keys = []
        if self.insulators is None:
            self.insulators = []

@dataclass
class Insulator:
    """Insulator component of transmission tower"""
    center_point: Point3D
    points: List[Point3D]
    insulator_type: str = "suspension"  # suspension, tension, etc.
    length: float = 0.0
    orientation: Optional[np.ndarray] = None

@dataclass
class Grid2DFeatures:
    """2D grid features as described in the paper"""
    dem: float  # Digital Elevation Model - minimum height
    dsm: float  # Digital Surface Model - maximum height  
    height_diff: float  # DSM - DEM
    point_density: float  # Points per square meter
    local_max_height: float  # Local maximum height value
    normalized_dsm: float  # nDSM for topography normalization
    
@dataclass
class Voxel3DFeatures:
    """3D dimensional features based on eigenvalue analysis"""
    eigenvalues: Tuple[float, float, float]  # λ1 > λ2 > λ3
    a1d: float  # Linearity measure: (√λ1 - √λ2) / √λ1
    a2d: float  # Planarity measure: (√λ2 - √λ3) / √λ1  
    a3d: float  # Sphericity measure: √λ3 / √λ1
    is_linear: bool  # True if a1d > linearity_threshold
    principal_direction: Optional[np.ndarray] = None  # Direction of first eigenvector

@dataclass
class ConnectivityAnalysis:
    """Results of topological connectivity analysis"""
    connected_power_lines: List[int]  # IDs of connected power line segments
    connected_towers: List[int]       # IDs of connected towers
    connectivity_score: float         # Overall connectivity measure [0,1]
    isolated_lines: List[int]         # IDs of isolated line segments
    isolated_towers: List[int]        # IDs of isolated towers

@dataclass
class TransmissionCorridor:
    """Complete transmission corridor containing all extracted objects"""
    power_lines: Optional[List[PowerLineSegment]] = None
    transmission_towers: Optional[List[TransmissionTower]] = None  # Renamed for consistency with main.py
    point_cloud: Optional[List[Point3D]] = None
    spatial_hash: Optional[SpatialHashGrid] = None
    processing_metadata: Optional[Dict[str, any]] = None
    
    def __post_init__(self):
        if self.power_lines is None:
            self.power_lines = []
        if self.transmission_towers is None:
            self.transmission_towers = []
        if self.processing_metadata is None:
            self.processing_metadata = {}