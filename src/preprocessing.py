"""
Preprocessing module for transmission line and pylon extraction
Implements noise removal and grid generation as described in the paper
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy import spatial
import statistics
from data_structures import Point3D, SpatialHashGrid, TransmissionCorridor

class NoiseRemover:
    """Statistical noise removal based on local point distribution"""
    
    def __init__(self, k_neighbors: int = 50, std_multiplier: float = 2.0):
        """
        Initialize noise remover
        
        Args:
            k_neighbors: Number of nearest neighbors to consider
            std_multiplier: Standard deviation multiplier for outlier detection
        """
        self.k_neighbors = k_neighbors
        self.std_multiplier = std_multiplier
    
    def remove_noise(self, points: List[Point3D]) -> List[Point3D]:
        """
        Remove statistical outliers from point cloud
        
        Args:
            points: Input point cloud
            
        Returns:
            Filtered point cloud with outliers removed
        """
        if len(points) < self.k_neighbors:
            return points
        
        # Convert to numpy array for efficient computation
        coords = np.array([[p.x, p.y, p.z] for p in points])
        
        # Build KD-tree for efficient nearest neighbor search
        tree = spatial.KDTree(coords)
        
        # Calculate mean distances to k nearest neighbors
        distances = []
        for i, point in enumerate(coords):
            # Find k+1 nearest neighbors (including the point itself)
            dists, indices = tree.query(point, k=self.k_neighbors + 1)
            # Exclude the point itself (distance 0)
            mean_dist = np.mean(dists[1:])
            distances.append(mean_dist)
        
        # Calculate statistical thresholds
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        threshold = mean_distance + self.std_multiplier * std_distance
        
        # Filter points based on threshold
        filtered_points = []
        for i, distance in enumerate(distances):
            if distance <= threshold:
                filtered_points.append(points[i])
        
        return filtered_points

class PointCloudPreprocessor:
    """
    Main preprocessing class that handles noise removal and spatial hashing
    """
    
    def __init__(self, 
                 grid_size_2d: float = 5.0,
                 voxel_size_3d: float = 0.5,
                 enable_noise_removal: bool = True,
                 noise_k_neighbors: int = 50,
                 noise_std_multiplier: float = 2.0):
        """
        Initialize preprocessor
        
        Args:
            grid_size_2d: Size of 2D grid cells (5m as per paper)
            voxel_size_3d: Size of 3D voxel cells (0.5m as per paper)
            enable_noise_removal: Whether to perform noise removal
            noise_k_neighbors: K neighbors for noise removal
            noise_std_multiplier: Standard deviation multiplier for noise removal
        """
        self.grid_size_2d = grid_size_2d
        self.voxel_size_3d = voxel_size_3d
        self.enable_noise_removal = enable_noise_removal
        
        # Initialize noise remover if enabled
        if self.enable_noise_removal:
            self.noise_remover = NoiseRemover(
                k_neighbors=noise_k_neighbors,
                std_multiplier=noise_std_multiplier
            )
        else:
            self.noise_remover = None
    
    def preprocess(self, points: List[Point3D]) -> TransmissionCorridor:
        """
        Complete preprocessing pipeline
        
        Args:
            points: Raw input point cloud
            
        Returns:
            TransmissionCorridor with preprocessed data
        """
        print(f"Starting preprocessing with {len(points)} points")
        
        # Step 1: Noise removal
        if self.enable_noise_removal and self.noise_remover:
            print("Removing noise points...")
            filtered_points = self.noise_remover.remove_noise(points)
            print(f"Noise removal: {len(points)} -> {len(filtered_points)} points")
        else:
            filtered_points = points.copy()
        
        # Step 2: Initialize spatial hash grid
        print("Initializing spatial hash grid...")
        spatial_hash = SpatialHashGrid(
            grid_size_2d=self.grid_size_2d,
            voxel_size_3d=self.voxel_size_3d
        )
        
        # Update global bounding box first
        spatial_hash.update_bounding_box(filtered_points)
        
        # Step 3: Insert points into spatial hash structure
        print("Inserting points into spatial hash structure...")
        for i, point in enumerate(filtered_points):
            spatial_hash.insert_point(point)
            
            if (i + 1) % 10000 == 0:
                print(f"Processed {i + 1}/{len(filtered_points)} points")
        
        print(f"Created {spatial_hash.get_grid_count()} 2D grids and {spatial_hash.get_voxel_count()} 3D voxels")
        
        # Step 4: Create transmission corridor object
        corridor = TransmissionCorridor(
            power_lines=[],
            transmission_towers=[],
            point_cloud=filtered_points,
            spatial_hash=spatial_hash
        )
        
        return corridor
    
    def preprocess_point_cloud(self, points: List[Point3D]) -> TransmissionCorridor:
        """Alias for preprocess method for compatibility"""
        return self.preprocess(points)
    
    def remove_noise(self, points: List[Point3D]) -> List[Point3D]:
        """Remove noise points using statistical method"""
        if self.enable_noise_removal and self.noise_remover:
            return self.noise_remover.remove_noise(points)
        else:
            return points.copy()
    
    def generate_spatial_hash_grid(self, points: List[Point3D]) -> SpatialHashGrid:
        """Generate spatial hash grid from point cloud"""
        spatial_hash = SpatialHashGrid(
            grid_size_2d=self.grid_size_2d,
            voxel_size_3d=self.voxel_size_3d
        )
        
        # Update global bounding box first
        spatial_hash.update_bounding_box(points)
        
        # Insert points into spatial hash structure
        for point in points:
            spatial_hash.insert_point(point)
        
        return spatial_hash
    
    def determine_power_line_height_divider(self, points: List[Point3D]) -> float:
        """Determine height threshold for separating power lines from ground objects"""
        if not points:
            return 10.0  # Default height divider
        
        analyzer = HeightHistogramAnalyzer()
        height_analysis = analyzer.analyze_height_distribution(points)
        
        # Return the calculated height divider or default value
        return height_analysis.get('height_divider', 10.0)
    
    def calculate_grid_statistics(self, spatial_hash: SpatialHashGrid) -> dict:
        """
        Calculate statistics about the grid structure
        
        Args:
            spatial_hash: Spatial hash grid structure
            
        Returns:
            Dictionary containing grid statistics
        """
        grids = spatial_hash.get_all_grids()
        voxels = spatial_hash.get_all_voxels()
        
        if not grids:
            return {}
        
        # Grid statistics
        grid_point_counts = [len(grid.points) for grid in grids]
        grid_height_diffs = [grid.height_diff for grid in grids if grid.height_diff is not None]
        grid_densities = [grid.point_density for grid in grids if grid.point_density is not None]
        
        # Voxel statistics
        voxel_point_counts = [len(voxel.points) for voxel in voxels]
        
        stats = {
            'total_grids': len(grids),
            'total_voxels': len(voxels),
            'grid_stats': {
                'mean_points_per_grid': np.mean(grid_point_counts) if grid_point_counts else 0,
                'max_points_per_grid': max(grid_point_counts) if grid_point_counts else 0,
                'mean_height_diff': np.mean(grid_height_diffs) if grid_height_diffs else 0,
                'max_height_diff': max(grid_height_diffs) if grid_height_diffs else 0,
                'mean_density': np.mean(grid_densities) if grid_densities else 0,
                'max_density': max(grid_densities) if grid_densities else 0,
            },
            'voxel_stats': {
                'mean_points_per_voxel': np.mean(voxel_point_counts) if voxel_point_counts else 0,
                'max_points_per_voxel': max(voxel_point_counts) if voxel_point_counts else 0,
            }
        }
        
        return stats

class HeightHistogramAnalyzer:
    """
    Analyzes height distribution to determine height divider for power line segmentation
    Based on Figure 21 analysis in the paper
    """
    
    def __init__(self, bin_size: float = 1.0):
        """
        Initialize height histogram analyzer
        
        Args:
            bin_size: Size of histogram bins in meters
        """
        self.bin_size = bin_size
    
    def analyze_height_distribution(self, points: List[Point3D]) -> dict:
        """
        Analyze height distribution to find power line height divider
        
        Args:
            points: Point cloud to analyze
            
        Returns:
            Dictionary containing height analysis results
        """
        if not points:
            return {}
        
        heights = [p.z for p in points]
        min_height = min(heights)
        max_height = max(heights)
        
        # Create histogram
        bins = np.arange(min_height, max_height + self.bin_size, self.bin_size)
        hist, bin_edges = np.histogram(heights, bins=bins)
        
        # Find peaks in the histogram (potential power line heights)
        peaks = self._find_peaks(hist, bin_edges[:-1])
        
        # Determine height divider (minimum gap between transmission lines and ground)
        # Based on China's Operation code: minimum 8 meters above ground
        height_divider = self._calculate_height_divider(hist, bin_edges, peaks)
        
        return {
            'histogram': hist,
            'bin_edges': bin_edges,
            'peaks': peaks,
            'height_divider': height_divider,
            'min_height': min_height,
            'max_height': max_height,
            'mean_height': np.mean(heights),
            'std_height': np.std(heights)
        }
    
    def _find_peaks(self, histogram: np.ndarray, bin_centers: np.ndarray, 
                   prominence_threshold: float = 0.1) -> List[Tuple[float, int]]:
        """Find peaks in histogram representing potential power line heights"""
        peaks = []
        
        # Simple peak detection
        for i in range(1, len(histogram) - 1):
            if (histogram[i] > histogram[i-1] and 
                histogram[i] > histogram[i+1] and 
                histogram[i] > max(histogram) * prominence_threshold):
                peaks.append((bin_centers[i], histogram[i]))
        
        return peaks
    
    def _calculate_height_divider(self, histogram: np.ndarray, 
                                 bin_edges: np.ndarray,
                                 peaks: List[Tuple[float, int]]) -> float:
        """
        Calculate height divider for power line segmentation
        Based on the first significant gap above ground level
        """
        if not peaks:
            # Default to 8m above minimum height (regulatory minimum)
            return bin_edges[0] + 8.0
        
        # Find first significant peak (likely first power line layer)
        peaks_sorted = sorted(peaks, key=lambda x: x[0])  # Sort by height
        
        # Look for the first significant gap
        # This represents the minimum height between transmission lines and ground objects
        first_peak_height = peaks_sorted[0][0]
        
        # Set divider at a reasonable margin below the first power line peak
        # According to paper analysis, this is typically around 8-16m range
        height_divider = max(bin_edges[0] + 8.0, first_peak_height - 5.0)
        
        return height_divider

def load_point_cloud_from_las(file_path: str) -> List[Point3D]:
    """
    Load point cloud from LAS file
    Note: This is a placeholder implementation
    In practice, you would use laspy or similar library
    
    Args:
        file_path: Path to LAS file
        
    Returns:
        List of Point3D objects
    """
    try:
        import laspy
        
        las_file = laspy.read(file_path)
        points = []
        
        for i in range(len(las_file.points)):
            point = Point3D(
                x=float(las_file.x[i]),
                y=float(las_file.y[i]),
                z=float(las_file.z[i]),
                intensity=float(las_file.intensity[i]) if hasattr(las_file, 'intensity') else None,
                return_number=int(las_file.return_number[i]) if hasattr(las_file, 'return_number') else None,
                classification=int(las_file.classification[i]) if hasattr(las_file, 'classification') else None
            )
            points.append(point)
        
        return points
        
    except ImportError:
        raise ImportError("laspy library is required to load LAS files. Install with: pip install laspy")

def load_point_cloud_from_txt(file_path: str, delimiter: str = ' ') -> List[Point3D]:
    """
    Load point cloud from text file
    Expected format: x y z [intensity] [return_number] [classification]
    
    Args:
        file_path: Path to text file
        delimiter: Column delimiter
        
    Returns:
        List of Point3D objects
    """
    points = []
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            try:
                parts = line.split(delimiter)
                if len(parts) < 3:
                    continue
                
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                intensity = float(parts[3]) if len(parts) > 3 else None
                return_number = int(parts[4]) if len(parts) > 4 else None
                classification = int(parts[5]) if len(parts) > 5 else None
                
                point = Point3D(
                    x=x, y=y, z=z,
                    intensity=intensity,
                    return_number=return_number,
                    classification=classification
                )
                points.append(point)
                
            except (ValueError, IndexError) as e:
                print(f"Warning: Skipping invalid line {line_num}: {line}")
                continue
    
    return points

def generate_synthetic_transmission_data(n_points: int = 100000, 
                                       corridor_length: float = 1000.0,
                                       n_towers: int = 5,
                                       n_lines: int = 3) -> List[Point3D]:
    """
    Generate synthetic transmission corridor data for testing
    
    Args:
        n_points: Total number of points to generate
        corridor_length: Length of transmission corridor in meters
        n_towers: Number of transmission towers
        n_lines: Number of transmission lines per span
        
    Returns:
        List of synthetic Point3D objects
    """
    points = []
    np.random.seed(42)  # For reproducible results
    
    # Tower positions along the corridor
    tower_positions = np.linspace(0, corridor_length, n_towers)
    tower_height = 45.0  # Average tower height
    line_height = 35.0   # Average line height
    
    # Generate towers
    for i, x_pos in enumerate(tower_positions):
        # Tower structure points
        tower_points = int(n_points * 0.1 / n_towers)  # 10% of points for towers
        
        for _ in range(tower_points):
            # Tower body (pyramid-like structure)
            height_ratio = np.random.uniform(0.1, 1.0)
            width = (1.0 - height_ratio) * 15.0 + 2.0  # Wider at bottom
            
            x = x_pos + np.random.uniform(-width/2, width/2)
            y = np.random.uniform(-width/2, width/2)
            z = height_ratio * tower_height
            
            points.append(Point3D(x=x, y=y, z=z, classification=1))  # Tower class
    
    # Generate transmission lines
    for span in range(n_towers - 1):
        span_points = int(n_points * 0.3 / (n_towers - 1))  # 30% of points for lines
        
        x_start = tower_positions[span]
        x_end = tower_positions[span + 1]
        span_length = x_end - x_start
        
        for line_idx in range(n_lines):
            # Line offset (parallel lines)
            y_offset = (line_idx - n_lines//2) * 8.0
            
            line_points_per_line = span_points // n_lines
            
            for _ in range(line_points_per_line):
                # Catenary curve for transmission line
                x = np.random.uniform(x_start, x_end)
                y = y_offset + np.random.normal(0, 1.0)  # Some lateral noise
                
                # Catenary sag
                x_normalized = (x - x_start) / span_length - 0.5  # -0.5 to 0.5
                sag = 5.0 * (1 - 4 * x_normalized**2)  # Parabolic approximation
                z = line_height - sag + np.random.normal(0, 0.5)
                
                points.append(Point3D(x=x, y=y, z=z, classification=2))  # Line class
    
    # Generate ground points
    ground_points = n_points - len(points)
    for _ in range(ground_points):
        x = np.random.uniform(0, corridor_length)
        y = np.random.uniform(-50, 50)
        z = np.random.uniform(0, 5) + np.random.normal(0, 2)  # Ground with noise
        
        points.append(Point3D(x=x, y=y, z=z, classification=0))  # Ground class
    
    # Add some vegetation/noise points
    vegetation_points = int(n_points * 0.1)
    for _ in range(vegetation_points):
        x = np.random.uniform(0, corridor_length)
        y = np.random.uniform(-100, 100)
        z = np.random.uniform(0, 25)  # Various heights
        
        points.append(Point3D(x=x, y=y, z=z, classification=3))  # Vegetation class
    
    return points


class SyntheticDataGenerator:
    """Generate synthetic transmission corridor data for testing"""
    
    def __init__(self):
        self.n_points = 100000
        self.corridor_length = 1000.0
        self.n_towers = 5
        self.n_lines = 3
    
    def generate_transmission_corridor_scene(self) -> List[Point3D]:
        """Generate complete synthetic transmission corridor scene"""
        return generate_synthetic_transmission_data(
            n_points=self.n_points,
            corridor_length=self.corridor_length,
            n_towers=self.n_towers,
            n_lines=self.n_lines
        )