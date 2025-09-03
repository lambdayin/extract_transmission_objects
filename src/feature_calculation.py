"""
Feature calculation module for transmission line and pylon extraction
Implements dimensional features and distribution features as described in the paper
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy import linalg
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from data_structures import Point3D, Grid2D, Voxel3D, SpatialHashGrid, GridKey, VoxelKey

class DimensionalFeatureCalculator:
    """
    Calculate dimensional features based on eigenvalue analysis
    Implements equations (3) from the paper for a1D, a2D, a3D
    """
    
    def __init__(self, min_points: int = 10):
        """
        Initialize dimensional feature calculator
        
        Args:
            min_points: Minimum points required for reliable eigenvalue calculation
        """
        self.min_points = min_points
    
    def calculate_covariance_matrix(self, points: List[Point3D]) -> np.ndarray:
        """
        Calculate 3D covariance matrix for point set
        
        Args:
            points: List of 3D points
            
        Returns:
            3x3 covariance matrix
        """
        if len(points) < 3:
            return np.eye(3)
        
        # Convert points to numpy array
        coords = np.array([[p.x, p.y, p.z] for p in points])
        
        # Center the points
        centroid = np.mean(coords, axis=0)
        centered = coords - centroid
        
        # Calculate covariance matrix
        cov_matrix = np.cov(centered.T)
        
        return cov_matrix
    
    def calculate_eigenvalues(self, covariance_matrix: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate and sort eigenvalues from covariance matrix
        
        Args:
            covariance_matrix: 3x3 covariance matrix
            
        Returns:
            Sorted eigenvalues (λ1 >= λ2 >= λ3)
        """
        eigenvalues = linalg.eigvals(covariance_matrix)
        eigenvalues = np.real(eigenvalues)  # Take real part (should be real anyway)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Sort in descending order
        
        # Ensure non-negative and non-zero
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        
        return tuple(eigenvalues)
    
    def calculate_dimensional_features(self, eigenvalues: Tuple[float, float, float]) -> Dict[str, float]:
        """
        Calculate dimensional features from eigenvalues
        Based on equation (3) in the paper
        
        Args:
            eigenvalues: Sorted eigenvalues (λ1, λ2, λ3)
            
        Returns:
            Dictionary with a1D, a2D, a3D features
        """
        lambda1, lambda2, lambda3 = eigenvalues
        
        # Avoid division by zero
        if lambda1 < 1e-10:
            return {'a1d': 0.0, 'a2d': 0.0, 'a3d': 0.0}
        
        sqrt_lambda1 = np.sqrt(lambda1)
        sqrt_lambda2 = np.sqrt(lambda2)
        sqrt_lambda3 = np.sqrt(lambda3)
        
        # Calculate dimensional features (Equation 3)
        a1d = (sqrt_lambda1 - sqrt_lambda2) / sqrt_lambda1  # Linearity
        a2d = (sqrt_lambda2 - sqrt_lambda3) / sqrt_lambda1  # Planarity
        a3d = sqrt_lambda3 / sqrt_lambda1                   # Sphericity
        
        return {
            'a1d': float(a1d),
            'a2d': float(a2d),
            'a3d': float(a3d)
        }
    
    def is_linear_structure(self, features: Dict[str, float], 
                          linearity_threshold: float = 0.7) -> bool:
        """
        Determine if point set has linear structure
        
        Args:
            features: Dimensional features dictionary
            linearity_threshold: Threshold for linearity detection
            
        Returns:
            True if structure is predominantly linear
        """
        return features['a1d'] > linearity_threshold
    
    def calculate_voxel_features(self, voxel: Voxel3D) -> Dict[str, float]:
        """
        Calculate dimensional features for a voxel
        
        Args:
            voxel: 3D voxel containing points
            
        Returns:
            Dictionary with dimensional features
        """
        if len(voxel.points) < self.min_points:
            return {'a1d': 0.0, 'a2d': 0.0, 'a3d': 0.0}
        
        # Calculate covariance matrix and eigenvalues
        cov_matrix = self.calculate_covariance_matrix(voxel.points)
        eigenvalues = self.calculate_eigenvalues(cov_matrix)
        features = self.calculate_dimensional_features(eigenvalues)
        
        # Update voxel properties
        voxel.eigenvalues = eigenvalues
        voxel.a1d = features['a1d']
        voxel.a2d = features['a2d']
        voxel.a3d = features['a3d']
        voxel.is_linear = self.is_linear_structure(features)
        
        return features

class GridFeatureCalculator:
    """
    Calculate features for 2D grids used in pylon detection
    """
    
    def __init__(self):
        pass
    
    def calculate_grid_features(self, grid: Grid2D, exclude_power_line_points: bool = True) -> Dict[str, float]:
        """
        Calculate comprehensive features for 2D grid
        Features are already calculated in Grid2D.__post_init__, but this provides updates
        
        Args:
            grid: 2D grid to analyze
            exclude_power_line_points: Whether to exclude power line points from DSM calculation
            
        Returns:
            Dictionary with grid features
        """
        if not grid.points:
            return {
                'dem': 0.0,
                'dsm': 0.0,
                'height_diff': 0.0,
                'point_density': 0.0,
                'local_max_height': 0.0,
                'continuous_height_distribution': 0.0,
                'vertical_continuity_score': 0.0
            }
        
        points = grid.points
        if exclude_power_line_points:
            # Filter out points classified as power lines if classification is available
            points = [p for p in grid.points if p.classification != 2]  # Assuming class 2 is power lines
            if not points:
                points = grid.points  # Fallback to all points
        
        z_values = [p.z for p in points]
        
        # Basic height features
        dem = min(z_values)
        dsm = max(z_values)
        height_diff = dsm - dem
        
        # Point density
        grid_area = (grid.x_max - grid.x_min) * (grid.y_max - grid.y_min)
        point_density = len(points) / grid_area if grid_area > 0 else 0
        
        # Continuous height distribution analysis
        continuous_score = self._calculate_continuous_height_distribution(points)
        
        # Vertical continuity score (important for pylon detection)
        vertical_score = self._calculate_vertical_continuity(points)
        
        features = {
            'dem': dem,
            'dsm': dsm,
            'height_diff': height_diff,
            'point_density': point_density,
            'local_max_height': dsm,
            'continuous_height_distribution': continuous_score,
            'vertical_continuity_score': vertical_score
        }
        
        # Update grid features
        grid.dem = dem
        grid.dsm = dsm
        grid.height_diff = height_diff
        grid.point_density = point_density
        grid.local_max_height = dsm
        
        return features
    
    def _calculate_continuous_height_distribution(self, points: List[Point3D], 
                                                bin_size: float = 2.0) -> float:
        """
        Calculate continuous height distribution score
        Higher score indicates more continuous vertical distribution (typical of pylons)
        
        Args:
            points: Points in the grid
            bin_size: Height bin size for analysis
            
        Returns:
            Continuous distribution score (0-1)
        """
        if len(points) < 5:
            return 0.0
        
        z_values = [p.z for p in points]
        min_z, max_z = min(z_values), max(z_values)
        
        if max_z - min_z < bin_size:
            return 0.0
        
        # Create height bins
        n_bins = int((max_z - min_z) / bin_size) + 1
        bins = np.linspace(min_z, max_z, n_bins + 1)
        hist, _ = np.histogram(z_values, bins=bins)
        
        # Calculate continuity score based on filled bins
        filled_bins = np.sum(hist > 0)
        continuity_score = filled_bins / len(hist)
        
        return continuity_score
    
    def _calculate_vertical_continuity(self, points: List[Point3D]) -> float:
        """
        Calculate vertical continuity score for pylon detection
        Pylons should have continuous vertical distribution
        
        Args:
            points: Points in the grid
            
        Returns:
            Vertical continuity score (0-1)
        """
        if len(points) < 10:
            return 0.0
        
        # Sort points by height
        sorted_points = sorted(points, key=lambda p: p.z)
        z_values = [p.z for p in sorted_points]
        
        # Calculate gaps in vertical distribution
        gaps = []
        for i in range(1, len(z_values)):
            gap = z_values[i] - z_values[i-1]
            gaps.append(gap)
        
        if not gaps:
            return 0.0
        
        # Calculate continuity based on gap statistics
        mean_gap = np.mean(gaps)
        max_gap = max(gaps)
        
        # Lower score if there are large gaps
        if max_gap > 10.0:  # Large gap indicates discontinuity
            continuity_score = max(0.0, 1.0 - (max_gap - mean_gap) / max_gap)
        else:
            continuity_score = 1.0 - (np.std(gaps) / (mean_gap + 1e-6))
        
        return max(0.0, min(1.0, continuity_score))

class NeighborhoodAnalyzer:
    """
    Analyze neighboring grids and voxels for feature enhancement
    """
    
    def __init__(self, spatial_hash: SpatialHashGrid):
        """
        Initialize neighborhood analyzer
        
        Args:
            spatial_hash: Spatial hash grid structure
        """
        self.spatial_hash = spatial_hash
    
    def calculate_height_similarity(self, grid_key: GridKey, radius: int = 1) -> float:
        """
        Calculate height similarity with neighboring grids
        Used for power line extraction
        
        Args:
            grid_key: Target grid key
            radius: Neighborhood radius
            
        Returns:
            Height similarity score (0-1)
        """
        target_grid = self.spatial_hash.get_grid(grid_key)
        if not target_grid or target_grid.dsm is None:
            return 0.0
        
        neighbors = self.spatial_hash.get_neighboring_grids(grid_key, radius)
        if not neighbors:
            return 0.0
        
        target_height = target_grid.dsm
        neighbor_heights = [grid.dsm for grid in neighbors if grid.dsm is not None]
        
        if not neighbor_heights:
            return 0.0
        
        # Calculate height similarity
        height_differences = [abs(target_height - h) for h in neighbor_heights]
        mean_diff = np.mean(height_differences)
        
        # Convert to similarity score (lower differences = higher similarity)
        similarity = max(0.0, 1.0 - mean_diff / 20.0)  # Normalize by 20m threshold
        
        return similarity
    
    def find_collinear_neighbors(self, voxel_key: VoxelKey, radius: int = 2) -> List[VoxelKey]:
        """
        Find neighboring voxels that could be part of the same line
        
        Args:
            voxel_key: Target voxel key
            radius: Search radius
            
        Returns:
            List of collinear neighbor voxel keys
        """
        target_voxel = self.spatial_hash.get_voxel(voxel_key)
        if not target_voxel or not target_voxel.is_linear:
            return []
        
        neighbors = self.spatial_hash.get_neighboring_voxels(voxel_key, radius)
        collinear_neighbors = []
        
        for neighbor in neighbors:
            if neighbor.is_linear:
                # Check if neighbors could be part of same line
                # This is a simplified check - in practice, you'd want more sophisticated analysis
                collinear_neighbors.append(neighbor.key)
        
        return collinear_neighbors

class CompassLineFilter:
    """
    Implement Compass Line Filter (CLF) for power line direction analysis
    Based on the paper's reference to CLF analysis
    """
    
    def __init__(self, n_directions: int = 36):
        """
        Initialize compass line filter
        
        Args:
            n_directions: Number of compass directions to analyze
        """
        self.n_directions = n_directions
        self.directions = np.linspace(0, 2*np.pi, n_directions, endpoint=False)
    
    def analyze_line_direction(self, points: List[Point3D]) -> Dict[str, float]:
        """
        Analyze principal direction of point set using compass line filter
        
        Args:
            points: Points representing a potential line segment
            
        Returns:
            Dictionary with direction analysis results
        """
        if len(points) < 2:
            return {'principal_direction': 0.0, 'direction_strength': 0.0}
        
        # Convert points to 2D for direction analysis (project to horizontal plane)
        coords_2d = np.array([[p.x, p.y] for p in points])
        
        # Calculate centroid
        centroid = np.mean(coords_2d, axis=0)
        
        # Calculate direction strengths
        direction_strengths = []
        
        for direction in self.directions:
            # Unit direction vector
            dir_vec = np.array([np.cos(direction), np.sin(direction)])
            
            # Project points onto this direction
            projections = []
            for coord in coords_2d:
                relative_pos = coord - centroid
                projection = np.dot(relative_pos, dir_vec)
                projections.append(projection)
            
            # Calculate spread along this direction
            projection_range = max(projections) - min(projections)
            
            # Calculate perpendicular spread (how well points align with direction)
            perp_distances = []
            for coord in coords_2d:
                relative_pos = coord - centroid
                # Distance from line through centroid in this direction
                perp_distance = abs(np.cross(relative_pos, dir_vec))
                perp_distances.append(perp_distance)
            
            perp_spread = np.mean(perp_distances)
            
            # Direction strength: high range along direction, low perpendicular spread
            if perp_spread > 0:
                strength = projection_range / (perp_spread + 1.0)
            else:
                strength = projection_range
            
            direction_strengths.append(strength)
        
        # Find principal direction
        best_idx = np.argmax(direction_strengths)
        principal_direction = self.directions[best_idx]
        direction_strength = direction_strengths[best_idx]
        
        # Normalize direction strength
        max_possible_strength = max(direction_strengths) if direction_strengths else 1.0
        normalized_strength = direction_strength / max_possible_strength if max_possible_strength > 0 else 0.0
        
        return {
            'principal_direction': principal_direction,
            'direction_strength': normalized_strength,
            'all_strengths': direction_strengths
        }

class FeatureCalculationEngine:
    """
    Main feature calculation engine that coordinates all feature calculations
    """
    
    def __init__(self, grid_size_2d: float = 5.0, voxel_size_3d: float = 0.5):
        """
        Initialize feature calculation engine
        
        Args:
            grid_size_2d: Size of 2D grid cells
            voxel_size_3d: Size of 3D voxel cells
        """
        self.grid_size_2d = grid_size_2d
        self.voxel_size_3d = voxel_size_3d
        self.dimensional_calculator = DimensionalFeatureCalculator()
        self.grid_calculator = GridFeatureCalculator()
        self.clf_analyzer = CompassLineFilter()
        
        # Cache for calculated features
        self._voxel_features_cache = {}
        self._grid_features_cache = {}
    
    def calculate_all_features(self, progress_callback=None) -> Dict[str, any]:
        """
        Calculate all features for grids and voxels
        
        Args:
            progress_callback: Optional callback function for progress reporting
            
        Returns:
            Dictionary with calculation results and statistics
        """
        print("Starting comprehensive feature calculation...")
        
        # Calculate 3D voxel features
        voxels = self.spatial_hash.get_all_voxels()
        print(f"Calculating dimensional features for {len(voxels)} voxels...")
        
        linear_voxels = 0
        for i, voxel in enumerate(voxels):
            features = self.dimensional_calculator.calculate_voxel_features(voxel)
            if voxel.is_linear:
                linear_voxels += 1
            
            if progress_callback and (i + 1) % 1000 == 0:
                progress_callback(f"Processed {i + 1}/{len(voxels)} voxels")
        
        # Calculate 2D grid features
        grids = self.spatial_hash.get_all_grids()
        print(f"Calculating grid features for {len(grids)} grids...")
        
        high_height_diff_grids = 0
        for i, grid in enumerate(grids):
            features = self.grid_calculator.calculate_grid_features(grid)
            
            # Count grids with large height differences (potential pylon areas)
            if features['height_diff'] > 15.0:  # Threshold from paper analysis
                high_height_diff_grids += 1
            
            if progress_callback and (i + 1) % 500 == 0:
                progress_callback(f"Processed {i + 1}/{len(grids)} grids")
        
        # Calculate neighborhood features
        print("Calculating neighborhood features...")
        for grid in grids:
            if grid.height_diff and grid.height_diff > 10.0:
                height_similarity = self.neighborhood_analyzer.calculate_height_similarity(grid.key)
                # Store height similarity in grid for later use
                setattr(grid, 'height_similarity', height_similarity)
        
        results = {
            'total_voxels': len(voxels),
            'linear_voxels': linear_voxels,
            'linearity_ratio': linear_voxels / len(voxels) if voxels else 0,
            'total_grids': len(grids),
            'high_height_diff_grids': high_height_diff_grids,
            'pylon_candidate_ratio': high_height_diff_grids / len(grids) if grids else 0
        }
        
        print(f"Feature calculation complete:")
        print(f"  - Linear voxels: {linear_voxels}/{len(voxels)} ({results['linearity_ratio']:.2%})")
        print(f"  - High height-diff grids: {high_height_diff_grids}/{len(grids)} ({results['pylon_candidate_ratio']:.2%})")
        
        return results
    
    def get_linear_voxels(self, min_linearity: float = 0.7) -> List[Voxel3D]:
        """
        Get all voxels with high linearity (potential power line segments)
        
        Args:
            min_linearity: Minimum linearity threshold
            
        Returns:
            List of linear voxels
        """
        linear_voxels = []
        for voxel in self.spatial_hash.get_all_voxels():
            if voxel.is_linear and voxel.a1d and voxel.a1d >= min_linearity:
                linear_voxels.append(voxel)
        
        return linear_voxels
    
    def get_pylon_candidate_grids(self, 
                                min_height_diff: float = 15.0,
                                min_point_density: float = 5.0,
                                min_continuity_score: float = 0.5) -> List[Grid2D]:
        """
        Get grids that are candidates for containing pylons
        
        Args:
            min_height_diff: Minimum height difference threshold
            min_point_density: Minimum point density threshold
            min_continuity_score: Minimum vertical continuity score
            
        Returns:
            List of pylon candidate grids
        """
        candidate_grids = []
        
        for grid in self.spatial_hash.get_all_grids():
            if (grid.height_diff and grid.height_diff >= min_height_diff and
                grid.point_density and grid.point_density >= min_point_density):
                
                # Check vertical continuity if available
                vertical_score = getattr(grid, 'vertical_continuity_score', 1.0)
                if vertical_score >= min_continuity_score:
                    candidate_grids.append(grid)
        
        return candidate_grids
    
    def calculate_2d_grid_features(self, grid: Grid2D) -> Dict[str, float]:
        """
        Calculate 2D grid features as described in paper
        
        Args:
            grid: 2D grid to analyze
            
        Returns:
            Dictionary of calculated features
        """
        if grid.key in self._grid_features_cache:
            return self._grid_features_cache[grid.key]
        
        features = self.grid_calculator.calculate_grid_features(grid)
        self._grid_features_cache[grid.key] = features
        return features
    
    def calculate_3d_dimensional_features(self, voxel: Voxel3D) -> Dict[str, float]:
        """
        Calculate 3D dimensional features for a voxel
        
        Args:
            voxel: 3D voxel to analyze
            
        Returns:
            Dictionary of calculated dimensional features
        """
        if voxel.key in self._voxel_features_cache:
            return self._voxel_features_cache[voxel.key]
        
        if len(voxel.points) < 3:
            return {'a1d': 0.0, 'a2d': 0.0, 'a3d': 0.0}
        
        # Calculate covariance matrix and eigenvalues
        cov_matrix = self.dimensional_calculator.calculate_covariance_matrix(voxel.points)
        eigenvalues = self.dimensional_calculator.calculate_eigenvalues(cov_matrix)
        features = self.dimensional_calculator.calculate_dimensional_features(eigenvalues)
        
        # Store eigenvalues in voxel
        voxel.eigenvalues = eigenvalues
        voxel.a1d = features['a1d']
        voxel.a2d = features['a2d'] 
        voxel.a3d = features['a3d']
        
        # Determine if voxel is linear (reduced threshold for better detection)
        linearity_threshold = 0.5  # Reduced from 0.7 to 0.5
        voxel.is_linear = features['a1d'] >= linearity_threshold
        
        # Calculate principal direction if linear
        if voxel.is_linear:
            coords = np.array([[p.x, p.y, p.z] for p in voxel.points])
            centroid = np.mean(coords, axis=0)
            centered = coords - centroid
            
            # SVD to get principal direction
            try:
                _, _, vt = np.linalg.svd(centered)
                voxel.principal_direction = vt[0]
            except:
                voxel.principal_direction = None
        
        # Cache the results
        self._voxel_features_cache[voxel.key] = features
        return features