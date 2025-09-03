"""
Transmission tower (pylon) extraction module implementing the five-step approach:
1. Identify 2D grids with large height differences
2. Use moving window to detect finer characteristics  
3. Check vertical continuity of points
4. Cluster neighboring grids belonging to same tower
5. Filter by horizontal distribution constraints
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
from sklearn.cluster import DBSCAN
from scipy import ndimage
import math

from data_structures import (
    Point3D, Grid2D, TransmissionTower, Insulator, SpatialHashGrid,
    GridKey, TransmissionCorridor
)
from feature_calculation import FeatureCalculationEngine

class PylonCandidateIdentifier:
    """
    Step 1: Identify grids with large height differences
    """
    
    def __init__(self, min_height_diff: float = 15.0):
        """
        Initialize pylon candidate identifier
        
        Args:
            min_height_diff: Minimum height difference for pylon candidates
        """
        self.min_height_diff = min_height_diff
    
    def identify_candidates(self, grids: List[Grid2D]) -> List[Grid2D]:
        """
        Identify grids with large height differences
        
        Args:
            grids: All 2D grids from spatial hash
            
        Returns:
            List of candidate grids with large height differences
        """
        candidates = []
        
        for grid in grids:
            if (grid.height_diff is not None and 
                grid.height_diff >= self.min_height_diff):
                candidates.append(grid)
        
        print(f"Identified {len(candidates)} grids with large height differences")
        return candidates

class MovingWindowAnalyzer:
    """
    Step 2: Use 2x2m moving window to detect finer characteristics
    """
    
    def __init__(self, window_size: float = 2.0, height_threshold: float = 8.0):
        """
        Initialize moving window analyzer
        
        Args:
            window_size: Size of moving window in meters
            height_threshold: Height threshold for window analysis
        """
        self.window_size = window_size
        self.height_threshold = height_threshold
    
    def analyze_with_moving_window(self, candidate_grids: List[Grid2D]) -> List[Grid2D]:
        """
        Analyze candidates using moving window approach
        
        Args:
            candidate_grids: Grids identified as candidates
            
        Returns:
            Refined list of candidate grids
        """
        refined_candidates = []
        
        for grid in candidate_grids:
            if self._analyze_grid_with_window(grid):
                refined_candidates.append(grid)
        
        print(f"Moving window analysis: {len(candidate_grids)} -> {len(refined_candidates)} grids")
        return refined_candidates
    
    def _analyze_grid_with_window(self, grid: Grid2D) -> bool:
        """
        Analyze a single grid with moving window
        """
        if not grid.points:
            return False
        
        # Create sub-windows within the grid
        grid_width = grid.x_max - grid.x_min
        grid_height = grid.y_max - grid.y_min
        
        n_windows_x = max(1, int(grid_width / self.window_size))
        n_windows_y = max(1, int(grid_height / self.window_size))
        
        valid_windows = 0
        total_windows = n_windows_x * n_windows_y
        
        for i in range(n_windows_x):
            for j in range(n_windows_y):
                x_min = grid.x_min + i * self.window_size
                x_max = min(grid.x_max, x_min + self.window_size)
                y_min = grid.y_min + j * self.window_size
                y_max = min(grid.y_max, y_min + self.window_size)
                
                # Get points in this window
                window_points = [
                    p for p in grid.points
                    if x_min <= p.x < x_max and y_min <= p.y < y_max
                ]
                
                if len(window_points) >= 3:
                    z_values = [p.z for p in window_points]
                    window_height_diff = max(z_values) - min(z_values)
                    
                    if window_height_diff >= self.height_threshold:
                        valid_windows += 1
        
        # Grid passes if sufficient windows have large height differences
        return valid_windows >= total_windows * 0.3  # At least 30% of windows

class VerticalContinuityChecker:
    """
    Step 3: Check vertical continuity of points in marked grids
    """
    
    def __init__(self, max_gap_threshold: float = 10.0, min_continuity_ratio: float = 0.6):
        """
        Initialize vertical continuity checker
        
        Args:
            max_gap_threshold: Maximum allowed gap in vertical distribution
            min_continuity_ratio: Minimum ratio of continuous height coverage
        """
        self.max_gap_threshold = max_gap_threshold
        self.min_continuity_ratio = min_continuity_ratio
    
    def check_vertical_continuity(self, candidate_grids: List[Grid2D]) -> List[Grid2D]:
        """
        Check vertical continuity of points in candidate grids
        
        Args:
            candidate_grids: Grids to check for vertical continuity
            
        Returns:
            Grids that pass vertical continuity test
        """
        continuous_grids = []
        
        for grid in candidate_grids:
            if self._has_vertical_continuity(grid):
                continuous_grids.append(grid)
        
        print(f"Vertical continuity check: {len(candidate_grids)} -> {len(continuous_grids)} grids")
        return continuous_grids
    
    def _has_vertical_continuity(self, grid: Grid2D) -> bool:
        """
        Check if grid has vertical continuity typical of pylons
        """
        if len(grid.points) < 10:
            return False
        
        # Sort points by height
        z_values = sorted([p.z for p in grid.points])
        
        # Check for large gaps in height distribution
        gaps = []
        for i in range(1, len(z_values)):
            gap = z_values[i] - z_values[i-1]
            gaps.append(gap)
        
        if not gaps:
            return False
        
        # Remove grids with excessive gaps (not continuous)
        max_gap = max(gaps)
        if max_gap > self.max_gap_threshold:
            return False
        
        # Check coverage continuity
        total_height = z_values[-1] - z_values[0]
        if total_height < 10.0:  # Too short for pylon
            return False
        
        # Analyze height bins for continuity
        n_bins = max(5, int(total_height / 2.0))  # 2m bins
        hist, bin_edges = np.histogram(z_values, bins=n_bins)
        
        # Calculate continuity ratio
        filled_bins = np.sum(hist > 0)
        continuity_ratio = filled_bins / len(hist)
        
        return continuity_ratio >= self.min_continuity_ratio

class GridClusterer:
    """
    Step 4: Cluster neighboring grids belonging to the same tower
    """
    
    def __init__(self, clustering_eps: float = 7.0, min_samples: int = 1):
        """
        Initialize grid clusterer
        
        Args:
            clustering_eps: Maximum distance between samples in same cluster
            min_samples: Minimum samples in cluster
        """
        self.clustering_eps = clustering_eps
        self.min_samples = min_samples
    
    def cluster_pylon_grids(self, 
                           continuous_grids: List[Grid2D],
                           spatial_hash: SpatialHashGrid) -> Dict[int, List[Grid2D]]:
        """
        Cluster neighboring grids that belong to the same tower
        
        Args:
            continuous_grids: Grids that passed vertical continuity check
            spatial_hash: Spatial hash for neighborhood analysis
            
        Returns:
            Dictionary mapping cluster ID to list of grids
        """
        if not continuous_grids:
            return {}
        
        # Extract grid centers for clustering
        grid_centers = []
        for grid in continuous_grids:
            center_x = (grid.x_min + grid.x_max) / 2
            center_y = (grid.y_min + grid.y_max) / 2
            grid_centers.append([center_x, center_y])
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=self.clustering_eps, min_samples=self.min_samples)
        cluster_labels = clustering.fit_predict(grid_centers)
        
        # Group grids by cluster
        clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            if label != -1:  # Exclude noise
                clusters[label].append(continuous_grids[i])
        
        print(f"Clustered {len(continuous_grids)} grids into {len(clusters)} tower clusters")
        return dict(clusters)
    
    def refine_clusters_by_height_similarity(self, 
                                           clusters: Dict[int, List[Grid2D]],
                                           height_threshold: float = 5.0) -> Dict[int, List[Grid2D]]:
        """
        Refine clusters by ensuring height similarity within clusters
        """
        refined_clusters = {}
        cluster_id = 0
        
        for original_id, grids in clusters.items():
            if len(grids) <= 1:
                refined_clusters[cluster_id] = grids
                cluster_id += 1
                continue
            
            # Group grids by similar height ranges
            height_groups = self._group_by_height_similarity(grids, height_threshold)
            
            for group in height_groups:
                if group:  # Non-empty group
                    refined_clusters[cluster_id] = group
                    cluster_id += 1
        
        return refined_clusters
    
    def _group_by_height_similarity(self, grids: List[Grid2D], threshold: float) -> List[List[Grid2D]]:
        """
        Group grids by height similarity
        """
        if not grids:
            return []
        
        # Calculate representative heights for each grid
        grid_heights = []
        for grid in grids:
            if grid.points:
                avg_height = np.mean([p.z for p in grid.points])
                grid_heights.append(avg_height)
            else:
                grid_heights.append(0.0)
        
        # Group grids with similar heights
        groups = []
        used = [False] * len(grids)
        
        for i, height in enumerate(grid_heights):
            if used[i]:
                continue
            
            group = [grids[i]]
            used[i] = True
            
            for j, other_height in enumerate(grid_heights):
                if not used[j] and abs(height - other_height) <= threshold:
                    group.append(grids[j])
                    used[j] = True
            
            groups.append(group)
        
        return groups

class HorizontalDistributionFilter:
    """
    Step 5: Filter by horizontal distribution constraints (Equation 4)
    """
    
    def __init__(self, max_wing_length: float = 15.0, margin: float = 5.0):
        """
        Initialize horizontal distribution filter
        
        Args:
            max_wing_length: Maximum wing length for HV towers
            margin: Additional margin for constraint R(m,n) ≤ (r + 5)
        """
        self.max_wing_length = max_wing_length
        self.margin = margin
    
    def filter_by_horizontal_distribution(self, 
                                        clusters: Dict[int, List[Grid2D]]) -> Dict[int, List[Grid2D]]:
        """
        Filter clusters by horizontal distribution constraints
        Based on Equation (4): R(m,n) ≤ (r + 5)
        
        Args:
            clusters: Clustered grids
            
        Returns:
            Filtered clusters that meet horizontal distribution constraints
        """
        valid_clusters = {}
        
        for cluster_id, grids in clusters.items():
            if self._validate_horizontal_distribution(grids):
                valid_clusters[cluster_id] = grids
        
        print(f"Horizontal distribution filter: {len(clusters)} -> {len(valid_clusters)} clusters")
        return valid_clusters
    
    def _validate_horizontal_distribution(self, grids: List[Grid2D]) -> bool:
        """
        Validate horizontal distribution for a cluster of grids
        """
        if not grids:
            return False
        
        # Collect all points from the cluster
        all_points = []
        for grid in grids:
            all_points.extend(grid.points)
        
        if len(all_points) < 10:
            return False
        
        # Calculate horizontal extent
        x_coords = [p.x for p in all_points]
        y_coords = [p.y for p in all_points]
        
        x_extent = max(x_coords) - min(x_coords)
        y_extent = max(y_coords) - min(y_coords)
        
        # Calculate radius in projected ground plane
        radius = max(x_extent, y_extent) / 2.0
        
        # Apply constraint: R(m,n) ≤ (r + 5)
        max_allowed_radius = self.max_wing_length + self.margin
        
        if radius > max_allowed_radius:
            return False  # Too large for a pylon
        
        # Additional check: minimum size (avoid too small clusters)
        if radius < 1.0:
            return False  # Too small for a pylon
        
        # Check aspect ratio (pylons shouldn't be extremely elongated)
        aspect_ratio = max(x_extent, y_extent) / (min(x_extent, y_extent) + 1e-6)
        if aspect_ratio > 5.0:
            return False  # Too elongated
        
        return True
    
    def calculate_cluster_properties(self, grids: List[Grid2D]) -> Dict[str, float]:
        """
        Calculate properties of a grid cluster
        """
        all_points = []
        for grid in grids:
            all_points.extend(grid.points)
        
        if not all_points:
            return {}
        
        x_coords = [p.x for p in all_points]
        y_coords = [p.y for p in all_points]
        z_coords = [p.z for p in all_points]
        
        properties = {
            'center_x': np.mean(x_coords),
            'center_y': np.mean(y_coords),
            'center_z': np.mean(z_coords),
            'x_extent': max(x_coords) - min(x_coords),
            'y_extent': max(y_coords) - min(y_coords),
            'height': max(z_coords) - min(z_coords),
            'num_points': len(all_points),
            'num_grids': len(grids)
        }
        
        properties['radius'] = max(properties['x_extent'], properties['y_extent']) / 2.0
        properties['wing_length'] = properties['radius']
        
        return properties

class TowerCenterCalculator:
    """
    Calculate precise center coordinates of towers using vertical slicing analysis
    Based on the algorithm described in Table 1 of the paper
    """
    
    def __init__(self, layer_height: float = 2.0, max_horizontal_deviation: float = 0.5):
        """
        Initialize tower center calculator
        
        Args:
            layer_height: Height of each horizontal slice (2m as in paper)
            max_horizontal_deviation: Maximum allowed horizontal deviation between layers
        """
        self.layer_height = layer_height
        self.max_horizontal_deviation = max_horizontal_deviation
    
    def calculate_tower_center(self, tower_points: List[Point3D]) -> Optional[Point3D]:
        """
        Calculate precise tower center using vertical slicing analysis
        Implements algorithm from Table 1 in the paper
        
        Args:
            tower_points: All points belonging to the tower
            
        Returns:
            Calculated center point of the tower
        """
        if len(tower_points) < 10:
            return None
        
        # Step 1: Divide points into horizontal layers
        layers = self._create_horizontal_layers(tower_points)
        if len(layers) < 3:
            return None
        
        # Step 2: Calculate center point for each layer
        layer_centers = []
        for layer_points in layers:
            if len(layer_points) >= 3:
                center = self._calculate_layer_center(layer_points)
                if center:
                    layer_centers.append(center)
        
        if len(layer_centers) < 2:
            return None
        
        # Step 3: Apply voting mechanism to find reliable center
        final_center = self._apply_voting_mechanism(layer_centers)
        
        return final_center
    
    def _create_horizontal_layers(self, points: List[Point3D]) -> List[List[Point3D]]:
        """
        Divide points into horizontal layers of specified height
        """
        if not points:
            return []
        
        z_values = [p.z for p in points]
        min_z = min(z_values)
        max_z = max(z_values)
        
        layers = []
        current_z = min_z
        
        while current_z < max_z:
            layer_points = [
                p for p in points 
                if current_z <= p.z < current_z + self.layer_height
            ]
            
            if layer_points:
                layers.append(layer_points)
            
            current_z += self.layer_height
        
        return layers
    
    def _calculate_layer_center(self, layer_points: List[Point3D]) -> Optional[Point3D]:
        """
        Calculate center point of a horizontal layer
        """
        if len(layer_points) < 3:
            return None
        
        # Calculate centroid
        x_coords = [p.x for p in layer_points]
        y_coords = [p.y for p in layer_points]
        z_coords = [p.z for p in layer_points]
        
        center_x = np.mean(x_coords)
        center_y = np.mean(y_coords)
        center_z = np.mean(z_coords)
        
        return Point3D(center_x, center_y, center_z)
    
    def _apply_voting_mechanism(self, layer_centers: List[Point3D]) -> Optional[Point3D]:
        """
        Apply voting mechanism to find most reliable center points
        Based on the algorithm in Table 1 of the paper
        """
        if len(layer_centers) < 2:
            return None
        
        # Calculate pairwise distances and angles between layer centers
        valid_centers = []
        
        for i in range(len(layer_centers)):
            votes = 0
            
            for j in range(len(layer_centers)):
                if i != j:
                    # Calculate horizontal distance
                    dx = layer_centers[i].x - layer_centers[j].x
                    dy = layer_centers[i].y - layer_centers[j].y
                    horizontal_distance = np.sqrt(dx**2 + dy**2)
                    
                    # Calculate angle with vertical
                    dz = abs(layer_centers[i].z - layer_centers[j].z)
                    if dz > 0:
                        angle = np.arctan(horizontal_distance / dz)
                        # Small angle indicates good vertical alignment
                        if angle < 0.1:  # ~6 degrees tolerance
                            votes += 1
            
            # Keep centers with good vertical alignment
            if votes >= len(layer_centers) * 0.3:  # At least 30% agreement
                valid_centers.append(layer_centers[i])
        
        if not valid_centers:
            valid_centers = layer_centers  # Fallback to all centers
        
        # Calculate final center as average of valid centers
        final_x = np.mean([c.x for c in valid_centers])
        final_y = np.mean([c.y for c in valid_centers])
        final_z = np.mean([c.z for c in valid_centers])
        
        return Point3D(final_x, final_y, final_z)

class PylonExtractor:
    """
    Main pylon extraction class that coordinates all five steps
    """
    
    def __init__(self,
                 min_height_diff: float = 15.0,
                 min_point_density: float = 5.0,
                 min_continuity_score: float = 0.5):
        """
        Initialize pylon extractor
        
        Args:
            min_height_diff: Minimum height difference for candidates
            min_point_density: Minimum point density
            min_continuity_score: Minimum vertical continuity score
        """
        self.min_height_diff = min_height_diff
        self.min_point_density = min_point_density
        self.min_continuity_score = min_continuity_score
        
        # Initialize processing components
        self.candidate_identifier = PylonCandidateIdentifier(min_height_diff)
        self.window_analyzer = MovingWindowAnalyzer()
        self.continuity_checker = VerticalContinuityChecker()
        self.grid_clusterer = GridClusterer()
        self.distribution_filter = HorizontalDistributionFilter()
        self.center_calculator = TowerCenterCalculator()
    
    def extract_pylons(self, 
                      corridor: TransmissionCorridor,
                      feature_engine: FeatureCalculationEngine) -> List[TransmissionTower]:
        """
        Complete five-step pylon extraction pipeline
        
        Args:
            corridor: Transmission corridor with spatial hash
            feature_engine: Feature calculation engine
            
        Returns:
            List of extracted transmission towers
        """
        print("Starting five-step pylon extraction...")
        
        # Get all grids from spatial hash
        if not corridor.spatial_hash:
            print("No spatial hash found in corridor")
            return []
            
        all_grids = corridor.spatial_hash.get_all_grids()
        print(f"Processing {len(all_grids)} total grids")
        
        # Step 1: Identify candidates with large height differences
        candidate_grids = self.candidate_identifier.identify_candidates(all_grids)
        print(f"Step 1 - Found {len(candidate_grids)} candidate grids")
        
        if not candidate_grids:
            print("No pylon candidates found")
            return []
        
        # Step 2: Moving window analysis
        refined_candidates = self.window_analyzer.analyze_with_moving_window(candidate_grids)
        print(f"Step 2 - {len(refined_candidates)} candidates after moving window analysis")
        
        if not refined_candidates:
            print("No candidates passed moving window analysis")
            return []
        
        # Step 3: Vertical continuity check
        continuous_grids = self.continuity_checker.check_vertical_continuity(refined_candidates)
        print(f"Step 3 - {len(continuous_grids)} grids passed vertical continuity check")
        
        if not continuous_grids:
            print("No grids passed vertical continuity check")
            return []
        
        # Step 4: Cluster neighboring grids
        grid_clusters = self.grid_clusterer.cluster_pylon_grids(
            continuous_grids, 
            corridor.spatial_hash
        )
        print(f"Step 4 - Found {len(grid_clusters)} grid clusters")
        
        if not grid_clusters:
            print("No valid grid clusters found")
            return []
        
        # Refine clusters by height similarity
        refined_clusters = self.grid_clusterer.refine_clusters_by_height_similarity(grid_clusters)
        print(f"Step 4 - {len(refined_clusters)} clusters after height refinement")
        
        # Step 5: Filter by horizontal distribution
        valid_clusters = self.distribution_filter.filter_by_horizontal_distribution(refined_clusters)
        print(f"Step 5 - {len(valid_clusters)} clusters passed horizontal distribution filter")
        
        if not valid_clusters:
            print("No clusters passed horizontal distribution filter")
            return []
        
        # Create transmission tower objects
        towers = []
        for cluster_id, grids in valid_clusters.items():
            tower = self._create_tower_from_cluster(cluster_id, grids)
            if tower:
                towers.append(tower)
        
        print(f"Pylon extraction complete: {len(towers)} towers extracted")
        
        return towers
    
    def _create_tower_from_cluster(self, 
                                  cluster_id: int, 
                                  grids: List[Grid2D]) -> Optional[TransmissionTower]:
        """
        Create transmission tower object from grid cluster
        """
        # Collect all points from the cluster
        all_points = []
        grid_keys = []
        
        for grid in grids:
            all_points.extend(grid.points)
            grid_keys.append(grid.key)
        
        if not all_points:
            return None
        
        # Calculate tower properties
        properties = self.distribution_filter.calculate_cluster_properties(grids)
        
        # Calculate precise center using vertical slicing
        center_point = self.center_calculator.calculate_tower_center(all_points)
        
        if not center_point:
            # Fallback to simple centroid
            x_coords = [p.x for p in all_points]
            y_coords = [p.y for p in all_points]
            z_coords = [p.z for p in all_points]
            
            center_point = Point3D(
                np.mean(x_coords),
                np.mean(y_coords),
                np.mean(z_coords)
            )
        
        # Determine tower type based on shape analysis
        tower_type = self._classify_tower_type(all_points, properties)
        
        # Calculate height
        z_coords = [p.z for p in all_points]
        height = max(z_coords) - min(z_coords)
        
        # Create tower object
        tower = TransmissionTower(
            center_point=center_point,
            height=height,
            points=all_points,
            grid_keys=grid_keys,
            wing_length=properties.get('wing_length', 10.0),
            tower_type=tower_type,
            insulators=[]  # Will be populated later
        )
        
        return tower
    
    def _classify_tower_type(self, 
                           tower_points: List[Point3D], 
                           properties: Dict[str, float]) -> str:
        """
        Classify tower type based on shape analysis
        Types: drum-like, goblet-like, zigzag, cat-head-like
        """
        if not properties:
            return "unknown"
        
        aspect_ratio = properties.get('height', 0) / (properties.get('radius', 1) * 2)
        wing_length = properties.get('wing_length', 0)
        
        # Simple classification based on geometric properties
        if aspect_ratio > 3.0 and wing_length < 8.0:
            return "drum-like"
        elif aspect_ratio > 2.5 and wing_length > 12.0:
            return "goblet-like"
        elif wing_length > 15.0:
            return "cat-head-like"
        else:
            return "zigzag"
    
    def analyze_pylon_statistics(self, towers: List[TransmissionTower]) -> Dict[str, any]:
        """
        Analyze statistics of extracted pylons
        """
        if not towers:
            return {}
        
        heights = [tower.height for tower in towers]
        wing_lengths = [tower.wing_length for tower in towers]
        point_counts = [len(tower.points) for tower in towers]
        
        # Count tower types
        type_counts = defaultdict(int)
        for tower in towers:
            type_counts[tower.tower_type or "unknown"] += 1
        
        stats = {
            'num_towers': len(towers),
            'height_stats': {
                'mean': np.mean(heights),
                'std': np.std(heights),
                'min': min(heights),
                'max': max(heights)
            },
            'wing_length_stats': {
                'mean': np.mean(wing_lengths),
                'std': np.std(wing_lengths),
                'min': min(wing_lengths),
                'max': max(wing_lengths)
            },
            'point_count_stats': {
                'mean': np.mean(point_counts),
                'std': np.std(point_counts),
                'min': min(point_counts),
                'max': max(point_counts),
                'total': sum(point_counts)
            },
            'tower_types': dict(type_counts)
        }
        
        return stats