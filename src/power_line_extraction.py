"""
Power line extraction module implementing the two-step approach described in the paper:
1. Local line segment extraction based on dimensional features
2. Global PL merging based on local collinearity and height approximation
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
import math

from .data_structures import (
    Point3D, Voxel3D, Grid2D, PowerLineSegment, SpatialHashGrid, 
    VoxelKey, GridKey, TransmissionCorridor
)
from .feature_calculation import FeatureCalculationEngine, CompassLineFilter

class LocalSegmentExtractor:
    """
    Extract local power line segments based on dimensional features
    Implements the local segmentation step from the paper
    """
    
    def __init__(self, 
                 linearity_threshold: float = 0.7,
                 min_segment_length: float = 5.0,
                 min_points_per_segment: int = 10,
                 clustering_eps: float = 2.0):
        """
        Initialize local segment extractor
        
        Args:
            linearity_threshold: Minimum linearity (a1D) for power line detection
            min_segment_length: Minimum length of valid line segments
            min_points_per_segment: Minimum points required for a segment
            clustering_eps: DBSCAN epsilon parameter for clustering
        """
        self.linearity_threshold = linearity_threshold
        self.min_segment_length = min_segment_length
        self.min_points_per_segment = min_points_per_segment
        self.clustering_eps = clustering_eps
        self.compass_filter = CompassLineFilter()
    
    def extract_local_segments(self, 
                             linear_voxels: List[Voxel3D],
                             height_threshold: float = 8.0) -> List[PowerLineSegment]:
        """
        Extract local power line segments from linear voxels
        
        Args:
            linear_voxels: Voxels with high linearity
            height_threshold: Minimum height above ground for power lines
            
        Returns:
            List of local power line segments
        """
        print(f"Extracting local segments from {len(linear_voxels)} linear voxels...")
        
        # Filter voxels by height threshold
        elevated_voxels = []
        for voxel in linear_voxels:
            if voxel.points:
                avg_height = np.mean([p.z for p in voxel.points])
                if avg_height >= height_threshold:
                    elevated_voxels.append(voxel)
        
        print(f"Found {len(elevated_voxels)} elevated linear voxels")
        
        # Cluster voxels into potential line segments
        voxel_clusters = self._cluster_linear_voxels(elevated_voxels)
        
        # Extract segments from each cluster
        segments = []
        for cluster_id, cluster_voxels in voxel_clusters.items():
            if cluster_id == -1:  # Noise cluster
                continue
                
            segment = self._create_segment_from_cluster(cluster_voxels)
            if segment and self._validate_segment(segment):
                segments.append(segment)
        
        print(f"Extracted {len(segments)} local segments")
        return segments
    
    def _cluster_linear_voxels(self, linear_voxels: List[Voxel3D]) -> Dict[int, List[Voxel3D]]:
        """
        Cluster linear voxels into groups that could form line segments
        """
        if not linear_voxels:
            return {}
        
        # Extract voxel centers for clustering
        voxel_centers = []
        for voxel in linear_voxels:
            if voxel.points:
                center_x = (voxel.x_min + voxel.x_max) / 2
                center_y = (voxel.y_min + voxel.y_max) / 2
                center_z = (voxel.z_min + voxel.z_max) / 2
                voxel_centers.append([center_x, center_y, center_z])
        
        if len(voxel_centers) < 2:
            return {}
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=self.clustering_eps, min_samples=2)
        cluster_labels = clustering.fit_predict(voxel_centers)
        
        # Group voxels by cluster
        clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            clusters[label].append(linear_voxels[i])
        
        return dict(clusters)
    
    def _create_segment_from_cluster(self, cluster_voxels: List[Voxel3D]) -> Optional[PowerLineSegment]:
        """
        Create a power line segment from clustered voxels
        """
        if len(cluster_voxels) < 2:
            return None
        
        # Collect all points from the cluster
        all_points = []
        voxel_keys = []
        grid_keys = set()
        
        for voxel in cluster_voxels:
            all_points.extend(voxel.points)
            voxel_keys.append(voxel.key)
            # Derive grid key from voxel key (approximate)
            # In practice, you'd maintain a proper mapping
            grid_key = GridKey(voxel.key.r // 10, voxel.key.c // 10)  # Approximate conversion
            grid_keys.add(grid_key)
        
        if len(all_points) < self.min_points_per_segment:
            return None
        
        # Fit line through points to get start and end points
        start_point, end_point, direction = self._fit_line_to_points(all_points)
        if start_point is None:
            return None
        
        # Calculate segment length
        length = start_point.distance_to(end_point)
        if length < self.min_segment_length:
            return None
        
        # Create segment
        segment = PowerLineSegment(
            points=all_points,
            start_point=start_point,
            end_point=end_point,
            direction=direction,
            length=length,
            grid_keys=list(grid_keys),
            voxel_keys=voxel_keys,
            is_candidate=True
        )
        
        return segment
    
    def _fit_line_to_points(self, points: List[Point3D]) -> Tuple[Optional[Point3D], Optional[Point3D], Optional[np.ndarray]]:
        """
        Fit a 3D line through points and determine start/end points
        """
        if len(points) < 2:
            return None, None, None
        
        # Convert to numpy array
        coords = np.array([[p.x, p.y, p.z] for p in points])
        
        # Calculate centroid
        centroid = np.mean(coords, axis=0)
        
        # Perform SVD to find principal direction
        centered = coords - centroid
        u, s, vt = np.linalg.svd(centered)
        direction = vt[0]  # First principal component
        
        # Project points onto the line direction
        projections = np.dot(centered, direction)
        
        # Find extreme points
        min_proj_idx = np.argmin(projections)
        max_proj_idx = np.argmax(projections)
        
        start_coords = coords[min_proj_idx]
        end_coords = coords[max_proj_idx]
        
        start_point = Point3D(start_coords[0], start_coords[1], start_coords[2])
        end_point = Point3D(end_coords[0], end_coords[1], end_coords[2])
        
        return start_point, end_point, direction
    
    def _validate_segment(self, segment: PowerLineSegment) -> bool:
        """
        Validate if a segment meets quality criteria
        """
        if segment.length < self.min_segment_length:
            return False
        
        if len(segment.points) < self.min_points_per_segment:
            return False
        
        # Check linearity of points
        if segment.direction is not None:
            coords = np.array([[p.x, p.y, p.z] for p in segment.points])
            centroid = np.mean(coords, axis=0)
            centered = coords - centroid
            
            # Calculate how well points align with the fitted direction
            projections = np.dot(centered, segment.direction)
            line_points = centroid + np.outer(projections, segment.direction)
            
            # Calculate perpendicular distances
            perp_distances = np.linalg.norm(centered - (line_points - centroid), axis=1)
            mean_perp_distance = np.mean(perp_distances)
            
            # Segment is valid if points are well-aligned
            return mean_perp_distance < 2.0  # 2m tolerance
        
        return True

class GlobalLineMerger:
    """
    Merge local segments into complete power lines based on collinearity and height approximation
    """
    
    def __init__(self,
                 collinearity_threshold: float = 0.9,
                 height_diff_threshold: float = 5.0,
                 gap_tolerance: float = 10.0,
                 direction_tolerance: float = 0.2):
        """
        Initialize global line merger
        
        Args:
            collinearity_threshold: Minimum collinearity score for merging
            height_diff_threshold: Maximum height difference between segments
            gap_tolerance: Maximum gap between segments for merging
            direction_tolerance: Maximum direction difference (radians)
        """
        self.collinearity_threshold = collinearity_threshold
        self.height_diff_threshold = height_diff_threshold
        self.gap_tolerance = gap_tolerance
        self.direction_tolerance = direction_tolerance
        self.compass_filter = CompassLineFilter()
    
    def merge_segments_to_lines(self, segments: List[PowerLineSegment]) -> List[PowerLineSegment]:
        """
        Merge local segments into complete power lines
        
        Args:
            segments: List of local power line segments
            
        Returns:
            List of merged power line segments representing complete lines
        """
        print(f"Merging {len(segments)} local segments into complete lines...")
        
        if not segments:
            return []
        
        # Analyze segment compatibility
        compatibility_matrix = self._calculate_compatibility_matrix(segments)
        
        # Find connected components (segments that should be merged)
        merged_groups = self._find_connected_components(segments, compatibility_matrix)
        
        # Merge each group into a complete line
        merged_lines = []
        for group in merged_groups:
            merged_line = self._merge_segment_group(group)
            if merged_line:
                merged_lines.append(merged_line)
        
        print(f"Merged into {len(merged_lines)} complete power lines")
        return merged_lines
    
    def _calculate_compatibility_matrix(self, segments: List[PowerLineSegment]) -> np.ndarray:
        """
        Calculate compatibility matrix between all segment pairs
        """
        n = len(segments)
        compatibility = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                score = self._calculate_segment_compatibility(segments[i], segments[j])
                compatibility[i][j] = score
                compatibility[j][i] = score
        
        return compatibility
    
    def _calculate_segment_compatibility(self, seg1: PowerLineSegment, seg2: PowerLineSegment) -> float:
        """
        Calculate compatibility score between two segments
        """
        # Check collinearity
        collinearity_score = self._calculate_collinearity(seg1, seg2)
        if collinearity_score < self.collinearity_threshold:
            return 0.0
        
        # Check height similarity
        height1 = np.mean([p.z for p in seg1.points])
        height2 = np.mean([p.z for p in seg2.points])
        height_diff = abs(height1 - height2)
        if height_diff > self.height_diff_threshold:
            return 0.0
        
        # Check gap between segments
        gap = self._calculate_segment_gap(seg1, seg2)
        if gap > self.gap_tolerance:
            return 0.0
        
        # Check direction similarity
        direction_score = self._calculate_direction_similarity(seg1, seg2)
        if direction_score < 0.8:  # High threshold for direction similarity
            return 0.0
        
        # Combine all scores
        combined_score = (collinearity_score + direction_score) / 2.0
        return combined_score
    
    def _calculate_collinearity(self, seg1: PowerLineSegment, seg2: PowerLineSegment) -> float:
        """
        Calculate collinearity score between two segments
        """
        if seg1.direction is None or seg2.direction is None:
            return 0.0
        
        # Calculate angle between directions
        dot_product = np.dot(seg1.direction, seg2.direction)
        # Handle numerical errors
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle = np.arccos(abs(dot_product))  # Use abs to handle opposite directions
        
        # Convert to collinearity score (0 = perpendicular, 1 = parallel)
        collinearity = 1.0 - (2.0 * angle / np.pi)
        return max(0.0, collinearity)
    
    def _calculate_segment_gap(self, seg1: PowerLineSegment, seg2: PowerLineSegment) -> float:
        """
        Calculate minimum gap between two segments
        """
        # Calculate distances between all endpoint combinations
        distances = [
            seg1.start_point.distance_to(seg2.start_point),
            seg1.start_point.distance_to(seg2.end_point),
            seg1.end_point.distance_to(seg2.start_point),
            seg1.end_point.distance_to(seg2.end_point)
        ]
        
        return min(distances)
    
    def _calculate_direction_similarity(self, seg1: PowerLineSegment, seg2: PowerLineSegment) -> float:
        """
        Calculate direction similarity between segments
        """
        if seg1.direction is None or seg2.direction is None:
            return 0.0
        
        # Use compass line filter for direction analysis
        combined_points = seg1.points + seg2.points
        direction_analysis = self.compass_filter.analyze_line_direction(combined_points)
        
        return direction_analysis['direction_strength']
    
    def _find_connected_components(self, segments: List[PowerLineSegment], 
                                 compatibility_matrix: np.ndarray) -> List[List[PowerLineSegment]]:
        """
        Find connected components of compatible segments
        """
        n = len(segments)
        visited = [False] * n
        components = []
        
        def dfs(node_idx, component):
            visited[node_idx] = True
            component.append(segments[node_idx])
            
            for neighbor_idx in range(n):
                if (not visited[neighbor_idx] and 
                    compatibility_matrix[node_idx][neighbor_idx] > self.collinearity_threshold):
                    dfs(neighbor_idx, component)
        
        for i in range(n):
            if not visited[i]:
                component = []
                dfs(i, component)
                if component:
                    components.append(component)
        
        return components
    
    def _merge_segment_group(self, segments: List[PowerLineSegment]) -> Optional[PowerLineSegment]:
        """
        Merge a group of compatible segments into a single line
        """
        if not segments:
            return None
        
        if len(segments) == 1:
            return segments[0]
        
        # Combine all points
        all_points = []
        all_voxel_keys = []
        all_grid_keys = set()
        
        for segment in segments:
            all_points.extend(segment.points)
            all_voxel_keys.extend(segment.voxel_keys)
            all_grid_keys.update(segment.grid_keys)
        
        # Refit line through all points
        start_point, end_point, direction = self._fit_merged_line(all_points)
        if start_point is None:
            return None
        
        # Calculate total length
        length = start_point.distance_to(end_point)
        
        # Create merged segment
        merged_segment = PowerLineSegment(
            points=all_points,
            start_point=start_point,
            end_point=end_point,
            direction=direction,
            length=length,
            grid_keys=list(all_grid_keys),
            voxel_keys=all_voxel_keys,
            is_candidate=True,
            line_id=None  # Will be assigned later
        )
        
        return merged_segment
    
    def _fit_merged_line(self, points: List[Point3D]) -> Tuple[Optional[Point3D], Optional[Point3D], Optional[np.ndarray]]:
        """
        Fit a line through merged points using robust methods
        """
        if len(points) < 2:
            return None, None, None
        
        # Convert to numpy array
        coords = np.array([[p.x, p.y, p.z] for p in points])
        
        # Use RANSAC for robust line fitting
        try:
            # Fit line in 2D projection first (more stable)
            X_2d = coords[:, :2]  # X, Y coordinates
            Z = coords[:, 2]      # Z coordinates
            
            # Fit X-Y relationship
            if np.std(coords[:, 0]) > np.std(coords[:, 1]):
                # X has more variation, predict Y from X
                X = coords[:, 0:1]
                y = coords[:, 1]
                ransac = RANSACRegressor(random_state=42)
                ransac.fit(X, y)
            else:
                # Y has more variation, predict X from Y
                X = coords[:, 1:2]
                y = coords[:, 0]
                ransac = RANSACRegressor(random_state=42)
                ransac.fit(X, y)
            
            # Find extreme points along the fitted line
            centroid = np.mean(coords, axis=0)
            
            # Use PCA for direction
            centered = coords - centroid
            u, s, vt = np.linalg.svd(centered)
            direction = vt[0]
            
            # Project points to find extremes
            projections = np.dot(centered, direction)
            min_idx = np.argmin(projections)
            max_idx = np.argmax(projections)
            
            start_coords = coords[min_idx]
            end_coords = coords[max_idx]
            
            start_point = Point3D(start_coords[0], start_coords[1], start_coords[2])
            end_point = Point3D(end_coords[0], end_coords[1], end_coords[2])
            
            return start_point, end_point, direction
            
        except Exception as e:
            print(f"Warning: Robust line fitting failed, falling back to simple method: {e}")
            # Fallback to simple PCA-based fitting
            return self._simple_line_fit(points)
    
    def _simple_line_fit(self, points: List[Point3D]) -> Tuple[Optional[Point3D], Optional[Point3D], Optional[np.ndarray]]:
        """Simple PCA-based line fitting as fallback"""
        coords = np.array([[p.x, p.y, p.z] for p in points])
        centroid = np.mean(coords, axis=0)
        centered = coords - centroid
        
        u, s, vt = np.linalg.svd(centered)
        direction = vt[0]
        
        projections = np.dot(centered, direction)
        min_idx = np.argmin(projections)
        max_idx = np.argmax(projections)
        
        start_coords = coords[min_idx]
        end_coords = coords[max_idx]
        
        start_point = Point3D(start_coords[0], start_coords[1], start_coords[2])
        end_point = Point3D(end_coords[0], end_coords[1], end_coords[2])
        
        return start_point, end_point, direction

class PowerLineExtractor:
    """
    Main power line extraction class that coordinates local and global extraction
    """
    
    def __init__(self,
                 linearity_threshold: float = 0.7,
                 height_threshold: float = 8.0,
                 min_segment_length: float = 10.0,
                 collinearity_threshold: float = 0.8):
        """
        Initialize power line extractor
        
        Args:
            linearity_threshold: Threshold for voxel linearity
            height_threshold: Minimum height for power lines
            min_segment_length: Minimum segment length
            collinearity_threshold: Threshold for segment merging
        """
        self.linearity_threshold = linearity_threshold
        self.height_threshold = height_threshold
        
        self.local_extractor = LocalSegmentExtractor(
            linearity_threshold=linearity_threshold,
            min_segment_length=min_segment_length
        )
        
        self.global_merger = GlobalLineMerger(
            collinearity_threshold=collinearity_threshold
        )
    
    def extract_power_lines(self, 
                           corridor: TransmissionCorridor,
                           feature_engine: FeatureCalculationEngine) -> List[PowerLineSegment]:
        """
        Complete power line extraction pipeline
        
        Args:
            corridor: Transmission corridor with spatial hash
            feature_engine: Feature calculation engine
            
        Returns:
            List of extracted power line segments
        """
        print("Starting power line extraction...")
        
        # Step 1: Get linear voxels
        linear_voxels = feature_engine.get_linear_voxels(min_linearity=self.linearity_threshold)
        print(f"Found {len(linear_voxels)} linear voxels")
        
        # Step 2: Extract local segments
        local_segments = self.local_extractor.extract_local_segments(
            linear_voxels, 
            height_threshold=self.height_threshold
        )
        
        if not local_segments:
            print("No local segments found")
            return []
        
        # Step 3: Merge segments into complete lines
        complete_lines = self.global_merger.merge_segments_to_lines(local_segments)
        
        # Step 4: Assign line IDs and finalize
        for i, line in enumerate(complete_lines):
            line.line_id = i + 1
        
        print(f"Power line extraction complete: {len(complete_lines)} lines extracted")
        
        # Update corridor with power lines
        corridor.power_lines = complete_lines
        
        return complete_lines
    
    def filter_low_voltage_lines(self, 
                               power_lines: List[PowerLineSegment],
                               min_height: float = 15.0,
                               min_length: float = 50.0) -> List[PowerLineSegment]:
        """
        Filter out low-voltage transmission lines
        
        Args:
            power_lines: List of extracted power lines
            min_height: Minimum average height for high-voltage lines
            min_length: Minimum length for high-voltage lines
            
        Returns:
            Filtered list of high-voltage power lines
        """
        hv_lines = []
        
        for line in power_lines:
            # Calculate average height
            avg_height = np.mean([p.z for p in line.points])
            
            # Filter by height and length criteria
            if avg_height >= min_height and line.length >= min_length:
                hv_lines.append(line)
        
        print(f"Filtered {len(power_lines)} -> {len(hv_lines)} high-voltage lines")
        return hv_lines
    
    def analyze_power_line_statistics(self, power_lines: List[PowerLineSegment]) -> Dict[str, any]:
        """
        Analyze statistics of extracted power lines
        """
        if not power_lines:
            return {}
        
        lengths = [line.length for line in power_lines]
        heights = []
        for line in power_lines:
            line_heights = [p.z for p in line.points]
            heights.extend(line_heights)
        
        total_points = sum(len(line.points) for line in power_lines)
        
        stats = {
            'num_lines': len(power_lines),
            'total_points': total_points,
            'length_stats': {
                'mean': np.mean(lengths),
                'std': np.std(lengths),
                'min': min(lengths),
                'max': max(lengths),
                'total': sum(lengths)
            },
            'height_stats': {
                'mean': np.mean(heights),
                'std': np.std(heights),
                'min': min(heights),
                'max': max(heights)
            }
        }
        
        return stats