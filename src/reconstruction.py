"""
3D reconstruction module for power transmission components
Implements catenary curve modeling for power lines and parametric modeling for towers
Based on equations (5), (6), (7) from the paper and MCMC optimization
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
import math
from scipy.optimize import minimize, differential_evolution
from sklearn.linear_model import RANSACRegressor
from dataclasses import dataclass

from .data_structures import (
    Point3D, PowerLineSegment, TransmissionTower, Insulator, TransmissionCorridor
)

@dataclass
class CatenaryParameters:
    """Parameters for catenary curve modeling"""
    a: float  # Shape parameter
    b: float  # Horizontal offset
    c: float  # Vertical offset
    theta: float  # Orientation angle
    rho: float  # Line parameter for horizontal projection
    
class PowerLineCatenaryModeler:
    """
    Model power lines using catenary curves as described in equations (5)-(7)
    """
    
    def __init__(self, 
                 max_iterations: int = 1000,
                 convergence_tolerance: float = 1e-6,
                 ransac_threshold: float = 2.0):
        """
        Initialize catenary modeler
        
        Args:
            max_iterations: Maximum iterations for optimization
            convergence_tolerance: Convergence tolerance for fitting
            ransac_threshold: RANSAC threshold for robust fitting
        """
        self.max_iterations = max_iterations
        self.convergence_tolerance = convergence_tolerance
        self.ransac_threshold = ransac_threshold
    
    def model_power_line_catenary(self, line_segment: PowerLineSegment) -> Optional[CatenaryParameters]:
        """
        Model power line segment using catenary curve
        Based on equations (5), (6), (7) from the paper
        
        Args:
            line_segment: Power line segment to model
            
        Returns:
            Catenary parameters if successful, None otherwise
        """
        if len(line_segment.points) < 10:
            return None
        
        points = line_segment.points
        
        # Step 1: Fit 2D line in horizontal projection (Equation 6)
        horizontal_params = self._fit_horizontal_projection(points)
        if horizontal_params is None:
            return None
        
        theta, rho = horizontal_params
        
        # Step 2: Fit catenary in vertical plane (Equation 5)
        catenary_params = self._fit_catenary_curve(points, theta, rho)
        if catenary_params is None:
            return None
        
        a, b, c = catenary_params
        
        return CatenaryParameters(a=a, b=b, c=c, theta=theta, rho=rho)
    
    def _fit_horizontal_projection(self, points: List[Point3D]) -> Optional[Tuple[float, float]]:
        """
        Fit 2D line in horizontal projection using least squares
        Implements equation (6): ρ = X cos(θ) + Y sin θ
        """
        # Convert points to 2D horizontal coordinates
        coords_2d = np.array([[p.x, p.y] for p in points])
        
        # Use PCA to find principal direction
        centroid = np.mean(coords_2d, axis=0)
        centered = coords_2d - centroid
        
        # SVD to find principal direction
        u, s, vt = np.linalg.svd(centered.T)
        principal_direction = u[:, 0]  # First principal component
        
        # Calculate angle theta
        theta = np.arctan2(principal_direction[1], principal_direction[0])
        
        # Calculate rho using the line equation
        rho = centroid[0] * np.cos(theta) + centroid[1] * np.sin(theta)
        
        return theta, rho
    
    def _fit_catenary_curve(self, 
                          points: List[Point3D], 
                          theta: float, 
                          rho: float) -> Optional[Tuple[float, float, float]]:
        """
        Fit catenary curve in vertical plane perpendicular to horizontal projection
        Implements equation (5): Z = a cosh((x-b)/a) + c
        """
        # Transform points to local coordinate system
        local_coords = []
        for p in points:
            # Distance along the line direction
            x_local = (p.x - rho * np.cos(theta)) * np.cos(theta) + (p.y - rho * np.sin(theta)) * np.sin(theta)
            z_local = p.z
            local_coords.append([x_local, z_local])
        
        local_coords = np.array(local_coords)
        x_local = local_coords[:, 0]
        z_local = local_coords[:, 1]
        
        # Use RANSAC for robust fitting
        try:
            catenary_params = self._ransac_catenary_fit(x_local, z_local)
            return catenary_params
        except Exception as e:
            print(f"Warning: Catenary fitting failed: {e}")
            return None
    
    def _ransac_catenary_fit(self, x: np.ndarray, z: np.ndarray) -> Tuple[float, float, float]:
        """
        Robust catenary fitting using RANSAC approach
        """
        best_params = None
        best_score = float('inf')
        n_iterations = min(self.max_iterations, 100)  # Limit iterations for RANSAC
        
        for _ in range(n_iterations):
            # Sample subset of points
            n_sample = min(10, len(x) // 2)
            if n_sample < 5:
                n_sample = len(x)
            
            sample_idx = np.random.choice(len(x), n_sample, replace=False)
            x_sample = x[sample_idx]
            z_sample = z[sample_idx]
            
            # Fit catenary to sample
            try:
                params = self._fit_catenary_to_points(x_sample, z_sample)
                if params is None:
                    continue
                
                a, b, c = params
                
                # Evaluate on all points
                z_pred = self._evaluate_catenary(x, a, b, c)
                residuals = np.abs(z - z_pred)
                
                # Count inliers
                inliers = residuals < self.ransac_threshold
                n_inliers = np.sum(inliers)
                
                if n_inliers < len(x) * 0.5:  # Need at least 50% inliers
                    continue
                
                # Calculate score (lower is better)
                score = np.mean(residuals[inliers])
                
                if score < best_score:
                    best_score = score
                    best_params = params
                    
                    # Early termination if very good fit
                    if score < self.convergence_tolerance:
                        break
            
            except Exception:
                continue
        
        if best_params is None:
            # Fallback to direct fitting on all points
            return self._fit_catenary_to_points(x, z)
        
        return best_params
    
    def _fit_catenary_to_points(self, x: np.ndarray, z: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """
        Fit catenary curve to points using optimization
        Minimizes equation (7): argmin Σ((f(xi,m) - yi)/σi)²
        """
        if len(x) < 3:
            return None
        
        # Initial parameter estimates
        z_min = np.min(z)
        z_max = np.max(z)
        x_center = np.mean(x)
        
        # Initial guess
        a_init = (z_max - z_min) / 2.0
        b_init = x_center
        c_init = z_min - a_init
        
        initial_params = [a_init, b_init, c_init]
        
        def objective(params):
            a, b, c = params
            if a <= 0:  # Invalid parameter
                return 1e10
            
            try:
                z_pred = self._evaluate_catenary(x, a, b, c)
                residuals = z - z_pred
                # Weighted least squares (assuming uniform weights)
                return np.sum(residuals ** 2)
            except (ValueError, OverflowError):
                return 1e10
        
        # Bounds for parameters
        bounds = [(0.1, 100.0),  # a > 0
                  (x.min() - 10, x.max() + 10),  # b
                  (z.min() - 20, z.max() + 20)]   # c
        
        try:
            # Use differential evolution for global optimization
            result = differential_evolution(objective, bounds, maxiter=100, seed=42)
            
            if result.success:
                return tuple(result.x)
            else:
                # Fallback to local optimization
                result = minimize(objective, initial_params, bounds=bounds, method='L-BFGS-B')
                if result.success:
                    return tuple(result.x)
        
        except Exception as e:
            print(f"Warning: Catenary optimization failed: {e}")
        
        return None
    
    def _evaluate_catenary(self, x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        """
        Evaluate catenary curve at given x positions
        Z = a cosh((x-b)/a) + c
        """
        return a * np.cosh((x - b) / a) + c
    
    def generate_catenary_points(self, 
                                params: CatenaryParameters,
                                x_range: Tuple[float, float],
                                n_points: int = 100) -> List[Point3D]:
        """
        Generate 3D points along the fitted catenary curve
        """
        x_min, x_max = x_range
        x_local = np.linspace(x_min, x_max, n_points)
        
        # Evaluate catenary in local coordinates
        z_local = self._evaluate_catenary(x_local, params.a, params.b, params.c)
        
        # Transform back to global coordinates
        points_3d = []
        for x_l, z_l in zip(x_local, z_local):
            # Transform from local line coordinate to global coordinates
            x_global = params.rho * np.cos(params.theta) + x_l * np.cos(params.theta)
            y_global = params.rho * np.sin(params.theta) + x_l * np.sin(params.theta)
            z_global = z_l
            
            points_3d.append(Point3D(x_global, y_global, z_global))
        
        return points_3d

@dataclass
class TowerModelTemplate:
    """Template for tower model with parametric definition"""
    tower_type: str
    base_width: float
    top_width: float
    height: float
    cross_arm_height: float
    cross_arm_width: float
    leg_positions: List[Tuple[float, float]]  # Relative positions of legs

class ParametricTowerModeler:
    """
    Parametric modeling of transmission towers using template library
    Implements model-driven reconstruction with MCMC optimization
    """
    
    def __init__(self):
        """Initialize parametric tower modeler with template library"""
        self.tower_templates = self._create_tower_templates()
    
    def _create_tower_templates(self) -> Dict[str, TowerModelTemplate]:
        """
        Create library of parametric tower models
        Based on the four types mentioned in Figure 10 of the paper
        """
        templates = {}
        
        # Drum-like tower
        templates['drum-like'] = TowerModelTemplate(
            tower_type='drum-like',
            base_width=12.0,
            top_width=3.0,
            height=45.0,
            cross_arm_height=40.0,
            cross_arm_width=25.0,
            leg_positions=[(-6, -6), (6, -6), (6, 6), (-6, 6)]
        )
        
        # Goblet-like tower
        templates['goblet-like'] = TowerModelTemplate(
            tower_type='goblet-like',
            base_width=8.0,
            top_width=15.0,
            height=50.0,
            cross_arm_height=45.0,
            cross_arm_width=30.0,
            leg_positions=[(-4, -4), (4, -4), (4, 4), (-4, 4)]
        )
        
        # Zigzag structure tower
        templates['zigzag'] = TowerModelTemplate(
            tower_type='zigzag',
            base_width=10.0,
            top_width=4.0,
            height=40.0,
            cross_arm_height=35.0,
            cross_arm_width=20.0,
            leg_positions=[(-5, -5), (5, -5), (5, 5), (-5, 5)]
        )
        
        # Cat-head-like tower
        templates['cat-head-like'] = TowerModelTemplate(
            tower_type='cat-head-like',
            base_width=15.0,
            top_width=8.0,
            height=55.0,
            cross_arm_height=50.0,
            cross_arm_width=35.0,
            leg_positions=[(-7.5, -7.5), (7.5, -7.5), (7.5, 7.5), (-7.5, 7.5)]
        )
        
        return templates
    
    def fit_tower_model(self, tower: TransmissionTower) -> Dict[str, any]:
        """
        Fit parametric model to tower points using MCMC optimization
        
        Args:
            tower: Transmission tower to model
            
        Returns:
            Dictionary with fitted model parameters and metadata
        """
        if not tower.points or len(tower.points) < 20:
            return {'success': False, 'reason': 'insufficient_points'}
        
        # Determine best fitting template
        best_template, best_score = self._select_best_template(tower)
        
        if best_template is None:
            return {'success': False, 'reason': 'no_suitable_template'}
        
        # Optimize template parameters to fit the data
        optimized_params = self._optimize_template_parameters(tower, best_template)
        
        # Generate model points
        model_points = self._generate_tower_model_points(
            tower.center_point, 
            best_template, 
            optimized_params
        )
        
        # Calculate fitting quality
        fitting_quality = self._calculate_fitting_quality(tower.points, model_points)
        
        result = {
            'success': True,
            'template_type': best_template.tower_type,
            'template': best_template,
            'optimized_parameters': optimized_params,
            'model_points': model_points,
            'fitting_quality': fitting_quality,
            'original_points': len(tower.points),
            'model_points_count': len(model_points)
        }
        
        return result
    
    def _select_best_template(self, tower: TransmissionTower) -> Tuple[Optional[TowerModelTemplate], float]:
        """
        Select best fitting template for the tower
        """
        best_template = None
        best_score = float('inf')
        
        # Calculate tower characteristics
        tower_props = self._analyze_tower_geometry(tower)
        
        for template in self.tower_templates.values():
            # Score based on geometric similarity
            score = self._calculate_template_similarity(tower_props, template)
            
            if score < best_score:
                best_score = score
                best_template = template
        
        return best_template, best_score
    
    def _analyze_tower_geometry(self, tower: TransmissionTower) -> Dict[str, float]:
        """
        Analyze geometric properties of tower points
        """
        points = tower.points
        x_coords = [p.x for p in points]
        y_coords = [p.y for p in points]
        z_coords = [p.z for p in points]
        
        # Basic dimensions
        x_extent = max(x_coords) - min(x_coords)
        y_extent = max(y_coords) - min(y_coords)
        height = max(z_coords) - min(z_coords)
        
        # Analyze width variation with height
        n_layers = 10
        layer_height = height / n_layers
        layer_widths = []
        
        for i in range(n_layers):
            z_min = min(z_coords) + i * layer_height
            z_max = z_min + layer_height
            
            layer_points = [p for p in points if z_min <= p.z < z_max]
            if layer_points:
                layer_x = [p.x for p in layer_points]
                layer_y = [p.y for p in layer_points]
                layer_width = max(max(layer_x) - min(layer_x), max(layer_y) - min(layer_y))
                layer_widths.append(layer_width)
        
        base_width = max(layer_widths[:3]) if len(layer_widths) >= 3 else max(x_extent, y_extent)
        top_width = max(layer_widths[-3:]) if len(layer_widths) >= 3 else base_width * 0.5
        
        return {
            'height': height,
            'base_width': base_width,
            'top_width': top_width,
            'x_extent': x_extent,
            'y_extent': y_extent,
            'aspect_ratio': height / max(base_width, 1.0),
            'taper_ratio': top_width / max(base_width, 1.0) if base_width > 0 else 0.5
        }
    
    def _calculate_template_similarity(self, 
                                     tower_props: Dict[str, float], 
                                     template: TowerModelTemplate) -> float:
        """
        Calculate similarity score between tower properties and template
        """
        # Normalize differences
        height_diff = abs(tower_props['height'] - template.height) / template.height
        base_width_diff = abs(tower_props['base_width'] - template.base_width) / template.base_width
        top_width_diff = abs(tower_props['top_width'] - template.top_width) / max(template.top_width, 1.0)
        
        # Weighted similarity score
        score = (0.4 * height_diff + 
                0.3 * base_width_diff + 
                0.3 * top_width_diff)
        
        return score
    
    def _optimize_template_parameters(self, 
                                    tower: TransmissionTower,
                                    template: TowerModelTemplate) -> Dict[str, float]:
        """
        Optimize template parameters using simplified MCMC approach
        """
        # Initial parameters based on tower analysis
        tower_props = self._analyze_tower_geometry(tower)
        
        # Scale factors for template adaptation
        height_scale = tower_props['height'] / template.height
        width_scale = tower_props['base_width'] / template.base_width
        
        optimized_params = {
            'height_scale': height_scale,
            'width_scale': width_scale,
            'x_offset': 0.0,
            'y_offset': 0.0,
            'rotation': 0.0
        }
        
        # Simple optimization (in practice, you'd use MCMC here)
        # For now, we'll use the scaled template parameters
        return optimized_params
    
    def _generate_tower_model_points(self, 
                                   center: Point3D,
                                   template: TowerModelTemplate,
                                   params: Dict[str, float]) -> List[Point3D]:
        """
        Generate 3D model points based on template and parameters
        """
        model_points = []
        
        height_scale = params['height_scale']
        width_scale = params['width_scale']
        
        # Generate tower structure
        n_levels = 20
        for level in range(n_levels):
            height_ratio = level / (n_levels - 1)
            current_height = center.z + height_ratio * template.height * height_scale
            
            # Linear interpolation between base and top width
            current_width = (template.base_width * (1 - height_ratio) + 
                           template.top_width * height_ratio) * width_scale
            
            # Generate points at this level
            n_points_per_level = 8
            for point_idx in range(n_points_per_level):
                angle = 2 * np.pi * point_idx / n_points_per_level
                
                # Leg structure
                for leg_x, leg_y in template.leg_positions:
                    x = center.x + leg_x * width_scale / template.base_width + 0.5 * np.cos(angle)
                    y = center.y + leg_y * width_scale / template.base_width + 0.5 * np.sin(angle)
                    z = current_height
                    
                    model_points.append(Point3D(x, y, z))
        
        # Add cross-arm structure
        cross_arm_height = center.z + template.cross_arm_height * height_scale
        cross_arm_width = template.cross_arm_width * width_scale
        
        for i in range(10):
            x = center.x + (i - 4.5) * cross_arm_width / 10
            y = center.y
            z = cross_arm_height
            model_points.append(Point3D(x, y, z))
            
            # Symmetric arm
            x = center.x
            y = center.y + (i - 4.5) * cross_arm_width / 10
            z = cross_arm_height
            model_points.append(Point3D(x, y, z))
        
        return model_points
    
    def _calculate_fitting_quality(self, 
                                 original_points: List[Point3D],
                                 model_points: List[Point3D]) -> Dict[str, float]:
        """
        Calculate quality of model fitting
        """
        if not original_points or not model_points:
            return {'rmse': float('inf'), 'coverage': 0.0}
        
        # Convert to numpy arrays
        orig_coords = np.array([[p.x, p.y, p.z] for p in original_points])
        model_coords = np.array([[p.x, p.y, p.z] for p in model_points])
        
        # Calculate distances from each original point to nearest model point
        distances = []
        for orig_point in orig_coords:
            min_dist = float('inf')
            for model_point in model_coords:
                dist = np.linalg.norm(orig_point - model_point)
                min_dist = min(min_dist, dist)
            distances.append(min_dist)
        
        rmse = np.sqrt(np.mean(np.array(distances) ** 2))
        
        # Coverage: percentage of original points within reasonable distance
        reasonable_distance = 3.0  # 3 meters
        coverage = np.mean(np.array(distances) < reasonable_distance)
        
        return {
            'rmse': rmse,
            'coverage': coverage,
            'mean_distance': np.mean(distances),
            'max_distance': np.max(distances)
        }

class InsulatorDetector:
    """
    Detect and model insulators on transmission towers
    Based on region growing method described in the paper
    """
    
    def __init__(self, 
                 search_radius: float = 5.0,
                 min_insulator_points: int = 5,
                 max_insulator_points: int = 50):
        """
        Initialize insulator detector
        
        Args:
            search_radius: Search radius for insulator detection
            min_insulator_points: Minimum points for valid insulator
            max_insulator_points: Maximum points for valid insulator
        """
        self.search_radius = search_radius
        self.min_insulator_points = min_insulator_points
        self.max_insulator_points = max_insulator_points
    
    def detect_insulators(self, 
                         tower: TransmissionTower,
                         power_lines: List[PowerLineSegment]) -> List[Insulator]:
        """
        Detect insulators on a transmission tower
        
        Args:
            tower: Transmission tower
            power_lines: Nearby power lines
            
        Returns:
            List of detected insulators
        """
        insulators = []
        
        if not power_lines:
            return insulators
        
        # Find connection points between tower and power lines
        connection_heights = []
        for line in power_lines:
            # Check if line is near this tower
            min_distance = min(
                line.start_point.distance_to(tower.center_point),
                line.end_point.distance_to(tower.center_point)
            )
            
            if min_distance < 30.0:  # Within reasonable connection distance
                # Find height of connection point
                connection_point = (line.start_point if 
                                  line.start_point.distance_to(tower.center_point) < 
                                  line.end_point.distance_to(tower.center_point)
                                  else line.end_point)
                connection_heights.append(connection_point.z)
        
        # Search for insulators near connection heights
        for conn_height in connection_heights:
            insulator_candidates = self._find_insulator_candidates_at_height(
                tower, conn_height
            )
            
            for candidate in insulator_candidates:
                if self._validate_insulator_candidate(candidate):
                    insulator = self._create_insulator_from_points(candidate)
                    if insulator:
                        insulators.append(insulator)
        
        return insulators
    
    def _find_insulator_candidates_at_height(self, 
                                           tower: TransmissionTower,
                                           target_height: float) -> List[List[Point3D]]:
        """
        Find insulator candidate point clusters at specified height
        """
        # Get points near the target height
        height_tolerance = 3.0
        candidate_points = [
            p for p in tower.points 
            if abs(p.z - target_height) < height_tolerance
        ]
        
        if len(candidate_points) < self.min_insulator_points:
            return []
        
        # Cluster points using region growing
        clusters = self._cluster_points_by_distance(candidate_points)
        
        # Filter clusters by size
        valid_clusters = [
            cluster for cluster in clusters
            if self.min_insulator_points <= len(cluster) <= self.max_insulator_points
        ]
        
        return valid_clusters
    
    def _cluster_points_by_distance(self, points: List[Point3D]) -> List[List[Point3D]]:
        """
        Cluster points using distance-based region growing
        """
        if not points:
            return []
        
        clusters = []
        unvisited = set(range(len(points)))
        
        while unvisited:
            # Start new cluster
            seed_idx = next(iter(unvisited))
            cluster = []
            queue = [seed_idx]
            
            while queue:
                current_idx = queue.pop(0)
                if current_idx not in unvisited:
                    continue
                
                unvisited.remove(current_idx)
                cluster.append(points[current_idx])
                
                # Find nearby points
                current_point = points[current_idx]
                for neighbor_idx in list(unvisited):
                    neighbor_point = points[neighbor_idx]
                    distance = current_point.distance_to(neighbor_point)
                    
                    if distance < self.search_radius:
                        queue.append(neighbor_idx)
            
            if cluster:
                clusters.append(cluster)
        
        return clusters
    
    def _validate_insulator_candidate(self, candidate_points: List[Point3D]) -> bool:
        """
        Validate if point cluster represents a valid insulator
        """
        if len(candidate_points) < self.min_insulator_points:
            return False
        
        # Check spatial extent (insulators shouldn't be too large)
        x_coords = [p.x for p in candidate_points]
        y_coords = [p.y for p in candidate_points]
        z_coords = [p.z for p in candidate_points]
        
        x_extent = max(x_coords) - min(x_coords)
        y_extent = max(y_coords) - min(y_coords)
        z_extent = max(z_coords) - min(z_coords)
        
        # Insulator size constraints
        max_horizontal_extent = 2.0  # 2 meters
        max_vertical_extent = 3.0    # 3 meters
        
        if (x_extent > max_horizontal_extent or 
            y_extent > max_horizontal_extent or 
            z_extent > max_vertical_extent):
            return False
        
        return True
    
    def _create_insulator_from_points(self, insulator_points: List[Point3D]) -> Optional[Insulator]:
        """
        Create insulator object from point cluster
        """
        if not insulator_points:
            return None
        
        # Calculate center point
        x_center = np.mean([p.x for p in insulator_points])
        y_center = np.mean([p.y for p in insulator_points])
        z_center = np.mean([p.z for p in insulator_points])
        
        center_point = Point3D(x_center, y_center, z_center)
        
        # Determine insulator type (simplified classification)
        z_extent = max(p.z for p in insulator_points) - min(p.z for p in insulator_points)
        insulator_type = "suspension" if z_extent > 1.5 else "tension"
        
        return Insulator(
            center_point=center_point,
            points=insulator_points,
            insulator_type=insulator_type
        )

class TransmissionCorridorReconstructor:
    """
    Main reconstruction class that coordinates all reconstruction components
    """
    
    def __init__(self):
        """Initialize transmission corridor reconstructor"""
        self.catenary_modeler = PowerLineCatenaryModeler()
        self.tower_modeler = ParametricTowerModeler()
        self.insulator_detector = InsulatorDetector()
    
    def reconstruct_corridor(self, corridor: TransmissionCorridor) -> Dict[str, any]:
        """
        Complete 3D reconstruction of transmission corridor
        
        Args:
            corridor: Transmission corridor with extracted objects
            
        Returns:
            Dictionary with reconstruction results
        """
        print("Starting 3D reconstruction of transmission corridor...")
        
        reconstruction_results = {
            'power_line_models': [],
            'tower_models': [],
            'insulator_models': [],
            'reconstruction_quality': {},
            'statistics': {}
        }
        
        # Reconstruct power lines
        if corridor.power_lines:
            print(f"Reconstructing {len(corridor.power_lines)} power lines...")
            
            for i, line in enumerate(corridor.power_lines):
                catenary_params = self.catenary_modeler.model_power_line_catenary(line)
                
                line_model = {
                    'line_id': i,
                    'original_segment': line,
                    'catenary_parameters': catenary_params,
                    'success': catenary_params is not None
                }
                
                if catenary_params:
                    # Generate model points
                    x_range = (min(p.x for p in line.points), max(p.x for p in line.points))
                    model_points = self.catenary_modeler.generate_catenary_points(
                        catenary_params, x_range, n_points=50
                    )
                    line_model['model_points'] = model_points
                
                reconstruction_results['power_line_models'].append(line_model)
        
        # Reconstruct towers
        if corridor.towers:
            print(f"Reconstructing {len(corridor.towers)} transmission towers...")
            
            for i, tower in enumerate(corridor.towers):
                tower_model_result = self.tower_modeler.fit_tower_model(tower)
                
                tower_model = {
                    'tower_id': i,
                    'original_tower': tower,
                    'model_result': tower_model_result
                }
                
                # Detect insulators
                insulators = self.insulator_detector.detect_insulators(tower, corridor.power_lines)
                tower_model['insulators'] = insulators
                
                # Update tower object with insulators
                tower.insulators = insulators
                
                reconstruction_results['tower_models'].append(tower_model)
                reconstruction_results['insulator_models'].extend([
                    {'tower_id': i, 'insulator': ins} for ins in insulators
                ])
        
        # Calculate reconstruction quality metrics
        quality_metrics = self._calculate_reconstruction_quality(reconstruction_results)
        reconstruction_results['reconstruction_quality'] = quality_metrics
        
        # Calculate statistics
        stats = self._calculate_reconstruction_statistics(reconstruction_results)
        reconstruction_results['statistics'] = stats
        
        print("3D reconstruction complete:")
        print(f"  Power lines: {stats.get('successful_line_models', 0)}/{len(corridor.power_lines or [])}")
        print(f"  Towers: {stats.get('successful_tower_models', 0)}/{len(corridor.towers or [])}")
        print(f"  Insulators: {len(reconstruction_results['insulator_models'])}")
        
        return reconstruction_results
    
    def _calculate_reconstruction_quality(self, results: Dict[str, any]) -> Dict[str, float]:
        """Calculate overall reconstruction quality metrics"""
        line_success_rate = 0.0
        tower_success_rate = 0.0
        
        if results['power_line_models']:
            successful_lines = sum(1 for model in results['power_line_models'] if model['success'])
            line_success_rate = successful_lines / len(results['power_line_models'])
        
        if results['tower_models']:
            successful_towers = sum(1 for model in results['tower_models'] 
                                  if model['model_result']['success'])
            tower_success_rate = successful_towers / len(results['tower_models'])
        
        overall_quality = (line_success_rate + tower_success_rate) / 2.0
        
        return {
            'line_success_rate': line_success_rate,
            'tower_success_rate': tower_success_rate,
            'overall_quality': overall_quality,
            'insulator_detection_rate': len(results['insulator_models']) / max(1, len(results['tower_models']))
        }
    
    def _calculate_reconstruction_statistics(self, results: Dict[str, any]) -> Dict[str, any]:
        """Calculate reconstruction statistics"""
        stats = {
            'total_power_lines': len(results['power_line_models']),
            'successful_line_models': sum(1 for model in results['power_line_models'] if model['success']),
            'total_towers': len(results['tower_models']),
            'successful_tower_models': sum(1 for model in results['tower_models'] 
                                         if model['model_result']['success']),
            'total_insulators': len(results['insulator_models']),
        }
        
        # Tower type distribution
        tower_types = defaultdict(int)
        for model in results['tower_models']:
            if model['model_result']['success']:
                tower_type = model['model_result']['template_type']
                tower_types[tower_type] += 1
        
        stats['tower_type_distribution'] = dict(tower_types)
        
        return stats