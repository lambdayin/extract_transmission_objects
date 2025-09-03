"""
Optimization module for refining extraction results based on topological relationships
between power lines and pylons. Implements the connectivity constraints described in the paper.
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
import math
from collections import defaultdict

from data_structures import (
    Point3D, PowerLineSegment, TransmissionTower, TransmissionCorridor
)

class TopologicalConstraintAnalyzer:
    """
    Analyzes topological relationships between power lines and pylons
    Based on the constraints described in Section 3.4 of the paper
    """
    
    def __init__(self,
                 max_connection_distance: float = 15.0,
                 direction_tolerance: float = 0.3,  # ~17 degrees
                 height_tolerance: float = 10.0):
        """
        Initialize topological constraint analyzer
        
        Args:
            max_connection_distance: Maximum distance for PL-tower connection
            direction_tolerance: Maximum angle difference in radians
            height_tolerance: Maximum height difference for connections
        """
        self.max_connection_distance = max_connection_distance
        self.direction_tolerance = direction_tolerance
        self.height_tolerance = height_tolerance
    
    def analyze_connectivity(self, 
                           power_lines: List[PowerLineSegment],
                           towers: List[TransmissionTower]) -> Dict[str, any]:
        """
        Analyze connectivity between power lines and towers
        
        Args:
            power_lines: Extracted power line segments
            towers: Extracted transmission towers
            
        Returns:
            Dictionary containing connectivity analysis results
        """
        if not power_lines or not towers:
            return {'connected_lines': [], 'connected_towers': [], 'connections': []}
        
        # Find connections between lines and towers
        connections = self._find_connections(power_lines, towers)
        
        # Analyze connection patterns
        connected_lines = set()
        connected_towers = set()
        
        for line_id, tower_id, distance, connection_point in connections:
            connected_lines.add(line_id)
            connected_towers.add(tower_id)
        
        analysis = {
            'total_connections': len(connections),
            'connected_lines': len(connected_lines),
            'connected_towers': len(connected_towers),
            'unconnected_lines': len(power_lines) - len(connected_lines),
            'unconnected_towers': len(towers) - len(connected_towers),
            'connections': connections,
            'connection_ratio': len(connections) / (len(power_lines) * len(towers)) if power_lines and towers else 0
        }
        
        return analysis
    
    def _find_connections(self, 
                         power_lines: List[PowerLineSegment],
                         towers: List[TransmissionTower]) -> List[Tuple[int, int, float, Point3D]]:
        """
        Find valid connections between power lines and towers
        
        Returns:
            List of (line_index, tower_index, distance, connection_point) tuples
        """
        connections = []
        
        for line_idx, line in enumerate(power_lines):
            for tower_idx, tower in enumerate(towers):
                connection_result = self._check_line_tower_connection(line, tower)
                
                if connection_result:
                    distance, connection_point = connection_result
                    connections.append((line_idx, tower_idx, distance, connection_point))
        
        return connections
    
    def _check_line_tower_connection(self, 
                                   line: PowerLineSegment, 
                                   tower: TransmissionTower) -> Optional[Tuple[float, Point3D]]:
        """
        Check if a power line is connected to a tower
        
        Args:
            line: Power line segment
            tower: Transmission tower
            
        Returns:
            (distance, connection_point) if connected, None otherwise
        """
        # Check distance between line endpoints and tower center
        start_distance = line.start_point.distance_to(tower.center_point)
        end_distance = line.end_point.distance_to(tower.center_point)
        
        min_distance = min(start_distance, end_distance)
        connection_point = line.start_point if start_distance < end_distance else line.end_point
        
        # Distance constraint
        if min_distance > self.max_connection_distance:
            return None
        
        # Height compatibility check
        height_diff = abs(connection_point.z - tower.center_point.z)
        if height_diff > self.height_tolerance:
            return None
        
        # Check if line passes near tower (alternative connection check)
        closest_point_on_line = self._find_closest_point_on_line(line, tower.center_point)
        line_distance = closest_point_on_line.distance_to(tower.center_point)
        
        if line_distance < self.max_connection_distance:
            return line_distance, closest_point_on_line
        
        return None
    
    def _find_closest_point_on_line(self, line: PowerLineSegment, point: Point3D) -> Point3D:
        """
        Find the closest point on a line segment to a given point
        """
        # Vector from start to end of line
        line_vec = np.array([
            line.end_point.x - line.start_point.x,
            line.end_point.y - line.start_point.y,
            line.end_point.z - line.start_point.z
        ])
        
        # Vector from line start to the point
        point_vec = np.array([
            point.x - line.start_point.x,
            point.y - line.start_point.y,
            point.z - line.start_point.z
        ])
        
        # Project point onto line
        line_length_sq = np.dot(line_vec, line_vec)
        if line_length_sq == 0:
            return line.start_point
        
        t = np.dot(point_vec, line_vec) / line_length_sq
        t = max(0, min(1, t))  # Clamp to line segment
        
        # Calculate closest point
        closest = np.array([line.start_point.x, line.start_point.y, line.start_point.z]) + t * line_vec
        
        return Point3D(closest[0], closest[1], closest[2])

class DirectionConsistencyChecker:
    """
    Check direction consistency between power lines and tower arrangements
    Constraint (b): dominant orientation should be parallel to connecting line
    """
    
    def __init__(self, direction_tolerance: float = 0.2):
        """
        Initialize direction consistency checker
        
        Args:
            direction_tolerance: Maximum allowed direction deviation in radians
        """
        self.direction_tolerance = direction_tolerance
    
    def check_direction_consistency(self,
                                  power_lines: List[PowerLineSegment],
                                  towers: List[TransmissionTower],
                                  connections: List[Tuple[int, int, float, Point3D]]) -> Dict[str, any]:
        """
        Check direction consistency between lines and tower arrangements
        
        Args:
            power_lines: Power line segments
            towers: Transmission towers
            connections: Connection relationships
            
        Returns:
            Direction consistency analysis results
        """
        consistent_lines = []
        inconsistent_lines = []
        
        for line_idx, line in enumerate(power_lines):
            # Find towers connected to this line
            connected_towers = [
                towers[tower_idx] for line_id, tower_idx, _, _ in connections
                if line_id == line_idx
            ]
            
            if len(connected_towers) >= 2:
                consistency = self._check_line_tower_direction_consistency(line, connected_towers)
                
                if consistency['is_consistent']:
                    consistent_lines.append(line_idx)
                else:
                    inconsistent_lines.append(line_idx)
        
        analysis = {
            'consistent_lines': consistent_lines,
            'inconsistent_lines': inconsistent_lines,
            'consistency_ratio': len(consistent_lines) / len(power_lines) if power_lines else 0,
            'total_analyzed': len(consistent_lines) + len(inconsistent_lines)
        }
        
        return analysis
    
    def _check_line_tower_direction_consistency(self,
                                              line: PowerLineSegment,
                                              connected_towers: List[TransmissionTower]) -> Dict[str, any]:
        """
        Check if line direction is consistent with connected towers
        """
        if len(connected_towers) < 2 or line.direction is None:
            return {'is_consistent': False, 'reason': 'insufficient_data'}
        
        # Calculate direction between towers
        tower1, tower2 = connected_towers[0], connected_towers[1]
        tower_direction = np.array([
            tower2.center_point.x - tower1.center_point.x,
            tower2.center_point.y - tower1.center_point.y,
            tower2.center_point.z - tower1.center_point.z
        ])
        
        # Normalize directions
        tower_direction_norm = tower_direction / (np.linalg.norm(tower_direction) + 1e-10)
        line_direction_norm = line.direction / (np.linalg.norm(line.direction) + 1e-10)
        
        # Calculate angle between directions
        dot_product = np.dot(tower_direction_norm, line_direction_norm)
        dot_product = np.clip(dot_product, -1.0, 1.0)  # Handle numerical errors
        
        angle = np.arccos(abs(dot_product))  # Use absolute value to handle opposite directions
        
        is_consistent = angle <= self.direction_tolerance
        
        return {
            'is_consistent': is_consistent,
            'angle_deviation': angle,
            'dot_product': abs(dot_product),
            'reason': 'consistent' if is_consistent else 'direction_mismatch'
        }

class InterferenceFilter:
    """
    Filter out interference objects like tall trees, signal poles, and low-voltage lines
    Based on connectivity analysis and topological constraints
    """
    
    def __init__(self):
        pass
    
    def filter_interference_objects(self,
                                   power_lines: List[PowerLineSegment],
                                   towers: List[TransmissionTower],
                                   connectivity_analysis: Dict[str, any]) -> Tuple[List[PowerLineSegment], List[TransmissionTower]]:
        """
        Filter out interference objects based on connectivity analysis
        
        Args:
            power_lines: Original power line segments
            towers: Original transmission towers
            connectivity_analysis: Results from connectivity analysis
            
        Returns:
            Tuple of (filtered_power_lines, filtered_towers)
        """
        # Get connected objects
        connections = connectivity_analysis.get('connections', [])
        connected_line_ids = set()
        connected_tower_ids = set()
        
        for line_id, tower_id, _, _ in connections:
            connected_line_ids.add(line_id)
            connected_tower_ids.add(tower_id)
        
        # Filter power lines: keep only those connected to towers
        filtered_lines = []
        for line_idx, line in enumerate(power_lines):
            if line_idx in connected_line_ids:
                # Additional validation for HV lines
                if self._validate_hv_power_line(line):
                    filtered_lines.append(line)
            else:
                # Check if it's an isolated HV line that should be kept
                if self._is_valid_isolated_line(line, towers):
                    filtered_lines.append(line)
        
        # Filter towers: keep only those connected to power lines or meeting HV criteria
        filtered_towers = []
        for tower_idx, tower in enumerate(towers):
            if tower_idx in connected_tower_ids:
                filtered_towers.append(tower)
            else:
                # Check if it's a valid isolated tower
                if self._is_valid_isolated_tower(tower):
                    filtered_towers.append(tower)
        
        return filtered_lines, filtered_towers
    
    def _validate_hv_power_line(self, line: PowerLineSegment) -> bool:
        """
        Validate if a power line meets HV criteria
        """
        # Height criterion
        avg_height = np.mean([p.z for p in line.points])
        if avg_height < 15.0:  # Minimum height for HV lines
            return False
        
        # Length criterion
        if line.length < 20.0:  # Minimum length for HV spans
            return False
        
        # Point density criterion (HV lines should have sufficient points)
        point_density = len(line.points) / line.length
        if point_density < 0.5:  # At least 0.5 points per meter
            return False
        
        return True
    
    def _is_valid_isolated_line(self, line: PowerLineSegment, towers: List[TransmissionTower]) -> bool:
        """
        Check if an isolated line should be kept (e.g., crossing lines)
        """
        # Check if line passes near any towers (potential crossing)
        for tower in towers:
            closest_point = self._find_closest_point_on_line_segment(line, tower.center_point)
            distance = closest_point.distance_to(tower.center_point)
            
            if distance < 50.0:  # Within reasonable distance of corridor
                avg_height = np.mean([p.z for p in line.points])
                if avg_height > 10.0 and line.length > 30.0:
                    return True
        
        return False
    
    def _is_valid_isolated_tower(self, tower: TransmissionTower) -> bool:
        """
        Check if an isolated tower should be kept
        """
        # Height and structure criteria for HV towers
        if tower.height < 20.0:  # Minimum height for HV towers
            return False
        
        if tower.wing_length < 5.0 or tower.wing_length > 20.0:  # Reasonable wing length
            return False
        
        # Point count criterion
        if len(tower.points) < 50:  # Minimum points for substantial tower
            return False
        
        return True
    
    def _find_closest_point_on_line_segment(self, line: PowerLineSegment, point: Point3D) -> Point3D:
        """Helper method to find closest point on line segment"""
        line_vec = np.array([
            line.end_point.x - line.start_point.x,
            line.end_point.y - line.start_point.y,
            line.end_point.z - line.start_point.z
        ])
        
        point_vec = np.array([
            point.x - line.start_point.x,
            point.y - line.start_point.y,
            point.z - line.start_point.z
        ])
        
        line_length_sq = np.dot(line_vec, line_vec)
        if line_length_sq == 0:
            return line.start_point
        
        t = np.dot(point_vec, line_vec) / line_length_sq
        t = max(0, min(1, t))
        
        closest = np.array([line.start_point.x, line.start_point.y, line.start_point.z]) + t * line_vec
        return Point3D(closest[0], closest[1], closest[2])

class TransmissionCorridorOptimizer:
    """
    Main optimization class that coordinates all optimization steps
    """
    
    def __init__(self,
                 max_connection_distance: float = 15.0,
                 direction_tolerance: float = 0.2,
                 enable_interference_filtering: bool = True):
        """
        Initialize transmission corridor optimizer
        
        Args:
            max_connection_distance: Maximum distance for line-tower connections
            direction_tolerance: Direction consistency tolerance
            enable_interference_filtering: Whether to filter interference objects
        """
        self.max_connection_distance = max_connection_distance
        self.direction_tolerance = direction_tolerance
        self.enable_interference_filtering = enable_interference_filtering
        
        # Initialize optimization components
        self.topology_analyzer = TopologicalConstraintAnalyzer(
            max_connection_distance=max_connection_distance,
            direction_tolerance=direction_tolerance
        )
        
        self.direction_checker = DirectionConsistencyChecker(
            direction_tolerance=direction_tolerance
        )
        
        self.interference_filter = InterferenceFilter()
    
    def optimize_extraction_results(self, corridor: TransmissionCorridor) -> Dict[str, any]:
        """
        Complete optimization pipeline for extraction results
        
        Args:
            corridor: Transmission corridor with extracted objects
            
        Returns:
            Dictionary with optimization results and statistics
        """
        print("Starting transmission corridor optimization...")
        
        if not corridor.power_lines or not corridor.transmission_towers:
            print("Warning: No power lines or towers to optimize")
            return {'status': 'no_objects', 'optimized_lines': 0, 'optimized_towers': 0}
        
        # Step 1: Analyze topological connectivity
        print("Analyzing topological connectivity...")
        connectivity_analysis = self.topology_analyzer.analyze_connectivity(
            corridor.power_lines, 
            corridor.transmission_towers
        )
        
        print(f"Connectivity analysis: {connectivity_analysis['total_connections']} connections found")
        print(f"Connected lines: {connectivity_analysis['connected_lines']}/{len(corridor.power_lines)}")
        print(f"Connected towers: {connectivity_analysis['connected_towers']}/{len(corridor.transmission_towers)}")
        
        # Step 2: Check direction consistency
        print("Checking direction consistency...")
        direction_analysis = self.direction_checker.check_direction_consistency(
            corridor.power_lines,
            corridor.transmission_towers,
            connectivity_analysis['connections']
        )
        
        print(f"Direction consistency: {direction_analysis['consistent_lines']} consistent lines")
        
        # Step 3: Filter interference objects
        if self.enable_interference_filtering:
            print("Filtering interference objects...")
            
            original_line_count = len(corridor.power_lines)
            original_tower_count = len(corridor.transmission_towers)
            
            filtered_lines, filtered_towers = self.interference_filter.filter_interference_objects(
                corridor.power_lines,
                corridor.transmission_towers,
                connectivity_analysis
            )
            
            # Update corridor with filtered objects
            corridor.power_lines = filtered_lines
            corridor.transmission_towers = filtered_towers
            
            print(f"Interference filtering:")
            print(f"  Lines: {original_line_count} -> {len(filtered_lines)}")
            print(f"  Towers: {original_tower_count} -> {len(filtered_towers)}")
        
        # Step 4: Re-analyze after filtering
        final_connectivity = self.topology_analyzer.analyze_connectivity(
            corridor.power_lines, 
            corridor.transmission_towers
        )
        
        # Compile optimization results
        optimization_results = {
            'status': 'completed',
            'original_counts': {
                'lines': len(corridor.power_lines) if not self.enable_interference_filtering 
                        else original_line_count,
                'towers': len(corridor.transmission_towers) if not self.enable_interference_filtering 
                         else original_tower_count
            },
            'optimized_counts': {
                'lines': len(corridor.power_lines),
                'towers': len(corridor.transmission_towers)
            },
            'connectivity_analysis': connectivity_analysis,
            'direction_analysis': direction_analysis,
            'final_connectivity': final_connectivity,
            'optimization_summary': {
                'lines_removed': (original_line_count - len(corridor.power_lines)) if self.enable_interference_filtering else 0,
                'towers_removed': (original_tower_count - len(corridor.transmission_towers)) if self.enable_interference_filtering else 0,
                'final_connection_ratio': final_connectivity.get('connection_ratio', 0.0),
                'direction_consistency_ratio': direction_analysis['consistency_ratio']
            }
        }
        
        print("Optimization complete:")
        print(f"  Final objects: {len(corridor.power_lines)} lines, {len(corridor.transmission_towers)} towers")
        print(f"  Connection ratio: {final_connectivity.get('connection_ratio', 0.0):.3f}")
        print(f"  Direction consistency: {direction_analysis['consistency_ratio']:.3f}")
        
        return optimization_results
    
    def validate_transmission_corridor(self, corridor: TransmissionCorridor) -> Dict[str, any]:
        """
        Validate the quality of extracted transmission corridor
        
        Args:
            corridor: Transmission corridor to validate
            
        Returns:
            Validation results
        """
        validation_results = {
            'is_valid': False,
            'quality_score': 0.0,
            'issues': [],
            'recommendations': []
        }
        
        if not corridor.power_lines:
            validation_results['issues'].append("No power lines detected")
            validation_results['recommendations'].append("Check height thresholds and linearity parameters")
        
        if not corridor.transmission_towers:
            validation_results['issues'].append("No transmission towers detected")
            validation_results['recommendations'].append("Check height difference thresholds and clustering parameters")
        
        if corridor.power_lines and corridor.transmission_towers:
            # Analyze connectivity
            connectivity = self.topology_analyzer.analyze_connectivity(
                corridor.power_lines, corridor.transmission_towers
            )
            
            # Calculate quality metrics
            connection_score = min(1.0, connectivity['connection_ratio'] * 2.0)  # Scale to 0-1
            
            # Check line-to-tower ratio (should be reasonable)
            line_tower_ratio = len(corridor.power_lines) / len(corridor.transmission_towers) if corridor.transmission_towers else 0
            ratio_score = 1.0 if 1.0 <= line_tower_ratio <= 5.0 else 0.5
            
            # Overall quality score
            validation_results['quality_score'] = (connection_score + ratio_score) / 2.0
            validation_results['is_valid'] = validation_results['quality_score'] >= 0.5
            
            if connectivity['connection_ratio'] < 0.3:
                validation_results['issues'].append("Low connectivity between lines and towers")
                validation_results['recommendations'].append("Adjust connection distance thresholds")
            
            if line_tower_ratio > 10:
                validation_results['issues'].append("Too many lines relative to towers")
                validation_results['recommendations'].append("Review line extraction parameters")
            elif line_tower_ratio < 0.5:
                validation_results['issues'].append("Too few lines relative to towers")
                validation_results['recommendations'].append("Review tower extraction parameters")
        
        return validation_results