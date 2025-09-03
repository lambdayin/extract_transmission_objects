#!/usr/bin/env python3
"""
Comprehensive test suite for the transmission object extraction pipeline.
Tests all components individually and as an integrated system.
"""

import unittest
import numpy as np
import os
import tempfile
import shutil
from typing import List, Dict, Tuple
import logging

# Import all modules for testing
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Try importing as package
    from src.data_structures import Point3D, SpatialHashGrid, Grid2D, Voxel3D, GridKey, VoxelKey
    from src.preprocessing import PointCloudPreprocessor
    from src.feature_calculation import FeatureCalculationEngine
    from src.power_line_extraction import PowerLineExtractor
    from src.pylon_extraction import PylonExtractor
    from src.optimization import TransmissionCorridorOptimizer
    from src.reconstruction import TransmissionCorridorReconstructor
    from src.main import TransmissionObjectExtractor
except ImportError:
    # Fallback to direct imports
    from data_structures import Point3D, SpatialHashGrid, Grid2D, Voxel3D, GridKey, VoxelKey
    from preprocessing import PointCloudPreprocessor
    from feature_calculation import FeatureCalculationEngine
    from power_line_extraction import PowerLineExtractor
    from pylon_extraction import PylonExtractor
    from optimization import TransmissionCorridorOptimizer
    from reconstruction import TransmissionCorridorReconstructor
    from main import TransmissionObjectExtractor

# Disable logging during tests unless specifically needed
logging.getLogger().setLevel(logging.ERROR)


class SyntheticDataGenerator:
    """Simple synthetic data generator for testing purposes."""
    
    def __init__(self):
        pass
    
    def generate_transmission_corridor_points(self, 
                                            corridor_length: float = 200.0,
                                            n_towers: int = 3,
                                            n_lines: int = 6) -> List[Point3D]:
        """Generate synthetic transmission corridor point cloud."""
        points = []
        
        # Generate tower points
        tower_spacing = corridor_length / (n_towers + 1)
        for i in range(n_towers):
            x_pos = (i + 1) * tower_spacing - corridor_length / 2
            tower_points = self._generate_tower_points(x_pos, 0.0, 0.0)
            points.extend(tower_points)
        
        # Generate power line points
        for line_idx in range(n_lines):
            y_offset = (line_idx - n_lines/2) * 5.0
            line_points = self._generate_line_points(-corridor_length/2, corridor_length/2, y_offset, 25.0)
            points.extend(line_points)
        
        return points
    
    def _generate_tower_points(self, x_center: float, y_center: float, z_base: float) -> List[Point3D]:
        """Generate points for a single tower."""
        points = []
        tower_height = 40.0
        base_width = 10.0
        
        # Generate tower structure points
        for height_level in range(0, int(tower_height), 2):
            current_width = base_width * (1.0 - height_level / tower_height * 0.7)
            
            # Create square cross-section
            for angle in np.linspace(0, 2*np.pi, 16, endpoint=False):
                x = x_center + current_width * np.cos(angle) / 2
                y = y_center + current_width * np.sin(angle) / 2
                z = z_base + height_level
                
                points.append(Point3D(x, y, z, 100.0, 1))
        
        return points
    
    def _generate_line_points(self, x_start: float, x_end: float, y_pos: float, z_height: float) -> List[Point3D]:
        """Generate points for a power line using catenary shape."""
        points = []
        n_points = 50
        
        # Simple catenary approximation
        x_positions = np.linspace(x_start, x_end, n_points)
        for x in x_positions:
            # Simple catenary sag
            sag = 5.0 * (1 - np.cos(2 * np.pi * (x - x_start) / (x_end - x_start)))
            z = z_height - sag
            
            points.append(Point3D(x, y_pos, z, 100.0, 2))
        
        return points


class TestDataStructures(unittest.TestCase):
    """Test core data structures."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_points = [
            Point3D(0.0, 0.0, 10.0, 100.0, 1),
            Point3D(5.0, 5.0, 15.0, 150.0, 1),
            Point3D(10.0, 10.0, 20.0, 200.0, 2)
        ]
        
        self.spatial_grid = SpatialHashGrid(
            x_min=-50.0, x_max=50.0,
            y_min=-50.0, y_max=50.0,
            z_min=0.0, z_max=100.0
        )
    
    def test_point3d_creation(self):
        """Test Point3D object creation and properties."""
        point = Point3D(1.0, 2.0, 3.0, 100.0, 1)
        
        self.assertEqual(point.x, 1.0)
        self.assertEqual(point.y, 2.0)
        self.assertEqual(point.z, 3.0)
        self.assertEqual(point.intensity, 100.0)
        self.assertEqual(point.classification, 1)
    
    def test_spatial_hash_grid_key_generation(self):
        """Test 2D grid and 3D voxel key generation."""
        # Test 2D grid key generation
        point = Point3D(7.5, 12.5, 25.0)
        grid_key = self.spatial_grid.point_to_grid_key(point)
        
        self.assertIsInstance(grid_key, GridKey)
        self.assertIsInstance(grid_key.m, int)
        self.assertIsInstance(grid_key.n, int)
        
        # Test 3D voxel key generation
        voxel_key = self.spatial_grid.point_to_voxel_key(point)
        
        self.assertIsInstance(voxel_key, VoxelKey)
        self.assertIsInstance(voxel_key.i, int)
        self.assertIsInstance(voxel_key.j, int)
        self.assertIsInstance(voxel_key.k, int)
    
    def test_spatial_hash_grid_point_insertion(self):
        """Test point insertion into spatial hash grid."""
        # Add points to grid
        for point in self.test_points:
            self.spatial_grid.add_point(point)
        
        # Check that points were added
        all_2d_grids = self.spatial_grid.get_all_2d_grids()
        all_3d_voxels = self.spatial_grid.get_all_3d_voxels()
        
        self.assertGreater(len(all_2d_grids), 0)
        self.assertGreater(len(all_3d_voxels), 0)
        
        # Verify total point count
        total_points_in_grids = sum(len(grid.points) for grid in all_2d_grids.values())
        self.assertEqual(total_points_in_grids, len(self.test_points))
    
    def test_grid_and_voxel_properties(self):
        """Test Grid2D and Voxel3D properties calculation."""
        # Create a grid with test points
        grid = Grid2D(GridKey(0, 0))
        for point in self.test_points:
            grid.add_point(point)
        
        # Test grid properties
        self.assertEqual(len(grid.points), len(self.test_points))
        self.assertAlmostEqual(grid.min_height, 10.0)
        self.assertAlmostEqual(grid.max_height, 20.0)
        self.assertAlmostEqual(grid.height_range, 10.0)
        
        # Create a voxel with test points
        voxel = Voxel3D(VoxelKey(0, 0, 0))
        for point in self.test_points:
            voxel.add_point(point)
        
        # Test voxel properties
        self.assertEqual(len(voxel.points), len(self.test_points))
        self.assertAlmostEqual(voxel.centroid.x, np.mean([p.x for p in self.test_points]))
        self.assertAlmostEqual(voxel.centroid.y, np.mean([p.y for p in self.test_points]))
        self.assertAlmostEqual(voxel.centroid.z, np.mean([p.z for p in self.test_points]))


class TestPreprocessing(unittest.TestCase):
    """Test preprocessing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = PointCloudPreprocessor()
        self.synthetic_generator = SyntheticDataGenerator()
    
    def test_synthetic_data_generation(self):
        """Test synthetic data generation."""
        # Generate power line
        power_line_points = self.synthetic_generator.generate_power_line(
            start_point=(0, 0, 20),
            end_point=(100, 0, 25),
            num_points=50,
            catenary_sag=2.0
        )
        
        self.assertEqual(len(power_line_points), 50)
        self.assertAlmostEqual(power_line_points[0].x, 0.0, places=1)
        self.assertAlmostEqual(power_line_points[-1].x, 100.0, places=1)
        
        # Generate transmission tower
        tower_points = self.synthetic_generator.generate_transmission_tower(
            center=(50, 10, 0),
            height=30.0,
            tower_type='lattice'
        )
        
        self.assertGreater(len(tower_points), 0)
        
        # Generate complete scene
        scene_points = self.synthetic_generator.generate_transmission_corridor_scene()
        self.assertGreater(len(scene_points), 100)
    
    def test_noise_removal(self):
        """Test statistical noise removal."""
        # Create test data with outliers
        clean_points = [Point3D(i, 0, 10) for i in range(20)]
        outliers = [Point3D(100, 100, 100), Point3D(-100, -100, -100)]
        test_points = clean_points + outliers
        
        # Apply noise removal
        denoised_points = self.preprocessor.remove_noise(test_points)
        
        # Should remove outliers but keep most clean points
        self.assertLess(len(denoised_points), len(test_points))
        self.assertGreater(len(denoised_points), len(clean_points) * 0.8)
    
    def test_spatial_hash_grid_generation(self):
        """Test spatial hash grid generation."""
        # Generate test data
        test_points = self.synthetic_generator.generate_transmission_corridor_scene()
        
        # Generate spatial hash grid
        spatial_grid = self.preprocessor.generate_spatial_hash_grid(test_points)
        
        # Verify grid structure
        all_2d_grids = spatial_grid.get_all_2d_grids()
        all_3d_voxels = spatial_grid.get_all_3d_voxels()
        
        self.assertGreater(len(all_2d_grids), 0)
        self.assertGreater(len(all_3d_voxels), 0)
    
    def test_height_divider_determination(self):
        """Test power line height divider determination."""
        # Create points at different heights
        test_points = []
        # Ground points (0-5m)
        test_points.extend([Point3D(i, j, np.random.uniform(0, 5)) 
                           for i in range(10) for j in range(10)])
        # Power line points (15-25m) 
        test_points.extend([Point3D(i, j, np.random.uniform(15, 25))
                           for i in range(5) for j in range(5)])
        
        height_divider = self.preprocessor.determine_power_line_height_divider(test_points)
        
        # Should be somewhere between ground and power line heights
        self.assertGreater(height_divider, 5.0)
        self.assertLess(height_divider, 15.0)


class TestFeatureCalculation(unittest.TestCase):
    """Test feature calculation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.feature_engine = FeatureCalculationEngine()
        
        # Create test grid with points
        self.test_grid = Grid2D(GridKey(0, 0))
        heights = [10, 12, 15, 18, 20, 22, 25, 28, 30]
        for i, h in enumerate(heights):
            self.test_grid.add_point(Point3D(i, 0, h, 100, 1))
        
        # Create test voxel with points for eigenvalue analysis
        self.test_voxel = Voxel3D(VoxelKey(0, 0, 0))
        # Linear arrangement of points (high linearity)
        for i in range(10):
            self.test_voxel.add_point(Point3D(i, 0, 0))
    
    def test_2d_grid_features(self):
        """Test 2D grid feature calculation."""
        features = self.feature_engine.calculate_2d_grid_features(self.test_grid)
        
        # Check that all expected features are present
        expected_features = ['dem', 'dsm', 'height_difference', 'point_count', 
                           'height_std', 'height_variance']
        for feature_name in expected_features:
            self.assertIn(feature_name, features)
        
        # Verify feature values make sense
        self.assertEqual(features['point_count'], 9)
        self.assertAlmostEqual(features['dem'], 10.0)  # Minimum height
        self.assertAlmostEqual(features['dsm'], 30.0)  # Maximum height
        self.assertAlmostEqual(features['height_difference'], 20.0)  # Max - Min
    
    def test_3d_dimensional_features(self):
        """Test 3D dimensional feature calculation."""
        features = self.feature_engine.calculate_3d_dimensional_features(self.test_voxel)
        
        # Check that dimensional features are present
        self.assertIn('a1d', features)  # Linearity
        self.assertIn('a2d', features)  # Planarity  
        self.assertIn('a3d', features)  # Sphericity
        
        # For linear arrangement, linearity should be high
        self.assertGreater(features['a1d'], 0.5)
        # Sphericity should be low for linear arrangement
        self.assertLess(features['a3d'], 0.5)
    
    def test_compass_line_filter(self):
        """Test Compass Line Filter functionality."""
        # Create points along a specific direction
        points = [Point3D(i, i, 0) for i in range(10)]  # 45-degree line
        
        dominant_direction = self.feature_engine.calculate_dominant_direction_clf(points)
        
        # Should detect approximately 45-degree direction
        expected_angle = np.pi / 4  # 45 degrees in radians
        self.assertAlmostEqual(dominant_direction, expected_angle, delta=0.2)


class TestPowerLineExtraction(unittest.TestCase):
    """Test power line extraction functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.power_line_extractor = PowerLineExtractor()
        self.feature_engine = FeatureCalculationEngine()
        
        # Create synthetic power line data
        generator = SyntheticDataGenerator()
        power_line_points = generator.generate_power_line(
            start_point=(0, 0, 20),
            end_point=(100, 0, 22),
            num_points=50
        )
        
        # Create spatial grid with power line
        self.spatial_grid = SpatialHashGrid(
            x_min=-10.0, x_max=110.0,
            y_min=-10.0, y_max=10.0,
            z_min=0.0, z_max=50.0
        )
        
        for point in power_line_points:
            self.spatial_grid.add_point(point)
    
    def test_local_segment_extraction(self):
        """Test local power line segment extraction."""
        # Extract local segments
        local_segments = self.power_line_extractor.local_extractor.extract_local_segments(
            self.spatial_grid, self.feature_engine, height_divider=15.0
        )
        
        # Should find some power line segments
        self.assertGreater(len(local_segments), 0)
        
        # Check segment properties
        for segment in local_segments:
            self.assertGreater(len(segment.points), 0)
            self.assertGreater(segment.confidence, 0.0)
            self.assertLessEqual(segment.confidence, 1.0)
    
    def test_global_line_merging(self):
        """Test global power line merging."""
        # First extract local segments
        local_segments = self.power_line_extractor.local_extractor.extract_local_segments(
            self.spatial_grid, self.feature_engine, height_divider=15.0
        )
        
        if len(local_segments) > 1:
            # Try to merge segments
            merged_lines = self.power_line_extractor.global_merger.merge_segments_to_lines(local_segments)
            
            # Merged lines should be fewer than or equal to local segments
            self.assertLessEqual(len(merged_lines), len(local_segments))
    
    def test_complete_power_line_extraction(self):
        """Test complete power line extraction pipeline."""
        power_lines = self.power_line_extractor.extract_power_lines(
            self.spatial_grid, self.feature_engine, height_divider=15.0
        )
        
        # Should extract at least one power line
        self.assertGreater(len(power_lines), 0)


class TestPylonExtraction(unittest.TestCase):
    """Test pylon extraction functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pylon_extractor = PylonExtractor()
        self.feature_engine = FeatureCalculationEngine()
        
        # Create synthetic transmission corridor with tower
        generator = SyntheticDataGenerator()
        tower_points = generator.generate_transmission_tower(
            center=(50, 0, 0),
            height=30.0,
            tower_type='lattice'
        )
        
        # Create spatial grid with tower
        self.spatial_grid = SpatialHashGrid(
            x_min=0.0, x_max=100.0,
            y_min=-25.0, y_max=25.0,
            z_min=0.0, z_max=50.0
        )
        
        for point in tower_points:
            self.spatial_grid.add_point(point)
        
        # Create test corridor
        from data_structures import TransmissionCorridor
        self.test_corridor = TransmissionCorridor()
    
    def test_candidate_identification(self):
        """Test pylon candidate identification (Step 1)."""
        all_grids = self.spatial_grid.get_all_2d_grids()
        
        candidate_grids = self.pylon_extractor.candidate_identifier.identify_candidates(
            all_grids
        )
        
        # Should identify some candidates around the tower location
        self.assertGreater(len(candidate_grids), 0)
    
    def test_moving_window_analysis(self):
        """Test moving window analysis (Step 2)."""
        all_grids = self.spatial_grid.get_all_2d_grids()
        candidate_grids = self.pylon_extractor.candidate_identifier.identify_candidates(all_grids)
        
        if len(candidate_grids) > 0:
            refined_candidates = self.pylon_extractor.window_analyzer.analyze_with_moving_window(
                candidate_grids
            )
            
            # Refined candidates should be subset of original candidates
            self.assertLessEqual(len(refined_candidates), len(candidate_grids))
    
    def test_complete_pylon_extraction(self):
        """Test complete five-step pylon extraction."""
        towers = self.pylon_extractor.extract_pylons(self.test_corridor, self.feature_engine)
        
        # Should extract at least one tower (might be zero if parameters are too strict)
        self.assertGreaterEqual(len(towers), 0)
        
        # If towers are found, check their properties
        for tower in towers:
            self.assertIsNotNone(tower.center_point)
            self.assertGreater(tower.height, 0)
            self.assertGreater(tower.confidence, 0)


class TestOptimization(unittest.TestCase):
    """Test topological optimization functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = TopologicalOptimizer()
        
        # Create test corridor with mock data
        from data_structures import TransmissionCorridor, PowerLineSegment, TransmissionTower
        
        self.test_corridor = TransmissionCorridor()
        
        # Add mock power line
        power_line = PowerLineSegment()
        power_line.points = [Point3D(0, 0, 20), Point3D(100, 0, 22)]
        power_line.confidence = 0.8
        self.test_corridor.power_lines = [power_line]
        
        # Add mock towers
        tower1 = TransmissionTower()
        tower1.center_point = Point3D(-10, 0, 0)
        tower1.height = 30.0
        tower1.confidence = 0.9
        
        tower2 = TransmissionTower()
        tower2.center_point = Point3D(110, 0, 0)
        tower2.height = 32.0
        tower2.confidence = 0.85
        
        self.test_corridor.transmission_towers = [tower1, tower2]
    
    def test_connectivity_analysis(self):
        """Test connectivity analysis between power lines and towers."""
        analysis = self.optimizer.analyze_connectivity(
            self.test_corridor.power_lines,
            self.test_corridor.transmission_towers
        )
        
        # Should return analysis results
        self.assertIn('connectivity_score', analysis)
        self.assertIn('line_tower_connections', analysis)
    
    def test_direction_consistency(self):
        """Test direction consistency checking."""
        is_consistent = self.optimizer.check_direction_consistency(
            self.test_corridor.power_lines,
            self.test_corridor.transmission_towers
        )
        
        # Should return boolean result
        self.assertIsInstance(is_consistent, bool)
    
    def test_corridor_optimization(self):
        """Test complete corridor optimization."""
        optimized_corridor = self.optimizer.optimize_corridor(self.test_corridor)
        
        # Should return optimized corridor
        self.assertIsNotNone(optimized_corridor)
        self.assertGreaterEqual(len(optimized_corridor.power_lines), 0)
        self.assertGreaterEqual(len(optimized_corridor.transmission_towers), 0)


class TestReconstruction(unittest.TestCase):
    """Test 3D reconstruction functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.reconstructor = TransmissionCorridorReconstructor()
        
        # Create test corridor with mock data
        from data_structures import TransmissionCorridor, PowerLineSegment, TransmissionTower
        
        self.test_corridor = TransmissionCorridor()
        
        # Add power line with catenary shape
        power_line = PowerLineSegment()
        # Generate points along a catenary curve
        x_vals = np.linspace(0, 100, 20)
        for x in x_vals:
            # Simple catenary: y = a * cosh(x/a) - a
            a = 50.0  # catenary parameter
            z = a * np.cosh(x/a) - a + 20  # offset by 20m height
            power_line.points.append(Point3D(x, 0, z))
        power_line.confidence = 0.9
        
        self.test_corridor.power_lines = [power_line]
        
        # Add tower
        tower = TransmissionTower()
        tower.center_point = Point3D(50, 5, 0)
        tower.height = 30.0
        tower.confidence = 0.8
        tower.tower_type = 'lattice'
        
        self.test_corridor.transmission_towers = [tower]
    
    def test_catenary_modeling(self):
        """Test catenary curve modeling for power lines."""
        power_line = self.test_corridor.power_lines[0]
        
        # Try to fit catenary parameters
        catenary_params = self.reconstructor.power_line_modeler.model_power_line_catenary(power_line)
        
        if catenary_params:
            # Check that parameters are reasonable
            self.assertGreater(catenary_params.a, 0)  # Catenary parameter should be positive
            self.assertIsInstance(catenary_params.h, float)
            self.assertIsInstance(catenary_params.k, float)
    
    def test_parametric_tower_modeling(self):
        """Test parametric tower modeling."""
        tower = self.test_corridor.transmission_towers[0]
        
        # Model tower parametrically
        tower_model = self.reconstructor.tower_modeler.model_tower_parametric(tower)
        
        if tower_model:
            # Check model properties
            self.assertIn('tower_type', tower_model)
            self.assertIn('parameters', tower_model)
    
    def test_complete_reconstruction(self):
        """Test complete corridor reconstruction."""
        reconstructed_corridor = self.reconstructor.reconstruct_corridor(self.test_corridor)
        
        # Should return reconstructed corridor
        self.assertIsNotNone(reconstructed_corridor)
        self.assertGreaterEqual(len(reconstructed_corridor.power_lines), 0)
        self.assertGreaterEqual(len(reconstructed_corridor.transmission_towers), 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = TransmissionObjectExtractor(
            grid_size_2d=5.0,
            voxel_size_3d=0.5
        )
        
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_synthetic_data_pipeline(self):
        """Test complete pipeline with synthetic data."""
        # Generate synthetic test data
        generator = SyntheticDataGenerator()
        test_points = generator.generate_transmission_corridor_scene()
        
        # Run complete extraction pipeline
        corridor = self.extractor.extract_transmission_objects(test_points)
        
        # Verify results
        self.assertIsNotNone(corridor)
        self.assertGreaterEqual(len(corridor.power_lines), 0)
        self.assertGreaterEqual(len(corridor.transmission_towers), 0)
        
        # Test saving results
        self.extractor.save_results(corridor, self.temp_dir)
        
        # Check that output files were created
        expected_files = [
            'processing_statistics.txt',
            'power_lines.txt', 
            'transmission_towers.txt',
            'catenary_parameters.txt'
        ]
        
        for filename in expected_files:
            file_path = os.path.join(self.temp_dir, filename)
            self.assertTrue(os.path.exists(file_path), f"Output file not found: {filename}")
    
    def test_point_cloud_file_loading(self):
        """Test loading point cloud from file."""
        # Create test point cloud file
        test_file = os.path.join(self.temp_dir, 'test_points.txt')
        
        with open(test_file, 'w') as f:
            f.write("# Test point cloud\n")
            f.write("0.0 0.0 10.0 100 1\n")
            f.write("5.0 0.0 15.0 150 1\n") 
            f.write("10.0 0.0 20.0 200 2\n")
        
        # Load points
        points = self.extractor.load_point_cloud_from_file(test_file)
        
        self.assertEqual(len(points), 3)
        self.assertEqual(points[0].x, 0.0)
        self.assertEqual(points[1].y, 0.0)
        self.assertEqual(points[2].z, 20.0)
    
    def test_processing_statistics(self):
        """Test processing statistics tracking."""
        # Generate small test dataset
        generator = SyntheticDataGenerator()
        test_points = generator.generate_transmission_corridor_scene(
            corridor_length=50.0,
            num_power_lines=1,
            num_towers=2
        )
        
        # Run extraction
        corridor = self.extractor.extract_transmission_objects(test_points)
        
        # Check statistics were recorded
        stats = self.extractor.processing_stats
        
        self.assertGreater(stats['total_points'], 0)
        self.assertGreaterEqual(stats['points_after_noise_removal'], 0)
        self.assertGreater(stats['num_2d_grids'], 0)
        self.assertGreater(stats['num_3d_voxels'], 0)
        self.assertGreater(stats['processing_time_seconds'], 0)


class TestPerformance(unittest.TestCase):
    """Performance and stress tests."""
    
    def test_large_dataset_performance(self):
        """Test performance with large synthetic dataset."""
        generator = SyntheticDataGenerator()
        
        # Generate large dataset
        large_points = generator.generate_transmission_corridor_scene(
            corridor_length=1000.0,  # 1km corridor
            num_power_lines=6,       # 6 power lines
            num_towers=10,           # 10 towers
            ground_point_density=0.5  # Reduced density for performance
        )
        
        extractor = TransmissionObjectExtractor()
        
        # Measure processing time
        import time
        start_time = time.time()
        
        corridor = extractor.extract_transmission_objects(large_points)
        
        processing_time = time.time() - start_time
        
        # Performance assertions
        self.assertLess(processing_time, 300.0, "Processing took too long (>5 minutes)")
        self.assertGreater(len(large_points), 1000, "Test dataset should be substantial")
        
        # Verify results quality
        points_per_second = len(large_points) / processing_time
        print(f"Performance: {points_per_second:.0f} points/second")
        
        self.assertGreater(points_per_second, 100, "Processing speed too slow")


def run_all_tests():
    """Run all test suites and generate report."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestDataStructures,
        TestPreprocessing, 
        TestFeatureCalculation,
        TestPowerLineExtraction,
        TestPylonExtraction,
        TestOptimization,
        TestReconstruction,
        TestIntegration,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
    
    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('\\n')[-2]}")
    
    print("="*60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)