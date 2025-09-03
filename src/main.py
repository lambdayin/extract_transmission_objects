#!/usr/bin/env python3
"""
Main program for transmission object extraction from UAV LiDAR point clouds.
Complete implementation of the algorithm described in the research paper:
"Automatic Extraction of High-Voltage Power Transmission Objects from UAV Lidar Point Clouds"

This program implements the full pipeline:
1. Point cloud preprocessing and spatial hash grid generation
2. Feature calculation (2D grid and 3D dimensional features)
3. Power line extraction (local segmentation + global merging)
4. Pylon extraction (five-step algorithm)
5. Topological optimization
6. 3D reconstruction with catenary curves and parametric models
"""

import os
import sys
import argparse
import time
from typing import List, Dict, Tuple, Optional
import numpy as np
import logging

# Import all our implemented modules
from data_structures import Point3D, SpatialHashGrid, TransmissionCorridor
from preprocessing import PointCloudPreprocessor, SyntheticDataGenerator
from feature_calculation import FeatureCalculationEngine
from power_line_extraction import PowerLineExtractor
from pylon_extraction import PylonExtractor
from optimization import TransmissionCorridorOptimizer
from reconstruction import TransmissionCorridorReconstructor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('transmission_extraction.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TransmissionObjectExtractor:
    """Main class for the complete transmission object extraction pipeline."""
    
    def __init__(self, 
                 grid_size_2d: float = 5.0,
                 voxel_size_3d: float = 0.5,
                 noise_removal_k: int = 20,
                 noise_removal_std_threshold: float = 2.0):
        """
        Initialize the transmission object extractor.
        
        Args:
            grid_size_2d: Size of 2D grid cells in meters (default 5m as per paper)
            voxel_size_3d: Size of 3D voxels in meters (default 0.5m as per paper)
            noise_removal_k: Number of nearest neighbors for noise removal
            noise_removal_std_threshold: Standard deviation threshold for noise removal
        """
        self.grid_size_2d = grid_size_2d
        self.voxel_size_3d = voxel_size_3d
        
        # Initialize all pipeline components
        self.preprocessor = PointCloudPreprocessor(
            grid_size_2d=grid_size_2d,
            voxel_size_3d=voxel_size_3d,
            noise_removal_k=noise_removal_k,
            noise_removal_std_threshold=noise_removal_std_threshold
        )
        
        self.feature_engine = FeatureCalculationEngine(
            grid_size_2d=grid_size_2d,
            voxel_size_3d=voxel_size_3d
        )
        
        self.power_line_extractor = PowerLineExtractor()
        self.pylon_extractor = PylonExtractor()
        self.optimizer = TransmissionCorridorOptimizer()
        self.reconstructor = TransmissionCorridorReconstructor()
        
        # Statistics tracking
        self.processing_stats = {
            'total_points': 0,
            'points_after_noise_removal': 0,
            'num_2d_grids': 0,
            'num_3d_voxels': 0,
            'num_power_line_segments': 0,
            'num_pylons': 0,
            'processing_time_seconds': 0.0
        }
    
    def load_point_cloud_from_file(self, file_path: str) -> List[Point3D]:
        """
        Load point cloud from file. Supports common formats.
        
        Args:
            file_path: Path to point cloud file
            
        Returns:
            List of Point3D objects
        """
        logger.info(f"Loading point cloud from {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Point cloud file not found: {file_path}")
        
        points = []
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.txt' or file_ext == '.xyz':
            # Simple ASCII format: x y z [intensity] [classification]
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                            intensity = float(parts[3]) if len(parts) > 3 else 0.0
                            classification = int(parts[4]) if len(parts) > 4 else 0
                            points.append(Point3D(x, y, z, intensity, classification))
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Skipping invalid line {line_num} in {file_path}: {e}")
                        continue
        
        elif file_ext == '.las' or file_ext == '.laz':
            try:
                import laspy
                las_file = laspy.read(file_path)
                for i in range(len(las_file.x)):
                    x, y, z = float(las_file.x[i]), float(las_file.y[i]), float(las_file.z[i])
                    intensity = float(las_file.intensity[i]) if hasattr(las_file, 'intensity') else 0.0
                    classification = int(las_file.classification[i]) if hasattr(las_file, 'classification') else 0
                    points.append(Point3D(x, y, z, intensity, classification))
            except ImportError:
                raise ImportError("laspy library required for LAS/LAZ files. Install with: pip install laspy")
        
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        logger.info(f"Loaded {len(points)} points from {file_path}")
        return points
    
    def extract_transmission_objects(self, points: List[Point3D]) -> TransmissionCorridor:
        """
        Execute the complete transmission object extraction pipeline.
        
        Args:
            points: Input point cloud as list of Point3D objects
            
        Returns:
            TransmissionCorridor containing all extracted objects and reconstruction
        """
        start_time = time.time()
        
        logger.info("Starting transmission object extraction pipeline")
        logger.info(f"Input: {len(points)} points")
        
        self.processing_stats['total_points'] = len(points)
        
        # Step 1: Preprocessing and spatial hash grid generation
        logger.info("Step 1: Preprocessing and spatial hash grid generation")
        denoised_points = self.preprocessor.remove_noise(points)
        self.processing_stats['points_after_noise_removal'] = len(denoised_points)
        
        spatial_grid = self.preprocessor.generate_spatial_hash_grid(denoised_points)
        
        # Determine power line height divider
        height_divider = self.preprocessor.determine_power_line_height_divider(denoised_points)
        logger.info(f"Power line height divider determined: {height_divider:.2f}m")
        
        # Step 2: Feature calculation
        logger.info("Step 2: Feature calculation")
        
        # Calculate 2D grid features
        grid_features = {}
        all_grids = spatial_grid.get_all_2d_grids()
        self.processing_stats['num_2d_grids'] = len(all_grids)
        
        for grid_key, grid in all_grids.items():
            features_2d = self.feature_engine.calculate_2d_grid_features(grid)
            grid_features[grid_key] = features_2d
        
        # Calculate 3D dimensional features for voxels
        voxel_features = {}
        all_voxels = spatial_grid.get_all_3d_voxels()
        self.processing_stats['num_3d_voxels'] = len(all_voxels)
        
        for voxel_key, voxel in all_voxels.items():
            if len(voxel.points) >= 3:  # Need minimum points for eigenvalue analysis
                features_3d = self.feature_engine.calculate_3d_dimensional_features(voxel)
                voxel_features[voxel_key] = features_3d
        
        logger.info(f"Calculated features for {len(grid_features)} 2D grids and {len(voxel_features)} 3D voxels")
        
        # Step 3: Power line extraction
        logger.info("Step 3: Power line extraction")
        corridor = TransmissionCorridor()
        
        power_line_segments = self.power_line_extractor.extract_power_lines(
            spatial_grid, self.feature_engine, height_divider
        )
        corridor.power_lines = power_line_segments
        self.processing_stats['num_power_line_segments'] = len(power_line_segments)
        
        logger.info(f"Extracted {len(power_line_segments)} power line segments")
        
        # Step 4: Pylon extraction
        logger.info("Step 4: Pylon extraction")
        transmission_towers = self.pylon_extractor.extract_pylons(corridor, self.feature_engine)
        corridor.transmission_towers = transmission_towers
        self.processing_stats['num_pylons'] = len(transmission_towers)
        
        logger.info(f"Extracted {len(transmission_towers)} transmission towers")
        
        # Step 5: Topological optimization
        logger.info("Step 5: Topological optimization")
        optimized_corridor = self.optimizer.optimize_extraction_results(corridor)
        
        # Log optimization results
        connectivity_analysis = self.optimizer.analyze_connectivity(
            optimized_corridor.power_lines, optimized_corridor.transmission_towers
        )
        logger.info(f"Topological optimization completed. Connectivity score: {connectivity_analysis.get('connectivity_score', 'N/A')}")
        
        # Step 6: 3D reconstruction
        logger.info("Step 6: 3D reconstruction")
        final_corridor = self.reconstructor.reconstruct_corridor(optimized_corridor)
        
        # Log reconstruction results
        reconstruction_stats = self._analyze_reconstruction_results(final_corridor)
        logger.info(f"3D reconstruction completed: {reconstruction_stats}")
        
        # Calculate total processing time
        end_time = time.time()
        self.processing_stats['processing_time_seconds'] = end_time - start_time
        
        logger.info("Transmission object extraction pipeline completed successfully")
        logger.info(f"Total processing time: {self.processing_stats['processing_time_seconds']:.2f} seconds")
        
        return final_corridor
    
    def _analyze_reconstruction_results(self, corridor: TransmissionCorridor) -> Dict[str, any]:
        """Analyze and summarize reconstruction results."""
        stats = {
            'power_lines_modeled': 0,
            'towers_modeled': 0,
            'insulators_detected': 0,
            'catenary_curves_fitted': 0
        }
        
        # Count power line models
        for line in corridor.power_lines:
            if hasattr(line, 'catenary_parameters') and line.catenary_parameters is not None:
                stats['catenary_curves_fitted'] += 1
            stats['power_lines_modeled'] += 1
        
        # Count tower models
        for tower in corridor.transmission_towers:
            if hasattr(tower, 'tower_model') and tower.tower_model is not None:
                stats['towers_modeled'] += 1
            if hasattr(tower, 'insulators') and tower.insulators:
                stats['insulators_detected'] += len(tower.insulators)
        
        return stats
    
    def save_results(self, corridor: TransmissionCorridor, output_dir: str):
        """
        Save extraction and reconstruction results to files.
        
        Args:
            corridor: Transmission corridor with all extracted objects
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save processing statistics
        stats_file = os.path.join(output_dir, 'processing_statistics.txt')
        with open(stats_file, 'w') as f:
            f.write("Transmission Object Extraction Results\n")
            f.write("=====================================\n\n")
            for key, value in self.processing_stats.items():
                f.write(f"{key}: {value}\n")
        
        # Save power line segments
        lines_file = os.path.join(output_dir, 'power_lines.txt')
        with open(lines_file, 'w') as f:
            f.write("# Power Line Segments\n")
            f.write("# Format: segment_id x1 y1 z1 x2 y2 z2 confidence\n")
            for i, line in enumerate(corridor.power_lines):
                if line.points and len(line.points) >= 2:
                    start_point = line.points[0]
                    end_point = line.points[-1]
                    f.write(f"{i} {start_point.x:.3f} {start_point.y:.3f} {start_point.z:.3f} ")
                    f.write(f"{end_point.x:.3f} {end_point.y:.3f} {end_point.z:.3f} {line.confidence:.3f}\n")
        
        # Save transmission towers
        towers_file = os.path.join(output_dir, 'transmission_towers.txt')
        with open(towers_file, 'w') as f:
            f.write("# Transmission Towers\n")
            f.write("# Format: tower_id center_x center_y center_z height tower_type confidence\n")
            for i, tower in enumerate(corridor.transmission_towers):
                if tower.center_point:
                    f.write(f"{i} {tower.center_point.x:.3f} {tower.center_point.y:.3f} {tower.center_point.z:.3f} ")
                    f.write(f"{tower.height:.3f} {tower.tower_type} {tower.confidence:.3f}\n")
        
        # Save catenary parameters
        catenary_file = os.path.join(output_dir, 'catenary_parameters.txt')
        with open(catenary_file, 'w') as f:
            f.write("# Catenary Curve Parameters\n")
            f.write("# Format: line_id a h k theta rho\n")
            for i, line in enumerate(corridor.power_lines):
                if hasattr(line, 'catenary_parameters') and line.catenary_parameters:
                    params = line.catenary_parameters
                    f.write(f"{i} {params.a:.6f} {params.h:.3f} {params.k:.3f} ")
                    f.write(f"{params.theta:.6f} {params.rho:.6f}\n")
        
        logger.info(f"Results saved to {output_dir}")
    
    def print_summary(self):
        """Print a summary of processing statistics."""
        print("\n" + "="*60)
        print("TRANSMISSION OBJECT EXTRACTION SUMMARY")
        print("="*60)
        print(f"Input points: {self.processing_stats['total_points']:,}")
        print(f"Points after noise removal: {self.processing_stats['points_after_noise_removal']:,}")
        print(f"2D grid cells processed: {self.processing_stats['num_2d_grids']:,}")
        print(f"3D voxels processed: {self.processing_stats['num_3d_voxels']:,}")
        print(f"Power line segments extracted: {self.processing_stats['num_power_line_segments']}")
        print(f"Transmission towers extracted: {self.processing_stats['num_pylons']}")
        print(f"Total processing time: {self.processing_stats['processing_time_seconds']:.2f} seconds")
        
        # Calculate efficiency metrics
        points_per_second = self.processing_stats['total_points'] / max(self.processing_stats['processing_time_seconds'], 1)
        print(f"Processing speed: {points_per_second:,.0f} points/second")
        print("="*60)


def main():
    """Main entry point for the transmission object extraction program."""
    parser = argparse.ArgumentParser(
        description="Extract transmission objects from UAV LiDAR point clouds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract objects from point cloud file
  python main.py --input data/transmission_corridor.txt --output results/

  # Generate synthetic data and extract objects
  python main.py --synthetic --output results/

  # Use custom parameters
  python main.py --input data/lidar.las --output results/ --grid-size 3.0 --voxel-size 0.3
        """
    )
    
    parser.add_argument('--input', '-i', type=str, 
                       help='Input point cloud file (.txt, .xyz, .las, .laz)')
    parser.add_argument('--output', '-o', type=str, default='results/',
                       help='Output directory for results (default: results/)')
    parser.add_argument('--synthetic', action='store_true',
                       help='Generate and use synthetic test data')
    parser.add_argument('--grid-size', type=float, default=5.0,
                       help='2D grid cell size in meters (default: 5.0)')
    parser.add_argument('--voxel-size', type=float, default=0.5,
                       help='3D voxel size in meters (default: 0.5)')
    parser.add_argument('--noise-k', type=int, default=20,
                       help='K nearest neighbors for noise removal (default: 20)')
    parser.add_argument('--noise-threshold', type=float, default=2.0,
                       help='Standard deviation threshold for noise removal (default: 2.0)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if not args.synthetic and not args.input:
        parser.error("Either --input or --synthetic must be specified")
    
    if args.input and not os.path.exists(args.input):
        parser.error(f"Input file does not exist: {args.input}")
    
    try:
        # Initialize extractor
        extractor = TransmissionObjectExtractor(
            grid_size_2d=args.grid_size,
            voxel_size_3d=args.voxel_size,
            noise_removal_k=args.noise_k,
            noise_removal_std_threshold=args.noise_threshold
        )
        
        # Load or generate point cloud
        if args.synthetic:
            logger.info("Generating synthetic test data")
            generator = SyntheticDataGenerator()
            points = generator.generate_transmission_corridor_scene()
        else:
            points = extractor.load_point_cloud_from_file(args.input)
        
        # Execute extraction pipeline
        corridor = extractor.extract_transmission_objects(points)
        
        # Save results
        extractor.save_results(corridor, args.output)
        
        # Print summary
        extractor.print_summary()
        
        logger.info("Program completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error during extraction: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())