#!/usr/bin/env python3
"""
快速测试脚本，只运行到线性体素检测
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_structures import Point3D
from preprocessing import PointCloudPreprocessor
from feature_calculation import FeatureCalculationEngine

def load_point_cloud(file_path: str):
    points = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                if ',' in line:
                    parts = line.strip().split(',')
                else:
                    parts = line.strip().split()
                
                if len(parts) >= 3:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    points.append(Point3D(x, y, z))
            except (ValueError, IndexError):
                continue
    return points

def main():
    # 加载数据
    print("加载点云数据...")
    points = load_point_cloud("./001Combined.txt")
    print(f"加载了 {len(points)} 个点")
    
    # 预处理
    print("预处理...")
    preprocessor = PointCloudPreprocessor(
        grid_size_2d=5.0,
        voxel_size_3d=0.5,
        noise_k_neighbors=20,
        noise_std_multiplier=2.0
    )
    
    denoised_points = preprocessor.remove_noise(points)
    print(f"噪声移除: {len(points)} -> {len(denoised_points)} 点")
    
    spatial_grid = preprocessor.generate_spatial_hash_grid(denoised_points)
    print(f"生成的网格: {spatial_grid.get_grid_count()} 个2D网格, {spatial_grid.get_voxel_count()} 个3D体素")
    
    height_divider = preprocessor.determine_power_line_height_divider(denoised_points)
    print(f"高度分割器: {height_divider:.2f}m")
    
    # 特征计算
    print("计算特征...")
    feature_engine = FeatureCalculationEngine(
        grid_size_2d=5.0,
        voxel_size_3d=0.5
    )
    
    # 计算3D体素特征
    all_voxels = spatial_grid.get_all_3d_voxels()
    print(f"处理 {len(all_voxels)} 个体素...")
    
    linear_voxel_count = 0
    elevated_linear_voxel_count = 0
    
    for i, (voxel_key, voxel) in enumerate(all_voxels.items()):
        if len(voxel.points) >= 3:
            features_3d = feature_engine.calculate_3d_dimensional_features(voxel)
            
            if voxel.is_linear:
                linear_voxel_count += 1
                
                # 检查是否在高度分割器之上
                avg_height = sum(p.z for p in voxel.points) / len(voxel.points)
                if avg_height > height_divider:
                    elevated_linear_voxel_count += 1
        
        if (i + 1) % 1000 == 0:
            print(f"处理了 {i + 1}/{len(all_voxels)} 个体素")
    
    print(f"检测结果:")
    print(f"  线性体素总数: {linear_voxel_count}")
    print(f"  高空线性体素: {elevated_linear_voxel_count}")
    print(f"  线性度: {linear_voxel_count / len(all_voxels) * 100:.1f}%")
    
    # 详细分析一些线性体素
    print(f"\n分析前10个线性体素:")
    linear_count = 0
    for voxel_key, voxel in all_voxels.items():
        if len(voxel.points) >= 3 and hasattr(voxel, 'is_linear') and voxel.is_linear:
            linear_count += 1
            avg_height = sum(p.z for p in voxel.points) / len(voxel.points)
            print(f"  体素 {linear_count}: {len(voxel.points)} 个点, 平均高度: {avg_height:.2f}m, a1d: {voxel.a1d:.3f}")
            
            if linear_count >= 10:
                break

if __name__ == "__main__":
    main()