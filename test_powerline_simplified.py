#!/usr/bin/env python3
"""
简化的输电线提取测试，跳过复杂的聚类算法
"""

import sys
import os
import numpy as np
from typing import List
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_structures import Point3D, PowerLineSegment
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

def simple_line_segment_extraction(linear_voxels, height_divider):
    """简化的线段提取，基于空间邻近性"""
    print(f"简化线段提取，处理 {len(linear_voxels)} 个线性体素...")
    
    power_line_segments = []
    
    # 按高度分组
    elevated_voxels = []
    for voxel in linear_voxels:
        if len(voxel.points) >= 3:
            avg_height = sum(p.z for p in voxel.points) / len(voxel.points)
            if avg_height > height_divider:
                elevated_voxels.append(voxel)
    
    print(f"高空线性体素: {len(elevated_voxels)}")
    
    # 简化方法：基于空间位置的局部聚类
    processed = set()
    
    for i, voxel in enumerate(elevated_voxels):
        if i in processed:
            continue
            
        # 收集所有该体素的点
        segment_points = list(voxel.points)
        processed.add(i)
        
        # 查找附近的体素 
        voxel_center = np.array([
            sum(p.x for p in voxel.points) / len(voxel.points),
            sum(p.y for p in voxel.points) / len(voxel.points),
            sum(p.z for p in voxel.points) / len(voxel.points)
        ])
        
        for j, other_voxel in enumerate(elevated_voxels[i+1:], i+1):
            if j in processed:
                continue
                
            other_center = np.array([
                sum(p.x for p in other_voxel.points) / len(other_voxel.points),
                sum(p.y for p in other_voxel.points) / len(other_voxel.points),
                sum(p.z for p in other_voxel.points) / len(other_voxel.points)
            ])
            
            # 距离检查 (简化的邻接检查)
            distance = np.linalg.norm(voxel_center - other_center)
            if distance < 3.0:  # 3米内认为是连续的
                segment_points.extend(other_voxel.points)
                processed.add(j)
        
        # 创建线段
        if len(segment_points) >= 5:  # 至少5个点才构成线段
            # 计算线段的方向和长度
            coords = np.array([[p.x, p.y, p.z] for p in segment_points])
            centroid = np.mean(coords, axis=0)
            
            # 简单的长度计算
            distances = np.linalg.norm(coords - centroid, axis=1)
            length = np.max(distances) * 2  # 近似长度
            
            if length > 2.0:  # 长度至少2米
                segment = PowerLineSegment(
                    points=segment_points,
                    confidence=min(1.0, len(segment_points) / 10.0)  # 基于点数的置信度
                )
                power_line_segments.append(segment)
        
        if len(power_line_segments) % 10 == 0 and len(power_line_segments) > 0:
            print(f"已提取 {len(power_line_segments)} 个线段...")
    
    return power_line_segments

def main():
    # 加载和预处理 (重复之前的步骤)
    print("加载点云数据...")
    points = load_point_cloud("./001Combined.txt")
    print(f"加载了 {len(points)} 个点")
    
    preprocessor = PointCloudPreprocessor(
        grid_size_2d=5.0,
        voxel_size_3d=0.5,
        noise_k_neighbors=20,
        noise_std_multiplier=2.0
    )
    
    denoised_points = preprocessor.remove_noise(points)
    spatial_grid = preprocessor.generate_spatial_hash_grid(denoised_points)
    height_divider = preprocessor.determine_power_line_height_divider(denoised_points)
    
    print(f"高度分割器: {height_divider:.2f}m")
    
    # 特征计算和线性体素提取
    feature_engine = FeatureCalculationEngine(grid_size_2d=5.0, voxel_size_3d=0.5)
    
    all_voxels = spatial_grid.get_all_3d_voxels()
    linear_voxels = []
    
    for voxel_key, voxel in all_voxels.items():
        if len(voxel.points) >= 3:
            features_3d = feature_engine.calculate_3d_dimensional_features(voxel)
            if voxel.is_linear:
                linear_voxels.append(voxel)
    
    print(f"线性体素数量: {len(linear_voxels)}")
    
    # 简化的线段提取
    power_line_segments = simple_line_segment_extraction(linear_voxels, height_divider)
    
    print(f"\n提取结果:")
    print(f"输电线段数量: {len(power_line_segments)}")
    
    if power_line_segments:
        print(f"前5个线段的详细信息:")
        for i, segment in enumerate(power_line_segments[:5]):
            print(f"  线段 {i+1}: {len(segment.points)} 个点, 置信度: {segment.confidence:.3f}")
            
            # 计算线段的空间范围
            x_coords = [p.x for p in segment.points]
            y_coords = [p.y for p in segment.points]
            z_coords = [p.z for p in segment.points]
            
            print(f"    X范围: {min(x_coords):.2f} - {max(x_coords):.2f}m")
            print(f"    Y范围: {min(y_coords):.2f} - {max(y_coords):.2f}m") 
            print(f"    Z范围: {min(z_coords):.2f} - {max(z_coords):.2f}m")

if __name__ == "__main__":
    main()