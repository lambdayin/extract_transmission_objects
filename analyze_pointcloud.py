#!/usr/bin/env python3
"""
分析点云数据的脚本，帮助调试检测算法
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_structures import Point3D
from preprocessing import PointCloudPreprocessor, HeightHistogramAnalyzer

def load_point_cloud(file_path: str) -> List[Point3D]:
    """加载点云数据"""
    points = []
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
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

def analyze_height_distribution(points: List[Point3D]):
    """分析高度分布"""
    heights = [p.z for p in points]
    
    print(f"点云统计信息:")
    print(f"  总点数: {len(points):,}")
    print(f"  高度范围: {min(heights):.2f}m - {max(heights):.2f}m")
    print(f"  高度差: {max(heights) - min(heights):.2f}m")
    print(f"  平均高度: {np.mean(heights):.2f}m")
    print(f"  高度标准差: {np.std(heights):.2f}m")
    
    # 创建高度直方图
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(heights, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('高度 (m)')
    plt.ylabel('点数量')
    plt.title('高度分布直方图')
    plt.grid(True, alpha=0.3)
    
    # 分析高度层次
    analyzer = HeightHistogramAnalyzer(bin_size=2.0)  # 2米为一个bin
    analysis = analyzer.analyze_height_distribution(points)
    
    print(f"\n高度分析结果:")
    print(f"  建议的高度分割器: {analysis.get('height_divider', 'N/A'):.2f}m")
    print(f"  检测到的峰值数量: {len(analysis.get('peaks', []))}")
    
    if analysis.get('peaks'):
        print("  峰值位置:")
        for i, (height, count) in enumerate(analysis['peaks']):
            print(f"    峰值 {i+1}: {height:.2f}m (点数: {count})")
    
    # 绘制带峰值标记的直方图
    plt.subplot(2, 2, 2)
    hist, bin_edges = analysis['histogram'], analysis['bin_edges']
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.bar(bin_centers, hist, width=bin_centers[1]-bin_centers[0], alpha=0.7, edgecolor='black')
    
    # 标记峰值
    if analysis.get('peaks'):
        for height, count in analysis['peaks']:
            plt.axvline(x=height, color='red', linestyle='--', alpha=0.8)
            plt.text(height, count, f'{height:.1f}m', rotation=90, va='bottom')
    
    # 标记高度分割器
    if 'height_divider' in analysis:
        plt.axvline(x=analysis['height_divider'], color='green', linestyle='-', linewidth=2)
        plt.text(analysis['height_divider'], max(hist)*0.8, 
                f'分割器\n{analysis["height_divider"]:.1f}m', 
                ha='center', va='center', bbox=dict(boxstyle='round', facecolor='lightgreen'))
    
    plt.xlabel('高度 (m)')
    plt.ylabel('点数量')
    plt.title('高度分布分析 (2m bins)')
    plt.grid(True, alpha=0.3)
    
    return analysis

def analyze_spatial_distribution(points: List[Point3D]):
    """分析空间分布"""
    x_coords = [p.x for p in points]
    y_coords = [p.y for p in points]
    z_coords = [p.z for p in points]
    
    print(f"\n空间分布统计:")
    print(f"  X范围: {min(x_coords):.2f} - {max(x_coords):.2f}m (跨度: {max(x_coords)-min(x_coords):.2f}m)")
    print(f"  Y范围: {min(y_coords):.2f} - {max(y_coords):.2f}m (跨度: {max(y_coords)-min(y_coords):.2f}m)")
    print(f"  Z范围: {min(z_coords):.2f} - {max(z_coords):.2f}m (跨度: {max(z_coords)-min(z_coords):.2f}m)")
    
    # XY平面投影
    plt.subplot(2, 2, 3)
    plt.scatter(x_coords[::10], y_coords[::10], c=z_coords[::10], s=1, cmap='viridis', alpha=0.6)
    plt.colorbar(label='高度 (m)')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('XY平面投影 (按高度着色)')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # 高度层次分析
    plt.subplot(2, 2, 4)
    # 将点按高度分层显示
    z_min, z_max = min(z_coords), max(z_coords)
    z_range = z_max - z_min
    
    # 分为三层：地面、中间、高空
    ground_height = z_min + z_range * 0.2
    mid_height = z_min + z_range * 0.7
    
    ground_points = [(p.x, p.y) for p in points if p.z <= ground_height]
    mid_points = [(p.x, p.y) for p in points if ground_height < p.z <= mid_height]
    high_points = [(p.x, p.y) for p in points if p.z > mid_height]
    
    if ground_points:
        x_g, y_g = zip(*ground_points)
        plt.scatter(x_g[::5], y_g[::5], c='brown', s=1, alpha=0.6, label=f'地面层 (<{ground_height:.1f}m)')
    
    if mid_points:
        x_m, y_m = zip(*mid_points)
        plt.scatter(x_m[::5], y_m[::5], c='orange', s=1, alpha=0.6, label=f'中间层 ({ground_height:.1f}-{mid_height:.1f}m)')
    
    if high_points:
        x_h, y_h = zip(*high_points)
        plt.scatter(x_h[::5], y_h[::5], c='red', s=1, alpha=0.8, label=f'高空层 (>{mid_height:.1f}m)')
    
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('按高度分层显示')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig('pointcloud_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'ground_points': len(ground_points),
        'mid_points': len(mid_points),
        'high_points': len(high_points),
        'ground_height_threshold': ground_height,
        'mid_height_threshold': mid_height
    }

def analyze_preprocessing_results(points: List[Point3D]):
    """分析预处理结果"""
    print(f"\n=== 预处理分析 ===")
    
    preprocessor = PointCloudPreprocessor(
        grid_size_2d=5.0,
        voxel_size_3d=0.5,
        noise_k_neighbors=20,
        noise_std_multiplier=2.0
    )
    
    # 噪声移除
    print("执行噪声移除...")
    denoised_points = preprocessor.remove_noise(points)
    print(f"噪声移除: {len(points)} -> {len(denoised_points)} 点 (移除了 {len(points) - len(denoised_points)} 点)")
    
    # 空间网格生成
    print("生成空间哈希网格...")
    spatial_grid = preprocessor.generate_spatial_hash_grid(denoised_points)
    
    print(f"生成的网格统计:")
    print(f"  2D网格数量: {spatial_grid.get_grid_count()}")
    print(f"  3D体素数量: {spatial_grid.get_voxel_count()}")
    
    # 高度分割器确定
    height_divider = preprocessor.determine_power_line_height_divider(denoised_points)
    print(f"  建议高度分割器: {height_divider:.2f}m")
    
    return {
        'original_points': len(points),
        'denoised_points': len(denoised_points),
        'num_grids': spatial_grid.get_grid_count(),
        'num_voxels': spatial_grid.get_voxel_count(),
        'height_divider': height_divider
    }

def main():
    # 分析点云文件
    file_path = "./001Combined.txt"
    
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return
    
    print("加载点云数据...")
    points = load_point_cloud(file_path)
    
    if not points:
        print("错误: 无法加载点云数据")
        return
    
    # 执行各种分析
    height_analysis = analyze_height_distribution(points)
    spatial_analysis = analyze_spatial_distribution(points)
    preprocessing_analysis = analyze_preprocessing_results(points)
    
    # 总结分析结果
    print(f"\n=== 诊断总结 ===")
    print(f"1. 数据规模: {len(points):,} 个点")
    print(f"2. 高度跨度: {max(p.z for p in points) - min(p.z for p in points):.2f}m")
    print(f"3. 建议高度分割器: {height_analysis.get('height_divider', 'N/A'):.2f}m")
    print(f"4. 检测到的高度峰值: {len(height_analysis.get('peaks', []))} 个")
    print(f"5. 高空点数量: {spatial_analysis['high_points']:,} ({spatial_analysis['high_points']/len(points)*100:.1f}%)")
    
    # 问题诊断
    print(f"\n=== 可能的问题 ===")
    
    if height_analysis.get('height_divider', 0) > 1900:  # 非常高的分割器
        print("⚠️  高度分割器设置过高，可能无法正确分离输电线和地面对象")
    
    if len(height_analysis.get('peaks', [])) < 2:
        print("⚠️  检测到的高度峰值过少，可能输电线不明显")
    
    if spatial_analysis['high_points'] < len(points) * 0.05:  # 高空点少于5%
        print("⚠️  高空点数量过少，可能输电线密度不足")
    
    preprocessing_noise_ratio = (preprocessing_analysis['original_points'] - preprocessing_analysis['denoised_points']) / preprocessing_analysis['original_points']
    if preprocessing_noise_ratio > 0.1:  # 超过10%的点被当作噪声移除
        print(f"⚠️  噪声移除比例过高 ({preprocessing_noise_ratio*100:.1f}%)，可能移除了有用的输电线点")

if __name__ == "__main__":
    main()