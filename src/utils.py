"""
Utilities Module
Helper functions for metrics calculation and result saving.
"""

import json
import numpy as np
import open3d as o3d
from typing import Dict, Tuple


def compute_rmse(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud
) -> float:
    """
    Compute RMSE (Root Mean Square Error) between two point clouds.
    
    Uses nearest neighbor distance from source to target.
    
    Args:
        source: Source point cloud
        target: Target point cloud
        
    Returns:
        RMSE value in mm
    """
    source_points = np.asarray(source.points)
    target_points = np.asarray(target.points)
    
    # Build KD-tree for target
    target_tree = o3d.geometry.KDTreeFlann(target)
    
    # Find nearest neighbor distances
    distances = []
    for point in source_points:
        [_, idx, dist] = target_tree.search_knn_vector_3d(point, 1)
        distances.append(np.sqrt(dist[0]))
    
    # Compute RMSE
    rmse = np.sqrt(np.mean(np.array(distances) ** 2))
    
    return rmse


def compute_inlier_ratio(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    threshold: float = 1.0
) -> float:
    """
    Compute inlier ratio (percentage of points within threshold).
    
    Args:
        source: Source point cloud
        target: Target point cloud
        threshold: Distance threshold in mm (default: 1.0mm)
        
    Returns:
        Inlier ratio (0.0 to 1.0)
    """
    source_points = np.asarray(source.points)
    target_points = np.asarray(target.points)
    
    # Build KD-tree for target
    target_tree = o3d.geometry.KDTreeFlann(target)
    
    # Count inliers
    inlier_count = 0
    for point in source_points:
        [_, idx, dist] = target_tree.search_knn_vector_3d(point, 1)
        if np.sqrt(dist[0]) <= threshold:
            inlier_count += 1
    
    inlier_ratio = inlier_count / len(source_points)
    
    return inlier_ratio


def evaluate_registration_quality(
    rmse: float,
    inlier_ratio: float
) -> str:
    """
    Evaluate registration quality based on clinical thresholds.
    
    Thresholds:
    - Highly Reliable: RMSE < 0.8mm AND inlier_ratio > 70%
    - Acceptable: RMSE < 1.5mm AND inlier_ratio > 50%
    - Failed: Otherwise
    
    Args:
        rmse: RMSE in mm
        inlier_ratio: Inlier ratio (0.0 to 1.0)
        
    Returns:
        Quality classification string
    """
    if rmse < 0.8 and inlier_ratio > 0.70:
        return "Highly Reliable ✓"
    elif rmse < 1.5 and inlier_ratio > 0.50:
        return "Acceptable (recommend visual inspection)"
    else:
        return "Failed ✗ (please re-select landmarks)"


def save_transform(
    matrix: np.ndarray,
    output_dir: str,
    prefix: str = "transformation"
) -> None:
    """
    Save transformation matrix in both .txt and .json formats.
    
    Args:
        matrix: 4x4 transformation matrix
        output_dir: Output directory
        prefix: Filename prefix (default: 'transformation')
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as human-readable text
    txt_path = os.path.join(output_dir, f"{prefix}_matrix.txt")
    np.savetxt(txt_path, matrix, fmt='%.6f', header='4x4 Transformation Matrix')
    
    # Save as JSON for programmatic use
    json_path = os.path.join(output_dir, f"{prefix}_matrix.json")
    transform_dict = {
        "matrix": matrix.tolist(),
        "rotation": matrix[:3, :3].tolist(),
        "translation": matrix[:3, 3].tolist()
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(transform_dict, f, indent=2, ensure_ascii=False)
    
    print(f"  Saved transformation matrix:")
    print(f"    - {txt_path}")
    print(f"    - {json_path}")


def print_registration_report(
    rmse: float,
    inlier_ratio: float,
    quality: str,
    landmark_count: int,
    icp_fitness: float,
    icp_rmse: float
) -> None:
    """
    Print a formatted registration quality report.
    
    Args:
        rmse: Overall RMSE
        inlier_ratio: Inlier ratio
        quality: Quality classification
        landmark_count: Number of landmark pairs used
        icp_fitness: ICP fitness score
        icp_rmse: ICP inlier RMSE
    """
    print("\n" + "=" * 60)
    print("REGISTRATION QUALITY REPORT")
    print("=" * 60)
    print(f"Quality Classification: {quality}")
    print(f"Landmarks Used:         {landmark_count} pairs")
    print(f"\nOverall Metrics:")
    print(f"  RMSE:                 {rmse:.3f} mm")
    print(f"  Inlier Ratio:         {inlier_ratio*100:.1f}% (within 1.0mm)")
    print(f"\nICP Refinement:")
    print(f"  ICP Fitness:          {icp_fitness:.4f}")
    print(f"  ICP Inlier RMSE:      {icp_rmse:.3f} mm")
    print("=" * 60 + "\n")
