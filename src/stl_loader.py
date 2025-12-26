"""
STL Loader Module
Handles STL file loading and point cloud preparation.
"""

import numpy as np
import open3d as o3d
from typing import Tuple


def load_stl(stl_path: str) -> o3d.geometry.TriangleMesh:
    """
    Load STL file using Open3D.
    
    Args:
        stl_path: Path to STL file
        
    Returns:
        open3d.geometry.TriangleMesh: Loaded mesh
        
    Raises:
        FileNotFoundError: If STL file doesn't exist
        RuntimeError: If mesh cannot be loaded
    """
    import os
    
    if not os.path.exists(stl_path):
        raise FileNotFoundError(f"STL file not found: {stl_path}")
    
    print(f"  Loading mesh from {os.path.basename(stl_path)}...")
    
    mesh = o3d.io.read_triangle_mesh(stl_path)
    
    if not mesh.has_vertices():
        raise RuntimeError(f"Failed to load mesh from {stl_path}")
    
    print(f"  Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    
    return mesh


def prepare_pointcloud_for_icp(
    mesh: o3d.geometry.TriangleMesh,
    target_points: int = 8000
) -> o3d.geometry.PointCloud:
    """
    Prepare point cloud from mesh for ICP registration.
    
    Steps:
    1. Sample points uniformly from mesh
    2. Remove statistical outliers
    3. Estimate normals
    4. Orient normals consistently
    
    Args:
        mesh: Input triangle mesh
        target_points: Target number of points (default: 8000)
        
    Returns:
        open3d.geometry.PointCloud: Processed point cloud with normals
    """
    print(f"  Sampling {target_points} points uniformly...")
    
    # Uniform sampling
    pcd = mesh.sample_points_uniformly(number_of_points=target_points)
    
    # Remove small isolated clusters and outliers
    print("  Removing statistical outliers...")
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=20,
        std_ratio=2.0
    )
    
    print(f"  After outlier removal: {len(pcd.points)} points")
    
    # Estimate normals
    print("  Estimating normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=2.0,  # 2mm radius for dental scans
            max_nn=30
        )
    )
    
    # Orient normals consistently toward centroid (outward facing)
    print("  Orienting normals consistently...")
    orient_normals_consistently(pcd)
    
    return pcd


def orient_normals_consistently(pcd: o3d.geometry.PointCloud) -> None:
    """
    Orient normals consistently toward the outside (away from centroid).
    
    Args:
        pcd: Point cloud with normals (modified in-place)
    """
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    
    # Compute centroid
    centroid = np.mean(points, axis=0)
    
    # Vectors from centroid to each point
    to_points = points - centroid
    
    # Check if normals point outward (dot product should be positive)
    dot_products = np.sum(normals * to_points, axis=1)
    
    # Flip normals that point inward
    flip_mask = dot_products < 0
    normals[flip_mask] *= -1
    
    pcd.normals = o3d.utility.Vector3dVector(normals)
    
    num_flipped = np.sum(flip_mask)
    print(f"    Flipped {num_flipped} normals to point outward")
